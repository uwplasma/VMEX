"""Free-boundary solve: NESTOR vacuum coupling around the core solver.

Implements the ``funct3d.f`` free-boundary block (VMEC2000) on top of the
fixed-boundary iteration of :mod:`vmec_jax.core.solver`:

- **Activation**: ``ivac`` starts at -1 and increments on every iteration
  with ``iter2 > 1`` and ``fsqr + fsqz <= 1e-3``; the first vacuum call
  promotes ``ivac`` 0 -> 1 (``vacuum.f``), prints the ``In VACUUM`` block,
  and triggers the soft-start restart (``restart_iter`` with ``irst = 2``:
  state <- best stored state, zero velocity, ``delt *= 0.9``,
  ``iter1 = iter2``, ``ijacob += 1``); ``eqsolve.f`` then prints the
  ``VACUUM PRESSURE TURNED ON`` banner and sets ``ivac = 2``.
- **Cadence**: ``ivacskip = mod(iter2 - iter1, nvacskip)`` (forced 0 while
  ``ivac <= 2``); on full steps (``ivacskip == 0``) the Green-function
  kernel/matrix is rebuilt and ``nvacskip = max(nvskip0,
  1/max(0.1, 1e11*(fsqr+fsqz)))``; on skip steps only the analytic source is
  refreshed against the cached matrix (``scalpot.f``).
- **Edge force**: ``bsqvac + presf(ns)`` enters the R/Z edge force rows via
  the :class:`~vmec_jax.core.solver.SolverRuntime` free-boundary seam
  (``lfreeb/bsqvac_edge/presf_ns_scale`` — see ``solver._evaluate``), the
  edge row is evolved (``jmax = ns``) and the ``rcon0/zcon0`` constraint
  baselines are damped by 0.9 per active iteration (``funct3d.f``).

The iteration itself runs the *same* traced body as the fixed-boundary
lanes, one jitted iteration per host step so the vacuum field, cadence
counters, and screen printing can interleave exactly like ``eqsolve.f``.

Known divergence from VMEC2000 (documented): at turn-on VMEC computes the
turn-on iteration's forces from the pre-restart geometry while evolving the
restored state; here the restart is applied *before* the turn-on iteration,
so that iteration's forces come from the restored state.  The golden
free-boundary fixture is chaotic/unconverged past turn-on, so trajectories
are compared structurally, not pointwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp

from . import profiles as _profiles
from .errors import MORE_ITER_FLAG, SUCCESSFUL_TERM_FLAG
from .fields import magnetic_fields, metric_elements
from .fourier import ModeTable
from .geometry import half_mesh_jacobian
from .input import VmecInput
from .mgrid import MgridField
from .printing import (
    FORCE_ITERATIONS_BANNER, screen_header, screen_line, stage_banner,
    vacuum_banner,
)
from .solver import (
    SolveResult, SolverRuntime, SpectralState,
    _finalize, _geometry, _initial_carry, _initial_state, _make_body,
    _result_from_carry, _zero_cache, prepare_runtime, resolution_from_input,
)
from .vacuum import (
    VacuumBasis, VacuumBoundary, make_vacuum_solver, vacuum_basis,
    vacuum_channels,
)

__all__ = [
    "FreeBoundaryState",
    "boundary_from_coefficients",
    "solve_free_boundary",
]

Array = Any
MU0 = 4.0e-7 * np.pi

#: funct3d.f vacuum activation threshold on fsqr + fsqz.
ACTIVATION_FSQ = 1.0e-3


# ---------------------------------------------------------------------------
# Boundary surface synthesis (NESTOR surface.f)
# ---------------------------------------------------------------------------


def boundary_from_coefficients(
    *,
    rmnc: np.ndarray,
    zmns: np.ndarray,
    rmns: np.ndarray | None,
    zmnc: np.ndarray | None,
    modes: ModeTable,
    basis: VacuumBasis,
) -> VacuumBoundary:
    """Sample the boundary surface on the NESTOR grid (``surface.f``).

    ``rmnc``... are wout-convention edge coefficients over the signed
    ``modes`` table.  Angles: ``theta/zeta`` from ``basis`` (per-period
    ``zeta``); ``xn = n*nfp`` so all v-derivatives are geometric-phi
    derivatives, exactly as ``surface.f``.
    """
    xm = np.asarray(modes.m, dtype=float)
    xn = np.asarray(modes.n, dtype=float) * float(basis.nfp)
    th = np.asarray(basis.theta, dtype=float)[:, None]
    ze = np.asarray(basis.zeta, dtype=float)[:, None]
    arg = th * xm[None, :] - ze * xn[None, :]
    cosmn = np.cos(arg)
    sinmn = np.sin(arg)

    rc = np.asarray(rmnc, dtype=float)
    zs = np.asarray(zmns, dtype=float)
    R = cosmn @ rc
    Z = sinmn @ zs
    Ru = -(sinmn * xm[None, :]) @ rc
    Rv = (sinmn * xn[None, :]) @ rc
    Zu = (cosmn * xm[None, :]) @ zs
    Zv = -(cosmn * xn[None, :]) @ zs
    ruu = -(cosmn * (xm * xm)[None, :]) @ rc
    ruv = (cosmn * (xm * xn)[None, :]) @ rc
    rvv = -(cosmn * (xn * xn)[None, :]) @ rc
    zuu = -(sinmn * (xm * xm)[None, :]) @ zs
    zuv = (sinmn * (xm * xn)[None, :]) @ zs
    zvv = -(sinmn * (xn * xn)[None, :]) @ zs
    if rmns is not None and zmnc is not None:
        rs = np.asarray(rmns, dtype=float)
        zc = np.asarray(zmnc, dtype=float)
        R = R + sinmn @ rs
        Z = Z + cosmn @ zc
        Ru = Ru + (cosmn * xm[None, :]) @ rs
        Rv = Rv - (cosmn * xn[None, :]) @ rs
        Zu = Zu - (sinmn * xm[None, :]) @ zc
        Zv = Zv + (sinmn * xn[None, :]) @ zc
        ruu = ruu - (sinmn * (xm * xm)[None, :]) @ rs
        ruv = ruv + (sinmn * (xm * xn)[None, :]) @ rs
        rvv = rvv - (sinmn * (xn * xn)[None, :]) @ rs
        zuu = zuu - (cosmn * (xm * xm)[None, :]) @ zc
        zuv = zuv + (cosmn * (xm * xn)[None, :]) @ zc
        zvv = zvv - (cosmn * (xn * xn)[None, :]) @ zc

    shape = (int(basis.ntheta3), int(basis.nzeta))
    return VacuumBoundary(
        R=R.reshape(shape), Z=Z.reshape(shape),
        Ru=Ru.reshape(shape), Zu=Zu.reshape(shape),
        Rv=Rv.reshape(shape), Zv=Zv.reshape(shape),
        ruu=ruu.reshape(shape), ruv=ruv.reshape(shape), rvv=rvv.reshape(shape),
        zuu=zuu.reshape(shape), zuv=zuv.reshape(shape), zvv=zvv.reshape(shape),
    )


def _edge_fourier(state: SpectralState, rt: SolverRuntime):
    """Edge-row wout-convention coefficients (``convert.f`` before vacuum)."""
    from .residuals import m1_constrained_to_physical
    from .transforms import physical_to_internal_scale

    setup = rt.setup
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos,
        modes=rt.modes, lthreed=setup.lthreed, lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = np.asarray(R_cos)[-1] * scale
    zmns = np.asarray(Z_sin)[-1] * scale
    if setup.lasym:
        rmns = np.asarray(R_sin)[-1] * scale
        zmnc = np.asarray(Z_cos)[-1] * scale
    else:
        rmns = zmnc = None
    return rmnc, zmns, rmns, zmnc


# ---------------------------------------------------------------------------
# Axis-filament plasma-current field (tolicu.f + belicu.f)
# ---------------------------------------------------------------------------


def axis_current_field(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    nfp: int,
    plascur: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Biot-Savart field of the net toroidal current on the magnetic axis.

    Port of the legacy parity-proven ``axis_current_field_vmec_filament``
    (VMEC ``tolicu.f`` axis filament across field periods + LIBSTELL
    ``bsc_b`` segment kernel with ``eps_sq`` regularization).  ``plascur``
    is VMEC's ``ctor`` (mu0*A, ``bcovar.f`` sign convention); the filament
    current is ``+plascur/mu0`` exactly as ``tolicu.f`` (the legacy port
    used the opposite sign because its ``plascur_edge_from_bcovar`` carried
    ``-signgs`` instead of ``bcovar.f``'s ``+signgs``).
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    axis_r = np.asarray(axis_r, dtype=float).reshape(-1)
    axis_z = np.asarray(axis_z, dtype=float).reshape(-1)
    ntheta, nv = R.shape
    current = float(plascur) / MU0
    if (not np.isfinite(current)) or current == 0.0:
        z = np.zeros_like(R)
        return z, z, z

    nfper = max(1, int(nfp))
    nvper = 64 if nv == 1 else nfper
    alvp = (2.0 * np.pi / float(max(1, nv))) / float(nfper)
    cosuv = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(nvper)
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    # tolicu.f: axis points over all periods (loop closed below).
    x0 = axis_r[None, :] * cosuv[None, :]
    y0 = axis_r[None, :] * sinuv[None, :]
    xpts = np.zeros((3, nvper * nv), dtype=float)
    for kper in range(nvper):
        sl = slice(kper * nv, (kper + 1) * nv)
        xpts[0, sl] = cosper[kper] * x0 - sinper[kper] * y0
        xpts[1, sl] = sinper[kper] * x0 + cosper[kper] * y0
        xpts[2, sl] = axis_z
    # bsc_construct('fil_loop'): drop zero-length segments, close the loop.
    keep = [0]
    for i in range(1, xpts.shape[1]):
        d = xpts[:, keep[-1]] - xpts[:, i]
        if float(d @ d) != 0.0:
            keep.append(i)
    xnod = xpts[:, keep]
    if float((xnod[:, -1] - xpts[:, 0]) @ (xnod[:, -1] - xpts[:, 0])) != 0.0:
        xnod = np.concatenate([xnod, xpts[:, :1]], axis=1)
    if xnod.shape[1] < 2:
        z = np.zeros_like(R)
        return z, z, z

    dxnod = xnod[:, 1:] - xnod[:, :-1]
    lsqnod = np.sum(dxnod * dxnod, axis=0)
    eps_sq = max(np.finfo(float).eps * float(np.min(lsqnod[lsqnod > 0.0])), np.finfo(float).tiny)

    cos1 = np.broadcast_to(cosuv[None, :], (ntheta, nv)).reshape(-1)
    sin1 = np.broadcast_to(sinuv[None, :], (ntheta, nv)).reshape(-1)
    rp = R.reshape(-1)
    xobs = np.stack([rp * cos1, rp * sin1, Z.reshape(-1)], axis=1)

    capRv = xobs[:, None, :] - xnod.T[None, :, :]
    capR = np.sqrt(np.maximum(eps_sq, np.sum(capRv * capRv, axis=2)))
    R1p2 = capR[:, :-1] + capR[:, 1:]
    denom = np.maximum(R1p2 * R1p2 - lsqnod[None, :], eps_sq)
    Rfactor = 2.0 * R1p2 / (capR[:, :-1] * capR[:, 1:] * denom)
    crossv = np.cross(dxnod.T[None, :, :], capRv[:, :-1, :])
    bxyz = (current * 1.0e-7) * np.sum(crossv * Rfactor[:, :, None], axis=1)

    br = cos1 * bxyz[:, 0] + sin1 * bxyz[:, 1]
    bp = -sin1 * bxyz[:, 0] + cos1 * bxyz[:, 1]
    return br.reshape((ntheta, nv)), bp.reshape((ntheta, nv)), bxyz[:, 2].reshape((ntheta, nv))


# ---------------------------------------------------------------------------
# External-field projection (bextern.f)
# ---------------------------------------------------------------------------


def external_field_channels(
    *,
    boundary: VacuumBoundary,
    br: np.ndarray,
    bp: np.ndarray,
    bz: np.ndarray,
    basis: VacuumBasis,
    signgs: int,
) -> dict[str, np.ndarray]:
    """``bextern.f``: covariant components, normal source, and metric.

    Returns ``bexu/bexv`` (covariant, geometric-phi convention), ``bexni``
    (the weighted normal source ``-B.n * wint * (2*pi)^2``), and the
    physical surface metric ``guu/guv/gvv``.
    """
    R = np.asarray(boundary.R, dtype=float)
    Ru = np.asarray(boundary.Ru, dtype=float)
    Zu = np.asarray(boundary.Zu, dtype=float)
    Rv = np.asarray(boundary.Rv, dtype=float)
    Zv = np.asarray(boundary.Zv, dtype=float)
    sgn = float(int(signgs))
    snr = sgn * R * Zu
    snv = sgn * (Ru * Zv - Rv * Zu)
    snz = -sgn * R * Ru
    bexu = Ru * br + Zu * bz
    bexv = Rv * br + Zv * bz + R * bp
    bexn = -(br * snr + bp * snv + bz * snz)
    wint2 = np.asarray(basis.wint, dtype=float).reshape(R.shape)
    bexni = bexn * wint2 * ((2.0 * np.pi) ** 2)
    return {
        "bexu": bexu,
        "bexv": bexv,
        "bexn": bexn,
        "bexni": bexni,
        "guu": Ru * Ru + Zu * Zu,
        "guv": Ru * Rv + Zu * Zv,
        "gvv": R * R + Rv * Rv + Zv * Zv,
    }


# ---------------------------------------------------------------------------
# Per-iteration plasma scalars (bcovar.f tails consumed by vacuum)
# ---------------------------------------------------------------------------


@jax.jit
def _vacuum_scalars(state: SpectralState, rt: SolverRuntime):
    """``(ctor, rbtor, axis_r, axis_z, bsq_edge_extrap, pres_edge)``.

    - ``ctor = signgs*2*pi*(1.5*buco(ns) - 0.5*buco(ns-1))`` with
      ``buco = <B_u>`` (``bcovar.f``/``calc_fbal``);
    - ``rbtor = 1.5*bvco(ns) - 0.5*bvco(ns-1)``;
    - ``axis_r/axis_z``: ``r1/z1(js=1, theta=0, :)`` — the ``raxis_nestor``
      arrays of ``funct3d.f``;
    - ``bsq_edge_extrap = 1.5*bsq(ns) - 0.5*bsq(ns-1)`` on the angular grid
      (``bsqsav(:,3)`` for the DEL-BSQ diagnostic).
    """
    setup = rt.setup
    s = setup.s_full
    _, geometry = _geometry(state, rt)
    jacobian = half_mesh_jacobian(geometry, s=s)
    metrics = metric_elements(geometry, s=s)
    fields = magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig,
        s=s, phips=setup.phips, phipf=setup.phipf, chips=setup.chips,
        signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv,
    )
    wint = jnp.asarray(rt.weights)  # (ntheta_eff,), zeta-constant wint
    buco = jnp.sum(fields.bsubu * wint[None, :, None], axis=(1, 2))
    bvco = jnp.sum(fields.bsubv * wint[None, :, None], axis=(1, 2))
    sgn = jnp.asarray(float(setup.signgs))
    ctor = sgn * (2.0 * jnp.pi) * (1.5 * buco[-1] - 0.5 * buco[-2])
    rbtor = 1.5 * bvco[-1] - 0.5 * bvco[-2]
    axis_r = geometry.R_even[0, 0, :]
    axis_z = geometry.Z_even[0, 0, :]
    bsq = fields.total_pressure
    bsq_edge_extrap = 1.5 * bsq[-1] - 0.5 * bsq[-2]
    return ctor, rbtor, axis_r, axis_z, bsq_edge_extrap, fields.pressure[-1]


@jax.jit
def _iter_lane(carry, rt: SolverRuntime):
    """One jitted eqsolve iteration (shared traced body; per-``rt`` lane)."""
    return _make_body(rt)(carry)


# ---------------------------------------------------------------------------
# Free-boundary driver state
# ---------------------------------------------------------------------------


@dataclass
class FreeBoundaryState:
    """Host cadence state + NESTOR cache (``funct3d.f`` module variables)."""

    ivac: int = -1
    nvacskip: int = 1
    nvskip0: int = 1
    turned_on: bool = False
    banner_pending: bool = False
    delbsq: float = 1.0
    bsqvac: np.ndarray | None = None
    # NESTOR cache (amatsav / bvecsav of scalpot.f):
    mode_matrix: Any = None
    bvec_nonsing: Any = None
    potvac: np.ndarray | None = None
    ctor: float = 0.0
    rbtor: float = 0.0
    vacuum_calls: int = 0
    full_updates: int = 0


def _resolve_mgrid(inp: VmecInput, mgrid_path: str | Path | None) -> Path:
    p = Path(str(mgrid_path if mgrid_path is not None else inp.mgrid_file)).expanduser()
    return p


def _vacuum_step(
    *,
    carry,
    rt: SolverRuntime,
    fb: FreeBoundaryState,
    basis: VacuumBasis,
    solver_vac,
    field: MgridField,
    ivacskip: int,
    emit,
    verbose: bool,
) -> np.ndarray:
    """One NESTOR update (``vacuum.f``): returns ``bsqvac`` on the grid."""
    ctor, rbtor, axis_r, axis_z, bsq3, pres_ns = _vacuum_scalars(carry.state, rt)
    fb.ctor = float(ctor)
    fb.rbtor = float(rbtor)

    rmnc, zmns, rmns, zmnc = _edge_fourier(carry.state, rt)
    boundary = boundary_from_coefficients(
        rmnc=rmnc, zmns=zmns, rmns=rmns, zmnc=zmnc, modes=rt.modes, basis=basis
    )
    shape = (basis.ntheta3, basis.nzeta)
    phi_geom = (np.asarray(basis.zeta) * basis.onp).reshape(shape)
    br_c, bp_c, bz_c = field.b_cyl(np.asarray(boundary.R), phi_geom, np.asarray(boundary.Z))
    br_a, bp_a, bz_a = axis_current_field(
        R=np.asarray(boundary.R), Z=np.asarray(boundary.Z),
        axis_r=np.asarray(axis_r), axis_z=np.asarray(axis_z),
        nfp=basis.nfp, plascur=fb.ctor,
    )
    br = np.asarray(br_c) + br_a
    bp = np.asarray(bp_c) + bp_a
    bz = np.asarray(bz_c) + bz_a
    ext = external_field_channels(
        boundary=boundary, br=br, bp=bp, bz=bz, basis=basis,
        signgs=int(rt.setup.signgs),
    )

    if int(ivacskip) == 0 or fb.mode_matrix is None:
        potvac, mode_matrix, bvec_nonsing, _rhs, _gsrc, _grp = solver_vac.full(
            boundary, jnp.asarray(ext["bexni"])
        )
        fb.mode_matrix = mode_matrix
        fb.bvec_nonsing = bvec_nonsing
        fb.full_updates += 1
    else:
        potvac, _rhs = solver_vac.skip(
            boundary, jnp.asarray(ext["bexni"]), fb.bvec_nonsing, fb.mode_matrix
        )
    fb.potvac = np.asarray(potvac)
    fb.vacuum_calls += 1

    bsqvac, bsubu_s, bsubv_s, _bsupu, _bsupv = vacuum_channels(
        basis=basis, potvac=potvac,
        bexu=jnp.asarray(ext["bexu"]), bexv=jnp.asarray(ext["bexv"]),
        guu=jnp.asarray(ext["guu"]), guv=jnp.asarray(ext["guv"]),
        gvv=jnp.asarray(ext["gvv"]),
    )
    bsqvac = np.asarray(bsqvac)

    if fb.ivac == 0:
        # vacuum.f first-call block: promote ivac and print grid/current info.
        fb.ivac = 1
        wint2 = np.asarray(basis.wint).reshape(shape)
        bsubuvac = float(np.sum(np.asarray(bsubu_s) * wint2)) * float(rt.setup.signgs) * 2.0 * np.pi
        bsubvvac = float(np.sum(np.asarray(bsubv_s) * wint2))
        if verbose:
            emit(
                f"\n  In VACUUM, np = {basis.nfp:2d}  mf = {basis.mf:2d}  nf = {basis.nf:2d}"
                f" nu = {basis.nu_full:2d}  nv = {basis.nzeta:4d}\n"
            )
            fac = 1.0e-6 / MU0
            emit(
                f"  2*pi * a * -BPOL(vac) = {bsubuvac*fac:10.2E}"
                f" TOROIDAL CURRENT = {fb.ctor*fac:10.2E}\n"
                f"  R * BTOR(vac) = {bsubvvac:10.2E}"
                f" R * BTOR(plasma) = {fb.rbtor:10.2E}\n"
            )

    # DEL-BSQ diagnostic (funct3d.f dbsq + printout.f delbsq).
    scale = float(np.asarray(rt.presf_ns_scale)) if rt.presf_ns_scale is not None else 0.0
    gcon_edge = bsqvac + float(pres_ns) * scale
    bsq3 = np.asarray(bsq3)
    wint2 = np.asarray(basis.wint).reshape(shape)
    den = float(np.sum(bsq3 * wint2))
    if den != 0.0:
        fb.delbsq = float(np.sum(np.abs(gcon_edge - bsq3) * wint2) / den)
    fb.bsqvac = bsqvac
    return bsqvac


def _presf_ns_scale(inp: VmecInput, ns: int) -> float:
    """funct3d.f: ``presf_ns = pmass(1)/pmass(hs*(ns-1.5)) * pres(ns)``."""
    hs = 1.0 / float(ns - 1)
    sedge = hs * (float(ns) - 1.5)
    kwargs = dict(pres_scale=float(inp.pres_scale), bloat=float(inp.bloat),
                  spres_ped=1.0)
    p_edge = float(np.asarray(_profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, sedge, **kwargs)))
    if p_edge == 0.0:
        return 0.0
    p_one = float(np.asarray(_profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, 1.0, **kwargs)))
    return p_one / p_edge


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def solve_free_boundary(
    inp: VmecInput,
    *,
    mgrid_path: str | Path | None = None,
    external_field: MgridField | None = None,
    resolution=None,
    ftol: float | None = None,
    max_iterations: int | None = None,
    verbose: bool = False,
    emit=print,
    error_on_no_convergence: bool = True,
) -> SolveResult:
    """Single-grid free-boundary solve (``eqsolve.f`` + ``funct3d.f`` IVAC0).

    ``external_field`` overrides the mgrid file (any
    :class:`~vmec_jax.core.mgrid.MgridField`-compatible object with a
    ``b_cyl(r, phi, z)`` method — e.g. a direct-coil Biot-Savart field).
    Raises :class:`~vmec_jax.core.errors.MgridNotFoundError` when the deck's
    mgrid file is missing and no field is supplied (callers such as the CLI
    implement VMEC2000's warn-and-fall-back-to-fixed-boundary policy).

    ``error_on_no_convergence=False`` returns the final state instead of
    raising when NITER is exhausted (useful against unconverged goldens).
    """
    if not bool(inp.lfreeb):
        raise ValueError("solve_free_boundary requires an LFREEB=T input")
    if external_field is None:
        path = _resolve_mgrid(inp, mgrid_path)
        data_extcur = np.atleast_1d(np.asarray(inp.extcur if inp.extcur is not None else [], dtype=float))
        from .mgrid import read_mgrid

        data = read_mgrid(path)  # raises MgridNotFoundError when missing
        extcur = np.zeros((data.nextcur,), dtype=float)
        n_copy = min(data_extcur.size, data.nextcur)
        extcur[:n_copy] = data_extcur[:n_copy]
        if str(data.mgrid_mode).upper().startswith("R") or str(data.mgrid_mode).upper().startswith("N"):
            raw = np.asarray(data.raw_coil_cur, dtype=float)
            extcur = np.divide(extcur, raw, out=extcur, where=raw != 0.0)
        external_field = MgridField.from_mgrid_data(data, extcur=extcur)

    if resolution is None:
        resolution = resolution_from_input(inp)
    rt = prepare_runtime(inp, resolution, ftol=ftol, max_iterations=max_iterations)
    ns = int(resolution.ns)
    dtype = rt.setup.s_full.dtype

    basis = vacuum_basis(
        mf=int(inp.mpol) + 1, nf=int(inp.ntor),
        ntheta3=int(resolution.ntheta3), nzeta=int(resolution.nzeta),
        nfp=int(resolution.nfp), lasym=bool(resolution.lasym),
        wint=np.asarray(rt.trig.wint, dtype=float),
    )
    solver_vac = make_vacuum_solver(basis, signgs=int(rt.setup.signgs))

    zeros_edge = jnp.zeros((basis.ntheta3, basis.nzeta), dtype=dtype)
    rt_fixed = replace(rt, lfreeb=False, bsqvac_edge=zeros_edge,
                       presf_ns_scale=jnp.asarray(0.0, dtype=dtype))
    rt_freeb = replace(
        rt, lfreeb=True, jmax=ns, bsqvac_edge=zeros_edge,
        presf_ns_scale=jnp.asarray(_presf_ns_scale(inp, ns), dtype=dtype),
    )

    fb = FreeBoundaryState(
        ivac=-1,
        nvacskip=max(1, int(inp.nvacskip)),
        nvskip0=max(1, int(inp.nvacskip)),
    )

    if verbose:
        emit(stage_banner(ns, resolution.mnmax, rt.ftol, rt.max_iterations), end="")
        emit(FORCE_ITERATIONS_BANNER, end="")
        emit(screen_header(lasym=resolution.lasym, lfreeb=True), end="")

    carry = _initial_carry(_initial_state(rt.setup), rt_fixed, ijacob=0)
    printed: set[int] = set()

    def _emit_due(final: bool) -> None:
        if not verbose:
            return
        upto = int(carry.iteration) if bool(carry.done) or final else int(carry.iteration) - 1
        trajectory = np.asarray(carry.trajectory[: max(upto, 0)])
        for it_p in range(1, upto + 1):
            due = (it_p == 1) or (it_p % rt.nstep == 0) or (final and it_p == upto)
            if not due or it_p in printed:
                continue
            row = trajectory[it_p - 1]
            if int(row[0]) != it_p:
                continue
            emit(screen_line(
                it_p, float(row[1]), float(row[2]), float(row[3]),
                float(row[7]), float(row[10]), float(row[9]),
                z_axis=float(row[8]) if resolution.lasym else None,
                del_bsq=float(fb.delbsq),
            ), end="")
            printed.add(it_p)

    max_passes = rt.max_iterations + 400
    for _ in range(max_passes):
        if bool(carry.done):
            break
        it = int(carry.iteration)
        iter1 = int(carry.iter1)
        fsq_rz = float(carry.fsqr) + float(carry.fsqz)

        # -- funct3d.f IVAC0 block (host) -----------------------------------
        if it > 1 and fsq_rz <= ACTIVATION_FSQ:
            fb.ivac += 1
        rt_use = rt_fixed
        if fb.ivac >= 0:
            # Damp the constraint baselines (funct3d: 0.9 per iteration).
            rt_fixed = replace(rt_fixed, rcon0=0.9 * rt_fixed.rcon0, zcon0=0.9 * rt_fixed.zcon0)
            rt_freeb = replace(rt_freeb, rcon0=0.9 * rt_freeb.rcon0, zcon0=0.9 * rt_freeb.zcon0)
            ivacskip = (it - iter1) % max(1, fb.nvacskip)
            if fb.ivac <= 2:
                ivacskip = 0
            if ivacskip == 0:
                fb.nvacskip = max(fb.nvskip0, int(1.0 / max(1.0e-1, 1.0e11 * fsq_rz)))
            bsqvac = _vacuum_step(
                carry=carry, rt=rt_freeb, fb=fb, basis=basis,
                solver_vac=solver_vac, field=external_field,
                ivacskip=ivacskip, emit=emit, verbose=verbose,
            )
            if fb.ivac >= 1 and not fb.turned_on:
                # funct3d.f soft start (restart_iter, irst = 2) applied on
                # the host: best state restored, velocity zeroed, delt*0.9,
                # iter1 = iter2, ijacob += 1.  Divergence from VMEC noted in
                # the module docstring (restart applied before this
                # iteration's force evaluation).
                fb.turned_on = True
                fb.banner_pending = True
                carry = replace(
                    carry,
                    state=carry.xstore,
                    xcdot=jax.tree.map(jnp.zeros_like, carry.xcdot),
                    time_step=carry.time_step * 0.9,
                    ijacob=carry.ijacob + 1,
                    iter1=carry.iteration,
                    # The preconditioner cache changes shape with jmax = ns;
                    # iter1 = iteration forces an immediate ns4 refresh, so
                    # the zeroed cache is never consumed.
                    cache=_zero_cache(rt_freeb),
                )
            if fb.ivac >= 1:
                rt_freeb = replace(rt_freeb, bsqvac_edge=jnp.asarray(bsqvac, dtype=dtype))
                rt_use = rt_freeb

        carry = _iter_lane(carry, rt_use)

        if fb.banner_pending:
            if verbose:
                emit(vacuum_banner(it), end="")
            fb.banner_pending = False
            fb.ivac = max(fb.ivac, 2)  # eqsolve.f: ivac = ivac + 1 after banner
        _emit_due(final=False)

    _emit_due(final=True)
    ier = int(carry.ier)
    if ier == MORE_ITER_FLAG and not error_on_no_convergence:
        result = _result_from_carry(carry, rt_freeb if fb.turned_on else rt_fixed)
        return replace(result, converged=False, ier_flag=MORE_ITER_FLAG)
    if ier == SUCCESSFUL_TERM_FLAG:
        return _result_from_carry(carry, rt_freeb if fb.turned_on else rt_fixed)
    return _finalize(carry, rt_freeb if fb.turned_on else rt_fixed)

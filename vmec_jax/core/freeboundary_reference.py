"""NumPy reference geometry and field channels for the NESTOR boundary.

These one-shot routines mirror VMEC2000 ``surface.f``, ``tolicu.f``, and
``bextern.f``.  The production iteration uses the fused JAX equivalents in
:mod:`vmec_jax.core.freeboundary`; keeping this reference path separate makes
the parity tests readable without mixing host validation code into the driver.
"""

from __future__ import annotations

import numpy as np

from .fourier import ModeTable
from .solver import SolverRuntime, SpectralState
from .vacuum import VacuumBasis, VacuumBoundary

MU0 = 4.0e-7 * np.pi

__all__ = [
    "axis_current_field",
    "boundary_from_coefficients",
    "external_field_channels",
]


def boundary_from_coefficients(
    *,
    rmnc: np.ndarray,
    zmns: np.ndarray,
    rmns: np.ndarray | None,
    zmnc: np.ndarray | None,
    modes: ModeTable,
    basis: VacuumBasis,
) -> VacuumBoundary:
    """Sample the boundary surface on the NESTOR grid (``surface.f``)."""
    xm = np.asarray(modes.m, dtype=float)
    xn = np.asarray(modes.n, dtype=float) * float(basis.nfp)
    th = np.asarray(basis.theta, dtype=float)[:, None]
    # basis.zeta is per-period; geometric phi is zeta / nfp.
    ze = np.asarray(basis.zeta, dtype=float)[:, None] * float(basis.onp)
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
        R=R.reshape(shape),
        Z=Z.reshape(shape),
        Ru=Ru.reshape(shape),
        Zu=Zu.reshape(shape),
        Rv=Rv.reshape(shape),
        Zv=Zv.reshape(shape),
        ruu=ruu.reshape(shape),
        ruv=ruv.reshape(shape),
        rvv=rvv.reshape(shape),
        zuu=zuu.reshape(shape),
        zuv=zuv.reshape(shape),
        zvv=zvv.reshape(shape),
    )


def _edge_fourier(state: SpectralState, rt: SolverRuntime):
    """Return edge-row WOUT coefficients (``convert.f`` before vacuum)."""
    from .residuals import m1_constrained_to_physical
    from .transforms import physical_to_internal_scale

    setup = rt.setup
    R_cos, Z_sin, R_sin, Z_cos = m1_constrained_to_physical(
        state.R_cos,
        state.Z_sin,
        state.R_sin,
        state.Z_cos,
        modes=rt.modes,
        lthreed=setup.lthreed,
        lasym=setup.lasym,
        lconm1=setup.lconm1,
    )
    scale = 1.0 / physical_to_internal_scale(rt.modes, rt.trig)
    rmnc = np.asarray(R_cos)[-1] * scale
    zmns = np.asarray(Z_sin)[-1] * scale
    if setup.lasym:
        return (
            rmnc,
            zmns,
            np.asarray(R_sin)[-1] * scale,
            np.asarray(Z_cos)[-1] * scale,
        )
    return rmnc, zmns, None, None


def axis_current_field(
    *,
    R: np.ndarray,
    Z: np.ndarray,
    axis_r: np.ndarray,
    axis_z: np.ndarray,
    nfp: int,
    plascur: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the Biot-Savart field of VMEC's magnetic-axis filament.

    This is the ``tolicu.f`` axis filament over all field periods followed by
    the LIBSTELL ``bsc_b`` segment kernel. ``plascur`` is ``ctor`` in
    ``mu0*A``; the physical filament current is ``plascur / mu0``.
    """
    R = np.asarray(R, dtype=float)
    Z = np.asarray(Z, dtype=float)
    axis_r = np.asarray(axis_r, dtype=float).reshape(-1)
    axis_z = np.asarray(axis_z, dtype=float).reshape(-1)
    ntheta, nv = R.shape
    current = float(plascur) / MU0
    if not np.isfinite(current) or current == 0.0:
        zeros = np.zeros_like(R)
        return zeros, zeros, zeros

    nfper = max(1, int(nfp))
    nvper = 64 if nv == 1 else nfper
    alvp = (2.0 * np.pi / float(max(1, nv))) / float(nfper)
    cosuv = np.cos(alvp * np.arange(nv, dtype=float))
    sinuv = np.sin(alvp * np.arange(nv, dtype=float))
    alp_per = 2.0 * np.pi / float(nvper)
    cosper = np.cos(alp_per * np.arange(nvper, dtype=float))
    sinper = np.sin(alp_per * np.arange(nvper, dtype=float))

    x0 = axis_r[None, :] * cosuv[None, :]
    y0 = axis_r[None, :] * sinuv[None, :]
    xpts = np.zeros((3, nvper * nv), dtype=float)
    for kper in range(nvper):
        sl = slice(kper * nv, (kper + 1) * nv)
        xpts[0, sl] = cosper[kper] * x0 - sinper[kper] * y0
        xpts[1, sl] = sinper[kper] * x0 + cosper[kper] * y0
        xpts[2, sl] = axis_z

    keep = [0]
    for i in range(1, xpts.shape[1]):
        delta = xpts[:, keep[-1]] - xpts[:, i]
        if float(delta @ delta) != 0.0:
            keep.append(i)
    xnod = xpts[:, keep]
    if float((xnod[:, -1] - xpts[:, 0]) @ (xnod[:, -1] - xpts[:, 0])) != 0.0:
        xnod = np.concatenate([xnod, xpts[:, :1]], axis=1)
    if xnod.shape[1] < 2:
        zeros = np.zeros_like(R)
        return zeros, zeros, zeros

    dxnod = xnod[:, 1:] - xnod[:, :-1]
    lsqnod = np.sum(dxnod * dxnod, axis=0)
    eps_sq = max(
        np.finfo(float).eps * float(np.min(lsqnod[lsqnod > 0.0])),
        np.finfo(float).tiny,
    )
    cos1 = np.broadcast_to(cosuv[None, :], (ntheta, nv)).reshape(-1)
    sin1 = np.broadcast_to(sinuv[None, :], (ntheta, nv)).reshape(-1)
    rp = R.reshape(-1)
    xobs = np.stack([rp * cos1, rp * sin1, Z.reshape(-1)], axis=1)

    cap_rv = xobs[:, None, :] - xnod.T[None, :, :]
    cap_r = np.sqrt(np.maximum(eps_sq, np.sum(cap_rv * cap_rv, axis=2)))
    r1p2 = cap_r[:, :-1] + cap_r[:, 1:]
    denom = np.maximum(r1p2 * r1p2 - lsqnod[None, :], eps_sq)
    factor = 2.0 * r1p2 / (cap_r[:, :-1] * cap_r[:, 1:] * denom)
    cross = np.cross(dxnod.T[None, :, :], cap_rv[:, :-1, :])
    bxyz = (current * 1.0e-7) * np.sum(cross * factor[:, :, None], axis=1)

    br = cos1 * bxyz[:, 0] + sin1 * bxyz[:, 1]
    bp = -sin1 * bxyz[:, 0] + cos1 * bxyz[:, 1]
    shape = (ntheta, nv)
    return br.reshape(shape), bp.reshape(shape), bxyz[:, 2].reshape(shape)


def external_field_channels(
    *,
    boundary: VacuumBoundary,
    br: np.ndarray,
    bp: np.ndarray,
    bz: np.ndarray,
    basis: VacuumBasis,
    signgs: int,
) -> dict[str, np.ndarray]:
    """Project a cylindrical external field into ``bextern.f`` channels."""
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
    wint = np.asarray(basis.wint, dtype=float).reshape(R.shape)
    return {
        "bexu": bexu,
        "bexv": bexv,
        "bexn": bexn,
        "bexni": bexn * wint * ((2.0 * np.pi) ** 2),
        "guu": Ru * Ru + Zu * Zu,
        "guv": Ru * Rv + Zu * Zv,
        "gvv": R * R + Rv * Rv + Zv * Zv,
    }

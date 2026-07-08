"""A/B equivalence tests: ``vmec_jax.core.preconditioner`` vs the legacy port.

Old implementations under test (left untouched; parity-proven vs VMEC2000):

- ``vmec_jax.preconditioner_1d_jax.lambda_preconditioner``      (lamcal.f90)
- ``vmec_jax.preconditioner_1d_jax.rz_preconditioner_matrices`` (precondn.f + scalfor.f assembly)
- ``vmec_jax.preconditioner_1d_jax.rz_preconditioner_apply``    (scalfor.f apply / tridslv)

New implementations:

- ``vmec_jax.core.preconditioner``: ``precondn`` / ``lamcal`` /
  ``scalfor_matrices`` / ``scalfor`` / ``tridiagonal_solve``.

Inputs are (a) synthetic, well-conditioned half-mesh fields with the right
shapes/signs, and (b) real solver states from short fixed-boundary runs of
``examples/data/input.solovev`` and ``examples/data/input.cth_like_fixed_bdy``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from vmec_jax.kernels.tomnsp import TomnspsRZL
from vmec_jax.preconditioner_1d_jax import (
    lambda_preconditioner as old_lambda_preconditioner,
    rz_preconditioner_apply as old_rz_preconditioner_apply,
    rz_preconditioner_matrices as old_rz_preconditioner_matrices,
)

from vmec_jax.core import preconditioner as newp

REPO_ROOT = Path(__file__).resolve().parents[2]

RTOL = 1e-12
ATOL = 1e-13

# (ns, mpol, ntor, ntheta, nzeta, nfp, lasym)
CASES = [
    (13, 6, 0, 18, 1, 1, False),
    (11, 7, 3, 22, 16, 3, False),
    (9, 5, 2, 20, 18, 2, True),
]
CASE_IDS = ["axisym", "nfp3-3d", "nfp2-lasym-3d"]


def _allclose(actual, desired, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(desired), rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Synthetic (well-conditioned) inputs with the VMEC shapes and signs
# ---------------------------------------------------------------------------


def _synthetic_case(case):
    """Build (bc, k, cfg, s, weights) shaped like the solver's bcovar payload."""
    ns, mpol, ntor, ntheta, nzeta, nfp, lasym = case
    lthreed = ntor > 0
    weights = newp.angular_integration_weights(ntheta=ntheta, nzeta=nzeta, lasym=lasym)
    ntheta_eff = int(weights.shape[0])
    shape = (ns, ntheta_eff, nzeta)
    rng = np.random.default_rng(abs(hash(case)) % (2**31))

    def uniform(lo, hi):
        return rng.uniform(lo, hi, size=shape)

    def smallnormal(scale=0.5):
        return scale * rng.standard_normal(size=shape)

    jac = SimpleNamespace(
        # Negative Jacobian, bounded away from zero (VMEC signgs = -1).
        sqrtg=-uniform(0.7, 1.7),
        r12=uniform(0.9, 1.4),
        tau=smallnormal(),
        rs=smallnormal(),
        zs=smallnormal(),
        ru12=smallnormal(),
        zu12=smallnormal(),
    )
    bc = SimpleNamespace(
        jac=jac,
        guu=uniform(0.8, 1.8),
        guv=0.3 * rng.standard_normal(size=shape),
        gvv=uniform(1.5, 2.5),
        bsq=uniform(1.2, 2.2),
        bsupv=uniform(0.6, 1.2),
        lamscale=0.7321,
    )
    k = SimpleNamespace(
        pr1_odd=smallnormal(),
        pz1_odd=smallnormal(),
        pru_even=smallnormal(),
        pru_odd=smallnormal(),
        pzu_even=smallnormal(),
        pzu_odd=smallnormal(),
    )
    cfg = SimpleNamespace(
        mpol=mpol,
        ntor=ntor,
        ntheta=ntheta,
        nzeta=nzeta,
        nfp=nfp,
        lasym=lasym,
        lthreed=lthreed,
    )
    s = np.linspace(0.0, 1.0, ns)
    return bc, k, cfg, s, weights


def _new_precondn_from_bc(bc, k, weights, s):
    """Call the new precondn for the R- and Z-force families (VMEC call sites)."""
    ns = int(np.asarray(s).shape[0])
    delta_s = float(np.asarray(s)[1] - np.asarray(s)[0])
    common = dict(
        r12_half=jnp.asarray(bc.jac.r12)[1:],
        bsq_half=jnp.asarray(bc.bsq)[1:],
        bsupv_half=jnp.asarray(bc.bsupv)[1:],
        sqrt_g_half=jnp.asarray(bc.jac.sqrtg)[1:],
        angular_weight=weights,
        delta_s=delta_s,
        ns=ns,
    )
    # R force <- Z geometry (VMEC: arm/ard/brm/brd/crd).
    coeffs_r = newp.precondn(
        dxds_half=jnp.asarray(bc.jac.zs)[1:],
        dxdu_half=jnp.asarray(bc.jac.zu12)[1:],
        dxdu_even_full=jnp.asarray(k.pzu_even),
        dxdu_odd_full=jnp.asarray(k.pzu_odd),
        x_odd_full=jnp.asarray(k.pz1_odd),
        **common,
    )
    # Z force <- R geometry (VMEC: azm/azd/bzm/bzd).
    coeffs_z = newp.precondn(
        dxds_half=jnp.asarray(bc.jac.rs)[1:],
        dxdu_half=jnp.asarray(bc.jac.ru12)[1:],
        dxdu_even_full=jnp.asarray(k.pru_even),
        dxdu_odd_full=jnp.asarray(k.pru_odd),
        x_odd_full=jnp.asarray(k.pr1_odd),
        **common,
    )
    return coeffs_r, coeffs_z, delta_s


def _new_matrices(coeffs_r, coeffs_z, cfg, ns, delta_s, jmax=None):
    kwargs = dict(
        delta_s=delta_s,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        nfp=int(cfg.nfp),
        ns=ns,
        jmax=jmax,
    )
    mats_r = newp.scalfor_matrices(coeffs_r, stabilize_edge_zc00=False, **kwargs)
    mats_z = newp.scalfor_matrices(coeffs_z, stabilize_edge_zc00=True, **kwargs)
    return mats_r, mats_z


def _random_frzl(rng, ns, mpol, nrange, lthreed, lasym):
    def arr():
        return jnp.asarray(rng.standard_normal((ns, mpol, nrange)))

    return TomnspsRZL(
        frcc=arr(),
        frss=arr() if lthreed else None,
        fzsc=arr(),
        fzcs=arr() if lthreed else None,
        flsc=arr(),
        flcs=arr() if lthreed else None,
        frsc=arr() if lasym else None,
        frcs=arr() if (lthreed and lasym) else None,
        fzcc=arr() if lasym else None,
        fzss=arr() if (lthreed and lasym) else None,
    )


def _r_channels(frzl):
    """R-force channels in the VMEC order (frcc, frss, frsc, frcs)."""
    names = ["frcc", "frss", "frsc", "frcs"]
    return [(name, getattr(frzl, name)) for name in names if getattr(frzl, name) is not None]


def _z_channels(frzl):
    """Z-force channels in the VMEC order (fzsc, fzcs, fzcc, fzss)."""
    names = ["fzsc", "fzcs", "fzcc", "fzss"]
    return [(name, getattr(frzl, name)) for name in names if getattr(frzl, name) is not None]


def _assert_ab_case(bc, k, cfg, s, weights, *, jmax_override=None, seed=7):
    """Full A/B: matrix elements, assembled matrices, and solved forces."""
    ns = int(np.asarray(s).shape[0])

    old_mats, old_jmin, old_jmax = old_rz_preconditioner_matrices(
        bc=bc, k=k, trig=None, s=s, cfg=cfg, jmax_override=jmax_override,
        use_precomputed=False, use_lax_tridi=False,
    )

    coeffs_r, coeffs_z, delta_s = _new_precondn_from_bc(bc, k, weights, s)

    # precondn matrix elements (VMEC arm/ard/brm/brd + azm/azd/bzm/bzd + crd).
    _allclose(coeffs_r.axm, old_mats["arm_parity"])
    _allclose(coeffs_r.axd, old_mats["ard_parity"])
    _allclose(coeffs_r.bxm, old_mats["brm_parity"])
    _allclose(coeffs_r.bxd, old_mats["brd_parity"])
    _allclose(coeffs_z.axm, old_mats["azm_parity"])
    _allclose(coeffs_z.axd, old_mats["azd_parity"])
    _allclose(coeffs_z.bxm, old_mats["bzm_parity"])
    _allclose(coeffs_z.bxd, old_mats["bzd_parity"])
    _allclose(coeffs_r.cx, old_mats["cxd_full"])
    _allclose(coeffs_z.cx, old_mats["cxd_full"])

    # scalfor tridiagonal assembly (ax/bx/dx per mode).
    mats_r, mats_z = _new_matrices(coeffs_r, coeffs_z, cfg, ns, delta_s, jmax=jmax_override)
    _allclose(mats_r.ax, old_mats["ar"])
    _allclose(mats_r.bx, old_mats["br"])
    _allclose(mats_r.dx, old_mats["dr"])
    _allclose(mats_z.ax, old_mats["az"])
    _allclose(mats_z.bx, old_mats["bz"])
    _allclose(mats_z.dx, old_mats["dz"])

    # jmin convention: m=0 solved from the axis, m>=1 from row 1 (VMEC jmin2).
    expected_jmin = np.where(np.arange(int(cfg.mpol)) > 0, 1, 0)[:, None] * np.ones(
        (int(cfg.mpol), int(cfg.ntor) + 1), dtype=np.int32
    )
    np.testing.assert_array_equal(np.asarray(old_jmin), expected_jmin)
    expected_jmax = ns - 1 if jmax_override is None else int(jmax_override)
    assert int(old_jmax) == expected_jmax

    # Solved preconditioned forces.
    rng = np.random.default_rng(seed)
    frzl = _random_frzl(
        rng, ns, int(cfg.mpol), int(cfg.ntor) + 1, bool(cfg.lthreed), bool(cfg.lasym)
    )
    old_out = old_rz_preconditioner_apply(
        frzl_in=frzl, mats=old_mats, jmax=old_jmax, cfg=cfg,
        use_precomputed=False, use_lax_tridi=False,
    )

    r_channels = _r_channels(frzl)
    z_channels = _z_channels(frzl)
    force_r = jnp.stack([value for _, value in r_channels], axis=-1)
    force_z = jnp.stack([value for _, value in z_channels], axis=-1)
    new_r = newp.scalfor(force_r, mats_r, jmax=old_jmax)
    new_z = newp.scalfor(force_z, mats_z, jmax=old_jmax)

    for idx, (name, _) in enumerate(r_channels):
        _allclose(new_r[..., idx], getattr(old_out, name), rtol=RTOL, atol=ATOL)
    for idx, (name, _) in enumerate(z_channels):
        _allclose(new_z[..., idx], getattr(old_out, name), rtol=RTOL, atol=ATOL)


def _assert_lamcal_case(bc, cfg, s, weights):
    old_faclam = old_lambda_preconditioner(bc=bc, trig=None, s=s, cfg=cfg)
    new_faclam = newp.lamcal(
        guu_half=bc.guu,
        guv_half=bc.guv,
        gvv_half=bc.gvv,
        sqrt_g_half=bc.jac.sqrtg,
        lamscale=bc.lamscale,
        angular_weight=weights,
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        nfp=int(cfg.nfp),
        lthreed=bool(cfg.lthreed),
    )
    _allclose(new_faclam, old_faclam)
    # Convention checks: axis row zero except the (0,0) chip/iota slot.
    new_np = np.asarray(new_faclam)
    assert np.all(new_np[0, 1:, :] == 0.0)
    assert np.all(new_np[0, 0, 1:] == 0.0)
    assert new_np[0, 0, 0] != 0.0


# ---------------------------------------------------------------------------
# Synthetic-input A/B tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_lamcal_matches_old(case):
    bc, _k, cfg, s, weights = _synthetic_case(case)
    _assert_lamcal_case(bc, cfg, s, weights)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_precondn_scalfor_matches_old_fixed_boundary_jmax(case):
    bc, k, cfg, s, weights = _synthetic_case(case)
    _assert_ab_case(bc, k, cfg, s, weights, jmax_override=None)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_precondn_scalfor_matches_old_edge_jmax(case):
    """jmax = ns activates the edge_pedestal and ZC(0,0) stabilization."""
    bc, k, cfg, s, weights = _synthetic_case(case)
    ns = int(np.asarray(s).shape[0])
    _assert_ab_case(bc, k, cfg, s, weights, jmax_override=ns)


def test_edge_pedestal_and_zc00_values():
    """The scalfor.f edge constants: 0.05 pedestal and the 0.25 ZC00 factor."""
    case = CASES[0]
    bc, k, cfg, s, weights = _synthetic_case(case)
    ns = int(np.asarray(s).shape[0])
    coeffs_r, coeffs_z, delta_s = _new_precondn_from_bc(bc, k, weights, s)

    kwargs = dict(
        delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns, jmax=ns,
    )
    plain = newp.scalfor_matrices(coeffs_z, stabilize_edge_zc00=False, **kwargs)
    stabilized = newp.scalfor_matrices(coeffs_z, stabilize_edge_zc00=True, **kwargs)
    # Only the (m,n) = (0,0) edge diagonal differs, by (1-mult_fac)/(1+pedestal).
    diff = np.asarray(stabilized.dx) - np.asarray(plain.dx)
    assert np.count_nonzero(diff) == 1
    mult_fac = min(0.25, 0.25 * delta_s * 15.0)
    expected = np.asarray(plain.dx)[ns - 1, 0, 0] * (1.0 - mult_fac) / (1.0 + 0.05)
    _allclose(np.asarray(stabilized.dx)[ns - 1, 0, 0], expected)

    # The pedestal itself: edge diagonal vs an assembly with jmax=ns and no
    # edge treatment possible (compare against re-deriving from coefficients).
    no_edge = newp.scalfor_matrices(
        coeffs_z, stabilize_edge_zc00=False,
        delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns + 1, jmax=ns,  # edge branch inactive (jmax < ns)
    )
    ped = np.asarray(plain.dx)[ns - 1] / np.asarray(no_edge.dx)[ns - 1]
    assert np.allclose(ped[0:2, :], 1.05, rtol=1e-12)
    assert np.allclose(ped[2:, :], 1.10, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tridiagonal solver vs dense reference
# ---------------------------------------------------------------------------


def test_tridiagonal_solve_matches_dense_numpy():
    rng = np.random.default_rng(42)
    ns, mpol, nrange, nrhs = 17, 8, 5, 3
    shape = (ns, mpol, nrange)
    superd = rng.uniform(-0.5, 0.5, size=shape)
    subd = rng.uniform(-0.5, 0.5, size=shape)
    diag = 3.0 + rng.uniform(0.0, 1.0, size=shape)  # diagonally dominant
    rhs = rng.standard_normal(size=shape + (nrhs,))

    solution = np.asarray(
        newp.tridiagonal_solve(
            jnp.asarray(superd), jnp.asarray(diag), jnp.asarray(subd), jnp.asarray(rhs)
        )
    )
    assert solution.shape == rhs.shape

    for m in range(mpol):
        for n in range(nrange):
            dense = (
                np.diag(diag[:, m, n])
                + np.diag(superd[:-1, m, n], k=1)
                + np.diag(subd[1:, m, n], k=-1)
            )
            expected = np.linalg.solve(dense, rhs[:, m, n, :])
            np.testing.assert_allclose(solution[:, m, n, :], expected, rtol=1e-11, atol=1e-12)


def test_tridiagonal_solve_broadcast_rank():
    """Shared (per-column) coefficients with extra trailing RHS axes."""
    rng = np.random.default_rng(3)
    ns, ncol = 9, 4
    superd = rng.uniform(-0.4, 0.4, size=(ns, ncol))
    subd = rng.uniform(-0.4, 0.4, size=(ns, ncol))
    diag = 2.5 + rng.uniform(0.0, 1.0, size=(ns, ncol))
    rhs = rng.standard_normal(size=(ns, ncol, 6))
    got = np.asarray(newp.tridiagonal_solve(superd, diag, subd, rhs))
    for c in range(ncol):
        dense = (
            np.diag(diag[:, c]) + np.diag(superd[:-1, c], k=1) + np.diag(subd[1:, c], k=-1)
        )
        np.testing.assert_allclose(
            got[:, c, :], np.linalg.solve(dense, rhs[:, c, :]), rtol=1e-11, atol=1e-12
        )


# ---------------------------------------------------------------------------
# jit compatibility
# ---------------------------------------------------------------------------


def test_jit_compatibility_end_to_end():
    case = CASES[1]
    bc, k, cfg, s, weights = _synthetic_case(case)
    ns = int(np.asarray(s).shape[0])
    delta_s = float(s[1] - s[0])

    @partial(jax.jit, static_argnames=("ns", "mpol", "ntor", "nfp", "jmax"))
    def preconditioned_force(
        dxds, dxdu, dxdu_e, dxdu_o, x_o, r12, bsq, bsupv, sqrtg, force,
        *, ns, mpol, ntor, nfp, jmax,
    ):
        coeffs = newp.precondn(
            dxds_half=dxds, dxdu_half=dxdu, dxdu_even_full=dxdu_e,
            dxdu_odd_full=dxdu_o, x_odd_full=x_o, r12_half=r12,
            bsq_half=bsq, bsupv_half=bsupv, sqrt_g_half=sqrtg,
            angular_weight=weights, delta_s=delta_s, ns=ns,
        )
        mats = newp.scalfor_matrices(
            coeffs, delta_s=delta_s, mpol=mpol, ntor=ntor, nfp=nfp, ns=ns, jmax=jmax,
        )
        return newp.scalfor(force, mats, jmax=jmax)

    rng = np.random.default_rng(11)
    force = jnp.asarray(rng.standard_normal((ns, int(cfg.mpol), int(cfg.ntor) + 1)))
    args = (
        jnp.asarray(bc.jac.zs)[1:], jnp.asarray(bc.jac.zu12)[1:],
        jnp.asarray(k.pzu_even), jnp.asarray(k.pzu_odd), jnp.asarray(k.pz1_odd),
        jnp.asarray(bc.jac.r12)[1:], jnp.asarray(bc.bsq)[1:],
        jnp.asarray(bc.bsupv)[1:], jnp.asarray(bc.jac.sqrtg)[1:], force,
    )
    kwargs = dict(ns=ns, mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp), jmax=ns - 1)
    jitted = preconditioned_force(*args, **kwargs)

    coeffs_r, _, _ = _new_precondn_from_bc(bc, k, weights, s)
    mats_r = newp.scalfor_matrices(
        coeffs_r, delta_s=delta_s, mpol=int(cfg.mpol), ntor=int(cfg.ntor),
        nfp=int(cfg.nfp), ns=ns, jmax=ns - 1,
    )
    eager = newp.scalfor(force, mats_r, jmax=ns - 1)
    _allclose(jitted, eager, rtol=RTOL, atol=1e-14)


def test_jit_compatibility_lamcal():
    case = CASES[1]
    bc, _k, cfg, s, weights = _synthetic_case(case)

    @jax.jit
    def jitted_lamcal(guu, guv, gvv, sqrtg, lamscale):
        return newp.lamcal(
            guu_half=guu, guv_half=guv, gvv_half=gvv, sqrt_g_half=sqrtg,
            lamscale=lamscale, angular_weight=weights,
            mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp),
            lthreed=bool(cfg.lthreed),
        )

    got = jitted_lamcal(
        jnp.asarray(bc.guu), jnp.asarray(bc.guv), jnp.asarray(bc.gvv),
        jnp.asarray(bc.jac.sqrtg), jnp.asarray(bc.lamscale),
    )
    eager = newp.lamcal(
        guu_half=bc.guu, guv_half=bc.guv, gvv_half=bc.gvv,
        sqrt_g_half=bc.jac.sqrtg, lamscale=bc.lamscale, angular_weight=weights,
        mpol=int(cfg.mpol), ntor=int(cfg.ntor), nfp=int(cfg.nfp),
        lthreed=bool(cfg.lthreed),
    )
    _allclose(got, eager, rtol=RTOL, atol=1e-14)


# ---------------------------------------------------------------------------
# Real-state A/B (short fixed-boundary runs)
# ---------------------------------------------------------------------------


def _real_state_inputs(input_name):
    vj = pytest.importorskip("vmec_jax")
    from vmec_jax.kernels.forces import vmec_forces_rz_from_wout

    input_path = REPO_ROOT / "examples" / "data" / input_name
    assert input_path.exists()
    run = vj.run_fixed_boundary(
        str(input_path),
        solver="vmec2000_iter",
        max_iter=10,
        multigrid_use_input_niter=False,
        verbose=False,
    )
    wout = vj.wout_from_fixed_boundary_run(run)
    k = vmec_forces_rz_from_wout(
        state=run.state, static=run.static, wout=wout, use_vmec_synthesis=True
    )
    bc = k.bc
    cfg = run.static.cfg
    s = np.asarray(run.static.s)
    weights = newp.angular_integration_weights(
        ntheta=int(cfg.ntheta), nzeta=int(cfg.nzeta), lasym=bool(cfg.lasym)
    )
    assert int(np.asarray(bc.guu).shape[1]) == int(weights.shape[0])
    return bc, k, cfg, s, weights


@pytest.mark.parametrize("input_name", ["input.solovev", "input.cth_like_fixed_bdy"])
def test_real_state_ab(input_name):
    bc, k, cfg, s, weights = _real_state_inputs(input_name)
    _assert_lamcal_case(bc, cfg, s, weights)
    _assert_ab_case(bc, k, cfg, s, weights, jmax_override=None, seed=23)

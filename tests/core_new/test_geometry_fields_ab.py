"""A/B equivalence tests: ``vmec_jax.core.{geometry,fields}`` vs the legacy kernels.

Old implementations under test (left untouched):

- ``vmec_jax.kernels.bcovar.vmec_bcovar_half_mesh_from_wout`` (bcovar.f,
  ``use_vmec_synthesis=True`` lane) — includes ``kernels.jacobian`` internally.
- ``vmec_jax.kernels.residue.vmec_force_norms_from_bcovar_dynamic`` (bcovar.f
  norms: wb/wp/vp/volume/fnorm/fnormL).
- ``vmec_jax.kernels.lforbal.currents_from_bcovar`` /
  ``plascur_edge_from_bcovar`` (fbal.f buco/bvco, bcovar.f ctor).
- ``vmec_jax.kernels.constraints.tcon_from_bcovar_precondn_diag`` (bcovar.f +
  precondn.f tcon).
- ``vmec_jax.kernels.parity`` block conversions (fnorm1 / ``rz_norm``
  reference, bcovar.f fnorm1).

New implementations:

- ``vmec_jax.core.geometry``  (real_space_geometry / half_mesh_jacobian /
  apply_lambda_axis_closure)
- ``vmec_jax.core.fields``    (metric_elements / magnetic_fields /
  energies_and_force_norms / preconditioned_force_norm / surface_currents /
  constraint_scaling)

Realistic spectral states are produced by running the legacy fixed-boundary
driver for a few (unconverged) iterations on sym 2D, sym 3D and lasym decks.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

# Pin the legacy kernels to their CPU DFT lane (deterministic A/B reference).
os.environ.setdefault("VMEC_JAX_TOMNSPS_FFT", "0")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import vmec_jax as vj
from vmec_jax.field import chips_from_wout_chipf
from vmec_jax.kernels.bcovar import vmec_bcovar_half_mesh_from_wout
from vmec_jax.kernels.constraints import tcon_from_bcovar_precondn_diag
from vmec_jax.kernels.lforbal import currents_from_bcovar, plascur_edge_from_bcovar
from vmec_jax.kernels.parity import (
    _signed_to_mn_cos_cached,
    _signed_to_mn_sin_cached,
    signed_maps_from_modes,
    vmec_m1_internal_to_physical_signed,
)
from vmec_jax.kernels.residue import vmec_force_norms_from_bcovar_dynamic
from vmec_jax.kernels.tomnsp import vmec_trig_tables

from vmec_jax.core.fields import (
    constraint_scaling,
    energies_and_force_norms,
    magnetic_fields,
    metric_elements,
    preconditioned_force_norm,
    surface_currents,
)
from vmec_jax.core.fourier import Resolution, mode_table, trig_tables
from vmec_jax.core.geometry import (
    apply_lambda_axis_closure,
    half_mesh_jacobian,
    real_space_geometry,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "examples" / "data"

RTOL = 1e-12
ATOL = 1e-13

TCON0 = 0.9

# name -> (max_iter, use_mass_profile)
CASES = {
    "solovev": (30, False),  # 2D sym, ncurr=0, pressure profile lane
    "cth_like_fixed_bdy": (25, True),  # 2D sym, nfp=5, ncurr=1 (add_fluxes lane)
    "li383_low_res": (25, True),  # 3D sym (lthreed: guv/gvv, lambda closure)
    "up_down_asymmetric_tokamak": (20, True),  # lasym
}

GAMMA_MASS = 5.0 / 3.0  # exercises pres = mass / vp**gamma


def _allclose(new, old, name):
    np.testing.assert_allclose(
        np.asarray(new), np.asarray(old), rtol=RTOL, atol=ATOL, err_msg=f"{name} mismatch"
    )


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request):
    """Run the legacy driver briefly and package old-side and new-side inputs."""
    name = request.param
    max_iter, use_mass = CASES[name]
    run = vj.run_fixed_boundary(str(DATA_DIR / f"input.{name}"), max_iter=max_iter, verbose=False)
    cfg, state, static, flux, prof = run.cfg, run.state, run.static, run.flux, run.profiles
    signgs = int(run.signgs)
    s = jnp.asarray(static.s)
    ns = int(s.shape[0])

    pres = jnp.asarray(prof["pressure"])
    if ns > 0:
        pres = pres.at[0].set(0.0)
    if use_mass:
        gamma = GAMMA_MASS
        mass = jnp.asarray(1e-2 * (2.0 - np.asarray(static.s)))
    else:
        gamma = 0.0
        mass = None

    chips = chips_from_wout_chipf(
        chipf=jnp.asarray(flux.chipf),
        phipf=jnp.asarray(flux.phipf),
        iotaf=None,
        iotas=None,
        assume_half_if_unknown=True,
    )
    ncurr = int(prof["ncurr"])

    wout_like = SimpleNamespace(
        phipf=jnp.asarray(flux.phipf),
        chipf=jnp.asarray(flux.chipf),
        phips=jnp.asarray(flux.phips),
        chips_eff=chips,
        nfp=int(cfg.nfp),
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        lasym=bool(cfg.lasym),
        signgs=signgs,
        ncurr=ncurr,
        lcurrent=(ncurr == 1),
        gamma=gamma,
        flux_is_internal=True,
    )

    # ------------------------------------------------------------------ old
    trig_old = vmec_trig_tables(
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        mmax=int(cfg.mpol) - 1,
        nmax=int(cfg.ntor),
        lasym=bool(cfg.lasym),
    )
    bc_old, aux_old = vmec_bcovar_half_mesh_from_wout(
        state=state,
        static=static,
        wout=wout_like,
        pres=pres,
        mass=mass,
        use_vmec_synthesis=True,
        trig=trig_old,
        return_parity_aux=True,
    )
    norms_old = vmec_force_norms_from_bcovar_dynamic(
        bc=bc_old, trig=trig_old, s=s, signgs=signgs
    )
    buco_old, bvco_old, _, _ = currents_from_bcovar(bc=bc_old, trig=trig_old, wout=wout_like, s=s)
    ctor_old_nestor = plascur_edge_from_bcovar(bc=bc_old, trig=trig_old, wout=wout_like, s=s)

    sqrt_s3 = jnp.sqrt(jnp.maximum(s, 0.0))[:, None, None]
    ru0_old = jnp.asarray(aux_old.pru_even) + sqrt_s3 * jnp.asarray(aux_old.pru_odd)
    zu0_old = jnp.asarray(aux_old.pzu_even) + sqrt_s3 * jnp.asarray(aux_old.pzu_odd)
    tcon_old = tcon_from_bcovar_precondn_diag(
        tcon0=TCON0,
        trig=trig_old,
        s=s,
        signgs=signgs,
        lasym=bool(cfg.lasym),
        bsq=bc_old.bsq,
        r12=bc_old.jac.r12,
        sqrtg=bc_old.jac.sqrtg,
        ru12=bc_old.jac.ru12,
        zu12=bc_old.jac.zu12,
        ru0=ru0_old,
        zu0=zu0_old,
    )

    # fnorm1 reference: the legacy solver ``ModeTransform.rz_norm`` formula
    # (bcovar.f fnorm1) written with the legacy parity block conversions.
    maps = signed_maps_from_modes(static.modes)
    rcc, rss = _signed_to_mn_cos_cached(jnp.asarray(state.Rcos), maps=maps)
    zsc, zcs = _signed_to_mn_sin_cached(jnp.asarray(state.Zsin), maps=maps)
    m_grid = np.arange(maps.mpol)[:, None]
    n_grid = np.arange(maps.nrange)[None, :]
    include_rcc = jnp.asarray(((m_grid > 0) | (n_grid > 0)).astype(float))
    sl = slice(1, None)
    rz_norm = jnp.sum(zsc[sl] * zsc[sl]) + jnp.sum(include_rcc * rcc[sl] * rcc[sl])
    if bool(cfg.lthreed):
        rz_norm = rz_norm + jnp.sum(rss[sl] * rss[sl]) + jnp.sum(zcs[sl] * zcs[sl])
    if bool(cfg.lasym):
        rsc, rcs = _signed_to_mn_sin_cached(jnp.asarray(state.Rsin), maps=maps)
        zcc, zss = _signed_to_mn_cos_cached(jnp.asarray(state.Zcos), maps=maps)
        rz_norm = (
            rz_norm
            + jnp.sum(rsc[sl] * rsc[sl])
            + jnp.sum(rcs[sl] * rcs[sl])
            + jnp.sum(zcc[sl] * zcc[sl])
            + jnp.sum(zss[sl] * zss[sl])
        )
    fnorm1_old = 1.0 / rz_norm

    # ------------------------------------------------------------------ new
    res = Resolution(
        mpol=int(cfg.mpol),
        ntor=int(cfg.ntor),
        ntheta=int(cfg.ntheta),
        nzeta=int(cfg.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
        ns=ns,
    )
    trig_new = trig_tables(res)
    modes_new = mode_table(int(cfg.mpol), int(cfg.ntor))

    # Input preparation with the legacy parity helper: undo the residue.f90
    # m=1 internal constraint (the new modules take physical-signed internal
    # coefficients; the constraint mapping belongs to the residuals module).
    R_cos, Z_sin, R_sin, Z_cos = vmec_m1_internal_to_physical_signed(
        Rcos=state.Rcos,
        Zsin=state.Zsin,
        Rsin=state.Rsin,
        Zcos=state.Zcos,
        modes=static.modes,
        lthreed=bool(cfg.lthreed),
        lasym=bool(cfg.lasym),
        lconm1=bool(cfg.lconm1),
    )
    new_inputs = dict(
        R_cos=jnp.asarray(R_cos),
        R_sin=jnp.asarray(R_sin),
        Z_cos=jnp.asarray(Z_cos),
        Z_sin=jnp.asarray(Z_sin),
        lambda_cos=jnp.asarray(state.Lcos),
        lambda_sin=apply_lambda_axis_closure(
            jnp.asarray(state.Lsin), modes=modes_new, ntor=int(cfg.ntor)
        ),
    )

    def new_pipeline(inputs):
        geom = real_space_geometry(**inputs, modes=modes_new, trig=trig_new, s=s)
        jac = half_mesh_jacobian(geom, s=s)
        mets = metric_elements(geom, s=s)
        mf = magnetic_fields(
            geometry=geom,
            jacobian=jac,
            metrics=mets,
            trig=trig_new,
            s=s,
            phips=jnp.asarray(flux.phips),
            phipf=jnp.asarray(flux.phipf),
            chips=chips,
            signgs=signgs,
            gamma=gamma,
            pressure=pres,
            mass=mass,
            ncurr=ncurr,
        )
        en = energies_and_force_norms(
            jacobian=jac, metrics=mets, fields=mf, trig=trig_new, s=s, signgs=signgs
        )
        tcon = constraint_scaling(
            tcon0=TCON0,
            geometry=geom,
            jacobian=jac,
            total_pressure=mf.total_pressure,
            trig=trig_new,
            s=s,
        )
        return geom, jac, mets, mf, en, tcon

    geom, jac, mets, mf, en, tcon_new = new_pipeline(new_inputs)
    cur_new = surface_currents(bsubu=mf.bsubu, bsubv=mf.bsubv, trig=trig_new, s=s, signgs=signgs)
    fnorm1_new = preconditioned_force_norm(
        R_cos=jnp.asarray(state.Rcos),
        Z_sin=jnp.asarray(state.Zsin),
        modes=modes_new,
        R_sin=jnp.asarray(state.Rsin) if bool(cfg.lasym) else None,
        Z_cos=jnp.asarray(state.Zcos) if bool(cfg.lasym) else None,
    )

    return SimpleNamespace(
        name=name,
        cfg=cfg,
        s=s,
        signgs=signgs,
        trig_new=trig_new,
        modes_new=modes_new,
        new_inputs=new_inputs,
        new_pipeline=new_pipeline,
        chips=chips,
        pres=pres,
        mass=mass,
        gamma=gamma,
        ncurr=ncurr,
        flux=flux,
        bc_old=bc_old,
        norms_old=norms_old,
        buco_old=buco_old,
        bvco_old=bvco_old,
        ctor_old_nestor=ctor_old_nestor,
        tcon_old=tcon_old,
        fnorm1_old=fnorm1_old,
        geom=geom,
        jac=jac,
        mets=mets,
        mf=mf,
        en=en,
        tcon_new=tcon_new,
        cur_new=cur_new,
        fnorm1_new=fnorm1_new,
    )


# ---------------------------------------------------------------------------
# geometry.py: half-mesh Jacobian (jacobian.f)
# ---------------------------------------------------------------------------


def test_half_mesh_jacobian_matches_old(case):
    old = case.bc_old.jac
    _allclose(case.jac.r12, old.r12, "r12")
    _allclose(case.jac.dR_ds, old.rs, "dR_ds (rs)")
    _allclose(case.jac.dZ_ds, old.zs, "dZ_ds (zs)")
    _allclose(case.jac.ru12, old.ru12, "ru12")
    _allclose(case.jac.zu12, old.zu12, "zu12")
    _allclose(case.jac.tau, old.tau, "tau")
    _allclose(case.jac.sqrt_g, old.sqrtg, "sqrt_g (gsqrt)")


def test_jacobian_sign_change_detection(case):
    # The driver states are healthy: no sign change.
    assert not bool(case.jac.jacobian_sign_changed)

    # A large m=1 spike on one interior surface makes the flux surfaces
    # cross: tau flips sign there (VMEC irst = 2 / bad_jacobian_flag).
    m = np.asarray(case.modes_new.m)
    k_m1 = int(np.nonzero(m == 1)[0][0])
    js_mid = int(case.s.shape[0]) // 2
    bad_inputs = dict(case.new_inputs)
    bad_inputs["R_cos"] = case.new_inputs["R_cos"].at[js_mid, k_m1].add(5.0)
    geom_bad = real_space_geometry(
        **bad_inputs, modes=case.modes_new, trig=case.trig_new, s=case.s
    )
    jac_bad = half_mesh_jacobian(geom_bad, s=case.s)
    assert bool(jac_bad.jacobian_sign_changed)


# ---------------------------------------------------------------------------
# fields.py: metric elements, B fields, bsq (bcovar.f)
# ---------------------------------------------------------------------------


def test_metric_elements_match_old(case):
    _allclose(case.mets.guu, case.bc_old.guu, "guu")
    _allclose(case.mets.guv, case.bc_old.guv, "guv")
    _allclose(case.mets.gvv, case.bc_old.gvv, "gvv")


def test_magnetic_fields_match_old(case):
    _allclose(case.mf.bsupu, case.bc_old.bsupu, "bsupu")
    _allclose(case.mf.bsupv, case.bc_old.bsupv, "bsupv")
    _allclose(case.mf.bsubu, case.bc_old.bsubu, "bsubu")
    _allclose(case.mf.bsubv, case.bc_old.bsubv, "bsubv")
    _allclose(case.mf.total_pressure, case.bc_old.bsq, "total_pressure (bsq)")
    _allclose(case.mf.lamscale, case.bc_old.lamscale, "lamscale")


def test_differential_volume_matches_old(case):
    _allclose(case.mf.vp, case.norms_old.vp, "vp (magnetic_fields)")
    _allclose(case.en.vp, case.norms_old.vp, "vp (energies)")


def test_pressure_from_mass_matches_old(case):
    if case.mass is None:
        pytest.skip("case runs the pressure-profile lane")
    # Old side embeds pres = mass/vp**gamma inside bsq; recover it.
    b2 = case.bc_old.bsupu * case.bc_old.bsubu + case.bc_old.bsupv * case.bc_old.bsubv
    pres_old = (case.bc_old.bsq - 0.5 * b2)[:, 0, 0]
    _allclose(case.mf.pressure, pres_old, "pressure (mass/vp**gamma)")


# ---------------------------------------------------------------------------
# fields.py: energies, force norms (bcovar.f), fnorm1
# ---------------------------------------------------------------------------


def test_energies_and_norms_match_old(case):
    old = case.norms_old
    _allclose(case.en.wb, old.wb, "wb")
    _allclose(case.en.wp, old.wp, "wp")
    _allclose(case.en.volume, old.volume, "volume")
    _allclose(case.en.energy_density, old.r2, "r2")
    _allclose(case.en.fnorm, old.fnorm, "fnorm")
    _allclose(case.en.fnormL, old.fnormL, "fnormL")
    _allclose(case.en.r1, old.r1, "r1")


def test_preconditioned_force_norm_matches_old(case):
    _allclose(case.fnorm1_new, case.fnorm1_old, "fnorm1")


# ---------------------------------------------------------------------------
# fields.py: surface currents (fbal.f / bcovar.f)
# ---------------------------------------------------------------------------


def test_surface_currents_match_old(case):
    _allclose(case.cur_new.buco, case.buco_old, "buco")
    _allclose(case.cur_new.bvco, case.bvco_old, "bvco")
    # bcovar.f: ctor = signgs*2*pi*(1.5*buco(ns) - 0.5*buco(ns-1)).  The old
    # kernel returns the NESTOR-convention value (opposite sign).
    _allclose(case.cur_new.ctor, -np.asarray(case.ctor_old_nestor), "ctor")
    bvco = np.asarray(case.bvco_old)
    _allclose(case.cur_new.rbtor, 1.5 * bvco[-1] - 0.5 * bvco[-2], "rbtor")
    _allclose(case.cur_new.rbtor0, 1.5 * bvco[1] - 0.5 * bvco[2], "rbtor0")


# ---------------------------------------------------------------------------
# fields.py: constraint scaling tcon (bcovar.f + precondn.f)
# ---------------------------------------------------------------------------


def test_constraint_scaling_matches_old(case):
    _allclose(case.tcon_new, case.tcon_old, "tcon")


# ---------------------------------------------------------------------------
# jit-compatibility and differentiability
# ---------------------------------------------------------------------------


def test_pipeline_is_jittable(case):
    def outputs(inputs):
        geom, jac, mets, mf, en, tcon = case.new_pipeline(inputs)
        return (
            jac.sqrt_g,
            jac.tau,
            jac.jacobian_sign_changed,
            mets.guu,
            mf.bsupu,
            mf.bsubv,
            mf.total_pressure,
            en.wb,
            en.fnorm,
            en.fnormL,
            tcon,
        )

    eager = outputs(case.new_inputs)
    jitted = jax.jit(outputs)(case.new_inputs)
    for idx, (a, b) in enumerate(zip(eager, jitted)):
        np.testing.assert_allclose(
            np.asarray(a), np.asarray(b), rtol=RTOL, atol=ATOL, err_msg=f"jit output {idx}"
        )


def test_grad_of_wb_wrt_spectral_coefficients(case):
    def wb_of_R_cos(R_cos):
        inputs = dict(case.new_inputs)
        inputs["R_cos"] = R_cos
        *_, en, _tcon = case.new_pipeline(inputs)
        return en.wb

    grad = jax.grad(wb_of_R_cos)(case.new_inputs["R_cos"])
    grad_np = np.asarray(grad)
    assert grad_np.shape == np.asarray(case.new_inputs["R_cos"]).shape
    assert np.all(np.isfinite(grad_np))
    assert np.any(grad_np != 0.0)

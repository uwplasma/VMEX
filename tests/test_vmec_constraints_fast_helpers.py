from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.vmec_constraints import (
    alias_gcon,
    faccon_from_signgs,
    precondn_diag_axd1_from_bcovar,
    tcon_from_bcovar_precondn_diag,
    tcon_from_tcon0_heuristic,
)
from vmec_jax.vmec_tomnsp import vmec_trig_tables


def test_faccon_from_signgs_matches_fixaray_indexing_and_guards():
    fac = np.asarray(faccon_from_signgs(mpol=6, signgs=-1))

    expected = np.zeros((6,), dtype=float)
    for m in range(1, 5):
        expected[m] = 0.25 / (((m + 1) * m) ** 2)
    np.testing.assert_allclose(fac, expected, rtol=0.0, atol=0.0)

    assert np.asarray(faccon_from_signgs(mpol=1, signgs=1)).tolist() == [0.0]
    with pytest.raises(ValueError, match="mpol must be positive"):
        faccon_from_signgs(mpol=0, signgs=1)


def test_tcon_heuristic_clamps_axis_and_edge_like_vmec_profile():
    trig = vmec_trig_tables(ntheta=8, nzeta=3, nfp=1, mmax=4, nmax=1, lasym=False)

    empty = np.asarray(tcon_from_tcon0_heuristic(tcon0=3.0, s=np.array([0.0]), trig=trig, lasym=False))
    np.testing.assert_allclose(empty, np.zeros((1,)))

    s = np.linspace(0.0, 1.0, 5)
    tcon = np.asarray(tcon_from_tcon0_heuristic(tcon0=3.0, s=s, trig=trig, lasym=True))

    hs = s[1] - s[0]
    tcon0 = 1.0
    ns = float(s.size)
    tcon_mul = tcon0 * (1.0 + ns * (1.0 / 60.0 + ns / (200.0 * 120.0)))
    tcon_mul = tcon_mul / ((4.0 * (float(trig.r0scale) ** 2)) ** 2)
    core = tcon_mul * (32.0 * hs) ** 2

    np.testing.assert_allclose(tcon[0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tcon[1:-1], core, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(tcon[-1], 0.5 * core, rtol=1e-14, atol=1e-14)


def test_precondn_diag_short_mesh_returns_zero_tcon_and_diagonals():
    trig = vmec_trig_tables(ntheta=8, nzeta=2, nfp=1, mmax=3, nmax=1, lasym=False)
    shape = (1, int(trig.ntheta3), 2)
    ones = np.ones(shape)
    s = np.array([0.0])

    ard1, azd1 = precondn_diag_axd1_from_bcovar(
        trig=trig,
        s=s,
        bsq=ones,
        r12=ones,
        sqrtg=ones,
        ru12=ones,
        zu12=ones,
    )
    tcon = tcon_from_bcovar_precondn_diag(
        tcon0=0.5,
        trig=trig,
        s=s,
        signgs=1,
        lasym=False,
        bsq=ones,
        r12=ones,
        sqrtg=ones,
        ru12=ones,
        zu12=ones,
        ru0=ones,
        zu0=ones,
    )

    np.testing.assert_allclose(np.asarray(ard1), np.zeros((1,)))
    np.testing.assert_allclose(np.asarray(azd1), np.zeros((1,)))
    np.testing.assert_allclose(np.asarray(tcon), np.zeros((1,)))


def test_precondn_diag_falls_back_to_dnorm3_when_weight_shape_differs():
    trig_base = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=3, nmax=1, lasym=False)
    trig = SimpleNamespace(
        cosmu=trig_base.cosmu,
        cosmui3=trig_base.cosmui3,
        cosnv=trig_base.cosnv,
        mscale=trig_base.mscale,
        r0scale=trig_base.r0scale,
        dnorm3=np.full((2, 1), 0.25),
    )
    s = np.linspace(0.0, 1.0, 3)
    shape = (3, 2, 1)
    bsq = np.full(shape, 2.0)
    r12 = np.full(shape, 1.5)
    sqrtg = np.full(shape, 0.75)
    ru12 = np.full(shape, 0.4)
    zu12 = np.full(shape, 0.6)

    ard1, azd1 = precondn_diag_axd1_from_bcovar(
        trig=trig,
        s=s,
        bsq=bsq,
        r12=r12,
        sqrtg=sqrtg,
        ru12=ru12,
        zu12=zu12,
    )

    hs = s[1] - s[0]
    pfactor = -4.0 * float(trig.r0scale) ** 2
    ptau = (pfactor * r12**2 * bsq * np.asarray(trig.dnorm3)[None, :, :]) / sqrtg
    ax_r = np.sum(ptau * (zu12 / hs) ** 2, axis=(1, 2))
    ax_z = np.sum(ptau * (ru12 / hs) ** 2, axis=(1, 2))
    ax_r[0] = 0.0
    ax_z[0] = 0.0
    expected_ard = ax_r + np.concatenate([ax_r[1:], np.zeros((1,))])
    expected_azd = ax_z + np.concatenate([ax_z[1:], np.zeros((1,))])

    np.testing.assert_allclose(np.asarray(ard1), expected_ard, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(azd1), expected_azd, rtol=1e-13, atol=1e-13)


def test_alias_gcon_rejects_inconsistent_grid_shapes():
    trig = vmec_trig_tables(ntheta=8, nzeta=4, nfp=1, mmax=3, nmax=1, lasym=False)
    ztemp = np.zeros((2, int(trig.ntheta3), 4))
    tcon = np.ones((2,))

    with pytest.raises(ValueError, match="theta size"):
        alias_gcon(
            ztemp=ztemp[:, :-1, :],
            trig=trig,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=False,
        )

    trig_bad_lasym = SimpleNamespace(
        **{
            name: getattr(trig, name)
            for name in (
                "cosmu",
                "sinmu",
                "cosmui",
                "sinmui",
                "cosnv",
                "sinnv",
                "ntheta1",
                "ntheta2",
                "ntheta3",
            )
        }
    )
    trig_bad_lasym.ntheta1 = int(trig.ntheta3) + 1
    with pytest.raises(ValueError, match="lasym=True requires"):
        alias_gcon(
            ztemp=ztemp,
            trig=trig_bad_lasym,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=True,
        )

    with pytest.raises(ValueError, match="nzeta"):
        alias_gcon(
            ztemp=ztemp[:, :, :-1],
            trig=trig,
            ntor=1,
            mpol=4,
            signgs=1,
            tcon=tcon,
            lasym=False,
        )

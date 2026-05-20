from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vmec_jax.namelist import InData
import vmec_jax.wout as wout_module
from vmec_jax.wout import (
    MU0,
    _bool_from_nc,
    _bss_scalxc_undo_factor,
    _bss_should_undo_scalxc,
    _chipf_from_chips,
    _compute_aspectratio,
    _compute_eqfor_beta,
    _compute_eqfor_betaxis,
    _compute_ctor_from_buco,
    _icurv_full_mesh_from_indata,
    _jxbforce_nyquist_limits,
    _nc_scalar,
    _pshalf_from_s,
    _read_wout_scalar_metadata,
    _safe_divide,
    _undo_bss_scalxc_if_enabled,
    _vmec_wint_from_trig,
    _wout_phi_profile_from_variables,
    assert_main_modes_match_wout,
)
from vmec_jax.wout_schema import WoutData as SchemaWoutData
from vmec_jax.wout_schema import _bool_from_nc as schema_bool_from_nc
from vmec_jax.wout_schema import _nc_scalar as schema_nc_scalar
from vmec_jax.wout_schema import assert_main_modes_match_wout as schema_assert_main_modes_match_wout


class _FakeNcVar:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self.value


def test_wout_half_mesh_and_flux_derivative_conventions() -> None:
    s_full = np.asarray([0.0, 0.25, 1.0])
    np.testing.assert_allclose(_pshalf_from_s(s_full), np.sqrt([0.125, 0.125, 0.625]))
    np.testing.assert_allclose(_pshalf_from_s(np.asarray([0.36])), [0.6])

    chips = np.asarray([0.0, 1.0, 4.0, 9.0])
    np.testing.assert_allclose(_chipf_from_chips(chips), [-0.5, 2.5, 6.5, 11.5])
    np.testing.assert_allclose(_chipf_from_chips(np.asarray([2.0, 5.0])), [5.0, 6.5])


def test_safe_divide_uses_unit_denominator_for_exact_zeros() -> None:
    num = np.asarray([2.0, 4.0, 6.0])
    den = np.asarray([1.0, 0.0, -2.0])
    np.testing.assert_allclose(_safe_divide(num, den), [2.0, 4.0, -3.0])


def test_wint_nyquist_and_scalxc_helper_edges(monkeypatch) -> None:
    trig = SimpleNamespace(cosmui3=np.asarray([[2.0], [4.0]]), mscale=np.asarray([2.0]), cosnv=np.zeros((5, 1)))
    np.testing.assert_allclose(_vmec_wint_from_trig(trig), np.ones((2, 5)) * np.asarray([[1.0], [2.0]]))
    assert _jxbforce_nyquist_limits(SimpleNamespace(ntheta2=4, cosnv=np.zeros((5, 1)))) == (3, 2)

    with pytest.raises(ValueError, match="cosmui3"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.ones((2, 1, 1)), mscale=np.asarray([1.0]), cosnv=np.zeros((1, 1))))
    with pytest.raises(ValueError, match="mscale"):
        _vmec_wint_from_trig(SimpleNamespace(cosmui3=np.ones((2, 1)), mscale=np.asarray([]), cosnv=np.zeros((1, 1))))

    s = np.asarray([0.0, 0.25, 1.0])
    np.testing.assert_allclose(_bss_scalxc_undo_factor(s).ravel(), [0.5, 0.5, 1.0])
    arr = np.ones((3, 1, 1))
    monkeypatch.delenv("VMEC_JAX_BSS_UNDO_SCALXC", raising=False)
    assert _bss_should_undo_scalxc() is False
    assert _undo_bss_scalxc_if_enabled(s, arr)[0] is arr
    monkeypatch.setenv("VMEC_JAX_BSS_UNDO_SCALXC", "1")
    assert _bss_should_undo_scalxc() is True
    np.testing.assert_allclose(_undo_bss_scalxc_if_enabled(s, arr)[0].ravel(), [0.5, 0.5, 1.0])


def test_current_profile_full_mesh_uses_vmec_half_mesh_normalization() -> None:
    s_full = np.asarray([0.0, 0.25, 1.0])
    indata = InData(
        scalars={
            "NCURR": 1,
            "CURTOR": 10.0,
            "PCURR_TYPE": "power_series",
            "AC": [2.0],
        },
        indexed={},
    )

    icurv = _icurv_full_mesh_from_indata(indata=indata, s_full=s_full, signgs=-1)
    expected_scale = -MU0 * 10.0 / (2.0 * np.pi) / 2.0
    # I(s_half)=2*s_half and VMEC explicitly zeroes the axis value.
    np.testing.assert_allclose(np.asarray(icurv), expected_scale * np.asarray([0.0, 0.25, 1.25]))

    no_current = InData(scalars={"NCURR": 0, "CURTOR": 10.0, "AC": [2.0]}, indexed={})
    np.testing.assert_allclose(np.asarray(_icurv_full_mesh_from_indata(indata=no_current, s_full=s_full, signgs=1)), 0.0)

    zero_edge = InData(scalars={"NCURR": 1, "CURTOR": 10.0, "AC": [0.0]}, indexed={})
    np.testing.assert_allclose(np.asarray(_icurv_full_mesh_from_indata(indata=zero_edge, s_full=s_full, signgs=1)), 0.0)


def test_eqfor_beta_aspect_and_ctor_match_vmec_normalizations() -> None:
    pres = np.asarray([0.0, 2.0, 4.0])
    vp = np.asarray([0.0, 5.0, 6.0])
    bsq = np.full((3, 2, 2), 20.0)
    r12 = np.full((3, 2, 2), 2.0)
    bsupv = np.full((3, 2, 2), 3.0)
    sqrtg = np.ones((3, 2, 2))
    wint = np.full((2, 2), 0.25)

    betapol, betator, betatot, betaxis = _compute_eqfor_beta(
        pres=pres,
        vp=vp,
        bsq=bsq,
        r12=r12,
        bsupv=bsupv,
        sqrtg=sqrtg,
        wint=wint,
        signgs=1,
    )

    hs = 0.5
    vnorm = (2.0 * np.pi) ** 2 * hs
    sump = vnorm * (vp[1] * pres[1] + vp[2] * pres[2])
    sumbtot = 2.0 * (vnorm * 40.0 - sump)
    sumbtor = vnorm * 72.0
    sumbpol = sumbtot - sumbtor
    assert betapol == pytest.approx(2.0 * sump / sumbpol)
    assert betator == pytest.approx(2.0 * sump / sumbtor)
    assert betatot == pytest.approx(2.0 * sump / sumbtot)
    assert betaxis == pytest.approx(1.5 * (2.0 / (20.0 / 5.0 - 2.0)) - 0.5 * (4.0 / (20.0 / 6.0 - 4.0)))
    assert _compute_eqfor_betaxis(pres=pres[:2], vp=vp[:2], bsq=bsq[:2], sqrtg=sqrtg[:2], wint=wint, signgs=1) == 0.0

    R = np.asarray([[[1.0, 1.0], [1.0, 1.0]], [[3.0, 3.0], [5.0, 5.0]]])
    Zu = np.ones_like(R)
    Aminor_p, Rmajor_p, aspect, volume_p, cross_area_p = _compute_aspectratio(R=R, Zu=Zu, wint=wint)
    assert cross_area_p == pytest.approx(2.0 * np.pi * 4.0)
    assert volume_p == pytest.approx(2.0 * np.pi * np.pi * 17.0)
    assert Rmajor_p == pytest.approx(volume_p / (2.0 * np.pi * cross_area_p))
    assert Aminor_p == pytest.approx(np.sqrt(cross_area_p / np.pi))
    assert aspect == pytest.approx(Rmajor_p / Aminor_p)

    with pytest.raises(ValueError, match="shape"):
        _compute_aspectratio(R=R[0], Zu=Zu[0], wint=wint)
    assert _compute_aspectratio(R=np.zeros_like(R), Zu=Zu, wint=wint)[:3] == (0.0, 0.0, 0.0)

    fixed_bdy = InData(scalars={}, indexed={})
    free_legacy = InData(scalars={"LFREEB": True, "ICTRL_PREC2D": 2}, indexed={})
    free_exact = InData(scalars={"LFREEB": True, "ICTRL_PREC2D": 1, "LHESS_EXACT": True}, indexed={})
    buco = np.asarray([1.0, 2.0, 4.0])
    assert _compute_ctor_from_buco(buco=buco, signgs=-1, indata=fixed_bdy) == pytest.approx(-2.0 * np.pi * 5.0 / MU0)
    assert _compute_ctor_from_buco(buco=buco, signgs=1, indata=free_legacy) == pytest.approx(2.0 * np.pi * 4.0 / MU0)
    assert _compute_ctor_from_buco(buco=buco, signgs=1, indata=free_exact) == pytest.approx(2.0 * np.pi * 4.0 / MU0)


def test_netcdf_scalar_helpers_handle_masked_and_fallback_values() -> None:
    assert _bool_from_nc(np.ma.array([1], mask=[False])) is True
    assert _bool_from_nc(np.ma.array([99], mask=[True])) is False
    assert _nc_scalar(np.ma.array([3.25], mask=[False])) == 3.25
    assert _nc_scalar(np.ma.array([3.25], mask=[False]), as_int=True) == 3
    assert _nc_scalar(object(), default=7.0) == 7.0
    assert _nc_scalar(object(), default=7.0, as_int=True) == 7


def test_wout_main_mode_order_contract_detects_mismatches() -> None:
    good = SimpleNamespace(
        path=Path("wout_good.nc"),
        mpol=2,
        ntor=1,
        nfp=3,
        xm=np.asarray([0, 0, 1, 1, 1]),
        xn=np.asarray([0, 3, -3, 0, 3]),
    )
    assert_main_modes_match_wout(wout=good)

    bad_m = SimpleNamespace(**{**good.__dict__, "xm": np.asarray([0, 1])})
    with pytest.raises(ValueError, match="Mode count mismatch"):
        assert_main_modes_match_wout(wout=bad_m)

    bad_order = SimpleNamespace(
        **{**good.__dict__, "xm": np.asarray([0, 0, 1, 1, 1]), "xn": np.asarray([0, -3, -3, 0, 3])}
    )
    with pytest.raises(ValueError, match="xn ordering"):
        assert_main_modes_match_wout(wout=bad_order)


def test_read_wout_scalar_metadata_defaults_and_validation() -> None:
    variables = {
        "ns": _FakeNcVar(np.asarray([3])),
        "mpol": _FakeNcVar(np.asarray([2])),
        "ntor": _FakeNcVar(np.asarray([0])),
        "nfp": _FakeNcVar(np.asarray([1])),
    }

    assert _read_wout_scalar_metadata(variables, path=Path("wout_minimal.nc")) == (3, 2, 0, 1, False, 1)

    variables["lasym__logical__"] = _FakeNcVar(np.asarray([1]))
    variables["signgs"] = _FakeNcVar(np.asarray([-1]))
    assert _read_wout_scalar_metadata(variables, path=Path("wout_asym.nc")) == (3, 2, 0, 1, True, -1)

    bad = {**variables, "ns": _FakeNcVar(np.asarray([0]))}
    with pytest.raises(ValueError, match="Incomplete or masked wout scalar metadata"):
        _read_wout_scalar_metadata(bad, path=Path("wout_bad.nc"))


def test_wout_phi_profile_uses_explicit_field_or_half_mesh_fallback() -> None:
    explicit = {"phi": _FakeNcVar(np.asarray([0.0, 0.25, 1.0]))}
    np.testing.assert_allclose(
        _wout_phi_profile_from_variables(explicit, ns=3, phipf=np.asarray([2.0, 4.0, 6.0])),
        [0.0, 0.25, 1.0],
    )

    phipf = np.asarray([2.0, 4.0, 6.0])
    np.testing.assert_allclose(
        _wout_phi_profile_from_variables({}, ns=3, phipf=phipf),
        [0.0, 2.0, 5.0],
    )
    np.testing.assert_allclose(_wout_phi_profile_from_variables({}, ns=1, phipf=np.asarray([2.0])), [0.0])


def test_wout_schema_symbols_remain_reexported_from_wout() -> None:
    assert wout_module.WoutData is SchemaWoutData
    assert wout_module._bool_from_nc is schema_bool_from_nc
    assert wout_module._nc_scalar is schema_nc_scalar
    assert wout_module.assert_main_modes_match_wout is schema_assert_main_modes_match_wout

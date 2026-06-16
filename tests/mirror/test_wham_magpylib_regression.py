from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vmec_jax.mirror import MirrorConfig, MirrorResolution, mirror_boundary_from_vacuum_flux_tube
from vmec_jax.mirror.validation.wham import (
    build_wham_loop_table,
    default_wham_fixture_path,
    load_wham_fixture,
    wham_on_axis_mirror_ratio,
    wham_reference_field,
    wham_vacuum_field_rz,
)

pytestmark = pytest.mark.mirror


def test_wham_packaged_fixture_matches_repository_validation_fixture():
    root = Path(__file__).resolve().parents[2]
    validation_fixture = root / "validation" / "mirror" / "wham_coils.json"

    assert json.loads(default_wham_fixture_path().read_text()) == json.loads(validation_fixture.read_text())


def test_wham_fixture_expands_deterministically():
    fixture = load_wham_fixture()
    table = build_wham_loop_table(fixture)

    assert fixture.source == "coil_model_WHAM-2.txt"
    assert fixture.coil_centers_z_m == (-0.98, 0.98)
    assert fixture.num_loops == 2 * 8 * 310
    assert table.radius_m.shape == (fixture.num_loops,)
    assert table.z_m.shape == (fixture.num_loops,)
    assert table.current_a.shape == (fixture.num_loops,)
    assert np.isclose(table.radius_m.min(), fixture.r_in_hf_m)
    assert np.isclose(table.radius_m.max(), fixture.r_out_hf_m)
    assert np.isclose(abs(table.z_m.min()), abs(table.z_m.max()))
    assert np.allclose(table.current_a, 2000.0 * 17.0 / 17.51)


def test_wham_vacuum_field_matches_stored_reference_table():
    fixture = load_wham_fixture()
    reference = wham_reference_field(fixture)
    computed = wham_vacuum_field_rz(fixture.reference_points_r_m, fixture.reference_points_z_m, fixture)

    assert np.allclose(computed.br, reference.br, rtol=1.0e-11, atol=1.0e-12)
    assert np.allclose(computed.bz, reference.bz, rtol=1.0e-11, atol=1.0e-12)
    assert wham_on_axis_mirror_ratio(fixture, num_points=101) == pytest.approx(58.729237882095454, rel=1.0e-12)


def test_wham_vacuum_field_has_expected_symmetry():
    fixture = load_wham_fixture()
    r = np.asarray([0.0, 0.04, 0.08])
    z = np.asarray([0.35, 0.35, 0.35])
    plus = wham_vacuum_field_rz(r, z, fixture)
    minus = wham_vacuum_field_rz(r, -z, fixture)

    assert np.allclose(plus.bz, minus.bz, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(plus.br, -minus.br, rtol=1.0e-12, atol=1.0e-12)
    assert plus.bmag[0] < plus.bmag[-1] or plus.bmag[0] > 0.0


def test_wham_vacuum_flux_boundary_is_positive_symmetric_and_resolution_consistent():
    fixture = load_wham_fixture()
    coarse = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=11, mpol=0), z_min=-0.8, z_max=0.8).build_grid()
    fine = MirrorConfig(MirrorResolution(ns=5, ntheta=1, nxi=21, mpol=0), z_min=-0.8, z_max=0.8).build_grid()
    midplane_field = wham_vacuum_field_rz(0.0, 0.0, fixture)
    psi_value = 0.5 * float(midplane_field.bz) * 0.25**2

    coarse_boundary = mirror_boundary_from_vacuum_flux_tube(psi_value, coarse.z, fixture)
    fine_boundary = mirror_boundary_from_vacuum_flux_tube(psi_value, fine.z, fixture)
    coarse_radius = coarse_boundary.radius_on_grid(coarse)
    fine_on_coarse = fine_boundary.radius(coarse.xi)

    assert np.all(coarse_radius > 0.0)
    assert np.allclose(coarse_radius, coarse_radius[::-1], rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(coarse_radius, fine_on_coarse, rtol=3.0e-3, atol=3.0e-4)
    assert coarse_radius[coarse.nxi // 2] > coarse_radius[0]


@pytest.mark.magpylib
def test_wham_vacuum_field_matches_magpylib_when_available():
    magpylib = pytest.importorskip("magpylib")
    if not hasattr(magpylib, "current") or not hasattr(magpylib.current, "Circle"):
        pytest.skip("magpylib current.Circle API not available")

    fixture = load_wham_fixture()
    table = build_wham_loop_table(fixture)
    points_r = np.asarray([0.0, 0.04, 0.08])
    points_z = np.asarray([0.0, -0.5, 0.2])
    points_xyz = np.column_stack([points_r, np.zeros_like(points_r), points_z])
    collection = magpylib.Collection()
    for radius, z_center, current in zip(table.radius_m, table.z_m, table.current_a):
        collection.add(magpylib.current.Circle(current=current, diameter=2.0 * radius, position=(0.0, 0.0, z_center)))

    magpy_b = np.asarray(collection.getB(points_xyz), dtype=float)
    computed = wham_vacuum_field_rz(points_r, points_z, fixture)

    assert np.allclose(computed.br, magpy_b[:, 0], rtol=2.0e-5, atol=2.0e-7)
    assert np.allclose(computed.bz, magpy_b[:, 2], rtol=2.0e-5, atol=2.0e-7)

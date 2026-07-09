"""A/B tests: ``vmec_jax.core.mgrid`` vs the legacy mgrid stack and ESSOS.

Covers (plan.md §8):

- netCDF round-trip (read -> write -> read) equality on the bundled
  ``mgrid_cth_like_lasym_small.nc`` fixture,
- interpolation parity vs the legacy parity-proven
  ``vmec_jax.external_fields.mgrid_jax`` kernel at rtol 1e-12 on random
  in-domain points, including non-trivial extcur scaling,
- jit equivalence and grad of ``|B|^2`` w.r.t. extcur,
- cross-read consistency with ``essos.mgrid.MGrid`` (same netCDF layout).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from vmec_jax.core.errors import MgridNotFoundError  # noqa: E402
from vmec_jax.core.mgrid import MgridData, MgridField, read_mgrid, write_mgrid  # noqa: E402
from vmec_jax.external_fields.mgrid_jax import (  # noqa: E402
    interpolate_mgrid_bfield_jax as legacy_interpolate,
)
from vmec_jax.solvers.free_boundary.mgrid import load_mgrid as legacy_load  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MGRID_PATH = REPO / "examples" / "data" / "mgrid_cth_like_lasym_small.nc"

assert MGRID_PATH.is_file(), f"missing fixture {MGRID_PATH}"


@pytest.fixture(scope="module")
def data() -> MgridData:
    return read_mgrid(MGRID_PATH)


def _random_points(data: MgridData, n: int = 200, seed: int = 1234):
    """Random strictly-in-domain cylindrical points, one full torus in phi."""

    rng = np.random.default_rng(seed)
    eps_r = 1e-6 * (data.rmax - data.rmin)
    eps_z = 1e-6 * (data.zmax - data.zmin)
    r = rng.uniform(data.rmin + eps_r, data.rmax - eps_r, size=n)
    z = rng.uniform(data.zmin + eps_z, data.zmax - eps_z, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return r, phi, z


# ---------------------------------------------------------------------------
# Read + round-trip
# ---------------------------------------------------------------------------


def test_read_matches_legacy_loader(data: MgridData) -> None:
    legacy = legacy_load(MGRID_PATH, load_fields=True)
    meta = legacy.metadata
    assert (data.ir, data.jz, data.kp) == (meta.ir, meta.jz, meta.kp)
    assert (data.nfp, data.nextcur) == (meta.nfp, meta.nextcur)
    assert (data.rmin, data.rmax, data.zmin, data.zmax) == (
        meta.rmin,
        meta.rmax,
        meta.zmin,
        meta.zmax,
    )
    assert data.mgrid_mode == meta.mgrid_mode
    assert data.coil_groups == meta.coil_groups
    assert data.raw_coil_cur == meta.raw_coil_cur
    np.testing.assert_array_equal(data.br, legacy.br)
    np.testing.assert_array_equal(data.bp, legacy.bp)
    np.testing.assert_array_equal(data.bz, legacy.bz)


def test_round_trip_read_write_read(data: MgridData, tmp_path: Path) -> None:
    out = tmp_path / "mgrid_roundtrip.nc"
    write_mgrid(out, data)
    back = read_mgrid(out)

    assert (back.ir, back.jz, back.kp) == (data.ir, data.jz, data.kp)
    assert (back.nfp, back.nextcur) == (data.nfp, data.nextcur)
    assert (back.rmin, back.rmax, back.zmin, back.zmax) == (
        data.rmin,
        data.rmax,
        data.zmin,
        data.zmax,
    )
    assert back.mgrid_mode == data.mgrid_mode
    assert back.coil_groups == data.coil_groups
    assert back.raw_coil_cur == data.raw_coil_cur
    np.testing.assert_array_equal(back.br, data.br)
    np.testing.assert_array_equal(back.bp, data.bp)
    np.testing.assert_array_equal(back.bz, data.bz)

    # And the written file is readable by the legacy VMEC2000-parity loader.
    legacy = legacy_load(out, load_fields=True)
    np.testing.assert_array_equal(legacy.br, data.br)
    assert legacy.metadata.coil_groups == data.coil_groups


def test_missing_file_raises_mgrid_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_mgrid.nc"
    with pytest.raises(MgridNotFoundError):
        read_mgrid(missing)
    with pytest.raises(MgridNotFoundError):
        MgridField.from_file(missing)


# ---------------------------------------------------------------------------
# Interpolation A/B vs legacy JAX kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extcur_scale", [1.0, -2.75])
def test_interpolation_matches_legacy(data: MgridData, extcur_scale: float) -> None:
    r, phi, z = _random_points(data)
    extcur = extcur_scale * (1.0 + np.arange(data.nextcur, dtype=float))
    field = MgridField.from_mgrid_data(data, extcur=extcur)
    br, bp, bz = field.b_cyl(r, phi, z)

    br_ref, bp_ref, bz_ref = legacy_interpolate(
        jnp.asarray(data.br),
        jnp.asarray(data.bp),
        jnp.asarray(data.bz),
        extcur=jnp.asarray(extcur),
        r=r,
        z=z,
        phi=phi,
        rmin=data.rmin,
        rmax=data.rmax,
        zmin=data.zmin,
        zmax=data.zmax,
        nfp=data.nfp,
    )
    np.testing.assert_allclose(np.asarray(br), np.asarray(br_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(bp), np.asarray(bp_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(bz), np.asarray(bz_ref), rtol=1e-12, atol=0.0)
    assert np.max(np.abs(np.asarray(bp))) > 0.0  # non-degenerate field


def test_extcur_scaling_is_linear(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=50, seed=7)
    base = 1.0 + np.arange(data.nextcur, dtype=float)
    f1 = MgridField.from_mgrid_data(data, extcur=base)
    f3 = MgridField.from_mgrid_data(data, extcur=3.0 * base)
    for a, b in zip(f1.b_cyl(r, phi, z), f3.b_cyl(r, phi, z)):
        np.testing.assert_allclose(3.0 * np.asarray(a), np.asarray(b), rtol=1e-13, atol=0.0)


def test_jit_equivalence(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=100, seed=42)
    field = MgridField.from_mgrid_data(data)  # extcur defaults to raw currents

    @jax.jit
    def eval_field(f: MgridField, rr, pp, zz):
        return f.b_cyl(rr, pp, zz)

    eager = field.b_cyl(r, phi, z)
    jitted = eval_field(field, r, phi, z)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14, atol=0.0)


def test_grad_wrt_extcur_finite_nonzero(data: MgridData) -> None:
    r, phi, z = _random_points(data, n=64, seed=3)
    field = MgridField.from_mgrid_data(data)

    def bsq_sum(extcur):
        f = MgridField.from_mgrid_data(data, extcur=extcur)
        br, bp, bz = f.b_cyl(r, phi, z)
        return jnp.sum(br**2 + bp**2 + bz**2)

    g = jax.grad(bsq_sum)(jnp.asarray(field.extcur))
    g_np = np.asarray(g)
    assert g_np.shape == (data.nextcur,)
    assert np.all(np.isfinite(g_np))
    assert np.max(np.abs(g_np)) > 0.0


# ---------------------------------------------------------------------------
# ESSOS cross-read
# ---------------------------------------------------------------------------


def test_essos_reads_same_grid_and_fields(data: MgridData) -> None:
    essos_mgrid = pytest.importorskip("essos.mgrid")
    eg = essos_mgrid.MGrid.from_file(MGRID_PATH)

    # ESSOS naming: nr/nz/nphi == ir/jz/kp; same extents and nfp.
    assert (eg.nr, eg.nz, eg.nphi, eg.nfp) == (data.ir, data.jz, data.kp, data.nfp)
    assert (eg.rmin, eg.rmax, eg.zmin, eg.zmax) == (
        data.rmin,
        data.rmax,
        data.zmin,
        data.zmax,
    )
    assert eg.n_ext_cur == data.nextcur
    assert eg.mode == data.mgrid_mode
    np.testing.assert_array_equal(
        np.asarray(eg.raw_coil_current), np.asarray(data.raw_coil_cur)
    )
    # ESSOS strips via _unpack (whitespace only) — same convention as ours.
    assert tuple(eg.coil_names) == data.coil_groups

    # Per-group field tables: ESSOS stores a list of (nphi, nz, nr) arrays,
    # ours is stacked (nextcur, kp, jz, ir) — identical per-group content.
    for i in range(data.nextcur):
        np.testing.assert_array_equal(np.asarray(eg.br_arr[i]), data.br[i])
        np.testing.assert_array_equal(np.asarray(eg.bp_arr[i]), data.bp[i])
        np.testing.assert_array_equal(np.asarray(eg.bz_arr[i]), data.bz[i])


def test_essos_reads_our_written_file(data: MgridData, tmp_path: Path) -> None:
    essos_mgrid = pytest.importorskip("essos.mgrid")
    out = tmp_path / "mgrid_for_essos.nc"
    write_mgrid(out, data)
    eg = essos_mgrid.MGrid.from_file(out)
    assert (eg.nr, eg.nz, eg.nphi, eg.nfp) == (data.ir, data.jz, data.kp, data.nfp)
    assert eg.n_ext_cur == data.nextcur
    for i in range(data.nextcur):
        np.testing.assert_array_equal(np.asarray(eg.br_arr[i]), data.br[i])
        np.testing.assert_array_equal(np.asarray(eg.bp_arr[i]), data.bp[i])
        np.testing.assert_array_equal(np.asarray(eg.bz_arr[i]), data.bz[i])

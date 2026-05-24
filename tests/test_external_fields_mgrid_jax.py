from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import (
    MGridFieldParams,
    interpolate_mgrid_bfield_jax,
    sample_external_field_cylindrical,
    sample_mgrid_field_cylindrical,
)
from vmec_jax.free_boundary import MGridData, MGridMetadata, interpolate_mgrid_bfield


def _affine_mgrid_params():
    from vmec_jax._compat import jnp

    rmin, rmax = 1.0, 2.0
    zmin, zmax = -0.8, 0.7
    nfp = 1
    nextcur, kp, jz, ir = 2, 6, 5, 4
    r_grid = jnp.linspace(rmin, rmax, ir)
    z_grid = jnp.linspace(zmin, zmax, jz)
    phi_grid = jnp.arange(kp, dtype=float) * ((2.0 * jnp.pi / nfp) / kp)

    coeffs = {
        "br": (
            jnp.asarray([1.2, -0.4]),
            jnp.asarray([0.7, 0.5]),
            jnp.asarray([0.3, -0.2]),
            jnp.asarray([0.1, 0.4]),
        ),
        "bphi": (
            jnp.asarray([-0.6, 0.2]),
            jnp.asarray([1.1, -0.3]),
            jnp.asarray([0.05, 0.4]),
            jnp.asarray([-0.2, 0.3]),
        ),
        "bz": (
            jnp.asarray([0.9, 0.8]),
            jnp.asarray([-0.2, 0.6]),
            jnp.asarray([0.15, 0.1]),
            jnp.asarray([0.0, -0.1]),
        ),
    }

    def build(component):
        a, b, c, d = coeffs[component]
        return (
            a[:, None, None, None] * r_grid[None, None, None, :]
            + b[:, None, None, None] * z_grid[None, None, :, None]
            + c[:, None, None, None] * phi_grid[None, :, None, None]
            + d[:, None, None, None]
        )

    params = MGridFieldParams(
        br=build("br"),
        bphi=build("bphi"),
        bz=build("bz"),
        extcur=jnp.asarray([0.8, -1.3]),
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=nfp,
    )
    return params, coeffs


def _expected_affine(coeffs, component, extcur, r, z, phi):
    a, b, c, d = coeffs[component]
    a = np.asarray(a)[:, None, None]
    b = np.asarray(b)[:, None, None]
    c = np.asarray(c)[:, None, None]
    d = np.asarray(d)[:, None, None]
    per_current = a * np.asarray(r)[None, ...] + b * np.asarray(z)[None, ...] + c * np.asarray(phi)[None, ...] + d
    return np.sum(np.asarray(extcur)[:, None, None] * per_current, axis=0)


def test_mgrid_jax_affine_values_match_exact_and_legacy_interpolator():
    enable_x64(True)
    params, coeffs = _affine_mgrid_params()
    R = np.asarray([[1.12, 1.44], [1.61, 1.83]])
    Z = np.asarray([[-0.52, -0.21], [0.05, 0.31]])
    phi = np.asarray([[0.21, 0.54], [0.73, 1.02]])

    actual = sample_mgrid_field_cylindrical(params, R, Z, phi)
    dispatch = sample_external_field_cylindrical("mgrid", None, params, R, Z, phi)
    expected = (
        _expected_affine(coeffs, "br", params.extcur, R, Z, phi),
        _expected_affine(coeffs, "bphi", params.extcur, R, Z, phi),
        _expected_affine(coeffs, "bz", params.extcur, R, Z, phi),
    )

    for got, got_dispatch, want in zip(actual, dispatch, expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=2.0e-14, atol=1.0e-13)
        np.testing.assert_allclose(got_dispatch, got, rtol=0.0, atol=0.0)

    legacy = interpolate_mgrid_bfield(
        MGridData(
            metadata=MGridMetadata(
                path="synthetic",
                ir=int(params.br.shape[3]),
                jz=int(params.br.shape[2]),
                kp=int(params.br.shape[1]),
                nfp=params.nfp,
                nextcur=int(params.br.shape[0]),
                rmin=params.rmin,
                rmax=params.rmax,
                zmin=params.zmin,
                zmax=params.zmax,
                mgrid_mode="S",
                coil_groups=(),
                raw_coil_cur=(),
            ),
            br=np.asarray(params.br),
            bp=np.asarray(params.bphi),
            bz=np.asarray(params.bz),
        ),
        r=R,
        z=Z,
        phi=phi,
        extcur=tuple(np.asarray(params.extcur)),
    )
    for got, want in zip(actual, legacy, strict=True):
        np.testing.assert_allclose(got, want, rtol=2.0e-14, atol=1.0e-13)


def test_mgrid_jax_gradient_wrt_extcur_matches_per_current_values():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params, coeffs = _affine_mgrid_params()
    R = jnp.asarray([1.22, 1.48])
    Z = jnp.asarray([-0.42, -0.11])
    phi = jnp.asarray([0.25, 0.65])

    def objective(extcur):
        trial = params.with_arrays(extcur=extcur)
        br, _bphi, _bz = sample_mgrid_field_cylindrical(trial, R, Z, phi)
        return jnp.sum(br)

    grad_extcur = np.asarray(jax.grad(objective)(params.extcur))
    a, b, c, d = coeffs["br"]
    expected = np.asarray(jnp.sum(a[:, None] * R[None, :] + b[:, None] * Z[None, :] + c[:, None] * phi[None, :] + d[:, None], axis=1))

    np.testing.assert_allclose(grad_extcur, expected, rtol=2.0e-14, atol=1.0e-13)


def test_mgrid_jax_gradient_wrt_field_value_matches_trilinear_weight():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params, _coeffs = _affine_mgrid_params()
    R = 1.31
    Z = -0.37
    phi = 0.41

    def objective(br_values):
        trial = params.with_arrays(br=br_values)
        br, _bphi, _bz = sample_mgrid_field_cylindrical(trial, R, Z, phi)
        return br

    grad_br = np.asarray(jax.grad(objective)(params.br))

    ir = int(params.br.shape[3])
    jz = int(params.br.shape[2])
    kp = int(params.br.shape[1])
    fr = (R - params.rmin) * ((ir - 1) / (params.rmax - params.rmin))
    fz = (Z - params.zmin) * ((jz - 1) / (params.zmax - params.zmin))
    fk = phi * (kp / (2.0 * np.pi / params.nfp))
    i0 = int(np.floor(fr))
    j0 = int(np.floor(fz))
    k0 = int(np.floor(fk))
    expected_weight = float(params.extcur[0]) * (1.0 - (fr - i0)) * (1.0 - (fz - j0)) * (1.0 - (fk - k0))

    np.testing.assert_allclose(grad_br[0, k0, j0, i0], expected_weight, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(np.sum(grad_br), np.sum(np.asarray(params.extcur)), rtol=1.0e-13, atol=1.0e-13)


def test_mgrid_jax_coordinate_derivative_matches_affine_coefficients():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    params, coeffs = _affine_mgrid_params()

    def objective(coords):
        R, Z, phi = coords
        br, _bphi, _bz = interpolate_mgrid_bfield_jax(
            params.br,
            params.bphi,
            params.bz,
            extcur=params.extcur,
            r=R,
            z=Z,
            phi=phi,
            rmin=params.rmin,
            rmax=params.rmax,
            zmin=params.zmin,
            zmax=params.zmax,
            nfp=params.nfp,
        )
        return br

    coords = jnp.asarray([1.37, -0.29, 0.52])
    exact = np.asarray(jax.grad(objective)(coords))
    a, b, c, _d = coeffs["br"]
    expected = np.asarray(
        [
            jnp.sum(params.extcur * a),
            jnp.sum(params.extcur * b),
            jnp.sum(params.extcur * c),
        ]
    )

    np.testing.assert_allclose(exact, expected, rtol=3.0e-13, atol=1.0e-13)

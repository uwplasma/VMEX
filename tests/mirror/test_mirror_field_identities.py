from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import IPrimeProfile, MirrorConfig, MirrorResolution, PsiPrimeProfile
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.energy import MU0, magnetic_energy_axisym
from vmec_jax.mirror.kernels.fields import (
    contravariant_fluxes_from_lambda,
    divergence_free_numerator,
    evaluate_axisym_field,
)
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry

pytestmark = pytest.mark.mirror


def _grid(*, ns=17, ntheta=1, nxi=33, mpol=0, length=1.3):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, ntheta=ntheta, nxi=nxi, mpol=mpol),
        z_min=-length,
        z_max=length,
    ).build_grid()


def test_flux_form_is_discretely_divergence_free_for_smooth_lambda():
    grid = _grid(ns=9, ntheta=17, nxi=25, mpol=4)
    s = grid.s_full[:, None, None]
    theta = grid.theta[None, :, None]
    xi = grid.xi[None, None, :]
    t3 = np.polynomial.chebyshev.chebval(xi, [0.0, 0.0, 0.0, 1.0])
    lam = s * (1.0 - s) * np.sin(2.0 * theta) * (1.0 - xi**2) * t3

    fluxes = contravariant_fluxes_from_lambda(
        lam,
        grid,
        psi_prime=PsiPrimeProfile.polynomial([0.2, -0.03]),
        i_prime=IPrimeProfile.constant(0.07),
    )
    residual = divergence_free_numerator(fluxes, grid)
    assert np.max(np.abs(residual)) < 2.0e-11


def test_lambda_flux_function_gauge_does_not_change_axisymmetric_field_or_energy():
    grid = _grid(ns=13, nxi=31, length=1.1)
    boundary = MirrorBoundary.polynomial_radius(r0=0.28, a2=0.12)
    base = MirrorStateAxisym.from_boundary(grid, boundary)
    s = grid.s_full[:, None]
    xi = grid.xi[None, :]
    lam = 0.04 * s * (1.0 - s) * xi * (1.0 - xi**2)
    gauge = 1.0 + 0.2 * grid.s_full[:, None] ** 2
    state = MirrorStateAxisym(a=base.a, lam=lam)
    shifted = MirrorStateAxisym(a=base.a, lam=lam + gauge)
    geom = evaluate_axisym_geometry(state, grid)
    psi = PsiPrimeProfile.constant(0.015)
    current = IPrimeProfile.constant(0.02)

    field = evaluate_axisym_field(state, grid, geom, psi_prime=psi, i_prime=current)
    shifted_field = evaluate_axisym_field(shifted, grid, geom, psi_prime=psi, i_prime=current)
    assert np.allclose(shifted_field.b_sup_theta, field.b_sup_theta)
    assert np.allclose(shifted_field.b_sup_xi, field.b_sup_xi)
    assert np.allclose(shifted_field.bmag, field.bmag)
    assert np.isclose(
        magnetic_energy_axisym(shifted_field, geom, grid),
        magnetic_energy_axisym(field, geom, grid),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_constant_axial_field_in_cylinder_has_expected_components_and_energy():
    radius = 0.31
    length = 1.6
    b0 = 2.3
    grid = _grid(ns=15, nxi=21, length=length)
    boundary = MirrorBoundary.constant_radius(radius)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    geom = evaluate_axisym_geometry(state, grid)
    psi = PsiPrimeProfile.constant(0.5 * radius**2 * b0)
    field = evaluate_axisym_field(state, grid, geom, psi_prime=psi, i_prime=IPrimeProfile.zero())

    assert np.allclose(field.b_r, 0.0, atol=2.0e-14)
    assert np.allclose(field.b_phi, 0.0, atol=2.0e-14)
    assert np.allclose(field.b_z, b0, atol=2.0e-14)
    assert np.allclose(field.bmag, b0, atol=2.0e-14)
    assert np.allclose(field.b_cov_xi, length * b0, atol=2.0e-14)
    assert np.allclose(field.b_x, 0.0, atol=2.0e-14)
    assert np.allclose(field.b_y, 0.0, atol=2.0e-14)

    expected_energy = b0**2 * geom.volume / (2.0 * MU0)
    assert np.isclose(magnetic_energy_axisym(field, geom, grid), expected_energy, rtol=2.0e-14)

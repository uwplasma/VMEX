from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import IPrimeProfile, MirrorConfig, MirrorResolution, PressureProfile, PsiPrimeProfile
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorState3D, MirrorStateAxisym
from vmec_jax.mirror.kernels.constraints import lambda_surface_average_3d, project_state_3d
from vmec_jax.mirror.kernels.energy import (
    magnetic_energy_3d,
    magnetic_energy_axisym,
    pressure_energy_3d,
    pressure_energy_axisym,
)
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field, evaluate_field_3d
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry, evaluate_geometry_3d

pytestmark = pytest.mark.mirror


def _grid(*, ns=17, ntheta=17, nxi=33, mpol=4, length=1.4):
    return MirrorConfig(
        resolution=MirrorResolution(ns=ns, ntheta=ntheta, nxi=nxi, mpol=mpol),
        z_min=-length,
        z_max=length,
    ).build_grid()


def test_cosine_modulated_boundary_geometry_matches_analytic_metrics():
    r0 = 0.29
    a2 = 0.12
    a4 = 0.03
    epsilon = 0.08
    mode = 2
    length = 1.3
    grid = _grid(ns=19, ntheta=25, nxi=37, mpol=5, length=length)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=r0, a2=a2, a4=a4, epsilon=epsilon, theta_mode=mode)
    state = MirrorState3D.from_boundary(grid, boundary)

    geom = evaluate_geometry_3d(state, grid)
    theta = grid.theta[None, :, None]
    xi = grid.xi[None, None, :]
    s = grid.s_full[:, None, None]
    axial = r0 * (1.0 + a2 * xi**2 + a4 * xi**4)
    axial_xi = r0 * (2.0 * a2 * xi + 4.0 * a4 * xi**3)
    modulation = 1.0 + epsilon * np.cos(mode * theta)
    a = axial * modulation
    a_theta = -epsilon * mode * axial * np.sin(mode * theta)
    a_xi = axial_xi * modulation

    assert np.allclose(geom.r[-1], boundary.radius_on_grid_3d(grid))
    assert np.all(geom.sqrtg > 0.0)
    assert np.allclose(geom.sqrtg, 0.5 * a**2 * length, atol=3.0e-12, rtol=3.0e-12)
    assert np.allclose(geom.g_thetatheta, s * (a_theta**2 + a**2), atol=3.0e-12, rtol=3.0e-12)
    assert np.allclose(geom.g_thetaxi, s * a_theta * a_xi, atol=3.0e-12, rtol=3.0e-12)
    assert np.allclose(geom.g_stheta, 0.5 * a * a_theta, atol=3.0e-12, rtol=3.0e-12)
    expected_volume = 0.5 * length * np.einsum("j,k,jk->", grid.w_theta, grid.w_xi, a[0] ** 2)
    assert np.isclose(geom.volume, expected_volume, atol=3.0e-13, rtol=3.0e-13)


def test_3d_constraints_fix_boundary_ends_axis_and_lambda_gauge():
    grid = _grid(ns=7, ntheta=13, nxi=17, mpol=4, length=1.0)
    boundary = MirrorBoundary.cosine_modulated_radius(r0=0.32, a2=-0.08, a4=0.03, epsilon=0.06, theta_mode=2)
    rng = np.random.default_rng(314159)
    state = MirrorState3D(
        a=0.2 + 0.08 * rng.random((grid.ns, grid.ntheta, grid.nxi)),
        lam=1.0 + rng.normal(size=(grid.ns, grid.ntheta, grid.nxi)),
    )

    projected = project_state_3d(state, grid, boundary)
    boundary_radius = boundary.radius_on_grid_3d(grid)
    assert np.allclose(projected.a[-1], boundary_radius)
    assert np.allclose(projected.a[:, :, 0], boundary_radius[:, 0][None, :])
    assert np.allclose(projected.a[:, :, -1], boundary_radius[:, -1][None, :])
    assert np.allclose(projected.a[0], projected.a[1])
    assert np.allclose(lambda_surface_average_3d(projected.lam, grid), 0.0, atol=3.0e-15)


def test_3d_field_and_energy_reduce_to_axisymmetric_case():
    radius = 0.27
    length = 1.1
    grid = _grid(ns=11, ntheta=17, nxi=29, mpol=4, length=length)
    boundary = MirrorBoundary.polynomial_radius(r0=radius, a2=0.15, a4=0.02)
    state_axisym = MirrorStateAxisym.from_boundary(grid, boundary)
    state_3d = MirrorState3D.from_boundary(grid, boundary)
    geom_axisym = evaluate_axisym_geometry(state_axisym, grid)
    geom_3d = evaluate_geometry_3d(state_3d, grid)

    assert np.allclose(geom_3d.sqrtg, geom_axisym.sqrtg[:, None, :])
    assert np.allclose(geom_3d.g_xixi, geom_axisym.g_xixi[:, None, :])

    psi = PsiPrimeProfile.constant(0.5 * radius**2)
    current = IPrimeProfile.constant(0.0)
    field_axisym = evaluate_axisym_field(state_axisym, grid, geom_axisym, psi_prime=psi, i_prime=current)
    field_3d = evaluate_field_3d(state_3d, grid, geom_3d, psi_prime=psi, i_prime=current)
    pressure = PressureProfile.polynomial([12.0, -6.0], gamma=2.0)

    assert np.allclose(field_3d.bmag, field_axisym.bmag[:, None, :])
    assert np.isclose(magnetic_energy_3d(field_3d, geom_3d, grid), magnetic_energy_axisym(field_axisym, geom_axisym, grid))
    assert np.isclose(pressure_energy_3d(pressure, geom_3d, grid), pressure_energy_axisym(pressure, geom_axisym, grid))

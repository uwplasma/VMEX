from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import IPrimeProfile, MirrorConfig, MirrorResolution, PressureProfile, PsiPrimeProfile
from vmec_jax.mirror.core.boundary import MirrorBoundary
from vmec_jax.mirror.core.state import MirrorStateAxisym
from vmec_jax.mirror.kernels.energy import pressure_energy_axisym, total_energy_axisym
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry
from vmec_jax.mirror.kernels.residuals import field_diagnostics

pytestmark = pytest.mark.mirror


def _cylinder_case(*, radius=0.24, length=1.2, ns=17, nxi=33):
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=ns, ntheta=1, nxi=nxi, mpol=0),
        z_min=-length,
        z_max=length,
    ).build_grid()
    boundary = MirrorBoundary.constant_radius(radius)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    return grid, state, evaluate_axisym_geometry(state, grid)


def test_pressure_energy_matches_cylinder_analytic_integral():
    radius = 0.34
    length = 1.5
    p0 = 1200.0
    gamma = 5.0 / 3.0
    grid, _, geom = _cylinder_case(radius=radius, length=length)
    pressure = PressureProfile.polynomial([p0, -p0], gamma=gamma)

    expected = geom.volume * (0.5 * p0) / (gamma - 1.0)
    assert np.isclose(pressure_energy_axisym(pressure, geom, grid), expected, rtol=2.0e-14, atol=2.0e-10)
    assert np.allclose(pressure.derivative(grid.s_full), -p0)


def test_total_energy_combines_magnetic_and_pressure_terms():
    radius = 0.29
    length = 1.0
    b0 = 1.7
    p0 = 25.0
    grid, state, geom = _cylinder_case(radius=radius, length=length)
    field = evaluate_axisym_field(
        state,
        grid,
        geom,
        psi_prime=PsiPrimeProfile.constant(0.5 * radius**2 * b0),
        i_prime=IPrimeProfile.zero(),
    )
    pressure = PressureProfile.constant(p0, gamma=2.0)
    energy = total_energy_axisym(field, pressure, geom, grid)

    assert energy.magnetic > 0.0
    assert np.isclose(energy.pressure, geom.volume * p0)
    assert np.isclose(energy.total, energy.magnetic + energy.pressure)


def test_field_diagnostics_reports_mirror_ratio_for_flared_area_field():
    radius = 0.25
    length = 1.2
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=13, ntheta=1, nxi=41, mpol=0),
        z_min=-length,
        z_max=length,
    ).build_grid()
    boundary = MirrorBoundary.polynomial_radius(r0=radius, a2=0.2)
    state = MirrorStateAxisym.from_boundary(grid, boundary)
    geom = evaluate_axisym_geometry(state, grid)
    field = evaluate_axisym_field(
        state,
        grid,
        geom,
        psi_prime=PsiPrimeProfile.constant(0.5 * radius**2),
        i_prime=IPrimeProfile.zero(),
    )
    diagnostics = field_diagnostics(field, grid)

    assert diagnostics.min_bmag > 0.0
    assert diagnostics.max_bmag >= diagnostics.min_bmag
    assert diagnostics.mirror_ratio > 1.0
    assert diagnostics.max_divergence_numerator == 0.0

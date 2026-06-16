from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.mirror import (
    IPrimeProfile,
    MirrorConfig,
    MirrorResolution,
    MirrorSolveOptions,
    PressureProfile,
    PsiPrimeProfile,
    mirror_boundary_from_on_axis_bz,
    on_axis_mirror_ratio,
    run_mirror_fixed_boundary,
    two_coil_on_axis_bz,
    two_coil_on_axis_mirror_ratio,
)
from vmec_jax.mirror.kernels.fields import evaluate_axisym_field
from vmec_jax.mirror.kernels.geometry import evaluate_axisym_geometry
from vmec_jax.mirror.validation.wham import circular_loop_field_rz

pytestmark = pytest.mark.mirror


def test_two_coil_on_axis_formula_matches_full_loop_on_axis_branch():
    z = np.linspace(-1.0, 1.0, 17)
    radius = 0.35
    separation = 2.0
    current = 1.0e6

    analytic = two_coil_on_axis_bz(z, coil_radius_m=radius, separation_m=separation, current_a=current)
    full_loop = circular_loop_field_rz(
        np.zeros_like(z), z + 0.5 * separation, loop_radius_m=radius, current_a=current
    ).bz
    full_loop += circular_loop_field_rz(
        np.zeros_like(z), z - 0.5 * separation, loop_radius_m=radius, current_a=current
    ).bz

    assert np.allclose(analytic, full_loop, rtol=2.0e-15, atol=2.0e-15)
    assert two_coil_on_axis_mirror_ratio(coil_radius_m=radius, separation_m=separation, current_a=current) > 1.0


def test_two_coil_flux_tube_mirror_axis_field_matches_analytic_bz():
    coil_radius = 0.35
    separation = 2.0
    current = 1.0e6
    midplane_radius = 0.3
    config = MirrorConfig(MirrorResolution(ns=9, ntheta=1, nxi=33, mpol=0), z_min=-1.0, z_max=1.0)
    grid = config.build_grid()
    analytic_bz = two_coil_on_axis_bz(
        grid.z,
        coil_radius_m=coil_radius,
        separation_m=separation,
        current_a=current,
    )
    psi_value = 0.5 * abs(
        float(two_coil_on_axis_bz(0.0, coil_radius_m=coil_radius, separation_m=separation, current_a=current))
    )
    psi_value *= midplane_radius**2
    boundary = mirror_boundary_from_on_axis_bz(psi_value, grid.z, analytic_bz)

    result = run_mirror_fixed_boundary(
        config,
        boundary,
        psi_prime=PsiPrimeProfile.constant(psi_value),
        i_prime=IPrimeProfile.zero(),
        pressure=PressureProfile.zero(),
        options=MirrorSolveOptions(optimizer="lbfgs", maxiter=4, tolerance=1.0e-10, mu0=1.0),
    )
    geometry = evaluate_axisym_geometry(result.state, result.grid)
    field = evaluate_axisym_field(
        result.state, result.grid, geometry, psi_prime=result.psi_prime, i_prime=result.i_prime
    )
    mirror_bz = field.b_z[0]

    assert np.allclose(mirror_bz, analytic_bz, rtol=3.0e-13, atol=3.0e-13)
    assert on_axis_mirror_ratio(mirror_bz) == pytest.approx(on_axis_mirror_ratio(analytic_bz), rel=3.0e-13)
    assert result.final_trace.mirror_ratio > 1.0

"""Implicit derivatives of converged free-boundary mirror equilibria."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.core.coils import CoilSet, two_coil_on_axis_bz  # noqa: E402
from vmec_jax.mirror import (  # noqa: E402
    FreeBoundaryAdjointConfig,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    build_vacuum_grid,
    free_boundary_adjoint,
    free_boundary_parameters,
    solve_free_boundary_cli,
)


@pytest.fixture(autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def test_free_boundary_adjoint_rejects_unconverged_and_3d_results() -> None:
    axisymmetric_grid = MirrorConfig(
        resolution=MirrorResolution(ns=3, nxi=5)
    ).build_grid()
    parameters = free_boundary_parameters(
        object(), axial_flux_derivative=0.1
    )
    with pytest.raises(ValueError, match="converged"):
        free_boundary_adjoint(
            type("Result", (), {"converged": False})(),
            parameters,
            axisymmetric_grid,
            lambda *_: 0.0,
        )

    nonaxisymmetric_grid = MirrorConfig(
        resolution=MirrorResolution(ns=3, mpol=1, ntheta=3, nxi=5)
    ).build_grid()
    with pytest.raises(ValueError, match="axisymmetry"):
        free_boundary_adjoint(
            type("Result", (), {"converged": True})(),
            parameters,
            nonaxisymmetric_grid,
            lambda *_: 0.0,
        )


@pytest.mark.full
def test_free_boundary_coil_adjoint_matches_central_difference() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    vacuum_grid = build_vacuum_grid(grid, nrho=5)
    dofs = np.zeros((2, 3, 3))
    dofs[:, 0, 2] = 0.9
    dofs[:, 1, 1] = 0.9
    dofs[:, 2, 0] = [-1.0, 1.0]
    coils = CoilSet(
        jnp.asarray(dofs), jnp.asarray([2.0e5, 2.0e5]), n_segments=64
    )
    on_axis = two_coil_on_axis_bz(
        jnp.asarray(grid.z), coil_radius=0.9, separation=2.0, current=2.0e5
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    initial_boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    solve_options = dict(
        outer_radius=0.1,
        axial_flux_derivative=flux,
        vacuum_backend="exterior",
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
        exterior_high_order_cap_panels=True,
        exterior_curved_side_geometry=True,
        require_convergence=True,
    )
    result = solve_free_boundary_cli(
        initial_boundary, grid, vacuum_grid, config, coils, **solve_options
    )
    parameters = free_boundary_parameters(coils, axial_flux_derivative=flux)

    def quantity(boundary, _state, _energy, _vacuum):
        return boundary.radius_scale[0, center]

    adjoint = free_boundary_adjoint(
        result,
        parameters,
        grid,
        quantity,
        config=FreeBoundaryAdjointConfig(
            axisymmetric_ntheta=8,
            exterior_order=6,
            spectral_side_density=True,
            spectral_cap_density=True,
            curved_side_geometry=True,
            rtol=1.0e-8,
        ),
    )
    dofs_direction = np.zeros_like(dofs)
    dofs_direction[:, 0, 2] = 0.01
    dofs_direction[:, 1, 1] = 0.01
    current_direction = jnp.asarray([1.0e4, -5.0e3])
    predicted = float(
        jnp.vdot(
            adjoint.gradient.coilset.base_curve_dofs, jnp.asarray(dofs_direction)
        )
        + jnp.vdot(
            adjoint.gradient.coilset.base_currents, current_direction
        )
    )

    epsilon = 1.0e-4
    values = []
    for sign in (-1.0, 1.0):
        varied_coils = coils.with_arrays(
            base_curve_dofs=coils.base_curve_dofs
            + sign * epsilon * jnp.asarray(dofs_direction),
            base_currents=coils.base_currents
            + sign * epsilon * current_direction,
        )
        varied = solve_free_boundary_cli(
            result.boundary,
            grid,
            vacuum_grid,
            config,
            varied_coils,
            initial_state=result.plasma_state,
            **solve_options,
        )
        values.append(float(quantity(varied.boundary, None, None, None)))
    finite_difference = (values[1] - values[0]) / (2.0 * epsilon)
    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert adjoint.converged
    assert adjoint.relative_residual < 1.0e-8
    np.testing.assert_allclose(predicted, finite_difference, rtol=2.0e-4)

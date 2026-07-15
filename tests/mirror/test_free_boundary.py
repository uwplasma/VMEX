"""Free-space mirror free-boundary equilibrium and diagnostic tests."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from vmec_jax.mirror import (  # noqa: E402
    BiMaxwellianPressureClosure,
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    MirrorState,
    TabulatedPressureClosure,
    solve_beta_scan_cli,
)
from vmec_jax.mirror.free_boundary import solve_axisymmetric_beta_scan_cli  # noqa: E402
from vmec_jax.mirror.output import (  # noqa: E402
    boundary_fourier_amplitudes,
    boundary_fourier_norms,
    summarize_axisymmetric_beta_scan,
    summarize_nonaxisymmetric_beta_scan,
)
from vmec_jax.mirror.output import (  # noqa: E402
    FreeBoundaryRestart,
    load_free_boundary_restart,
    save_free_boundary_restart,
)
from vmec_jax.mirror.forces import MU0  # noqa: E402
from vmec_jax.mirror.geometry import magnetic_field_squared  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _enable_solver_jit():
    previous = bool(jax.config.jax_disable_jit)
    jax.config.update("jax_disable_jit", False)
    yield
    jax.config.update("jax_disable_jit", previous)


def test_free_boundary_restart_roundtrip_is_compact_and_grid_checked(tmp_path) -> None:
    config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=9))
    plasma_grid = config.build_grid()
    boundary = MirrorBoundary.from_radius(0.3, plasma_grid)
    state = MirrorState.from_boundary(boundary, plasma_grid)
    restart = FreeBoundaryRestart(
        boundary=boundary,
        plasma_state=state,
        mass_scale=1.25,
    )

    path = save_free_boundary_restart(tmp_path / "beta_003", restart)
    loaded = load_free_boundary_restart(path, plasma_grid)

    assert path.suffix == ".npz"
    assert path.stat().st_size < 4096
    np.testing.assert_array_equal(loaded.boundary.radius_scale, boundary.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.radius_scale, state.radius_scale)
    np.testing.assert_array_equal(loaded.plasma_state.lambda_stream, state.lambda_stream)
    assert loaded.mass_scale == restart.mass_scale

    mismatched = MirrorConfig(resolution=MirrorResolution(ns=9, mpol=0, ntheta=1, nxi=9)).build_grid()
    with pytest.raises(ValueError, match="plasma state"):
        load_free_boundary_restart(path, mismatched)


def test_boundary_fourier_amplitudes_are_grid_independent() -> None:
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 7, endpoint=False)
    axial = jnp.linspace(-1.0, 1.0, 5)
    radius = 0.2 + 0.01 * jnp.cos(theta)[:, None] * (1.0 - axial**2)[None, :] + 0.004 * jnp.sin(2.0 * theta)[:, None]
    amplitudes = boundary_fourier_amplitudes(MirrorBoundary(radius))

    np.testing.assert_allclose(amplitudes[0], 0.2, atol=3.0e-17)
    np.testing.assert_allclose(amplitudes[1], 0.01 * (1.0 - axial**2), atol=5.0e-17)
    np.testing.assert_allclose(amplitudes[2], 0.004, atol=5.0e-17)
    np.testing.assert_allclose(amplitudes[3], 0.0, atol=5.0e-17)


def test_boundary_fourier_norms_do_not_use_a_symmetry_zero() -> None:
    grid = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=2, ntheta=7, nxi=9)
    ).build_grid()
    theta = jnp.asarray(grid.theta)[:, None]
    xi = jnp.asarray(grid.xi)[None, :]
    boundary = MirrorBoundary(0.2 + 0.03 * xi * jnp.cos(theta))

    l2, maximum = boundary_fourier_norms(boundary, grid)
    core_l2, core_maximum = boundary_fourier_norms(
        boundary, grid, central_fraction=0.75
    )

    np.testing.assert_allclose(l2[1], 0.03 / np.sqrt(3.0), rtol=2.0e-14)
    np.testing.assert_allclose(maximum[1], 0.03, rtol=2.0e-14)
    expected_interior = 0.03 * 0.75 / np.sqrt(3.0)
    np.testing.assert_allclose(core_l2[1], expected_interior, rtol=2.0e-14)
    np.testing.assert_allclose(core_maximum[1], 0.03 * 0.75, rtol=2.0e-14)
    np.testing.assert_allclose(boundary_fourier_amplitudes(boundary)[1, grid.nxi // 2], 0.0, atol=5e-17)


def _external_mirror_field(points):
    """Curl-free, divergence-free paraxial mirror field."""

    points = jnp.asarray(points)
    x, y, z = jnp.moveaxis(points, -1, 0)
    curvature = 0.02
    return jnp.stack(
        (
            -curvature * x * z,
            -curvature * y * z,
            0.08 + curvature * (z**2 - 0.5 * (x**2 + y**2)),
        ),
        axis=-1,
    )


def _on_axis_mirror_field(z, **_unused):
    return 0.08 + 0.02 * jnp.asarray(z) ** 2


def _nonaxisymmetric_mirror_field(points):
    field = _external_mirror_field(points)
    return field.at[..., 0].add(0.004)


@pytest.mark.full
def test_unbounded_exterior_free_boundary_beta_scan_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=200,
    )
    plasma_grid = config.build_grid()
    on_axis = _on_axis_mirror_field(
        jnp.asarray(plasma_grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = plasma_grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    betas = jnp.asarray([0.0, 0.10, 0.25, 0.50])
    results = solve_axisymmetric_beta_scan_cli(
        MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid),
        plasma_grid,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        exterior_ntheta=8,
        exterior_order=6,
        exterior_spectral_side_density=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    diagnostics = summarize_axisymmetric_beta_scan(
        results,
        betas,
        plasma_grid,
        reference_field=float(on_axis[center]),
    )
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        betas,
        rtol=2.0e-8,
        atol=1.0e-12,
    )
    center_radii = np.asarray([item.center_radius for item in diagnostics])
    field_ratios = np.asarray([item.diamagnetic_field_ratio for item in diagnostics])
    assert np.all(np.diff(center_radii) > 0.0)
    assert np.all(np.diff(field_ratios) < 0.0)
    assert center_radii[-1] > 1.07 * center_radii[0]
    assert field_ratios[-1] < 0.77
    assert all(np.isfinite(float(item.center_vacuum_side_field)) for item in diagnostics)


@pytest.mark.full
def test_nonaxisymmetric_exterior_free_boundary_equilibrium_converges() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=1, ntheta=3, nxi=5),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=300,
    )
    grid = config.build_grid()
    on_axis = _on_axis_mirror_field(
        jnp.asarray(grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.2**2
    base = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    boundary = MirrorBoundary(
        base.radius_scale + 0.03 * jnp.asarray(grid.xi)[None, :] * jnp.cos(jnp.asarray(grid.theta)[:, None])
    )
    betas = jnp.asarray([0.0, 0.50])
    results = solve_beta_scan_cli(
        boundary,
        grid,
        config,
        _nonaxisymmetric_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        current_derivative=1.0e-3 * jnp.asarray(grid.s),
        exterior_order=6,
        exterior_spectral_side_density=True,
    )

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(
        float(result.plasma_staggered_weak_force.maximum) <= config.ftol
        for result in results
    )
    assert all(float(result.normalized_divergence_rms) < 1.0e-12 for result in results)
    assert all(
        float(jnp.max(jnp.abs(result.plasma_state.lambda_stream))) > 1.0e-5
        for result in results
    )
    assert all(float(result.interface.vacuum_b_normal_rms) < 1.0e-12 for result in results)
    assert all(float(result.interface.normal_stress_rms) < 1.0e-12 for result in results)
    assert all(float(result.vacuum_field.neumann_result.compatibility_error) < 2.0e-3 for result in results)
    assert all(float(result.vacuum_field.neumann_result.condition_number) < 5.0 for result in results)

    diagnostics = summarize_nonaxisymmetric_beta_scan(
        results,
        betas,
        grid,
        reference_field=float(on_axis[center]),
    )
    achieved_betas = np.asarray([item.achieved_reference_beta for item in diagnostics])
    np.testing.assert_allclose(achieved_betas, betas, rtol=2.0e-8, atol=1.0e-12)
    mean_radii = np.asarray([item.center_mean_radius for item in diagnostics])
    mean_fields = np.asarray([item.center_mean_field for item in diagnostics])
    mode_one = np.asarray([item.center_boundary_modes[1] for item in diagnostics])
    mode_one_l2 = np.asarray([item.boundary_mode_l2[1] for item in diagnostics])
    mode_one_max = np.asarray([item.boundary_mode_max[1] for item in diagnostics])
    mode_one_core_l2 = np.asarray(
        [item.boundary_mode_core_l2[1] for item in diagnostics]
    )
    assert np.all(np.diff(mean_radii) > 0.0)
    assert np.all(np.diff(mean_fields) < 0.0)
    assert np.all(mode_one > 1.0e-4)
    assert mode_one[-1] > 1.2 * mode_one[0]
    assert np.all(mode_one_l2 > 1.0e-2)
    assert np.all(mode_one_max > 2.5e-2)
    assert np.all(mode_one_core_l2 > 1.0e-3)
    assert np.all(mode_one / mode_one_max < 1.2e-1)
    assert all(float(item.plasma_volume) > 0.0 for item in diagnostics)
    assert all(float(item.plasma_energy) > 0.0 for item in diagnostics)


@pytest.mark.full
def test_unbounded_exterior_beta_observables_converge_with_resolution() -> None:
    observables = []
    compatibility = []
    betas = jnp.asarray([0.0, 0.10, 0.50])
    for ns, nxi, ntheta_panel in ((5, 7, 8), (7, 13, 12), (9, 17, 16)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=500,
        )
        plasma_grid = config.build_grid()
        on_axis = _on_axis_mirror_field(
            jnp.asarray(plasma_grid.z),
            coil_radius=0.9,
            separation=2.0,
            current=2.0e5,
        )
        center = plasma_grid.nxi // 2
        flux = 0.5 * on_axis[center] * 0.25**2
        results = solve_axisymmetric_beta_scan_cli(
            MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid),
            plasma_grid,
            config,
            _external_mirror_field,
            betas,
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            exterior_ntheta=ntheta_panel,
            exterior_order=8,
        )
        assert all(result.converged for result in results)
        assert all(float(result.variational_max) <= config.ftol for result in results)
        observables.append(
            np.asarray(
                [
                    [
                        float(result.boundary.radius_scale[0, center]),
                        float(jnp.sqrt(result.plasma_b_squared[0, 0, center])),
                    ]
                    for result in results
                ]
            )
        )
        compatibility.append(
            np.asarray([float(result.vacuum_field.neumann_result.compatibility_error) for result in results])
        )

    relative_change = np.abs((observables[-1] - observables[-2]) / observables[-1])
    assert np.max(relative_change[:2]) < 5.0e-4
    assert np.max(relative_change[2]) < 5.0e-3
    assert np.max(compatibility[-1]) < 3.0e-9
    assert np.all(compatibility[-1] < compatibility[0])
    assert float(results[-1].boundary.radius_scale[0, center]) > 1.07 * float(
        results[0].boundary.radius_scale[0, center]
    )


def test_two_coil_anisotropic_free_boundary_calibrates_perpendicular_beta() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    plasma_grid = config.build_grid()
    on_axis = _on_axis_mirror_field(
        jnp.asarray(plasma_grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = plasma_grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, plasma_grid)
    target_pressure = 0.01 * on_axis[center] ** 2 / (2.0 * MU0)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )

    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        plasma_grid,
        config,
        _external_mirror_field,
        jnp.asarray([0.0, 0.01]),
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    result = results[-1]
    energy = result.plasma_energy
    b_squared = magnetic_field_squared(energy.field, energy.geometry)
    central_b = jnp.sqrt(b_squared[0, 0, center])
    central_perpendicular = result.mass_scale * closure.moments(0.0, central_b).perpendicular
    anisotropy = jnp.max(jnp.abs(energy.moments_half.perpendicular - energy.moments_half.parallel)) / jnp.max(
        energy.moments_half.parallel
    )

    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert float(result.interface.normal_stress_rms) < 2.0e-12
    np.testing.assert_allclose(central_perpendicular, target_pressure, rtol=2.0e-12)
    assert float(anisotropy) > 0.04
    assert bool(jnp.all(energy.indicators_half.valid))
    assert float(result.boundary.radius_scale[0, center]) > float(results[0].boundary.radius_scale[0, center])


def test_tabulated_pressure_free_boundary_matches_sampled_bimaxwellian() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=5, mpol=0, ntheta=1, nxi=7),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=1000,
    )
    grid = config.build_grid()
    on_axis = _on_axis_mirror_field(jnp.asarray(grid.z))
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    reference = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )
    s_nodes = jnp.linspace(0.0, 1.0, 5)
    b_nodes = jnp.linspace(0.04, 0.18, 9)
    closure = TabulatedPressureClosure(
        s_nodes,
        b_nodes,
        reference.parallel_pressure(s_nodes[:, None], b_nodes[None, :]),
        gamma=0.0,
    )
    betas = jnp.asarray([0.0, 0.01])
    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    reference_results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=reference,
    )
    result = results[-1]
    diagnostic = summarize_axisymmetric_beta_scan(results, betas, grid, reference_field=float(on_axis[center]))[-1]
    reference_diagnostic = summarize_axisymmetric_beta_scan(
        reference_results,
        betas,
        grid,
        reference_field=float(on_axis[center]),
    )[-1]

    assert result.converged
    assert float(result.variational_max) <= config.ftol
    assert bool(jnp.all(result.plasma_energy.indicators_half.valid))
    np.testing.assert_allclose(diagnostic.achieved_reference_beta, 0.01, rtol=2e-8)
    np.testing.assert_allclose(
        diagnostic.center_radius,
        reference_diagnostic.center_radius,
        rtol=5.0e-5,
    )
    np.testing.assert_allclose(
        diagnostic.diamagnetic_field_ratio,
        reference_diagnostic.diamagnetic_field_ratio,
        rtol=5.0e-5,
    )


@pytest.mark.full
def test_anisotropic_free_boundary_observables_converge_with_resolution() -> None:
    observables = []
    normal_fields = []
    for ns, nxi in ((5, 7), (7, 13), (9, 17)):
        config = MirrorConfig(
            resolution=MirrorResolution(ns=ns, mpol=0, ntheta=1, nxi=nxi),
            z_min=-0.8,
            z_max=0.8,
            ftol=1.0e-12,
            max_iterations=2000,
        )
        grid = config.build_grid()
        on_axis = _on_axis_mirror_field(
            jnp.asarray(grid.z),
            coil_radius=0.9,
            separation=2.0,
            current=2.0e5,
        )
        center = grid.nxi // 2
        flux = 0.5 * on_axis[center] * 0.25**2
        boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
        closure = BiMaxwellianPressureClosure(
            mass_coefficients=jnp.asarray([1.0, -1.0]),
            hot_fraction_coefficients=jnp.asarray([0.2]),
            temperature_ratio=0.7,
            critical_field=float(on_axis[center]),
            gamma=0.0,
        )
        results = solve_axisymmetric_beta_scan_cli(
            boundary,
            grid,
            config,
            _external_mirror_field,
            jnp.asarray([0.0, 0.10]),
            axial_flux_derivative=flux,
            reference_field=float(on_axis[center]),
            pressure_closure=closure,
        )
        result = results[-1]
        diagnostic = summarize_axisymmetric_beta_scan(
            results,
            jnp.asarray([0.0, 0.10]),
            grid,
            reference_field=float(on_axis[center]),
        )[-1]
        anisotropy = jnp.max(
            jnp.abs(result.plasma_energy.moments_half.perpendicular - result.plasma_energy.moments_half.parallel)
        ) / jnp.max(result.plasma_energy.moments_half.parallel)
        observables.append(
            np.asarray(
                [
                    diagnostic.center_radius,
                    diagnostic.diamagnetic_field_ratio,
                    diagnostic.volume_averaged_beta,
                    anisotropy,
                ]
            )
        )
        normal_fields.append(float(result.interface.vacuum_b_normal_rms))
        assert result.converged
        assert float(result.variational_max) <= config.ftol
        assert bool(jnp.all(result.plasma_energy.indicators_half.valid))

    relative_change = np.abs(observables[-1] - observables[-2]) / np.abs(observables[-1])
    assert np.max(relative_change) < 1.0e-3
    assert normal_fields[2] < normal_fields[1] < normal_fields[0]


@pytest.mark.full
def test_anisotropic_high_beta_scan_remains_elliptic_and_diamagnetic() -> None:
    config = MirrorConfig(
        resolution=MirrorResolution(ns=7, mpol=0, ntheta=1, nxi=13),
        z_min=-0.8,
        z_max=0.8,
        ftol=1.0e-12,
        max_iterations=2000,
    )
    grid = config.build_grid()
    on_axis = _on_axis_mirror_field(
        jnp.asarray(grid.z),
        coil_radius=0.9,
        separation=2.0,
        current=2.0e5,
    )
    center = grid.nxi // 2
    flux = 0.5 * on_axis[center] * 0.25**2
    boundary = MirrorBoundary.from_axis_field(flux, on_axis, grid)
    closure = BiMaxwellianPressureClosure(
        mass_coefficients=jnp.asarray([1.0, -1.0]),
        hot_fraction_coefficients=jnp.asarray([0.2]),
        temperature_ratio=0.7,
        critical_field=float(on_axis[center]),
        gamma=0.0,
    )
    betas = jnp.asarray([0.0, 0.10, 0.25, 0.50])
    results = solve_axisymmetric_beta_scan_cli(
        boundary,
        grid,
        config,
        _external_mirror_field,
        betas,
        axial_flux_derivative=flux,
        reference_field=float(on_axis[center]),
        pressure_closure=closure,
    )
    diagnostics = summarize_axisymmetric_beta_scan(results, betas, grid, reference_field=float(on_axis[center]))

    assert all(result.converged for result in results)
    assert all(float(result.variational_max) <= config.ftol for result in results)
    assert all(bool(jnp.all(result.plasma_energy.indicators_half.valid)) for result in results[1:])
    assert np.all(np.diff([item.center_radius for item in diagnostics]) > 0.0)
    assert np.all(np.diff([item.diamagnetic_field_ratio for item in diagnostics]) < 0.0)
    np.testing.assert_allclose(
        [item.achieved_reference_beta for item in diagnostics],
        betas,
        rtol=2.0e-8,
        atol=1.0e-12,
    )

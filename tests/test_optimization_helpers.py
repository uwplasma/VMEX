import numpy as np
import jax.numpy as jnp

from vmec_jax.boundary import BoundaryCoeffs
from vmec_jax.modes import vmec_mode_table
from vmec_jax.optimization import (
    apply_boundary_params,
    boundary_param_names,
    boundary_param_specs,
    gauss_newton_least_squares,
    surface_indices_from_s,
)


def test_boundary_param_specs_and_apply():
    modes = vmec_mode_table(mpol=2, ntor=1)
    k = modes.K
    boundary = BoundaryCoeffs(
        R_cos=np.linspace(1.0, 2.0, k),
        R_sin=np.zeros(k),
        Z_cos=np.zeros(k),
        Z_sin=np.linspace(0.1, 0.2, k),
    )

    specs = boundary_param_specs(
        boundary,
        modes,
        max_mode=1,
        min_coeff=0.0,
        include=("rc", "zs"),
        fix=("rc00",),
    )
    names = boundary_param_names(specs)

    assert "rc00" not in names
    assert any(name.startswith("rc1") for name in names)
    assert any(name.startswith("zs1") for name in names)

    params = jnp.ones((len(specs),))
    updated = apply_boundary_params(boundary, specs, params)

    # rc00 should remain unchanged
    assert np.isclose(updated.R_cos[0], boundary.R_cos[0])
    # At least one other coefficient should change
    assert not np.allclose(np.asarray(updated.R_cos), np.asarray(boundary.R_cos))


def test_surface_indices_from_s():
    s_half = np.array([0.1, 0.3, 0.5, 0.7])
    indices, selected = surface_indices_from_s(s_half, [0.28, 3])
    assert indices == [1, 2]
    np.testing.assert_allclose(selected, np.array([0.3, 0.5]))


def test_gauss_newton_least_squares_solves_linear_problem():
    def residual(x):
        x = np.asarray(x, dtype=float)
        return np.array([x[0] - 1.0, 2.0 * x[1] - 2.0], dtype=float)

    def jacobian(_x):
        return np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)

    result = gauss_newton_least_squares(
        residual,
        jacobian,
        np.array([0.0, 0.0], dtype=float),
        max_nfev=5,
        ftol=1e-12,
        gtol=1e-12,
        xtol=1e-12,
        verbose=0,
    )

    np.testing.assert_allclose(result["x"], np.array([1.0, 1.0]), atol=1e-12, rtol=0.0)
    assert result["success"]
    assert result["objective"] <= 1e-20

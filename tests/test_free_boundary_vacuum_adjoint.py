from __future__ import annotations

import numpy as np
import pytest

from vmec_jax._compat import enable_x64
from vmec_jax.external_fields import CoilFieldParams, sample_coil_field_cylindrical
from vmec_jax.free_boundary_adjoint import dense_vacuum_residual, dense_vacuum_solve_jax


def _well_conditioned_matrix():
    from vmec_jax._compat import jnp

    A = jnp.asarray(
        [
            [3.0, 0.2, -0.1],
            [0.4, 2.5, 0.3],
            [-0.2, 0.1, 2.2],
        ]
    )
    b = jnp.asarray([1.0, -0.4, 0.7])
    return A, b


def test_dense_vacuum_solve_matches_jnp_linalg_solve():
    from vmec_jax._compat import jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()

    actual = dense_vacuum_solve_jax(A, b)
    expected = jnp.linalg.solve(A, b)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-14)
    np.testing.assert_allclose(dense_vacuum_residual(A, actual, b), np.zeros_like(np.asarray(b)), atol=1.0e-14)


def test_dense_vacuum_vjp_wrt_b_matches_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(rhs):
        x = dense_vacuum_solve_jax(A, rhs)
        return jnp.vdot(cotangent, x)

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A.T, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def test_dense_vacuum_gradient_wrt_rhs_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    direction = jnp.asarray([0.2, -0.1, 0.4])
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A, b + scale * direction)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_gradient_wrt_matrix_parameter_matches_finite_difference():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A, b = _well_conditioned_matrix()
    dA = jnp.asarray(
        [
            [0.0, 0.2, 0.0],
            [-0.1, 0.0, 0.3],
            [0.0, 0.1, 0.0],
        ]
    )
    cotangent = jnp.asarray([0.3, -0.2, 0.5])

    def objective(scale):
        x = dense_vacuum_solve_jax(A + scale * dA, b)
        return jnp.vdot(cotangent, x)

    exact = jax.grad(objective)(0.0)
    eps = 1.0e-6
    fd = (objective(eps) - objective(-eps)) / (2.0 * eps)

    np.testing.assert_allclose(exact, fd, rtol=2.0e-9, atol=1.0e-11)


def test_dense_vacuum_symmetric_mode_uses_symmetric_transpose_solve():
    pytest.importorskip("jax")
    from vmec_jax._compat import jax, jnp

    enable_x64(True)
    A = jnp.asarray([[3.0, 0.2], [0.2, 2.0]])
    b = jnp.asarray([0.7, -0.1])
    cotangent = jnp.asarray([0.4, 0.5])

    def objective(rhs):
        return jnp.vdot(cotangent, dense_vacuum_solve_jax(A, rhs, symmetric=True))

    grad_b = jax.grad(objective)(b)
    expected = jnp.linalg.solve(A, cotangent)

    np.testing.assert_allclose(grad_b, expected, rtol=1.0e-13, atol=1.0e-13)


def _toy_coil_vacuum_response(*, current_scale: float = 0.0, radius_shift: float = 0.0):
    """Small direct-coil -> vacuum-linear-solve chain for adjoint checks."""

    from vmec_jax._compat import jnp

    radius = 1.15 + 0.02 * radius_shift
    dofs = jnp.zeros((1, 3, 3), dtype=float)
    dofs = dofs.at[0, 0, 2].set(radius)
    dofs = dofs.at[0, 1, 1].set(radius)
    params = CoilFieldParams(
        base_curve_dofs=dofs,
        base_currents=jnp.asarray([3.0e7 * (1.0 + 0.01 * current_scale)], dtype=float),
        n_segments=96,
        regularization_epsilon=1.0e-9,
    )
    R = jnp.asarray([0.24, 0.37, 0.51], dtype=float)
    Z = jnp.asarray([0.11, -0.17, 0.23], dtype=float)
    phi = jnp.asarray([0.0, 0.4, 0.9], dtype=float)
    br, bphi, bz = sample_coil_field_cylindrical(params, R, Z, phi)
    rhs = jnp.stack(
        (
            br[0] + 0.3 * bphi[1],
            bz[1] - 0.2 * br[2],
            bphi[2] + 0.5 * bz[0],
        )
    )
    A = jnp.asarray(
        [
            [2.7, 0.2, -0.1],
            [0.1, 2.2, 0.3],
            [-0.2, 0.4, 2.5],
        ],
        dtype=float,
    )
    x = dense_vacuum_solve_jax(A, rhs)
    return 0.5 * jnp.vdot(x, x) + 0.1 * jnp.vdot(rhs, rhs)


def test_dense_vacuum_adjoint_chain_wrt_coil_current_matches_finite_difference():
    """Validate a direct-coil field feeding an implicit vacuum solve."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda scale: _toy_coil_vacuum_response(current_scale=scale))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_vacuum_response(current_scale=eps)
        - _toy_coil_vacuum_response(current_scale=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-10)


def test_dense_vacuum_adjoint_chain_wrt_coil_geometry_matches_finite_difference():
    """Validate the same chain for a Fourier curve coefficient perturbation."""

    pytest.importorskip("jax")
    from vmec_jax._compat import jax

    enable_x64(True)

    exact = jax.grad(lambda shift: _toy_coil_vacuum_response(radius_shift=shift))(0.0)
    eps = 1.0e-4
    fd = (
        _toy_coil_vacuum_response(radius_shift=eps)
        - _toy_coil_vacuum_response(radius_shift=-eps)
    ) / (2.0 * eps)

    assert abs(float(exact)) > 1.0e-8
    np.testing.assert_allclose(exact, fd, rtol=2.0e-6, atol=1.0e-10)

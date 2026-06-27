"""Small native spline-control residual problems for free-boundary prototypes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np
import jax
from jax.scipy.sparse.linalg import cg as jax_cg

from vmec_jax._compat import jnp
from vmec_jax.state import VMECState

from .native_state import (
    free_boundary_native_spline_vector_projected_residual_jax,
    free_boundary_native_spline_vector_to_vmec_state_jax,
)


@dataclass(frozen=True)
class FreeBoundaryNativeSplineResidualProblem:
    """A residual written in packed native spline-control coordinates.

    ``residual_fn`` still receives a decoded ``VMECState`` so existing VMEC
    kernels can be reused. The returned residual is packed back into the native
    vector basis: interior VMEC rows plus reduced LCFS spline controls.
    """

    template_state: VMECState
    projection: dict[str, Any]
    residual_fn: Callable[[VMECState], Any]
    edge_metric: str = "pullback"

    def __post_init__(self) -> None:
        if not isinstance(self.template_state, VMECState):
            raise TypeError("template_state must be a VMECState")
        if not bool(self.projection.get("enabled", False)):
            raise ValueError("projection must be enabled")
        if not callable(self.residual_fn):
            raise TypeError("residual_fn must be callable")

    def decode(self, vector: Any) -> VMECState:
        """Decode native spline-control unknowns to a full VMEC state."""

        return free_boundary_native_spline_vector_to_vmec_state_jax(
            vector,
            self.template_state,
            self.projection,
        )

    def residual(self, vector: Any):
        """Evaluate the residual in packed native coordinates."""

        return free_boundary_native_spline_vector_projected_residual_jax(
            vector,
            self.template_state,
            self.projection,
            self.residual_fn,
            edge_metric=self.edge_metric,
        )


class FreeBoundaryNativeSplineDenseStep(NamedTuple):
    """One dense Gauss-Newton step for a small native residual problem."""

    vector: Any
    residual: Any
    jacobian: Any
    step: Any
    next_vector: Any
    residual_l2: float
    step_l2: float
    damping: float


class FreeBoundaryNativeSplineDenseSolve(NamedTuple):
    """Result from a tiny dense native spline-control residual solve."""

    vector: Any
    residual: Any
    n_iter: int
    converged: bool
    residual_l2: float
    history: tuple[dict[str, float | int], ...]


class FreeBoundaryNativeSplineMatrixFreeStep(NamedTuple):
    """One matrix-free normal-equation step for a native residual problem."""

    vector: Any
    residual: Any
    step: Any
    next_vector: Any
    residual_l2: float
    step_l2: float
    damping: float
    cg_info: Any


class FreeBoundaryNativeSplineMatrixFreeSolve(NamedTuple):
    """Result from a small matrix-free native residual solve."""

    vector: Any
    residual: Any
    n_iter: int
    converged: bool
    residual_l2: float
    history: tuple[dict[str, float | int], ...]


def free_boundary_native_spline_dense_gauss_newton_step_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    damping: float = 0.0,
) -> FreeBoundaryNativeSplineDenseStep:
    """Take one dense Gauss-Newton step in native coordinates.

    This helper intentionally forms the dense Jacobian, so it is only suitable
    for small manufactured or smoke problems. Production strict solves should
    use matrix-free JVP/VJP, implicit differentiation, or an adjoint.
    """

    if not isinstance(problem, FreeBoundaryNativeSplineResidualProblem):
        raise TypeError("problem must be a FreeBoundaryNativeSplineResidualProblem")
    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and nonnegative")
    values = jnp.asarray(vector)
    residual_fn = lambda candidate: problem.residual(candidate)
    residual = residual_fn(values)
    jacobian = jnp.asarray(jax.jacfwd(residual_fn)(values))
    lhs = jacobian.T @ jacobian
    if damping_value > 0.0:
        lhs = lhs + jnp.asarray(damping_value, dtype=lhs.dtype) * jnp.eye(
            int(lhs.shape[0]),
            dtype=lhs.dtype,
        )
    rhs = -(jacobian.T @ residual)
    step = jnp.linalg.solve(lhs, rhs)
    next_vector = values + step
    return FreeBoundaryNativeSplineDenseStep(
        vector=values,
        residual=residual,
        jacobian=jacobian,
        step=step,
        next_vector=next_vector,
        residual_l2=float(jnp.linalg.norm(residual)),
        step_l2=float(jnp.linalg.norm(step)),
        damping=damping_value,
    )


def free_boundary_native_spline_matrix_free_normal_step_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    damping: float = 1.0e-10,
    tol: float = 1.0e-10,
    maxiter: int | None = None,
) -> FreeBoundaryNativeSplineMatrixFreeStep:
    """Take one matrix-free damped normal-equation step.

    The linear operator is ``J.T @ J + damping * I``. Products with ``J`` and
    ``J.T`` come from ``jax.linearize`` and ``jax.vjp`` instead of a dense
    Jacobian. This is still a prototype, but it is the scalable direction for
    native spline-control solves.
    """

    if not isinstance(problem, FreeBoundaryNativeSplineResidualProblem):
        raise TypeError("problem must be a FreeBoundaryNativeSplineResidualProblem")
    damping_value = float(damping)
    if not np.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and nonnegative")
    tol_value = float(tol)
    if not np.isfinite(tol_value) or tol_value < 0.0:
        raise ValueError("tol must be finite and nonnegative")
    if maxiter is not None and int(maxiter) <= 0:
        raise ValueError("maxiter must be positive when supplied")

    values = jnp.asarray(vector)
    residual_fn = lambda candidate: problem.residual(candidate)
    residual, jvp = jax.linearize(residual_fn, values)
    _residual_for_vjp, vjp = jax.vjp(residual_fn, values)

    def jt(target):
        return vjp(target)[0]

    rhs = -jt(residual)
    damping_arr = jnp.asarray(damping_value, dtype=values.dtype)

    def normal_matvec(step):
        return jt(jvp(step)) + damping_arr * step

    step, info = jax_cg(
        normal_matvec,
        rhs,
        tol=tol_value,
        maxiter=None if maxiter is None else int(maxiter),
    )
    next_vector = values + step
    return FreeBoundaryNativeSplineMatrixFreeStep(
        vector=values,
        residual=residual,
        step=step,
        next_vector=next_vector,
        residual_l2=float(jnp.linalg.norm(residual)),
        step_l2=float(jnp.linalg.norm(step)),
        damping=damping_value,
        cg_info=info,
    )


def free_boundary_native_spline_matrix_free_normal_solve_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    max_iter: int = 8,
    ftol: float = 1.0e-12,
    damping: float = 1.0e-10,
    linear_tol: float = 1.0e-10,
    linear_maxiter: int | None = None,
) -> FreeBoundaryNativeSplineMatrixFreeSolve:
    """Solve a small native residual problem with matrix-free normal steps."""

    if int(max_iter) < 0:
        raise ValueError("max_iter must be nonnegative")
    target = float(ftol)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("ftol must be finite and nonnegative")
    current = jnp.asarray(vector)
    history: list[dict[str, float | int]] = []
    residual = problem.residual(current)
    residual_l2 = float(jnp.linalg.norm(residual))
    converged = residual_l2 <= target
    n_iter = 0
    while (not converged) and n_iter < int(max_iter):
        step = free_boundary_native_spline_matrix_free_normal_step_jax(
            problem,
            current,
            damping=damping,
            tol=linear_tol,
            maxiter=linear_maxiter,
        )
        current = step.next_vector
        residual = problem.residual(current)
        residual_l2 = float(jnp.linalg.norm(residual))
        n_iter += 1
        history.append(
            {
                "iter": int(n_iter),
                "residual_l2": residual_l2,
                "step_l2": float(step.step_l2),
            }
        )
        converged = residual_l2 <= target
    return FreeBoundaryNativeSplineMatrixFreeSolve(
        vector=current,
        residual=residual,
        n_iter=int(n_iter),
        converged=bool(converged),
        residual_l2=residual_l2,
        history=tuple(history),
    )


def free_boundary_native_spline_dense_gauss_newton_solve_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    max_iter: int = 8,
    ftol: float = 1.0e-12,
    damping: float = 0.0,
) -> FreeBoundaryNativeSplineDenseSolve:
    """Solve a tiny native residual problem with dense Gauss-Newton steps."""

    if int(max_iter) < 0:
        raise ValueError("max_iter must be nonnegative")
    target = float(ftol)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("ftol must be finite and nonnegative")
    current = jnp.asarray(vector)
    history: list[dict[str, float | int]] = []
    residual = problem.residual(current)
    residual_l2 = float(jnp.linalg.norm(residual))
    converged = residual_l2 <= target
    n_iter = 0
    while (not converged) and n_iter < int(max_iter):
        step = free_boundary_native_spline_dense_gauss_newton_step_jax(
            problem,
            current,
            damping=damping,
        )
        current = step.next_vector
        residual = problem.residual(current)
        residual_l2 = float(jnp.linalg.norm(residual))
        n_iter += 1
        history.append(
            {
                "iter": int(n_iter),
                "residual_l2": residual_l2,
                "step_l2": float(step.step_l2),
            }
        )
        converged = residual_l2 <= target
    return FreeBoundaryNativeSplineDenseSolve(
        vector=current,
        residual=residual,
        n_iter=int(n_iter),
        converged=bool(converged),
        residual_l2=residual_l2,
        history=tuple(history),
    )

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
    preconditioner_info: dict[str, Any]


class FreeBoundaryNativeSplineMatrixFreeSolve(NamedTuple):
    """Result from a small matrix-free native residual solve."""

    vector: Any
    residual: Any
    n_iter: int
    converged: bool
    residual_l2: float
    history: tuple[dict[str, float | int | bool], ...]


def free_boundary_native_spline_force_blocks_to_state_residual(
    *,
    template_state: VMECState,
    force_blocks: Any,
    mode_context: Any,
    lambda_update_scale: Any = 1.0,
) -> VMECState:
    """Map VMEC force blocks into the native residual state's signed basis.

    ``force_blocks`` is normally a VMEC ``TomnspsRZL`` payload from the force
    evaluator. The mode transform is supplied by the residual driver so this
    helper follows the same R/Z/lambda block mapping used by VMEC-style updates.
    """

    if not isinstance(template_state, VMECState):
        raise TypeError("template_state must be a VMECState")
    required = ("frcc", "fzsc", "flsc")
    missing = [name for name in required if not hasattr(force_blocks, name)]
    if missing:
        raise TypeError(f"force_blocks missing required fields: {', '.join(missing)}")
    for name in ("mn_cos_to_signed_physical", "mn_sin_to_signed_physical"):
        if not callable(getattr(mode_context, name, None)):
            raise TypeError(f"mode_context must provide {name}")

    zeros = jnp.zeros_like(jnp.asarray(template_state.Rcos))
    rcos = mode_context.mn_cos_to_signed_physical(
        getattr(force_blocks, "frcc"),
        getattr(force_blocks, "frss", None),
    )
    zsin = mode_context.mn_sin_to_signed_physical(
        getattr(force_blocks, "fzsc"),
        getattr(force_blocks, "fzcs", None),
    )
    lambda_scale = jnp.asarray(lambda_update_scale, dtype=jnp.asarray(zsin).dtype)
    lsin = (
        mode_context.mn_sin_to_signed_physical_lambda(
            getattr(force_blocks, "flsc"),
            getattr(force_blocks, "flcs", None),
        )
        if callable(getattr(mode_context, "mn_sin_to_signed_physical_lambda", None))
        else mode_context.mn_sin_to_signed_physical(
            getattr(force_blocks, "flsc"),
            getattr(force_blocks, "flcs", None),
        )
    )
    lsin = lsin * lambda_scale
    if bool(template_state.layout.lasym):
        rsin = mode_context.mn_sin_to_signed_physical(
            getattr(force_blocks, "frsc", None),
            getattr(force_blocks, "frcs", None),
        )
        zcos = mode_context.mn_cos_to_signed_physical(
            getattr(force_blocks, "fzcc", None),
            getattr(force_blocks, "fzss", None),
        )
        lcos = (
            mode_context.mn_cos_to_signed_physical_lambda(
                getattr(force_blocks, "flcc", None),
                getattr(force_blocks, "flss", None),
            )
            if callable(getattr(mode_context, "mn_cos_to_signed_physical_lambda", None))
            else mode_context.mn_cos_to_signed_physical(
                getattr(force_blocks, "flcc", None),
                getattr(force_blocks, "flss", None),
            )
        )
        lcos = lcos * lambda_scale
    else:
        rsin = zeros
        zcos = zeros
        lcos = zeros
    return VMECState(
        layout=template_state.layout,
        Rcos=rcos,
        Rsin=rsin,
        Zcos=zcos,
        Zsin=zsin,
        Lcos=lcos,
        Lsin=lsin,
    )


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
    preconditioner: str | None = None,
    preconditioner_probes: int = 4,
    preconditioner_floor: float = 1.0e-12,
) -> FreeBoundaryNativeSplineMatrixFreeStep:
    """Take one matrix-free damped normal-equation step.

    The linear operator is ``J.T @ J + damping * I``. Products with ``J`` and
    ``J.T`` come from ``jax.linearize`` and ``jax.vjp`` instead of a dense
    Jacobian. An optional deterministic Hutchinson diagonal estimate can be
    passed to CG as a Jacobi preconditioner. This is still a prototype, but it
    is the scalable direction for native spline-control solves.
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
    preconditioner_mode = "none" if preconditioner is None else str(preconditioner).strip().lower()
    if preconditioner_mode in {"", "false", "off", "no", "identity"}:
        preconditioner_mode = "none"
    if preconditioner_mode not in {"none", "hutchinson_diag", "jacobi", "diag", "diagonal"}:
        raise ValueError("preconditioner must be 'none' or 'hutchinson_diag'")
    probes = int(preconditioner_probes)
    if probes <= 0:
        if preconditioner_mode == "none":
            probes = 0
        else:
            raise ValueError("preconditioner_probes must be positive when a preconditioner is enabled")
    floor_value = float(preconditioner_floor)
    if not np.isfinite(floor_value) or floor_value <= 0.0:
        raise ValueError("preconditioner_floor must be finite and positive")

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

    cg_preconditioner = None
    preconditioner_info: dict[str, Any] = {
        "mode": preconditioner_mode,
        "probes": int(probes),
        "floor": float(floor_value),
        "diagonal_estimated": False,
    }
    if preconditioner_mode != "none":
        diagonal = _hutchinson_normal_diagonal_estimate(
            normal_matvec,
            values,
            probes=probes,
            floor=floor_value,
        )

        def cg_preconditioner(target):
            return target / diagonal

        preconditioner_info = {
            "mode": "hutchinson_diag",
            "requested_mode": preconditioner_mode,
            "probes": int(probes),
            "floor": float(floor_value),
            "diagonal_estimated": True,
            "diagonal_l2": float(jnp.linalg.norm(diagonal)),
            "diagonal_linf": float(jnp.max(jnp.abs(diagonal))),
            "diagonal_min_abs": float(jnp.min(jnp.abs(diagonal))),
        }

    step, info = jax_cg(
        normal_matvec,
        rhs,
        tol=tol_value,
        maxiter=None if maxiter is None else int(maxiter),
        M=cg_preconditioner,
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
        preconditioner_info=preconditioner_info,
    )


def _hutchinson_normal_diagonal_estimate(
    normal_matvec: Callable[[Any], Any],
    template: Any,
    *,
    probes: int,
    floor: float,
):
    """Estimate ``diag(A)`` for a matrix-free normal operator.

    Rademacher probes give ``diag(A) = E[z * A z]``. The deterministic sign
    sequence keeps profiling reproducible and avoids carrying PRNG state
    through this low-level prototype helper.
    """

    values = jnp.asarray(template)
    if values.ndim != 1:
        raise ValueError("native matrix-free vectors must be one-dimensional")
    count = int(probes)
    if count <= 0:
        raise ValueError("probes must be positive")
    index = jnp.arange(values.shape[0], dtype=jnp.int32)
    accum = jnp.zeros_like(values)
    for probe in range(count):
        hashed = (index * (2 * int(probe) + 3) + 17 * (int(probe) + 1)) % 4
        signs = jnp.where(hashed < 2, 1.0, -1.0).astype(values.dtype)
        accum = accum + signs * normal_matvec(signs)
    diagonal = accum / jnp.asarray(count, dtype=values.dtype)
    floor_arr = jnp.asarray(float(floor), dtype=values.dtype)
    return jnp.maximum(jnp.abs(diagonal), floor_arr)


def free_boundary_native_spline_matrix_free_normal_solve_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    max_iter: int = 8,
    ftol: float = 1.0e-12,
    damping: float = 1.0e-10,
    linear_tol: float = 1.0e-10,
    linear_maxiter: int | None = None,
    preconditioner: str | None = None,
    preconditioner_probes: int = 4,
    preconditioner_floor: float = 1.0e-12,
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
            preconditioner=preconditioner,
            preconditioner_probes=preconditioner_probes,
            preconditioner_floor=preconditioner_floor,
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


def free_boundary_native_spline_matrix_free_line_search_solve_jax(
    problem: FreeBoundaryNativeSplineResidualProblem,
    vector: Any,
    *,
    max_iter: int = 8,
    ftol: float = 1.0e-12,
    damping: float = 1.0e-10,
    linear_tol: float = 1.0e-10,
    linear_maxiter: int | None = None,
    max_backtracks: int = 8,
    shrink: float = 0.5,
    accept_ratio: float = 1.0,
    preconditioner: str | None = None,
    preconditioner_probes: int = 4,
    preconditioner_floor: float = 1.0e-12,
) -> FreeBoundaryNativeSplineMatrixFreeSolve:
    """Solve a native residual problem with safeguarded matrix-free steps.

    Each nonlinear iteration computes the same damped normal-equation step as
    ``free_boundary_native_spline_matrix_free_normal_step_jax`` and then
    backtracks along that direction until the residual norm does not exceed
    ``accept_ratio * current_norm``. If every trial is worse, the solve stops
    without accepting a degrading update.
    """

    if int(max_iter) < 0:
        raise ValueError("max_iter must be nonnegative")
    target = float(ftol)
    if not np.isfinite(target) or target < 0.0:
        raise ValueError("ftol must be finite and nonnegative")
    if int(max_backtracks) < 0:
        raise ValueError("max_backtracks must be nonnegative")
    shrink_value = float(shrink)
    if not np.isfinite(shrink_value) or not (0.0 < shrink_value < 1.0):
        raise ValueError("shrink must be finite and between 0 and 1")
    accept_ratio_value = float(accept_ratio)
    if not np.isfinite(accept_ratio_value) or accept_ratio_value <= 0.0:
        raise ValueError("accept_ratio must be finite and positive")

    current = jnp.asarray(vector)
    history: list[dict[str, float | int | bool]] = []
    residual = problem.residual(current)
    residual_l2 = float(jnp.linalg.norm(residual))
    converged = residual_l2 <= target
    accepted_iter = 0
    attempt_iter = 0
    while (not converged) and accepted_iter < int(max_iter):
        step = free_boundary_native_spline_matrix_free_normal_step_jax(
            problem,
            current,
            damping=damping,
            tol=linear_tol,
            maxiter=linear_maxiter,
            preconditioner=preconditioner,
            preconditioner_probes=preconditioner_probes,
            preconditioner_floor=preconditioner_floor,
        )
        step_l2 = float(step.step_l2)
        before_l2 = float(residual_l2)
        if step_l2 <= np.finfo(float).tiny:
            history.append(
                {
                    "iter": int(attempt_iter + 1),
                    "accepted": False,
                    "alpha": 0.0,
                    "backtracks": 0,
                    "residual_l2_before": before_l2,
                    "residual_l2": before_l2,
                    "step_l2": step_l2,
                    "reason": 0,
                }
            )
            break

        best_vector = current
        best_residual = residual
        best_l2 = before_l2
        best_alpha = 0.0
        best_backtracks = 0
        accepted = False
        alpha = 1.0
        for backtracks in range(int(max_backtracks) + 1):
            alpha_arr = jnp.asarray(alpha, dtype=current.dtype)
            trial_vector = current + alpha_arr * step.step
            trial_residual = problem.residual(trial_vector)
            trial_l2 = float(jnp.linalg.norm(trial_residual))
            if trial_l2 < best_l2:
                best_vector = trial_vector
                best_residual = trial_residual
                best_l2 = trial_l2
                best_alpha = float(alpha)
                best_backtracks = int(backtracks)
            if trial_l2 <= accept_ratio_value * before_l2:
                best_vector = trial_vector
                best_residual = trial_residual
                best_l2 = trial_l2
                best_alpha = float(alpha)
                best_backtracks = int(backtracks)
                accepted = True
                break
            alpha *= shrink_value

        # If strict non-increase failed but one trial improved, accept the best
        # improving trial. This avoids rejecting useful steps due to a very
        # small accept-ratio margin while still preventing uphill moves.
        if not accepted and best_l2 < before_l2:
            accepted = True

        attempt_iter += 1
        history.append(
            {
                "iter": int(attempt_iter),
                "accepted": bool(accepted),
                "alpha": float(best_alpha),
                "backtracks": int(best_backtracks),
                "residual_l2_before": before_l2,
                "residual_l2": float(best_l2),
                "step_l2": step_l2,
                "accepted_step_l2": float(best_alpha * step_l2),
            }
        )
        if not accepted:
            break

        current = best_vector
        residual = best_residual
        residual_l2 = float(best_l2)
        accepted_iter += 1
        converged = residual_l2 <= target

    return FreeBoundaryNativeSplineMatrixFreeSolve(
        vector=current,
        residual=residual,
        n_iter=int(accepted_iter),
        converged=bool(converged),
        residual_l2=float(residual_l2),
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

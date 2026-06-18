"""Linear correction helpers for mirror residual-Newton solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualLinearSolve:
    """One reduced residual-Newton linear correction and diagnostics."""

    step_y: np.ndarray
    istop: int
    iterations: int
    residual_norm: float
    normal_residual_norm: float
    condition_estimate: float


def _sparse_linalg():
    try:
        from scipy.sparse.linalg import LinearOperator, lsmr, lsqr
    except Exception as exc:  # pragma: no cover
        raise ImportError("optimizer='residual_newton' requires scipy.sparse.linalg least-squares solvers") from exc
    return LinearOperator, lsmr, lsqr


def solve_full_krylov_step(
    *,
    matvec_y: Callable[[np.ndarray], np.ndarray],
    precondition_y: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    preconditioner_kind: str,
    linear_solver_kind: str,
    tolerance: float,
    linear_maxiter: int,
) -> ResidualLinearSolve:
    """Solve the full reduced correction with right-preconditioned LSMR/LSQR."""
    LinearOperator, lsmr, lsqr = _sparse_linalg()
    size = int(np.asarray(rhs).size)
    if preconditioner_kind == "none":
        operator = LinearOperator((size, size), matvec=matvec_y, rmatvec=matvec_y, dtype=float)
    else:

        def matvec_preconditioned(vector_z):
            return matvec_y(precondition_y(vector_z))

        def rmatvec_preconditioned(vector_y):
            return precondition_y(matvec_y(vector_y))

        operator = LinearOperator(
            (size, size),
            matvec=matvec_preconditioned,
            rmatvec=rmatvec_preconditioned,
            dtype=float,
        )

    if linear_solver_kind == "lsqr":
        linear_result = lsqr(
            operator,
            rhs,
            atol=tolerance,
            btol=tolerance,
            iter_lim=linear_maxiter,
        )
        residual_norm = float(linear_result[3])
        normal_residual_norm = float(linear_result[7])
        condition_estimate = float(linear_result[6])
    else:
        linear_result = lsmr(
            operator,
            rhs,
            atol=tolerance,
            btol=tolerance,
            maxiter=linear_maxiter,
        )
        residual_norm = float(linear_result[3])
        normal_residual_norm = float(linear_result[4])
        condition_estimate = float(linear_result[6])

    step_y_raw = np.asarray(linear_result[0], dtype=float)
    step_y = step_y_raw if preconditioner_kind == "none" else precondition_y(step_y_raw)
    return ResidualLinearSolve(
        step_y=step_y,
        istop=int(linear_result[1]),
        iterations=int(linear_result[2]),
        residual_norm=residual_norm,
        normal_residual_norm=normal_residual_norm,
        condition_estimate=condition_estimate,
    )


def solve_block_lsmr_step(
    *,
    matvec_y: Callable[[np.ndarray], np.ndarray],
    precondition_y: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    num_a: int,
    preconditioner_kind: str,
    tolerance: float,
    radius_linear_maxiter: int,
    lambda_linear_maxiter: int,
) -> ResidualLinearSolve:
    """Solve radius and lambda diagonal correction blocks with matrix-free LSMR."""
    LinearOperator, lsmr, _lsqr = _sparse_linalg()
    rhs = np.asarray(rhs, dtype=float)
    size = int(rhs.size)
    num_a = int(num_a)
    step_y = np.zeros(size, dtype=float)
    block_istops: list[int] = []
    block_iterations: list[int] = []
    block_residual_norms: list[float] = []
    block_normal_residual_norms: list[float] = []
    block_condition_estimates: list[float] = []

    def lift_block(vector_block: np.ndarray, *, start: int, stop: int) -> np.ndarray:
        vector_full = np.zeros(size, dtype=float)
        vector_full[start:stop] = np.asarray(vector_block, dtype=float)
        return vector_full if preconditioner_kind == "none" else precondition_y(vector_full)

    def solve_block(start: int, stop: int, *, maxiter: int) -> np.ndarray:
        block_size = int(stop - start)
        if block_size <= 0:
            return np.zeros(size, dtype=float)

        def matvec_block(vector_block):
            return matvec_y(lift_block(vector_block, start=start, stop=stop))[start:stop]

        def rmatvec_block(vector_block):
            vector_full = np.zeros(size, dtype=float)
            vector_full[start:stop] = np.asarray(vector_block, dtype=float)
            result_full = matvec_y(vector_full)
            if preconditioner_kind != "none":
                result_full = precondition_y(result_full)
            return result_full[start:stop]

        operator_block = LinearOperator(
            (block_size, block_size),
            matvec=matvec_block,
            rmatvec=rmatvec_block,
            dtype=float,
        )
        linear_result = lsmr(
            operator_block,
            rhs[start:stop],
            atol=tolerance,
            btol=tolerance,
            maxiter=maxiter,
        )
        block_istops.append(int(linear_result[1]))
        block_iterations.append(int(linear_result[2]))
        block_residual_norms.append(float(linear_result[3]))
        block_normal_residual_norms.append(float(linear_result[4]))
        block_condition_estimates.append(float(linear_result[6]))
        return lift_block(np.asarray(linear_result[0], dtype=float), start=start, stop=stop)

    step_y += solve_block(0, num_a, maxiter=radius_linear_maxiter)
    step_y += solve_block(num_a, size, maxiter=lambda_linear_maxiter)
    return ResidualLinearSolve(
        step_y=step_y,
        istop=max(block_istops) if block_istops else 0,
        iterations=int(sum(block_iterations)),
        residual_norm=float(np.linalg.norm(block_residual_norms)),
        normal_residual_norm=float(np.linalg.norm(block_normal_residual_norms)),
        condition_estimate=float(max(block_condition_estimates)) if block_condition_estimates else 0.0,
    )

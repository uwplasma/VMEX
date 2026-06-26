"""Reduced-control linear steps for free-boundary geometry updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReducedControlStep:
    """One least-squares update in a reduced boundary-control basis."""

    control_delta: np.ndarray
    predicted_delta: np.ndarray
    residual_after: np.ndarray
    labels: tuple[str, ...]
    rank: int
    singular_values: np.ndarray
    condition_number: float | None
    target_l2: float
    predicted_l2: float
    residual_l2: float
    residual_rel: float | None
    control_l2: float
    control_linf: float
    ridge: float
    rcond: float | None
    trust_radius: float | None
    trust_scale: float

    @property
    def control_delta_by_label(self) -> dict[str, float]:
        """Return the reduced-control step keyed by label."""

        return {str(label): float(value) for label, value in zip(self.labels, self.control_delta, strict=False)}

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-friendly scalar diagnostics."""

        out = asdict(self)
        out["control_delta"] = [float(value) for value in self.control_delta]
        out["predicted_delta"] = [float(value) for value in self.predicted_delta]
        out["residual_after"] = [float(value) for value in self.residual_after]
        out["labels"] = list(self.labels)
        out["singular_values"] = [float(value) for value in self.singular_values]
        out["control_delta_by_label"] = self.control_delta_by_label
        return out


def _rank_and_condition(jacobian: np.ndarray, *, rcond: float | None) -> tuple[int, np.ndarray, float | None]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    finite = singular_values[np.isfinite(singular_values)]
    if finite.size == 0:
        return 0, singular_values, None
    smax = float(np.max(finite))
    smin = float(np.min(finite))
    if rcond is None:
        tol = max(jacobian.shape) * np.finfo(float).eps * smax
    else:
        tol = max(float(rcond) * smax, np.finfo(float).eps)
    rank = int(np.sum(finite > tol))
    condition = None if smin <= 0.0 else float(smax / smin)
    return rank, singular_values, condition


def reduced_control_least_squares_step(
    jacobian: Any,
    target_delta: Any,
    *,
    labels: tuple[str, ...] | list[str] | None = None,
    ridge: float = 0.0,
    rcond: float | None = None,
    trust_radius: float | None = None,
) -> ReducedControlStep:
    """Fit a full boundary displacement with reduced control variables.

    ``jacobian`` maps reduced controls into the full boundary-coefficient
    vector. The returned ``control_delta`` solves
    ``min ||J c - target_delta||^2 + ridge ||c||^2`` and is optionally scaled to
    satisfy ``||c|| <= trust_radius``. This helper is deliberately independent
    of VMEC state objects so it can be reused by diagnostics, CLI prototypes,
    and future differentiable native-control solves.
    """

    jac = np.asarray(jacobian, dtype=float)
    target = np.asarray(target_delta, dtype=float).reshape(-1)
    if jac.ndim != 2:
        raise ValueError("jacobian must be two-dimensional")
    if jac.shape[0] != target.size:
        raise ValueError("jacobian row count must match target_delta size")
    if jac.shape[1] == 0:
        raise ValueError("jacobian must have at least one control column")
    if not (np.all(np.isfinite(jac)) and np.all(np.isfinite(target))):
        raise ValueError("jacobian and target_delta must be finite")
    ridge_value = float(ridge)
    if not np.isfinite(ridge_value) or ridge_value < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if rcond is not None and (not np.isfinite(float(rcond)) or float(rcond) < 0.0):
        raise ValueError("rcond must be finite and nonnegative when supplied")
    trust = None if trust_radius is None else float(trust_radius)
    if trust is not None and (not np.isfinite(trust) or trust <= 0.0):
        raise ValueError("trust_radius must be positive and finite when supplied")

    if labels is None:
        label_tuple = tuple(f"control_{idx}" for idx in range(jac.shape[1]))
    else:
        label_tuple = tuple(str(label) for label in labels)
        if len(label_tuple) != jac.shape[1]:
            raise ValueError("labels length must match the number of control columns")

    lhs = jac
    rhs = target
    if ridge_value > 0.0:
        lhs = np.vstack([jac, np.sqrt(ridge_value) * np.eye(jac.shape[1])])
        rhs = np.concatenate([target, np.zeros(jac.shape[1], dtype=float)])
    control_delta, _residuals, _rank_augmented, _sv_augmented = np.linalg.lstsq(lhs, rhs, rcond=rcond)

    trust_scale = 1.0
    control_norm = float(np.linalg.norm(control_delta))
    if trust is not None and control_norm > trust:
        trust_scale = float(trust / max(control_norm, np.finfo(float).tiny))
        control_delta = control_delta * trust_scale

    predicted = jac @ control_delta
    residual = target - predicted
    rank, singular_values, condition = _rank_and_condition(jac, rcond=rcond)
    target_l2 = float(np.linalg.norm(target))
    predicted_l2 = float(np.linalg.norm(predicted))
    residual_l2 = float(np.linalg.norm(residual))
    control_l2 = float(np.linalg.norm(control_delta))
    control_linf = float(np.max(np.abs(control_delta))) if control_delta.size else 0.0
    residual_rel = None if target_l2 <= np.finfo(float).tiny else float(residual_l2 / target_l2)
    return ReducedControlStep(
        control_delta=np.asarray(control_delta, dtype=float),
        predicted_delta=np.asarray(predicted, dtype=float),
        residual_after=np.asarray(residual, dtype=float),
        labels=label_tuple,
        rank=rank,
        singular_values=np.asarray(singular_values, dtype=float),
        condition_number=condition,
        target_l2=target_l2,
        predicted_l2=predicted_l2,
        residual_l2=residual_l2,
        residual_rel=residual_rel,
        control_l2=control_l2,
        control_linf=control_linf,
        ridge=ridge_value,
        rcond=None if rcond is None else float(rcond),
        trust_radius=trust,
        trust_scale=float(trust_scale),
    )

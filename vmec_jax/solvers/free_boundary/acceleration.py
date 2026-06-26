"""Small acceleration helpers for free-boundary vacuum-pressure updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AndersonPressureHistory:
    """Previous raw fixed-point output used by Anderson(1) mixing.

    VMEC free-boundary coupling updates a boundary vacuum pressure,
    ``bsqvac = |B|^2 / 2``.  Anderson(1) uses the last raw pressure update and
    its residual to damp the next update when the vacuum-pressure fixed point is
    slowly cycling.
    """

    previous_output: np.ndarray | None = None
    previous_residual: np.ndarray | None = None


@dataclass(frozen=True)
class AndersonPressureResult:
    """Result of one vacuum-pressure mixing attempt."""

    pressure: np.ndarray
    history: AndersonPressureHistory
    applied: bool
    reset: bool
    reason: str
    theta: float | None
    theta_unclamped: float | None
    residual_norm: float
    old_pressure_norm: float
    delta_residual_norm: float | None


def _as_pressure_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr


def _flat_dot(left: np.ndarray, right: np.ndarray) -> float:
    left_1d = np.asarray(left, dtype=float).reshape(-1)
    right_1d = np.asarray(right, dtype=float).reshape(-1)
    return float(np.dot(left_1d, right_1d))


def _norm(value: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=float).reshape(-1)))


def _empty_history() -> AndersonPressureHistory:
    return AndersonPressureHistory(previous_output=None, previous_residual=None)


def anderson1_vacuum_pressure_update(
    *,
    old_pressure: Any | None,
    raw_pressure: Any,
    history: AndersonPressureHistory | None = None,
    residual_rtol: float = 1.0e-9,
    theta_min: float = 0.0,
    theta_max: float = 2.0,
) -> AndersonPressureResult:
    """Return an Anderson(1)-mixed vacuum-pressure update.

    Parameters
    ----------
    old_pressure:
        Pressure currently coupled into the force balance.  ``None`` is treated
        as the first update from a zero pressure.
    raw_pressure:
        New pressure returned by the vacuum solve before mixing.
    history:
        Previous raw output/residual pair.  Incompatible or non-finite history
        is discarded automatically.
    residual_rtol:
        Relative residual threshold below which the history is cleared.
    theta_min, theta_max:
        Clamp for the scalar Anderson coefficient.  VMEC++ uses ``[0, 2]``.
    """

    raw = _as_pressure_array(raw_pressure, name="raw_pressure")
    if not np.all(np.isfinite(raw)):
        return AndersonPressureResult(
            pressure=raw,
            history=_empty_history(),
            applied=False,
            reset=True,
            reason="raw_pressure_nonfinite",
            theta=None,
            theta_unclamped=None,
            residual_norm=float("nan"),
            old_pressure_norm=float("nan"),
            delta_residual_norm=None,
        )

    reset = False
    if old_pressure is None:
        old = np.zeros_like(raw)
        reset = True
        reason_no_mix = "initialized"
    else:
        old = _as_pressure_array(old_pressure, name="old_pressure")
        if old.shape != raw.shape or not np.all(np.isfinite(old)):
            old = np.zeros_like(raw)
            reset = True
            reason_no_mix = "old_pressure_reset"
        else:
            reason_no_mix = "no_previous_history"

    residual = raw - old
    residual_norm = _norm(residual)
    old_norm = _norm(old)
    threshold = max(float(residual_rtol), 0.0) * max(1.0, old_norm)
    if residual_norm <= threshold:
        return AndersonPressureResult(
            pressure=raw,
            history=_empty_history(),
            applied=False,
            reset=True,
            reason="small_residual",
            theta=None,
            theta_unclamped=None,
            residual_norm=residual_norm,
            old_pressure_norm=old_norm,
            delta_residual_norm=None,
        )

    new_history = AndersonPressureHistory(
        previous_output=np.asarray(raw, dtype=float).copy(),
        previous_residual=np.asarray(residual, dtype=float).copy(),
    )

    prev_output = None if history is None else history.previous_output
    prev_residual = None if history is None else history.previous_residual
    if prev_output is None or prev_residual is None:
        return AndersonPressureResult(
            pressure=raw,
            history=new_history,
            applied=False,
            reset=reset,
            reason=reason_no_mix,
            theta=None,
            theta_unclamped=None,
            residual_norm=residual_norm,
            old_pressure_norm=old_norm,
            delta_residual_norm=None,
        )

    prev_output_arr = np.asarray(prev_output, dtype=float)
    prev_residual_arr = np.asarray(prev_residual, dtype=float)
    if (
        prev_output_arr.shape != raw.shape
        or prev_residual_arr.shape != raw.shape
        or not np.all(np.isfinite(prev_output_arr))
        or not np.all(np.isfinite(prev_residual_arr))
    ):
        return AndersonPressureResult(
            pressure=raw,
            history=new_history,
            applied=False,
            reset=True,
            reason="history_reset",
            theta=None,
            theta_unclamped=None,
            residual_norm=residual_norm,
            old_pressure_norm=old_norm,
            delta_residual_norm=None,
        )

    delta_residual = residual - prev_residual_arr
    delta_norm = _norm(delta_residual)
    denom = _flat_dot(delta_residual, delta_residual)
    if denom <= 0.0 or not np.isfinite(denom):
        return AndersonPressureResult(
            pressure=raw,
            history=new_history,
            applied=False,
            reset=reset,
            reason="singular_delta_residual",
            theta=None,
            theta_unclamped=None,
            residual_norm=residual_norm,
            old_pressure_norm=old_norm,
            delta_residual_norm=delta_norm,
        )

    theta_unclamped = _flat_dot(residual, delta_residual) / denom
    theta_lo = min(float(theta_min), float(theta_max))
    theta_hi = max(float(theta_min), float(theta_max))
    theta = float(np.clip(theta_unclamped, theta_lo, theta_hi))
    mixed = raw - theta * (raw - prev_output_arr)
    if not np.all(np.isfinite(mixed)):
        return AndersonPressureResult(
            pressure=raw,
            history=new_history,
            applied=False,
            reset=True,
            reason="mixed_pressure_nonfinite",
            theta=theta,
            theta_unclamped=float(theta_unclamped),
            residual_norm=residual_norm,
            old_pressure_norm=old_norm,
            delta_residual_norm=delta_norm,
        )

    return AndersonPressureResult(
        pressure=np.asarray(mixed, dtype=float),
        history=new_history,
        applied=True,
        reset=reset,
        reason="applied",
        theta=theta,
        theta_unclamped=float(theta_unclamped),
        residual_norm=residual_norm,
        old_pressure_norm=old_norm,
        delta_residual_norm=delta_norm,
    )

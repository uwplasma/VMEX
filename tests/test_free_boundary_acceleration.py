from __future__ import annotations

import numpy as np
import pytest

from vmec_jax.free_boundary_acceleration import (
    AndersonPressureHistory,
    anderson1_vacuum_pressure_update,
)


def test_anderson_pressure_first_update_initializes_history_without_mixing() -> None:
    raw = np.asarray([[1.0, 2.0], [3.0, 4.0]])

    result = anderson1_vacuum_pressure_update(old_pressure=None, raw_pressure=raw)

    assert result.applied is False
    assert result.reset is True
    assert result.reason == "initialized"
    np.testing.assert_allclose(result.pressure, raw)
    np.testing.assert_allclose(result.history.previous_output, raw)
    np.testing.assert_allclose(result.history.previous_residual, raw)


def test_anderson_pressure_second_update_mixes_with_clamped_theta() -> None:
    first_raw = np.asarray([1.0, 2.0])
    first = anderson1_vacuum_pressure_update(old_pressure=None, raw_pressure=first_raw)
    second_raw = np.asarray([0.6, 1.2])

    second = anderson1_vacuum_pressure_update(
        old_pressure=first.pressure,
        raw_pressure=second_raw,
        history=first.history,
    )

    assert second.applied is True
    assert second.reason == "applied"
    assert second.theta == pytest.approx(2.0 / 7.0)
    np.testing.assert_allclose(second.pressure, [0.7142857142857143, 1.4285714285714286])
    np.testing.assert_allclose(second.history.previous_output, second_raw)


def test_anderson_pressure_small_residual_clears_history() -> None:
    old = np.asarray([1.0, 2.0])
    raw = old + 1.0e-12

    result = anderson1_vacuum_pressure_update(
        old_pressure=old,
        raw_pressure=raw,
        history=AndersonPressureHistory(previous_output=old, previous_residual=np.ones_like(old)),
        residual_rtol=1.0e-9,
    )

    assert result.applied is False
    assert result.reset is True
    assert result.reason == "small_residual"
    assert result.history.previous_output is None
    assert result.history.previous_residual is None


def test_anderson_pressure_shape_change_resets_history() -> None:
    history = AndersonPressureHistory(
        previous_output=np.asarray([1.0, 2.0]),
        previous_residual=np.asarray([1.0, 2.0]),
    )

    result = anderson1_vacuum_pressure_update(
        old_pressure=np.zeros((2, 2)),
        raw_pressure=np.ones((2, 2)),
        history=history,
    )

    assert result.applied is False
    assert result.reset is True
    assert result.reason == "history_reset"
    np.testing.assert_allclose(result.pressure, np.ones((2, 2)))

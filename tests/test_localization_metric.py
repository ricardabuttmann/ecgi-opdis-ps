from __future__ import annotations

import numpy as np
import pytest

from ecgi.metrics import localization_error


def _step_at(index: int, n_time: int = 6) -> np.ndarray:
    trace = np.ones(n_time)
    trace[index + 1 :] = 0.0
    return trace


def _fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_mm = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [20.0, 0.0, 0.0], [21.0, 0.0, 0.0]]
    )
    truth = np.zeros((4, 6))
    truth[0] = _step_at(1)
    reconstruction = np.zeros((4, 6))
    reconstruction[0] = _step_at(1)
    reconstruction[2] = _step_at(1)
    return reconstruction, truth, points_mm


def test_localization_error_averages_tied_earliest_regions():
    reconstruction, truth, points_mm = _fields()

    assert localization_error(reconstruction, truth, points_mm, 5.0) == pytest.approx(10.0)


def test_localization_error_is_invariant_to_node_order():
    reconstruction, truth, points_mm = _fields()
    permutation = np.array([2, 0, 3, 1])

    original = localization_error(reconstruction, truth, points_mm, 5.0)
    reordered = localization_error(
        reconstruction[permutation], truth[permutation], points_mm[permutation], 5.0
    )

    assert reordered == pytest.approx(original)


def test_localization_error_uses_only_exact_earliest_ties():
    reconstruction, truth, points_mm = _fields()
    reconstruction[2] = _step_at(2)

    assert localization_error(reconstruction, truth, points_mm, 5.0) == pytest.approx(0.0)

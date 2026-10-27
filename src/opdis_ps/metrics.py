"""The five reconstruction metrics reported in the paper."""

from __future__ import annotations

import numpy as np

from ecgi.metrics.metrics import (
    activation_time_error,
    localization_error,
    relative_error_global,
    spatial_cc,
    temporal_cc,
)

METRIC_PROTOCOL = "paper-2026-tied-front-mean-v1"


def evaluate_reconstruction(
    reconstruction: np.ndarray,
    truth: np.ndarray,
    points_mm: np.ndarray,
    dt_ms: float,
) -> dict[str, float]:
    """Return the five paper metrics under ``METRIC_PROTOCOL``."""
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if reconstruction.shape != truth.shape:
        raise ValueError(
            f"reconstruction shape {reconstruction.shape} does not match truth {truth.shape}"
        )
    points_mm = np.asarray(points_mm, dtype=np.float64)
    if points_mm.shape != (truth.shape[0], 3):
        raise ValueError(
            f"points_mm shape {points_mm.shape} does not match {(truth.shape[0], 3)}"
        )
    return {
        "relative_error_global": relative_error_global(reconstruction, truth),
        "activation_time_error_ms": activation_time_error(
            reconstruction, truth, dt_ms
        ).mean,
        "localization_error_mm": localization_error(
            reconstruction, truth, points_mm, dt_ms
        ),
        "spatial_cc": spatial_cc(reconstruction, truth).mean,
        "temporal_cc": temporal_cc(reconstruction, truth).mean,
    }

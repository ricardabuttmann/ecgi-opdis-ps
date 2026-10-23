"""Path-free subset of the production dataset helpers used during inference."""

from __future__ import annotations

import numpy as np


def average_reference(signal: np.ndarray) -> np.ndarray:
    """Subtract the node-wise mean independently at every time sample."""
    return signal - signal.mean(axis=0, keepdims=True)

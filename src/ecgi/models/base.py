"""Shared model contract for deterministic point reconstruction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InverseModel(ABC):
    """Shared contract for the deterministic inverse models."""

    @abstractmethod
    def fit(self, train_data: Any) -> None:
        """Train on the training split. No-op for training-free methods (e.g. Tikhonov)."""

    @abstractmethod
    def reconstruct(self, case: Any, geometry: Any) -> np.ndarray:
        """case (b + average-referenced forward operator case.forward_A) -> x_hat [n_epi, n_time]."""

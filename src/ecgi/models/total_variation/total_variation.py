"""Training-free spatial total-variation model for ECGI."""

from __future__ import annotations

from typing import Any

import numpy as np

from ecgi.models.base import InverseModel
from ecgi.models.total_variation.operators import template_tv_gradient
from ecgi.models.total_variation.solver import TvSolverResult, solve_total_variation


class TotalVariation(InverseModel):
    """Deterministic convex TV reconstruction on the fixed UVC template."""

    def __init__(
        self,
        *,
        alpha: float,
        epsilon: float = 1e-6,
        max_iterations: int = 32000,
        tolerance: float = 1e-5,
    ):
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.last_result_: TvSolverResult | None = None

    def fit(self, train_data: Any) -> None:
        """Training-free model; alpha selection is a separate validation procedure."""
        return None

    def reconstruct(self, case: Any, geometry: Any) -> np.ndarray:
        template = getattr(geometry, "template", None)
        if template is None:
            raise ValueError("TV reconstruction requires fixed UVC template geometry")
        b = case.b if case.b.ndim == 2 else case.b[:, None]
        A = np.asarray(case.forward_A, dtype=np.float64)
        gradient = template_tv_gradient(template)
        self.last_result_ = solve_total_variation(
            A,
            b,
            gradient,
            alpha=self.alpha,
            epsilon=self.epsilon,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
        )
        return self.last_result_.x_hat

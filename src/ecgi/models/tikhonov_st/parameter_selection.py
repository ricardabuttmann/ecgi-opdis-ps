"""Joint ``lambda_s`` selection across the rotated temporal modes.

After the temporal rotation the estimator is block diagonal: rotated mode ``k`` has its own reduced
Gram eigenvalues ``w^(k)`` and coefficients ``beta^(k) = Q_k^T b_k``, but all blocks share the one
spatial regularization weight ``lambda_s``. GCV must therefore be formed over the whole rotated
problem rather than per mode, otherwise each mode would select its own lambda and the reconstruction
would no longer minimize the declared spatiotemporal objective.

For the block-diagonal hat matrix ``H = diag(H_k)`` with ``H_k = G_k (G_k + lambda^2 I)^-1``:

    ||(I - H) B||_F^2 = sum_k ||(I - H_k) b_k||^2
    tr(I - H)         = sum_k tr(I - H_k)
    GCV(lambda)       = (m * n_blocks) * sum_k ||(I - H_k) b_k||^2 / ( sum_k tr(I - H_k) )^2

The per-block terms reuse the filter-factor algebra of the spatial-only baseline, including its
treatment of rank-truncated modes: those are invisible to the estimator, so they contribute their
full lambda-independent energy to the numerator and one degree of freedom each to the trace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModeSpectrum:
    """Reduced-Gram spectrum of one rotated temporal mode, after effective-rank truncation."""

    eigenvalues: np.ndarray  # w^(k), retained and strictly positive, shape (rank_k,)
    coefficients: np.ndarray  # beta^(k) = Q_k^T b_k, shape (rank_k,)
    residual_floor_sq: float  # energy of b_k outside the retained measurement subspace

    def __post_init__(self) -> None:
        if self.eigenvalues.ndim != 1 or self.coefficients.ndim != 1:
            raise ValueError("eigenvalues and coefficients must be one-dimensional")
        if self.eigenvalues.shape != self.coefficients.shape:
            raise ValueError(
                f"eigenvalues {self.eigenvalues.shape} and coefficients "
                f"{self.coefficients.shape} must have the same length"
            )
        if not np.isfinite(self.residual_floor_sq) or self.residual_floor_sq < 0.0:
            raise ValueError(
                f"residual_floor_sq must be finite and non-negative, got {self.residual_floor_sq}"
            )


def gcv_curve(modes: list[ModeSpectrum], lambdas: np.ndarray, *, n_observations: int) -> np.ndarray:
    """GCV over ``lambdas`` for the block-diagonal rotated problem."""
    if not modes:
        raise ValueError("modes must contain at least one rotated temporal mode")
    max_rank = max(mode.eigenvalues.size for mode in modes)
    if n_observations < max_rank:
        raise ValueError(
            f"n_observations={n_observations} cannot be smaller than retained rank={max_rank}"
        )

    lam_sq = np.asarray(lambdas, dtype=np.float64)[:, None] ** 2
    numerator = np.zeros(lam_sq.shape[0], dtype=np.float64)
    trace = np.zeros(lam_sq.shape[0], dtype=np.float64)
    for mode in modes:
        w = np.clip(mode.eigenvalues, 0.0, None)[None, :]
        f = lam_sq / (w + lam_sq)  # filter factors, (n_grid, rank_k)
        numerator += mode.residual_floor_sq + np.sum(f**2 * mode.coefficients[None, :] ** 2, axis=1)
        trace += (n_observations - mode.eigenvalues.size) + np.sum(f, axis=1)

    total_observations = n_observations * len(modes)
    return total_observations * numerator / np.where(trace < 1e-300, 1e-300, trace) ** 2


def select_lambda_gcv_blocks(
    modes: list[ModeSpectrum],
    lambda_range: tuple[float, float],
    n_grid: int = 1000,
    *,
    n_observations: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Minimize the joint GCV functional on a log grid; returns ``(lambda, grid, gcv)``."""
    lambdas = np.geomspace(lambda_range[0], lambda_range[1], n_grid)
    gcv = gcv_curve(modes, lambdas, n_observations=n_observations)
    return float(lambdas[int(np.argmin(gcv))]), lambdas, gcv

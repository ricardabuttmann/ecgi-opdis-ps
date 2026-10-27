"""Deterministic primal-dual solver for spatial-TV ECGI reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from ecgi.models.total_variation.operators import isotropic_total_variation


@dataclass(frozen=True)
class TvSolverResult:
    """TV solution and convergence diagnostics."""

    x_hat: np.ndarray
    iterations: int
    converged: bool
    relative_change: float
    objective: float


def _operator_norm_upper_bound(operator: sp.spmatrix) -> float:
    """Return ``sqrt(||K||_1 ||K||_inf)``, a deterministic spectral-norm upper bound."""
    absolute = abs(operator)
    norm_one = float(np.asarray(absolute.sum(axis=0)).max())
    norm_inf = float(np.asarray(absolute.sum(axis=1)).max())
    bound = np.sqrt(norm_one * norm_inf)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError("gradient operator must have a finite positive norm")
    return float(bound)


def _validate_inputs(
    A: np.ndarray,
    b: np.ndarray,
    gradient: sp.spmatrix,
    alpha: float,
    epsilon: float,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    """Validate and normalize solver inputs."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b = b[:, None]
    if A.ndim != 2 or b.ndim != 2 or A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have compatible shapes (m, n) and (m, n_time)")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A and b must be finite")
    n_nodes = A.shape[1]
    if gradient.shape != (2 * n_nodes, n_nodes):
        raise ValueError(f"gradient shape must be {(2 * n_nodes, n_nodes)}, got {gradient.shape}")
    if alpha <= 0.0 or not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite and positive, got {alpha}")
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError(f"tolerance must be finite and positive, got {tolerance}")
    return A, b, gradient.tocsr()


def solve_total_variation(
    A: np.ndarray,
    b: np.ndarray,
    gradient: sp.spmatrix,
    *,
    alpha: float,
    epsilon: float = 1e-6,
    max_iterations: int = 32000,
    tolerance: float = 1e-5,
) -> TvSolverResult:
    """Minimize ``0.5||A X-b||_F^2 + alpha TV(X) + 0.5 epsilon||X||_F^2``.

    Chambolle--Pock/PDHG handles the non-smooth isotropic TV term through projection onto
    per-node L2 balls. The primal proximal step is solved exactly with a Woodbury reduction:
    only ``I + tau/c A A^T`` in electrode space is Cholesky-factorized, where
    ``c = 1 + tau*epsilon``.
    """
    A, b, gradient = _validate_inputs(A, b, gradient, alpha, epsilon, max_iterations, tolerance)
    n_nodes, n_time = A.shape[1], b.shape[1]
    norm_bound = _operator_norm_upper_bound(gradient)
    tau = sigma = 0.99 / norm_bound
    c = 1.0 + tau * epsilon

    reduced = np.eye(A.shape[0]) + (tau / c) * (A @ A.T)
    reduced_factor = scipy.linalg.cho_factor(reduced, lower=True, check_finite=False)
    At_b = A.T @ b

    def prox_data(rhs: np.ndarray) -> np.ndarray:
        """Apply prox of the quadratic data term without an epicardial-space factorization."""
        base = rhs + tau * At_b
        correction = scipy.linalg.cho_solve(reduced_factor, A @ base, check_finite=False)
        return base / c - (tau / c**2) * (A.T @ correction)

    x = np.zeros((n_nodes, n_time), dtype=np.float64)
    x_bar = x.copy()
    dual = np.zeros((2, n_nodes, n_time), dtype=np.float64)
    relative_change = float("inf")
    converged = False
    def project_dual(candidate: np.ndarray) -> np.ndarray:
        """Project dual blocks onto the per-node radius-alpha L2 balls."""
        candidate_norm = np.linalg.norm(candidate, axis=0)
        return candidate / np.maximum(1.0, candidate_norm / alpha)[None, :, :]

    for _iteration in range(1, max_iterations + 1):
        dual += sigma * np.asarray(gradient @ x_bar).reshape(2, n_nodes, n_time)
        dual = project_dual(dual)

        previous = x
        dual_flat = dual.reshape(2 * n_nodes, n_time)
        x = prox_data(previous - tau * (gradient.T @ dual_flat))
        x_bar = 2.0 * x - previous
        relative_change = float(np.linalg.norm(x - previous) / max(1.0, float(np.linalg.norm(x))))
        if relative_change <= tolerance:
            converged = True
            break

    residual = A @ x - b
    objective = (
        0.5 * float(np.sum(residual**2))
        + alpha * isotropic_total_variation(x, gradient)
        + 0.5 * epsilon * float(np.sum(x**2))
    )
    return TvSolverResult(
        x_hat=x,
        iterations=_iteration,
        converged=converged,
        relative_change=relative_change,
        objective=objective,
    )

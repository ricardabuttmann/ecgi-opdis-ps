"""Tikhonov baseline: order 0/1/2 spatial regularization for the ECGI inverse problem.

    x_hat = argmin_x ||A x - b||^2 + lambda^2 x^T M_order x

where M_0 = I, M_1 = L is the surface Dirichlet-energy matrix, and M_2 = L^T L is the
curvature-energy matrix (L is symmetric in the current template discretization).

Solved through the dual (reduced) formulation because m = n_electrodes (176 on the released
template) is far smaller than n_epi (1800 template state): with G = A M^-1 A^T (m x m) and a
single sparse factorization of the
regularization operator M, one eigendecomposition of G then yields both the lambda selection and
the closed-form solution x_lambda for the chosen lambda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from ecgi.forward.transfer_matrix import _cholmod_usable
from ecgi.models.base import InverseModel
from ecgi.models.tikhonov.parameter_selection import select_lambda_gcv, select_lambda_lcurve


def _factorize_spd(M: sp.spmatrix):
    """Factorize an SPD matrix for repeated solves (CHOLMOD, else SuperLU fallback)."""
    Mc = M.tocsc()
    if _cholmod_usable():
        from sksparse.cholmod import cholesky

        factor = cholesky(Mc)
        return lambda rhs: factor(rhs)
    lu = sp.linalg.splu(Mc)
    return lambda rhs: lu.solve(np.asarray(rhs))


def regularization_operator(
    L: sp.spmatrix, order: int, lambda_min: float, ridge_rel: float = 1e-3
) -> sp.csr_matrix:
    """Quadratic penalty matrix for order 0 (amplitude), 1 (gradient), or 2 (curvature).

    Specifically, M_0 = I, M_1 = L so that x^T L x is the discrete surface Dirichlet energy,
    and M_2 = L^T L (equal to L @ L here) so that x^T M_2 x = ||L x||^2.

    The constant lies exactly in the kernel of L (and thus of L^2), which makes M singular for
    the inversion M^-1 A^T. Ridge = ridge_rel * lambda_min(L)^order lifts ONLY that null mode
    (ridge_rel << 1 relative to the smallest TRUE eigenvalue of M). An absolute ridge would be
    either too small (numerical noise dominates) or too large (it corrupts genuine
    regularization directions), because L^2 squares the eigenvalues and so shrinks the gap to
    the null eigenvalue by orders of magnitude.
    """
    n = L.shape[0]
    if order == 0:
        return sp.identity(n, format="csr")
    M = L if order == 1 else (L @ L).tocsr()
    ridge = ridge_rel * lambda_min**order
    return (M + ridge * sp.identity(n, format="csr")).tocsr()


def effective_rank_mask(eigenvalues: np.ndarray, rank_tol: float) -> np.ndarray:
    """Select numerically resolved reduced-Gram modes relative to its largest eigenvalue."""
    w = np.asarray(eigenvalues, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("eigenvalues must be a non-empty one-dimensional array")
    if not np.isfinite(w).all():
        raise ValueError("eigenvalues must all be finite")
    if not 0.0 < rank_tol < 1.0:
        raise ValueError(f"rank_tol must lie strictly between 0 and 1, got {rank_tol}")
    w_max = float(np.max(w))
    if w_max <= 0.0:
        raise ValueError("reduced Gram matrix has no positive eigenvalue")
    return w > w_max * rank_tol


@dataclass
class TikhonovResult:
    """Diagnostics from the last reconstruct call (for inspecting lambda selection)."""

    x_hat: np.ndarray  # (n_epi, n_time)
    lambda_: float
    lambda_grid: np.ndarray
    criterion: np.ndarray  # GCV values, or (rho, eta) packed as (2, n) for the L-curve
    method: str
    retained_rank: int
    n_observations: int
    rank_tol: float


class Tikhonov(InverseModel):
    """Generalized Tikhonov, order 0/1/2, with GCV (default) or L-curve lambda selection."""

    def __init__(
        self,
        order: int = 2,
        method: str = "gcv",
        lambda_range: tuple[float, float] = (1e-8, 1e2),
        rank_tol: float = 1e-6,
    ):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be 0, 1 or 2, got {order}")
        if method not in ("gcv", "lcurve"):
            raise ValueError(f"method must be 'gcv' or 'lcurve', got {method}")
        self.order = order
        self.method = method
        self.lambda_range = lambda_range
        self.rank_tol = rank_tol
        self.last_result_: TikhonovResult | None = None

    def fit(self, train_data: Any) -> None:
        """Training-free -- Tikhonov needs no training data."""
        return None

    def reconstruct(self, case: Any, geometry: Any) -> np.ndarray:
        b = case.b if case.b.ndim == 2 else case.b[:, None]  # (n_electrodes, n_time)
        A = (
            case.forward_A
        )  # (n_electrodes, n_epi) -- average-referenced forward operator for THIS case
        M = regularization_operator(geometry.laplacian, self.order, geometry.lambda_min)

        solve_M = _factorize_spd(M)
        V = solve_M(A.T)  # (n_epi, m) = M^-1 A^T, independent of lambda
        G = A @ V  # (m, m) reduced Gram matrix
        G = 0.5 * (G + G.T)  # symmetrize away round-off before eigh
        w, Q = np.linalg.eigh(G)  # G = Q diag(w) Q^T, m ~ 200 so this is cheap

        # Effective-rank truncation. G = C(A M^-1 A^T)C is heavily rank-deficient: the constant
        # average-reference mode is an exact zero eigenvalue, and the forward operator cannot
        # resolve most electrode directions (severe ECGI ill-posedness), so a large block of w_i
        # sits at the numerical-noise floor (some slightly negative). These modes carry no signal,
        # yet their tiny non-zero beta divided by (w_i + lambda^2) ~ machine-eps would blow the
        # solution up by orders of magnitude (and GCV, minimizing only the residual, happily drives
        # lambda to the grid floor because those modes never penalize it). Keep only the modes the
        # operator actually resolves (w_i > w_max * rank_tol) and drop the rest from BOTH the lambda
        # selection and the reconstruction -- a TSVD+Tikhonov hybrid.
        keep = effective_rank_mask(w, self.rank_tol)
        w, Q = w[keep], Q[:, keep]
        beta = Q.T @ b  # (rank, n_time)
        # The truncated orthogonal complement is invisible to the estimator, hence H=0 there.
        # Keep its complete energy in the measurement-space residual used by GCV/L-curve. Computing
        # it via Pythagoras avoids retaining the discarded eigenvectors.
        residual_floor_sq = max(0.0, float(np.sum(b**2) - np.sum(beta**2)))

        if self.method == "gcv":
            lam, grid, gcv = select_lambda_gcv(
                w,
                beta,
                self.lambda_range,
                n_observations=A.shape[0],
                residual_floor_sq=residual_floor_sq,
            )
            criterion = gcv
        else:
            lam, grid, rho, eta = select_lambda_lcurve(
                w, beta, self.lambda_range, residual_floor_sq=residual_floor_sq
            )
            criterion = np.stack([rho, eta])

        # x_lambda = V (G + lambda^2 I)^-1 b = V Q ( beta / (w + lambda^2) ); w > 0 after truncation.
        z = Q @ (beta / (w[:, None] + lam**2))
        x_hat = V @ z  # (n_epi, n_time)

        self.last_result_ = TikhonovResult(
            x_hat=x_hat,
            lambda_=float(lam),
            lambda_grid=grid,
            criterion=criterion,
            method=self.method,
            retained_rank=int(w.size),
            n_observations=int(A.shape[0]),
            rank_tol=float(self.rank_tol),
        )
        return x_hat

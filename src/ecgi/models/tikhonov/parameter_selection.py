"""Regularization-parameter (lambda) selection for Tikhonov, in the reduced electrode space.

Both selectors work from the eigendecomposition of the reduced Gram matrix
    G = A M^-1 A^T = Q diag(w) Q^T        (m x m, m = n_electrodes)
and beta = Q^T b. In this basis every quantity needed for lambda selection is a cheap
per-eigenvalue expression, so a whole lambda grid costs only O(m * n_grid) vector ops --
no linear system is solved per lambda.

Filter factor per eigenvalue:  f_i(lambda) = lambda^2 / (w_i + lambda^2)
    residual  ||A x_lambda - b||^2 = sum_i f_i^2 * s_i,      s_i = sum_t beta_{i,t}^2
    seminorm  ||L x_lambda||^2     = sum_i w_i * s_i / (w_i + lambda^2)^2
    influence trace  tr(I - H) = sum_i f_i        (H = G (G + lambda^2 I)^-1)
"""

from __future__ import annotations

import numpy as np


def _row_energy(beta: np.ndarray) -> np.ndarray:
    """s_i = sum over time of beta_{i,t}^2  -> (m,); aggregates multi-time b."""
    return np.sum(beta**2, axis=1)


def select_lambda_gcv(
    w: np.ndarray,
    beta: np.ndarray,
    lambda_range: tuple[float, float],
    n_grid: int = 1000,
    *,
    n_observations: int | None = None,
    residual_floor_sq: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Generalized Cross-Validation (Golub, Heath & Wahba 1979).

    Minimizes G(lambda) = m * ||(I - H) b||^2 / tr(I - H)^2, a smooth scalar function with
    a well-defined minimum -- far more robust for an automated pipeline than L-curve corner
    detection, which hunts a corner on a discrete, noisy curve. Evaluated on a fine log grid
    and the minimizer returned (deterministic, no optimizer edge cases).
    """
    w = np.clip(w, 0.0, None)
    s = _row_energy(beta)
    m = w.size if n_observations is None else n_observations
    if m < w.size:
        raise ValueError(f"n_observations={m} cannot be smaller than retained rank={w.size}")
    if not np.isfinite(residual_floor_sq) or residual_floor_sq < 0.0:
        raise ValueError(
            f"residual_floor_sq must be finite and non-negative, got {residual_floor_sq}"
        )
    lambdas = np.geomspace(lambda_range[0], lambda_range[1], n_grid)
    f = lambdas[:, None] ** 2 / (w[None, :] + lambdas[:, None] ** 2)  # (n_grid, m)
    # H is zero on the truncated measurement subspace. Those components therefore contribute
    # their complete, lambda-independent energy to ||(I-H)b||^2 and one degree of freedom each to
    # tr(I-H). Omitting them changes the GCV functional after rank truncation and is especially
    # optimistic for AWGN, whose energy is present in every measurement direction.
    unresolved_dof = m - w.size
    num = residual_floor_sq + np.sum(f**2 * s[None, :], axis=1)
    den = (unresolved_dof + np.sum(f, axis=1)) ** 2
    gcv = m * num / np.where(den < 1e-300, 1e-300, den)
    idx = int(np.argmin(gcv))
    return float(lambdas[idx]), lambdas, gcv


def lcurve_rho_eta(
    w: np.ndarray,
    beta: np.ndarray,
    lambdas: np.ndarray,
    *,
    residual_floor_sq: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Data misfit rho = ||A x - b|| and solution seminorm eta = ||L x|| over a lambda grid."""
    w = np.clip(w, 0.0, None)
    s = _row_energy(beta)
    denom = w[None, :] + lambdas[:, None] ** 2
    rho = np.sqrt(
        residual_floor_sq + np.sum((lambdas[:, None] ** 2 / denom) ** 2 * s[None, :], axis=1)
    )
    eta = np.sqrt(np.sum(w[None, :] * s[None, :] / denom**2, axis=1))
    return rho, eta


def select_lambda_lcurve(
    w: np.ndarray,
    beta: np.ndarray,
    lambda_range: tuple[float, float],
    n_grid: int = 200,
    *,
    residual_floor_sq: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """L-curve corner via max curvature of the (log rho, log eta) curve (Hansen 1992).

    Kept as an alternative to GCV. Monotonicity is enforced first, a cubic spline is fitted
    through (log lambda, log rho/eta), and the curvature is evaluated analytically on a fine
    grid with the boundaries excluded (one-sided end differences there are unstable and tend
    to pick a grid edge instead of the true corner). GCV is the default; this is more fragile.
    """
    from scipy.interpolate import CubicSpline

    lambdas = np.geomspace(lambda_range[0], lambda_range[1], n_grid)
    rho, eta = lcurve_rho_eta(w, beta, lambdas, residual_floor_sq=residual_floor_sq)
    rho_mono = np.maximum.accumulate(rho)
    eta_mono = np.maximum.accumulate(eta[::-1])[::-1]

    log_lam = np.log(lambdas)
    cs_rho = CubicSpline(log_lam, np.log(rho_mono + 1e-300))
    cs_eta = CubicSpline(log_lam, np.log(eta_mono + 1e-300))
    t = np.linspace(log_lam[0], log_lam[-1], 20 * n_grid)
    d1r, d2r = cs_rho(t, 1), cs_rho(t, 2)
    d1e, d2e = cs_eta(t, 1), cs_eta(t, 2)
    curvature = (d1r * d2e - d1e * d2r) / np.where(
        (d1r**2 + d1e**2) ** 1.5 < 1e-300, 1e-300, (d1r**2 + d1e**2) ** 1.5
    )
    margin = t.size // 10
    best_t = t[margin:-margin][np.argmax(curvature[margin:-margin])]
    idx = int(np.argmin(np.abs(log_lam - best_t)))
    return float(lambdas[idx]), lambdas, rho, eta

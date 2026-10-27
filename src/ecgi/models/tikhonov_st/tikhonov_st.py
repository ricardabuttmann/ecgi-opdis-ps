"""Spatiotemporal Tikhonov baseline: spatial order 0/1/2 plus a temporal difference penalty.

    x_hat = argmin_X ||A X - B||_F^2 + lambda_s^2 tr(X^T M_order X)
            + lambda_t^2 ||X D_t^T||_F^2

where M_order is the same amplitude/gradient/curvature penalty matrix as in the spatial baseline.

This exists to answer one question the current benchmark cannot: both learned methods couple time
(the Transformer attends over the sequence, the diffusion prior is a 3D U-Net over ``[T, Z, phi]``)
while every classical control regularizes frame by frame. Without a classical control that also
couples time, the learned methods' advantage on activation time and localization cannot be
separated from plain temporal smoothness.

It is a separate model package rather than an option on ``ecgi.models.tikhonov`` because the
spatial-only ``order1_gcv`` configuration is a frozen five-SNR Test baseline; adding a parameter
there would change its variant labels and provenance records.

The solve reuses the spatial baseline's reduced (dual) formulation per rotated temporal mode -- see
``temporal.py`` for the rotation and ``parameter_selection.py`` for the joint lambda selection.
Setting ``rho = 0`` reproduces the spatial-only baseline up to floating-point round-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp

from ecgi.models.base import InverseModel

# The single factorization implementation is shared deliberately: a second copy would be free to
# drift away from the spatial baseline's CHOLMOD/SuperLU fallback behaviour.
from ecgi.models.tikhonov.tikhonov import (
    _factorize_spd,
    effective_rank_mask,
    regularization_operator,
)
from ecgi.models.tikhonov_st.parameter_selection import ModeSpectrum, select_lambda_gcv_blocks
from ecgi.models.tikhonov_st.temporal import TEMPORAL_ORDERS, temporal_gram_eigendecomposition


@dataclass
class TikhonovSTResult:
    """Diagnostics from the last reconstruct call."""

    x_hat: np.ndarray  # (n_epi, n_time)
    lambda_: float
    rho: float
    order: int
    order_t: int
    lambda_grid: np.ndarray
    gcv: np.ndarray
    n_observations: int
    n_time: int
    n_distinct_shifts: int
    retained_ranks: np.ndarray = field(repr=False)  # per rotated temporal mode


def _shifted_operator(m_spatial: sp.csr_matrix, shift: float) -> sp.csr_matrix:
    """``M_s + shift * I``: the rotated mode's regularization operator (shift = rho * mu_k)."""
    if shift == 0.0:
        return m_spatial
    return (m_spatial + shift * sp.identity(m_spatial.shape[0], format="csr")).tocsr()


def _reduced_spectrum(
    forward: np.ndarray, m_spatial: sp.csr_matrix, shift: float, rank_tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Truncated eigendecomposition of the reduced Gram ``G = A (M_s + shift I)^-1 A^T``."""
    solve_m = _factorize_spd(_shifted_operator(m_spatial, shift))
    gram = forward @ solve_m(forward.T)
    gram = 0.5 * (gram + gram.T)  # symmetrize away round-off before eigh
    w, q = np.linalg.eigh(gram)
    keep = effective_rank_mask(w, rank_tol)
    return w[keep], q[:, keep]


class TikhonovSpatioTemporal(InverseModel):
    """Generalized Tikhonov with an added temporal difference penalty, GCV-selected ``lambda_s``.

    ``rho = lambda_t^2 / lambda_s^2`` is a fixed model setting rather than a per-case quantity: it
    is selected once on the validation split, mirroring how TV's alpha is selected, and only
    ``lambda_s`` is chosen per case by GCV. Selecting both per case would need a 2-D GCV search,
    which is both far more expensive and markedly less stable than the 1-D one.
    """

    def __init__(
        self,
        order: int = 1,
        order_t: int = 1,
        rho: float = 0.0,
        lambda_range: tuple[float, float] = (1e-8, 1e2),
        rank_tol: float = 1e-6,
    ):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be 0, 1 or 2, got {order}")
        if order_t not in TEMPORAL_ORDERS:
            raise ValueError(f"order_t must be one of {TEMPORAL_ORDERS}, got {order_t}")
        if not np.isfinite(rho) or rho < 0.0:
            raise ValueError(f"rho must be finite and non-negative, got {rho!r}")
        self.order = order
        self.order_t = order_t
        self.rho = float(rho)
        self.lambda_range = lambda_range
        self.rank_tol = rank_tol
        self.last_result_: TikhonovSTResult | None = None

    def fit(self, train_data: Any) -> None:
        """Training-free; ``rho`` is selected on validation, not fitted to training data."""
        return None

    def reconstruct(self, case: Any, geometry: Any) -> np.ndarray:
        b = case.b if case.b.ndim == 2 else case.b[:, None]  # (n_electrodes, n_time)
        forward = case.forward_A  # (n_electrodes, n_epi), average-referenced, per case
        n_observations, n_time = b.shape
        if n_time <= self.order_t:
            # Degrading silently to the spatial-only baseline would report a spatiotemporal method
            # that never applied a temporal penalty.
            raise ValueError(
                f"a temporal penalty of order {self.order_t} needs more than {self.order_t} time "
                f"samples, got {n_time}"
            )

        m_spatial = regularization_operator(geometry.laplacian, self.order, geometry.lambda_min)
        mu, rotation = temporal_gram_eigendecomposition(n_time, self.order_t)
        b_rotated = b @ rotation  # (n_electrodes, n_time), one column per rotated temporal mode
        shifts = self.rho * mu

        # Pass 1: per-mode spectra for the joint lambda selection. Only (w, Q) is kept, never the
        # (n_epi, m) solve of each mode, whose combined size would be hundreds of megabytes.
        spectra: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        modes: list[ModeSpectrum] = []
        for k in range(n_time):
            shift = float(shifts[k])
            if shift not in spectra:
                spectra[shift] = _reduced_spectrum(forward, m_spatial, shift, self.rank_tol)
            w, q = spectra[shift]
            beta = q.T @ b_rotated[:, k]
            modes.append(
                ModeSpectrum(
                    eigenvalues=w,
                    coefficients=beta,
                    # Energy outside the retained measurement subspace, via Pythagoras so the
                    # discarded eigenvectors never have to be kept.
                    residual_floor_sq=max(
                        0.0, float(b_rotated[:, k] @ b_rotated[:, k] - beta @ beta)
                    ),
                )
            )

        lam, grid, gcv = select_lambda_gcv_blocks(
            modes, self.lambda_range, n_observations=n_observations
        )

        # Pass 2: reconstruct each rotated mode. Refactorizing costs one extra factorization per
        # distinct shift but needs only a single right-hand side here, unlike pass 1's m columns.
        solvers = {shift: _factorize_spd(_shifted_operator(m_spatial, shift)) for shift in spectra}
        x_rotated = np.empty((forward.shape[1], n_time), dtype=np.float64)
        for k, mode in enumerate(modes):
            w, q = spectra[float(shifts[k])]
            z = q @ (mode.coefficients / (w + lam**2))
            x_rotated[:, k] = solvers[float(shifts[k])](forward.T @ z).ravel()

        x_hat = x_rotated @ rotation.T  # rotate back; U is orthogonal

        self.last_result_ = TikhonovSTResult(
            x_hat=x_hat,
            lambda_=float(lam),
            rho=self.rho,
            order=self.order,
            order_t=self.order_t,
            lambda_grid=grid,
            gcv=gcv,
            n_observations=int(n_observations),
            n_time=int(n_time),
            n_distinct_shifts=len(spectra),
            retained_ranks=np.array([mode.eigenvalues.size for mode in modes]),
        )
        return x_hat

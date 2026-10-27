"""Temporal difference operators and the rotation that decouples the spatiotemporal solve.

The spatiotemporal objective couples time only through ``X D_t^T``:

    min_X  ||A X - B||_F^2 + lambda_s^2 ||L^order X||_F^2 + lambda_t^2 ||X D_t^T||_F^2

Because the temporal term acts from the right and the spatial term from the left, diagonalizing
the temporal Gram ``D_t^T D_t = U diag(mu) U^T`` and rotating ``X_tilde = X U``, ``B_tilde = B U``
splits the problem into ``n_time`` independent spatial problems (``U`` is orthogonal, so the data
and spatial terms are invariant under the rotation):

    min_x_k  ||A x_k - b_k||^2 + lambda_s^2 ( ||L^order x_k||^2 + rho * mu_k ||x_k||^2 )

with ``rho = lambda_t^2 / lambda_s^2``. Each rotated mode is therefore an ordinary generalized
Tikhonov problem whose regularization operator is shifted by ``rho * mu_k * I``.

Time is treated as an open interval, not a cycle: the analysis window is a beat segment, so the
difference operators must not wrap the last sample onto the first the way the periodic ``phi``
direction of the template does. Differences use unit sample spacing; any ``dt`` factor is absorbed
into ``rho``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

TEMPORAL_ORDERS = (1, 2)


def temporal_difference_operator(n_time: int, order: int) -> sp.csr_matrix:
    """Forward difference operator ``D_t`` of ``order`` 1 or 2 on ``n_time`` open samples.

    Order 1 returns the ``(n_time-1, n_time)`` first difference, whose kernel is the constant
    (temporally flat) signal. Order 2 returns the ``(n_time-2, n_time)`` second difference, whose
    kernel additionally contains the linear ramp. Neither wraps around the window.
    """
    if order not in TEMPORAL_ORDERS:
        raise ValueError(f"order must be one of {TEMPORAL_ORDERS}, got {order}")
    if n_time <= order:
        raise ValueError(f"n_time must exceed order {order} to form a difference, got {n_time}")

    stencil = np.array([-1.0, 1.0]) if order == 1 else np.array([1.0, -2.0, 1.0])
    n_rows = n_time - order
    rows = np.repeat(np.arange(n_rows), stencil.size)
    cols = (np.arange(n_rows)[:, None] + np.arange(stencil.size)[None, :]).ravel()
    data = np.tile(stencil, n_rows)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_time))


def temporal_gram_eigendecomposition(n_time: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu, U)`` with ``D_t^T D_t = U diag(mu) U^T``, ``mu`` ascending and non-negative.

    ``n_time`` is on the order of a hundred samples, so the dense symmetric eigendecomposition is
    negligible next to the per-mode spatial solves it enables. Eigenvalues are clipped at zero:
    ``D_t^T D_t`` is positive semidefinite by construction and its kernel modes come out of LAPACK
    at the negative noise floor, which would otherwise turn into a negative regularization shift.
    """
    d = temporal_difference_operator(n_time, order)
    gram = (d.T @ d).toarray()
    gram = 0.5 * (gram + gram.T)  # symmetrize away round-off before eigh
    mu, u = np.linalg.eigh(gram)
    return np.clip(mu, 0.0, None), u

"""Average-referencing primitives for ECGI measurement space.

Electrode potentials are defined only up to a common additive constant, so the measurement lives
in the ``(m-1)``-dimensional zero-mean subspace. This module provides the small, deterministic
building blocks the spectral samplers use to work in that subspace:

* ``average_reference_projector`` -- the projector ``Q = I - 11.T/m``;
* ``average_reference_basis`` -- a deterministic orthonormal Helmert basis for ``range(Q)``;
* ``rms_matched_referenced_variance`` -- the subspace noise variance matching a realized
  post-reference per-entry RMS.

Only small dense electrode matrices are formed here.
"""

from __future__ import annotations

import numpy as np


def average_reference_projector(n_electrodes: int) -> np.ndarray:
    """Return ``Q = I - 11.T/m`` in float64."""
    if n_electrodes < 2:
        raise ValueError(f"average referencing needs at least two electrodes, got {n_electrodes}")
    one = np.ones(n_electrodes, dtype=np.float64)
    return np.eye(n_electrodes, dtype=np.float64) - np.outer(one, one) / n_electrodes


def average_reference_basis(n_electrodes: int) -> np.ndarray:
    """Return a deterministic orthonormal Helmert basis for ``range(Q)``.

    The result has shape ``[m, m-1]``, satisfies ``R.T R = I`` and ``R R.T = Q``, and avoids
    eigensolver-dependent signs or rotations in diagnostics.
    """
    if n_electrodes < 2:
        raise ValueError(f"average referencing needs at least two electrodes, got {n_electrodes}")
    basis = np.zeros((n_electrodes, n_electrodes - 1), dtype=np.float64)
    for column in range(n_electrodes - 1):
        k = column + 1
        scale = np.sqrt(k * (k + 1.0))
        basis[:k, column] = 1.0 / scale
        basis[k, column] = -k / scale
    return basis


def rms_matched_referenced_variance(noise_rms: float, n_electrodes: int) -> float:
    """Return the subspace variance matching a realized post-reference per-entry RMS.

    Evaluation noise is projected and then globally normalized so
    ``||epsilon||_F^2 = m * n_time * noise_rms^2`` exactly.  Rotational symmetry in the
    ``(m-1) * n_time`` dimensional referenced subspace therefore gives the second moment

    ``E[epsilon_t epsilon_t.T] = noise_rms^2 * m/(m-1) * Q``.

    The norm-conditioned draw is not Gaussian and its time frames are not independent; this
    function matches its covariance, not its full joint distribution.
    """
    if n_electrodes < 2:
        raise ValueError(f"average referencing needs at least two electrodes, got {n_electrodes}")
    if not np.isfinite(noise_rms) or noise_rms < 0:
        raise ValueError(f"noise_rms must be non-negative and finite, got {noise_rms!r}")
    return float(noise_rms**2 * n_electrodes / (n_electrodes - 1))

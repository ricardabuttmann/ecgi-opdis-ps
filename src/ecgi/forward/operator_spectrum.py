"""SVD geometry required by DDRM for the average-referenced ECGI operator.

Average referencing makes the electrode-space covariance singular. DDRM therefore works in the
reduced Helmert coordinates

    y_tilde = R.T @ y,    H_tilde = R.T @ H,

where the measurement noise is isotropic and an economy SVD of H_tilde gives the independent
spectral coordinates used by the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecgi.forward.likelihood_geometry import (
    average_reference_basis,
    rms_matched_referenced_variance,
)

def reduce_operator(operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Helmert basis R and reduced operator R.T @ H."""
    operator = np.asarray(operator, dtype=np.float64)
    if operator.ndim != 2:
        raise ValueError(f"operator must be two-dimensional, got shape {operator.shape}")
    if not np.isfinite(operator).all():
        raise ValueError("operator contains non-finite values")
    basis = average_reference_basis(operator.shape[0])
    return basis, basis.T @ operator


def measurement_sigma(noise_rms: float, n_electrodes: int) -> float:
    """Return the measurement-noise scale in reduced Helmert coordinates."""
    return float(np.sqrt(rms_matched_referenced_variance(noise_rms, n_electrodes)))


@dataclass(frozen=True)
class OperatorSpectrum:
    """Economy SVD of the reduced operator H_tilde = U diag(s) Vr.T."""

    reduced_basis: np.ndarray
    left_vectors: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray

    def resolved_mask(self, relative_floor: float = 0.0) -> np.ndarray:
        """Return the singular directions above the numerical stability floor."""
        if not 0.0 <= relative_floor < 1.0:
            raise ValueError(f"relative_floor must lie in [0, 1), got {relative_floor}")
        if self.singular_values.size == 0:
            return np.zeros(0, dtype=bool)
        return self.singular_values > relative_floor * float(self.singular_values[0])


def operator_spectrum(operator: np.ndarray) -> OperatorSpectrum:
    """Reduce the operator and return its economy SVD."""
    basis, reduced = reduce_operator(operator)
    left, singular_values, right = np.linalg.svd(reduced, full_matrices=False)
    return OperatorSpectrum(
        reduced_basis=basis,
        left_vectors=left,
        singular_values=singular_values,
        right_vectors=right,
    )

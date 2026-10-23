"""Measurement-space whitening for discrepancy-aware ECGI likelihoods.

Constructs a reduced Helmert-space whitener from measurement noise and a frozen discrepancy
covariance. Conditional artifacts additionally provide the OpDis-PS operator correction and affine
offset.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ecgi.forward.likelihood_geometry import average_reference_basis


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reduced_whitener(
    discrepancy_covariance: np.ndarray, sigma_epsilon: float, ridge: float
) -> np.ndarray:
    """Return ``W = Sigma_total^{-1/2}`` in the reduced Helmert space, symmetric ``[m-1, m-1]``.

    ``Sigma_total = (sigma_epsilon^2 + ridge^2) I + Sigma_d``. The ridge keeps ``Sigma_total``
    invertible at ``sigma_epsilon = 0`` (clean): ``Sigma_d`` is rank-deficient, so a bare
    ``Sigma_d^{-1/2}`` divides by zero in the many directions the discrepancy does not occupy. At
    finite SNR ``sigma_epsilon^2`` already floors those directions, so ``ridge`` may be 0 there.
    """
    cov = np.asarray(discrepancy_covariance, dtype=np.float64)
    p = cov.shape[0]
    floor = float(sigma_epsilon) ** 2 + float(ridge) ** 2
    total = cov + floor * np.eye(p)
    eigvals, eigvecs = np.linalg.eigh(total)  # symmetric -> real eigenvalues, orthonormal vectors
    # Only require a ridge when Sigma_total is actually not positive definite. A shrunk Sigma_d is
    # typically already PD (identity shrinkage), so a bare covariance with sigma_eps = ridge = 0 is
    # legal; a rank-deficient one at floor = 0 is what fails here, with the clean-condition remedy.
    if eigvals.min() <= 0.0:
        raise ValueError(
            f"Sigma_total = (sigma_eps^2 + ridge^2) I + Sigma_d is not positive definite (min "
            f"eigenvalue {eigvals.min():.3e}); a rank-deficient Sigma_d needs a positive ridge for "
            "the clean condition (sigma_eps = 0)."
        )
    return (eigvecs * eigvals**-0.5) @ eigvecs.T


@dataclass(frozen=True)
class MeasurementWhitening:
    """The frozen reduced discrepancy covariance ``Sigma_d`` and the identity that binds it."""

    discrepancy_covariance: np.ndarray  # Sigma_d = Cov(d - mu_d), reduced Helmert [m-1, m-1]
    ridge: float  # isotropic reduced-space floor for clean invertibility; 0 is legal at finite SNR
    artifact_sha256: str
    artifact_path: str
    # mu_d = E[d~], the systematic discrepancy, reduced [m-1]. The producer's covariance is centred,
    # so this bias must be subtracted from the measurement or the whitened residual is not zero-mean
    # (and a small ridge amplifies any mu_d that sits in a low-variance direction). None = no bias.
    mean_discrepancy: np.ndarray | None = None
    covariance_key: str = "covariance"
    metadata: dict = field(default_factory=dict)

    @property
    def n_reduced(self) -> int:
        return int(self.discrepancy_covariance.shape[0])

    def identity(self) -> dict:
        """Provenance block: what a run manifest stores so the exact covariance is reloadable."""
        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_path": self.artifact_path,
            "ridge": self.ridge,
            "n_reduced": self.n_reduced,
            "mean_subtracted": self.mean_discrepancy is not None,
            "covariance_key": self.covariance_key,
            **{
                k: v for k, v in self.metadata.items() if k in ("scientific_run_id", "execution_id")
            },
        }

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        ridge: float,
        covariance_key: str = "covariance",
        mean_key: str | None = "mu_d",
    ) -> MeasurementWhitening:
        """Load ``Sigma_d`` and ``mu_d`` from a shipped discrepancy-covariance ``.npz`` artifact.

        The training-time producer (not part of this public package) stores the Ledoit-Wolf reduced
        covariance under ``covariance`` and the systematic mean under ``mu_d``. When present, the
        stored ``reduced_basis`` is validated against the Helmert basis for its electrode count, which
        catches a covariance carried over from a different electrode layout.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"discrepancy-covariance artifact {path} not found -- it ships with this "
                "repository under artifacts/<operator-discrepancy>/"
            )
        payload = np.load(path, allow_pickle=False)
        cov = np.asarray(payload[covariance_key], dtype=np.float64)
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError(
                f"{covariance_key} in {path} is not a square matrix: shape {cov.shape}"
            )
        cov = 0.5 * (cov + cov.T)
        p = cov.shape[0]
        if float(ridge) < 0.0:
            raise ValueError(f"ridge must be non-negative, got {ridge}")

        keys = set(payload.files)
        mean = None
        if mean_key is not None and mean_key in keys:
            mean = np.asarray(payload[mean_key], dtype=np.float64).reshape(-1)
            if mean.shape != (p,):
                raise ValueError(f"{mean_key} has shape {mean.shape}, expected ({p},)")
        if "reduced_basis" in keys:
            stored = np.asarray(payload["reduced_basis"], dtype=np.float64)
            expected = average_reference_basis(p + 1)
            if stored.shape != expected.shape or not np.allclose(stored, expected, atol=1e-8):
                raise ValueError(
                    f"reduced_basis in {path} is not the Helmert basis for {p + 1} electrodes -- the "
                    "covariance was estimated on a different electrode layout"
                )
        metadata_keys = (
            "scientific_run_id",
            "execution_id",
            "dataset",
            "conditional_rank",
        )
        metadata = {key: str(payload[key]) for key in metadata_keys if key in keys}
        return cls(cov, float(ridge), _sha256(path), str(path), mean, covariance_key, metadata)


@dataclass(frozen=True)
class ConditionalApproximationError:
    """Linear conditional approximation-error model in reduced measurement coordinates.

    The training diagnostic fits ``d = mu_d + B(s - mu_s) + r`` for ``s = R^T Hx``.  Therefore
    ``y = (I+B)Hx + (mu_d-B mu_s) + eps+r`` and only the residual covariance ``Sigma_r`` belongs in
    the noise model.  Keeping this separate from :class:`MeasurementWhitening` prevents the old
    marginal-``Sigma_d`` mode from silently acquiring an operator correction.
    """

    regression: np.ndarray
    signal_mean: np.ndarray
    discrepancy_mean: np.ndarray
    residual_whitening: MeasurementWhitening
    conditional_model: str = "full"
    conditional_rank: int | None = None

    @property
    def n_reduced(self) -> int:
        return self.residual_whitening.n_reduced

    @property
    def affine_offset(self) -> np.ndarray:
        return self.discrepancy_mean - self.regression @ self.signal_mean

    def identity(self) -> dict:
        return {
            **self.residual_whitening.identity(),
            "conditional_operator_corrected": True,
            "equation": "d = mu_d + B (s - mu_s) + r",
            "conditional_model": self.conditional_model,
            "conditional_rank": self.conditional_rank,
        }

    @classmethod
    def from_npz(cls, path: str | Path, *, ridge: float) -> ConditionalApproximationError:
        residual = MeasurementWhitening.from_npz(
            path,
            ridge=ridge,
            covariance_key="conditional_covariance",
            mean_key=None,
        )
        payload = np.load(Path(path), allow_pickle=False)
        required = {"conditional_regression", "signal_mean", "mu_d"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(
                f"conditional approximation-error artifact {path} is missing {missing}; the shipped "
                "artifacts/<operator-discrepancy>/ file should contain these keys"
            )
        p = residual.n_reduced
        regression = np.asarray(payload["conditional_regression"], dtype=np.float64)
        signal_mean = np.asarray(payload["signal_mean"], dtype=np.float64).reshape(-1)
        discrepancy_mean = np.asarray(payload["mu_d"], dtype=np.float64).reshape(-1)
        if regression.shape != (p, p):
            raise ValueError(
                f"conditional_regression has shape {regression.shape}, expected ({p}, {p})"
            )
        if signal_mean.shape != (p,) or discrepancy_mean.shape != (p,):
            raise ValueError(
                "conditional means do not match the residual covariance dimension: "
                f"signal_mean={signal_mean.shape}, mu_d={discrepancy_mean.shape}, expected ({p},)"
            )
        for name, value in (
            ("conditional_regression", regression),
            ("signal_mean", signal_mean),
            ("mu_d", discrepancy_mean),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} in {path} contains non-finite values")
        conditional_model = (
            str(payload["conditional_model"]) if "conditional_model" in payload else "full"
        )
        stored_rank = int(payload["conditional_rank"]) if "conditional_rank" in payload else -1
        conditional_rank = None if stored_rank < 0 else stored_rank
        if conditional_rank is not None:
            numerical_rank = int(np.linalg.matrix_rank(regression, tol=1e-10))
            if numerical_rank > conditional_rank:
                raise ValueError(
                    f"conditional_regression in {path} has numerical rank {numerical_rank}, "
                    f"exceeding declared rank {conditional_rank}"
                )
        return cls(
            regression,
            signal_mean,
            discrepancy_mean,
            residual,
            conditional_model,
            conditional_rank,
        )


def conditional_corrected_measurement_operator(
    operator: np.ndarray,
    measurement: np.ndarray,
    model: ConditionalApproximationError,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``H -> (I+B)H`` and subtract ``mu_d-B mu_s`` before residual whitening.

    The returned arrays remain in average-referenced electrode coordinates so they compose with
    the existing measurement-whitening path without changing the sampler.
    """

    operator = np.asarray(operator, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    m = operator.shape[0]
    if model.n_reduced != m - 1:
        raise ValueError(
            f"conditional model is {model.n_reduced}-dimensional but the operator has {m} "
            f"electrodes (reduced dimension {m - 1})"
        )
    basis = average_reference_basis(m)
    corrected_reduced = (np.eye(model.n_reduced) + model.regression) @ (basis.T @ operator)
    corrected_measurement = basis.T @ measurement - model.affine_offset[:, None]
    return basis @ corrected_reduced, basis @ corrected_measurement


def whitened_measurement_operator(
    operator: np.ndarray,
    measurement: np.ndarray,
    mw: MeasurementWhitening,
    sigma_epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten operator rows and measurement by ``W = Sigma_total^{-1/2}``, in electrode space.

    Returns ``(operator_w, measurement_w)`` with ``operator_w = W_elec H`` and
    ``measurement_w = W_elec (y - R mu_d)``, ``W_elec = R W R^T``. The systematic discrepancy
    ``R mu_d`` is subtracted first (the approximation-error bias correction), so the whitened residual
    is zero-mean; without it a small ridge over-weights the systematic directions. After the sampler's
    own ``R^T`` reduction the noise is unit-variance, so the caller runs DDRM with ``sigma_y = 1``.
    ``operator`` and ``measurement`` must be average-referenced (electrode rows sum to zero), which
    ``case.forward_A`` and ``case.b`` already are; whitening then leaves the constant direction alone.
    """
    operator = np.asarray(operator, dtype=np.float64)
    measurement = np.asarray(measurement, dtype=np.float64)
    m = operator.shape[0]
    if mw.n_reduced != m - 1:
        raise ValueError(
            f"Sigma_d is {mw.n_reduced}x{mw.n_reduced} but the operator has {m} electrodes "
            f"(reduced dimension m-1 = {m - 1})"
        )
    basis = average_reference_basis(m)  # R, [m, m-1], orthonormal columns
    whitener = reduced_whitener(mw.discrepancy_covariance, sigma_epsilon, mw.ridge)  # [m-1, m-1]
    w_elec = (
        basis @ whitener @ basis.T
    )  # [m, m], acts as W on the referenced subspace, 0 on constant
    if mw.mean_discrepancy is not None:
        measurement = (
            measurement - (basis @ np.asarray(mw.mean_discrepancy, dtype=np.float64))[:, None]
        )
    return w_elec @ operator, w_elec @ measurement

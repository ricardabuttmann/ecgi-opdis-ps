"""Strict, path-independent loading of one public ECGI example bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ExampleCase:
    """Arrays shared by every released inverse method."""

    measurement: np.ndarray
    forward: np.ndarray
    truth: np.ndarray
    points_mm: np.ndarray
    laplacian: sp.csr_matrix
    template_z: np.ndarray
    template_phi: np.ndarray
    template_shape: tuple[int, int]
    lambda_min: float
    dt_ms: float
    target_snr_db: float
    realized_snr_db: float
    noise_norm: float
    case_id: str
    operator_discrepancy: str


PUBLIC_EXAMPLES = (
    "lung-conductivity",
    "subcutaneous-fat",
    "lbbb-lung-conductivity",
)
OPERATOR_DISCREPANCIES = ("lung-conductivity", "subcutaneous-fat")


def public_example_path(name: str, root: str | Path = "examples") -> Path:
    """Resolve one of the bundled paper examples."""
    if name not in PUBLIC_EXAMPLES:
        raise ValueError(f"unknown example {name!r}; available: {PUBLIC_EXAMPLES}")
    return Path(root) / f"{name}.npz"


def _scalar(data: np.lib.npyio.NpzFile, name: str) -> float:
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"{name} must be scalar, got shape {value.shape}")
    result = float(value.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def load_example_case(path: str | Path) -> ExampleCase:
    """Load the documented NPZ schema without pickle or private path discovery."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {
            "schema_version",
            "case_id",
            "measurement",
            "forward",
            "truth",
            "points_mm",
            "laplacian_data",
            "laplacian_indices",
            "laplacian_indptr",
            "laplacian_shape",
            "template_z",
            "template_phi",
            "template_shape",
            "lambda_min",
            "dt_ms",
            "target_snr_db",
            "realized_snr_db",
            "noise_norm",
            "operator_discrepancy",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")

        schema_version = int(np.asarray(data["schema_version"]).item())
        if schema_version != 2:
            raise ValueError(f"unsupported example schema {schema_version!r}")

        measurement = np.asarray(data["measurement"], dtype=np.float64)
        forward = np.asarray(data["forward"], dtype=np.float64)
        truth = np.asarray(data["truth"], dtype=np.float64)
        points_mm = np.asarray(data["points_mm"], dtype=np.float64)
        shape = tuple(int(value) for value in np.asarray(data["laplacian_shape"]).reshape(-1))
        if len(shape) != 2:
            raise ValueError(f"laplacian_shape must contain two entries, got {shape}")
        laplacian = sp.csr_matrix(
            (
                np.asarray(data["laplacian_data"], dtype=np.float64),
                np.asarray(data["laplacian_indices"], dtype=np.int64),
                np.asarray(data["laplacian_indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
        template_z = np.asarray(data["template_z"], dtype=np.float64)
        template_phi = np.asarray(data["template_phi"], dtype=np.float64)
        template_shape = tuple(
            int(value) for value in np.asarray(data["template_shape"]).reshape(-1)
        )
        lambda_min = _scalar(data, "lambda_min")
        dt_ms = _scalar(data, "dt_ms")
        target_snr_db = _scalar(data, "target_snr_db")
        realized_snr_db = _scalar(data, "realized_snr_db")
        noise_norm = _scalar(data, "noise_norm")
        case_id = str(np.asarray(data["case_id"]).item())
        operator_discrepancy = str(np.asarray(data["operator_discrepancy"]).item())

    if measurement.ndim != 2 or forward.ndim != 2 or truth.ndim != 2:
        raise ValueError("measurement, forward, and truth must all be two-dimensional")
    if forward.shape[0] != measurement.shape[0]:
        raise ValueError("forward rows must match the number of measurement electrodes")
    if forward.shape[1] != truth.shape[0]:
        raise ValueError("forward columns must match the number of truth nodes")
    if measurement.shape[1] != truth.shape[1]:
        raise ValueError("measurement and truth must contain the same time samples")
    if laplacian.shape != (truth.shape[0], truth.shape[0]):
        raise ValueError("laplacian shape must match the number of truth nodes")
    if points_mm.shape != (truth.shape[0], 3):
        raise ValueError("points_mm must contain one 3D coordinate per truth node")
    if len(template_shape) != 2 or np.prod(template_shape) != truth.shape[0]:
        raise ValueError("template_shape must be two-dimensional and match the state size")
    if template_z.shape != (truth.shape[0],):
        raise ValueError("template_z must contain one coordinate per state entry")
    if template_phi.shape != template_z.shape:
        raise ValueError("template_z and template_phi must have matching shapes")
    if not all(
        np.isfinite(array).all()
        for array in (
            measurement,
            forward,
            truth,
            points_mm,
            laplacian.data,
            template_z,
            template_phi,
        )
    ):
        raise ValueError("example arrays must contain only finite values")
    if lambda_min <= 0 or dt_ms <= 0:
        raise ValueError("lambda_min and dt_ms must be positive")
    if noise_norm <= 0:
        raise ValueError("noise_norm must be positive for the finite-SNR public case")
    if not case_id or "/" in case_id or "\\" in case_id:
        raise ValueError(f"invalid public case_id {case_id!r}")
    if operator_discrepancy not in OPERATOR_DISCREPANCIES:
        raise ValueError(
            "invalid operator_discrepancy "
            f"{operator_discrepancy!r}; expected {OPERATOR_DISCREPANCIES}"
        )

    return ExampleCase(
        measurement=measurement,
        forward=forward,
        truth=truth,
        points_mm=points_mm,
        laplacian=laplacian,
        template_z=template_z,
        template_phi=template_phi,
        template_shape=template_shape,
        lambda_min=lambda_min,
        dt_ms=dt_ms,
        target_snr_db=target_snr_db,
        realized_snr_db=realized_snr_db,
        noise_norm=noise_norm,
        case_id=case_id,
        operator_discrepancy=operator_discrepancy,
    )

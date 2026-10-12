"""Indexed loading and validation for the accepted-paper dataset packages."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from opdis_ps.data import ExampleCase

FORWARD_SPECS = ("nominal", "conductivity", "subcut_fat_mild")
DISCREPANCY_LABELS = {
    "nominal": "nominal",
    "conductivity": "lung-conductivity",
    "subcut_fat_mild": "subcutaneous-fat",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_package_checksums(root: str | Path) -> dict[str, Any]:
    """Validate the package-wide SHA256SUMS file without trusting its paths."""
    root = Path(root).resolve()
    checksum_path = root / "SHA256SUMS"
    checked = 0
    total_bytes = 0
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative:
            raise ValueError(f"{checksum_path}:{line_number}: invalid checksum line")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError(f"{checksum_path}:{line_number}: unsafe or missing path {relative!r}")
        if _sha256(path) != f"sha256:{digest}":
            raise ValueError(f"{path}: SHA-256 mismatch")
        checked += 1
        total_bytes += path.stat().st_size
    if checked == 0:
        raise ValueError(f"{checksum_path}: no checksum records")
    return {"status": "ok", "file_count": checked, "size_bytes": total_bytes}


class ShardField:
    """Resolve one public identifier to one row in a memory-mapped NumPy shard."""

    def __init__(self, directory: Path, identity: str) -> None:
        self.directory = directory
        self.identity = identity
        manifest = _read_json(directory / "manifest.json")
        index_record = manifest.get("index", {})
        index_path = directory / str(index_record.get("path", "index.jsonl"))
        if _sha256(index_path) != index_record.get("sha256"):
            raise ValueError(f"{index_path}: digest differs from field manifest")
        rows = _read_jsonl(index_path)
        self.rows = {str(row[identity]): row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError(f"{index_path}: duplicate {identity}")
        self.arrays: dict[str, np.ndarray] = {}

    def get(self, identity: str) -> np.ndarray:
        try:
            row = self.rows[identity]
        except KeyError as exc:
            raise KeyError(f"{self.directory}: unknown {self.identity} {identity!r}") from exc
        shard = str(row["shard"])
        if shard not in self.arrays:
            path = self.directory / shard
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"{path}: unsafe or missing shard")
            self.arrays[shard] = np.load(path, mmap_mode="r", allow_pickle=False)
        return np.asarray(self.arrays[shard][int(row["offset"])])


def add_paper_noise(
    measurement_clean: np.ndarray,
    *,
    snr_db: float | None,
    noise_seed: int,
) -> tuple[np.ndarray, float | None, float]:
    """Apply the frozen average-referenced ``awgn_iid`` convention with an explicit seed."""
    clean = np.asarray(measurement_clean, dtype=np.float64)
    if clean.ndim != 2 or not np.isfinite(clean).all():
        raise ValueError("measurement_clean must be a finite [electrode, time] array")
    if snr_db is None:
        return clean.copy(), None, 0.0
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite or None")
    clean_norm = float(np.linalg.norm(clean))
    if clean_norm == 0.0:
        raise ValueError("cannot add relative noise to a zero-norm measurement")
    rng = np.random.default_rng(noise_seed)
    direction = rng.standard_normal(clean.shape)
    direction -= direction.mean(axis=0, keepdims=True)
    degrees_of_freedom = clean.size - clean.shape[1]
    target_noise_norm = clean_norm * 10.0 ** (-snr_db / 20.0)
    noise = direction * (target_noise_norm / math.sqrt(degrees_of_freedom))
    noise_norm = float(np.linalg.norm(noise))
    realized_snr_db = 20.0 * math.log10(clean_norm / noise_norm)
    return clean + noise, realized_snr_db, noise_norm


class PaperDataset:
    """Load cases from one extracted paper package using only public identifiers."""

    def __init__(self, root: str | Path, *, geometry_root: str | Path | None = None) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        manifest = _read_json(manifest_path)
        self.package_id = str(manifest.get("package_id"))
        self.release_version = str(manifest.get("release_version", ""))
        self.manifest_sha256 = _sha256(manifest_path)
        self.cases = {str(row["case_id"]): row for row in _read_jsonl(self.root / "cases.jsonl")}
        self.noise = {str(row["case_id"]): row for row in _read_jsonl(self.root / "noise.jsonl")}
        self.operators = {
            str(row["operator_id"]): row for row in _read_jsonl(self.root / "operators.jsonl")
        }
        if set(self.noise) != set(self.cases):
            raise ValueError("noise.jsonl does not exactly cover cases.jsonl")

        self.hsp = ShardField(self.root / "hsp", "case_id")
        self.measurements = {
            spec: ShardField(self.root / "measurements" / spec.replace("_", "-"), "case_id")
            for spec in FORWARD_SPECS
            if (self.root / "measurements" / spec.replace("_", "-")).is_dir()
        }
        self.nominal_operators = ShardField(self.root / "operators" / "nominal", "operator_id")

        geometry_base = self.root if (self.root / "geometry").is_dir() else None
        if geometry_base is None and geometry_root is not None:
            geometry_base = Path(geometry_root).resolve()
        if geometry_base is None:
            raise ValueError("this package requires paper-evaluation-v1 as geometry_root")
        geometry_manifest = _read_json(geometry_base / "geometry" / "manifest.json")
        geometry_file = geometry_base / "geometry" / str(geometry_manifest["file"]["path"])
        if _sha256(geometry_file) != geometry_manifest["file"]["sha256"]:
            raise ValueError(f"{geometry_file}: digest differs from geometry manifest")
        with np.load(geometry_file, allow_pickle=False) as arrays:
            self.points_mm = np.asarray(arrays["points_mm"], dtype=np.float64)
            self.faces = np.asarray(arrays["faces"], dtype=np.int64)
            self.template_z = np.asarray(arrays["template_z"], dtype=np.float64)
            self.template_phi = np.asarray(arrays["template_phi"], dtype=np.float64)
            self.template_shape = tuple(int(value) for value in arrays["template_shape"])
            self.time_ms = np.asarray(arrays["time_ms"], dtype=np.float64)
            self.grid_weights = np.asarray(arrays["grid_weights"], dtype=np.float64)
            shape = tuple(int(value) for value in arrays["laplacian_shape"])
            self.laplacian = sp.csr_matrix(
                (
                    np.asarray(arrays["laplacian_data"], dtype=np.float64),
                    np.asarray(arrays["laplacian_indices"], dtype=np.int64),
                    np.asarray(arrays["laplacian_indptr"], dtype=np.int64),
                ),
                shape=shape,
            )
            self.lambda_min = float(np.asarray(arrays["lambda_min"]).item())
        time_steps = np.diff(self.time_ms)
        if time_steps.size == 0 or not np.allclose(time_steps, time_steps[0]):
            raise ValueError("geometry time_ms must be uniformly spaced")
        self.dt_ms = float(time_steps[0])

    def load_case(
        self,
        case_id: str,
        *,
        forward_spec: str,
        snr_db: float | None = 40.0,
    ) -> ExampleCase:
        """Join HSP, clean measurement, nominal operator, geometry and frozen noise metadata."""
        if forward_spec not in FORWARD_SPECS:
            raise ValueError(f"unsupported forward_spec {forward_spec!r}; expected {FORWARD_SPECS}")
        try:
            case = self.cases[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown case_id {case_id!r}") from exc
        if forward_spec not in case.get("measurement_specs", []):
            raise ValueError(f"{case_id}: no {forward_spec!r} measurement in this package")
        if forward_spec not in self.measurements:
            raise ValueError(f"package does not contain {forward_spec!r} measurement shards")

        geometry_id = str(case["geometry_id"])
        operator_id = f"{geometry_id}-nominal"
        if operator_id not in self.operators:
            raise ValueError(f"{case_id}: nominal operator {operator_id!r} is not released")
        truth = np.asarray(self.hsp.get(case_id), dtype=np.float64)
        clean = np.asarray(self.measurements[forward_spec].get(case_id), dtype=np.float64)
        forward = np.asarray(self.nominal_operators.get(operator_id), dtype=np.float64)
        noise = self.noise[case_id]
        if noise.get("kind") != "awgn_iid":
            raise ValueError(f"{case_id}: unsupported frozen noise kind {noise.get('kind')!r}")
        measurement, realized_snr_db, noise_norm = add_paper_noise(
            clean,
            snr_db=snr_db,
            noise_seed=int(noise["noise_seed"]),
        )
        return ExampleCase(
            measurement=measurement,
            measurement_clean=clean,
            forward=forward,
            truth=truth,
            points_mm=self.points_mm,
            faces=self.faces,
            grid_weights=self.grid_weights,
            time_ms=self.time_ms,
            laplacian=self.laplacian,
            template_z=self.template_z,
            template_phi=self.template_phi,
            template_shape=self.template_shape,
            lambda_min=self.lambda_min,
            dt_ms=self.dt_ms,
            target_snr_db=float("inf") if snr_db is None else float(snr_db),
            realized_snr_db=float("inf") if realized_snr_db is None else realized_snr_db,
            noise_norm=noise_norm,
            case_id=case_id,
            geometry_id=geometry_id,
            forward_spec=forward_spec,
            noise_kind="awgn_iid",
            noise_seed=int(noise["noise_seed"]),
            operator_discrepancy=DISCREPANCY_LABELS[forward_spec],
        )

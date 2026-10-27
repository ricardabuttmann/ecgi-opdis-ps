"""Resumable cohort execution and transparent metric aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from opdis_ps.config import load_method_config
from opdis_ps.dataset import DISCREPANCY_LABELS, PaperDataset
from opdis_ps.metrics import METRIC_PROTOCOL
from opdis_ps.pretrained_models import resolve_method_model
from opdis_ps.runner import run_loaded_case

METRICS = (
    "relative_error_global",
    "activation_time_error_ms",
    "localization_error_mm",
    "spatial_cc",
    "temporal_cc",
)
CASE_COLUMNS = (
    "case_id",
    "geometry_id",
    "operator_id",
    "population",
    "block_id",
    "partition",
    "method",
    "model_id",
    "metric_protocol",
    "forward_spec",
    "target_snr_db",
    "realized_snr_db",
    *METRICS,
    "runtime_seconds",
)
SUMMARY_COLUMNS = (
    "population",
    "block_id",
    "method",
    "model_id",
    "forward_spec",
    "target_snr_db",
    "case_count",
    *(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "std")),
)
COHORT_RUN_FILENAME = "cohort-run.json"


def select_case_ids(
    dataset: PaperDataset,
    *,
    cohort: str,
    forward_spec: str,
    case_ids: Iterable[str] = (),
    limit: int | None = None,
) -> list[str]:
    """Select public case IDs in package order using release-safe metadata."""
    requested = set(case_ids)
    unknown = requested - set(dataset.cases)
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    known_cohorts = {"all", "healthy", "pathology"} | {
        str(row["block_id"]) for row in dataset.cases.values() if row.get("block_id")
    }
    if cohort not in known_cohorts:
        raise ValueError(f"unknown cohort {cohort!r}; available: {sorted(known_cohorts)}")

    selected = []
    for case_id, row in dataset.cases.items():
        population = str(row.get("population", ""))
        block_id = row.get("block_id")
        cohort_matches = (
            cohort == "all" or cohort == population or (block_id is not None and cohort == block_id)
        )
        if requested and case_id not in requested:
            continue
        if cohort_matches and forward_spec in row.get("measurement_specs", []):
            selected.append(case_id)
    if not selected:
        raise ValueError(f"selection has no cases with forward_spec={forward_spec!r}")
    return selected[:limit]


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_tree_sha256(root: Path) -> str:
    """Hash public Python source paths and bytes in a stable order."""
    digest = hashlib.sha256()
    source_root = root / "src"
    paths = sorted(path for path in source_root.rglob("*.py") if path.is_file())
    if not paths:
        raise ValueError(f"no public Python sources found under {source_root}")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _runtime_contract_sha256(root: Path) -> str:
    """Hash the public cohort entry point and frozen dependency declarations."""
    digest = hashlib.sha256()
    relative_paths = ("scripts/run_cohort.py", "pyproject.toml", "requirements-tested.txt")
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing public runtime-contract file: {path}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def case_ids_sha256(case_ids: Iterable[str]) -> str:
    """Hash an ordered public case list without depending on JSON formatting."""
    digest = hashlib.sha256()
    for case_id in case_ids:
        digest.update(str(case_id).encode("utf-8"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def method_assets(
    method: str,
    *,
    forward_spec: str,
    config_root: str | Path,
    models_root: str | Path | None = None,
    replicate: int | str = 0,
) -> dict[str, str]:
    """Hash the config, checkpoint, and likelihood artifact that define one public method run."""
    config_root = Path(config_root).resolve()
    config_path = config_root / f"{method}.toml"
    settings = load_method_config(config_root, method).get("settings", {})
    repo_root = config_root.parents[1]
    paths = {"config": config_path}
    selected_model = resolve_method_model(method, models_root, replicate)
    if selected_model is not None:
        paths["checkpoint"] = selected_model.path
    artifact_name = settings.get("discrepancy", settings.get("conditional_artifact"))
    if artifact_name is not None:
        if forward_spec == "nominal":
            raise ValueError(f"{method} requires a discrepancy forward specification")
        paths["likelihood_artifact"] = (
            repo_root / "artifacts" / DISCREPANCY_LABELS[forward_spec] / str(artifact_name)
        )
    result = {role: _sha256(path) for role, path in paths.items()}
    if selected_model is not None:
        result["model_id"] = selected_model.model_id
        result["model_package_manifest"] = selected_model.package_manifest_sha256
    result["implementation"] = _source_tree_sha256(repo_root)
    result["runtime_contract"] = _runtime_contract_sha256(repo_root)
    return result


def _load_completed(
    path: Path,
    *,
    case_id: str,
    method: str,
    forward_spec: str,
    snr_db: float | None,
    package_id: str,
    package_manifest_sha256: str,
    operator_id: str,
    assets: dict[str, str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(record, dict):
        return None
    expected_snr = math.inf if snr_db is None else float(snr_db)
    try:
        stored_snr = float(record.get("target_snr_db", math.nan))
        for metric in METRICS:
            float(record[metric])
    except (KeyError, TypeError, ValueError):
        return None
    identity_matches = (
        record.get("cohort_schema_version") == 1
        and record.get("metric_protocol") == METRIC_PROTOCOL
        and record.get("case_id") == case_id
        and record.get("method") == method
        and record.get("forward_spec") == forward_spec
        and record.get("package_id") == package_id
        and record.get("package_manifest_sha256") == package_manifest_sha256
        and record.get("operator_id") == operator_id
        and record.get("method_assets") == assets
        and stored_snr == expected_snr
    )
    if not identity_matches:
        return None
    return record


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute one transparent mean and sample standard deviation per release metadata group."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("population", ""),
            row.get("block_id", ""),
            row["method"],
            row.get("model_id", ""),
            row["forward_spec"],
            row["target_snr_db"],
        )
        groups.setdefault(key, []).append(row)

    summaries = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        summary = dict(zip(SUMMARY_COLUMNS[:6], key, strict=True))
        summary["case_count"] = len(group)
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
            )
        summaries.append(summary)
    return summaries


def run_cohort(
    dataset: PaperDataset,
    *,
    case_ids: list[str],
    method: str,
    forward_spec: str,
    snr_db: float | None,
    output_root: str | Path,
    config_root: str | Path,
    models_root: str | Path | None = None,
    replicate: int | str = 0,
    force: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run or resume one method and keep per-case and aggregate CSV files current."""
    output_root = Path(output_root)
    assets = method_assets(
        method,
        forward_spec=forward_spec,
        config_root=config_root,
        models_root=models_root,
        replicate=replicate,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / COHORT_RUN_FILENAME
    case_hash = case_ids_sha256(case_ids)
    expected_run_identity = {
        "package_id": dataset.package_id,
        "package_release_version": dataset.release_version,
        "package_manifest_sha256": dataset.manifest_sha256,
        "method": method,
        "model_id": assets.get("model_id", ""),
        "metric_protocol": METRIC_PROTOCOL,
        "method_assets": assets,
        "forward_spec": forward_spec,
        "target_snr_db": math.inf if snr_db is None else float(snr_db),
        "selected_case_count": len(case_ids),
        "case_ids_sha256": case_hash,
    }
    if not force and run_path.exists():
        try:
            existing_run = json.loads(run_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(
                f"{run_path}: invalid existing cohort identity; choose another --output or "
                "pass --force to replace it"
            ) from error
        if not isinstance(existing_run, dict) or any(
            existing_run.get(key) != value for key, value in expected_run_identity.items()
        ):
            raise ValueError(
                f"{run_path}: existing cohort has a different identity; choose another --output "
                "or pass --force to replace it"
            )
    existing_records: dict[str, dict[str, Any] | None] = {}
    for case_id in case_ids:
        case_row = dataset.cases[case_id]
        metrics_path = output_root / "cases" / case_id / method / "metrics.json"
        record = (
            None
            if force
            else _load_completed(
                metrics_path,
                case_id=case_id,
                method=method,
                forward_spec=forward_spec,
                snr_db=snr_db,
                package_id=dataset.package_id,
                package_manifest_sha256=dataset.manifest_sha256,
                operator_id=f"{case_row['geometry_id']}-nominal",
                assets=assets,
            )
        )
        if not force and metrics_path.exists() and record is None:
            raise ValueError(
                f"{metrics_path}: existing result has a different or invalid identity; "
                "choose another --output or pass --force to replace it"
            )
        existing_records[case_id] = record

    run_record = {
        "schema_name": "opdis_ps.cohort_run",
        "schema_version": 1,
        "status": "running",
        **expected_run_identity,
        "completed_case_count": 0,
    }
    _write_json_atomic(run_path, run_record)
    rows: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        case_row = dataset.cases[case_id]
        metrics_path = output_root / "cases" / case_id / method / "metrics.json"
        record = existing_records[case_id]
        action = "skipping" if record is not None else ("recomputing" if force else "running")
        if on_progress is not None:
            on_progress(f"[{index}/{len(case_ids)}] {action} {case_id}")
        if record is None:
            case = dataset.load_case(case_id, forward_spec=forward_spec, snr_db=snr_db)
            record = run_loaded_case(
                method,
                case,
                output_root / "cases" / case_id,
                config_root,
                models_root=models_root,
                replicate=replicate,
            )
            record.update(
                {
                    "cohort_schema_version": 1,
                    "metric_protocol": METRIC_PROTOCOL,
                    "package_id": dataset.package_id,
                    "package_release_version": dataset.release_version,
                    "package_manifest_sha256": dataset.manifest_sha256,
                    "method_assets": assets,
                    "population": case_row.get("population", ""),
                    "block_id": case_row.get("block_id") or "",
                    "partition": case_row.get("partition", ""),
                }
            )
            _write_json_atomic(metrics_path, record)
        rows.append(record)
        _write_csv(output_root / "results.csv", rows, CASE_COLUMNS)
        _write_csv(output_root / "summary.csv", aggregate_results(rows), SUMMARY_COLUMNS)
        run_record["completed_case_count"] = index
        _write_json_atomic(run_path, run_record)
    run_record["status"] = "complete"
    _write_json_atomic(run_path, run_record)
    return rows, aggregate_results(rows)

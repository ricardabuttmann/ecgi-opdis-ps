"""Validate and aggregate independent training-replicate cohort results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from opdis_ps.cohort import (
    CASE_COLUMNS,
    COHORT_RUN_FILENAME,
    METRICS,
    aggregate_results,
    case_ids_sha256,
)
from opdis_ps.pretrained_models import METHOD_FAMILIES

FINAL_COLUMNS = (
    "population",
    "block_id",
    "method",
    "forward_spec",
    "target_snr_db",
    "case_count",
    "replicate_count",
    *(column for metric in METRICS for column in (f"{metric}_mean", f"{metric}_training_sd")),
)
REPLICATE_COLUMNS = (
    "replicate",
    "model_id",
    "population",
    "block_id",
    "method",
    "forward_spec",
    "target_snr_db",
    "case_count",
    *(column for metric in METRICS for column in (f"{metric}_mean", f"{metric}_case_sd")),
)
_CASE_IDENTITY_FIELDS = (
    "case_id",
    "geometry_id",
    "operator_id",
    "population",
    "block_id",
    "partition",
    "method",
    "forward_spec",
    "target_snr_db",
    "realized_snr_db",
)
_NUMERIC_CASE_FIELDS = (
    "target_snr_db",
    "realized_snr_db",
    *METRICS,
    "runtime_seconds",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing cohort table: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty cohort table: {path}")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing case provenance: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid case provenance: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid case provenance: {path}")
    return value


def _group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("population", "")),
        str(row.get("block_id", "")),
        str(row["method"]),
        str(row["forward_spec"]),
        float(row["target_snr_db"]),
        int(row["case_count"]),
    )


def _summary_with_verified_means(
    results: list[dict[str, str]], summary: list[dict[str, str]], source: Path
) -> list[dict[str, Any]]:
    recomputed = aggregate_results(results)
    expected = {
        (_group_key(row), str(row.get("model_id", ""))): row for row in recomputed
    }
    observed = {(_group_key(row), str(row.get("model_id", ""))): row for row in summary}
    if len(observed) != len(summary) or set(observed) != set(expected):
        raise ValueError(f"{source}: summary groups do not match results.csv")
    for key, expected_row in expected.items():
        observed_row = observed[key]
        for metric in METRICS:
            for suffix in ("mean", "std"):
                column = f"{metric}_{suffix}"
                try:
                    actual = float(observed_row[column])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{source}: invalid {column!r}") from error
                wanted = float(expected_row[column])
                matches = (
                    math.isnan(actual) and math.isnan(wanted)
                    if math.isnan(wanted)
                    else math.isfinite(actual)
                    and np.isclose(actual, wanted, rtol=1e-12, atol=1e-12)
                )
                if not matches:
                    raise ValueError(f"{source}: {column} does not match results.csv")
    return recomputed


def _verify_result_row(record: dict[str, Any], row: dict[str, str], source: Path) -> None:
    for column in CASE_COLUMNS:
        if column in _NUMERIC_CASE_FIELDS:
            try:
                left = float(record[column])
                right = float(row[column])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{source}: missing or invalid {column!r}") from error
            if not np.isclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(f"{source}: {column} differs from results.csv")
        elif str(record.get(column, "")) != str(row.get(column, "")):
            raise ValueError(f"{source}: {column} differs from results.csv")


def _checkpoint_provenance(
    run_root: Path,
    results: list[dict[str, str]],
    *,
    method: str,
    model_id: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    checkpoint_hashes = set()
    contracts = []
    complete_assets = []
    for row in results:
        path = run_root / "cases" / row["case_id"] / method / "metrics.json"
        record = _read_json(path)
        assets = record.get("method_assets")
        if not isinstance(assets, dict):
            raise ValueError(f"{path}: missing method_assets")
        _verify_result_row(record, row, path)
        checkpoint = str(assets.get("checkpoint", ""))
        if (
            record.get("case_id") != row["case_id"]
            or record.get("method") != method
            or record.get("model_id") != model_id
            or assets.get("model_id") != model_id
            or record.get("model_sha256") != checkpoint
            or not checkpoint.startswith("sha256:")
        ):
            raise ValueError(f"{path}: inconsistent checkpoint provenance")
        checkpoint_hashes.add(checkpoint)
        complete_assets.append(assets)
        contracts.append(
            {
                "cohort_schema_version": record.get("cohort_schema_version"),
                "package_id": record.get("package_id"),
                "package_release_version": record.get("package_release_version"),
                "package_manifest_sha256": record.get("package_manifest_sha256"),
                "model_package": record.get("model_package"),
                "model_package_manifest_sha256": record.get("model_package_manifest_sha256"),
                "config": assets.get("config"),
                "implementation": assets.get("implementation"),
                "runtime_contract": assets.get("runtime_contract"),
                "likelihood_artifact": assets.get("likelihood_artifact"),
                "model_package_manifest": assets.get("model_package_manifest"),
                "device": record.get("device"),
                "runtime_environment": record.get("runtime_environment"),
            }
        )
    if len(checkpoint_hashes) != 1:
        raise ValueError(f"{run_root}: cases do not use one checkpoint")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError(f"{run_root}: cases do not share one evaluation contract")
    if any(assets != complete_assets[0] for assets in complete_assets[1:]):
        raise ValueError(f"{run_root}: cases do not share one method-asset identity")
    return next(iter(checkpoint_hashes)), contracts[0], complete_assets[0]


def _verify_complete_run(
    run_root: Path,
    results: list[dict[str, str]],
    *,
    method: str,
    model_id: str,
    method_assets: dict[str, Any],
) -> dict[str, Any]:
    path = run_root / COHORT_RUN_FILENAME
    record = _read_json(path)
    case_ids = [row["case_id"] for row in results]
    expected = {
        "schema_name": "opdis_ps.cohort_run",
        "schema_version": 1,
        "status": "complete",
        "method": method,
        "model_id": model_id,
        "forward_spec": results[0]["forward_spec"],
        "selected_case_count": len(results),
        "completed_case_count": len(results),
        "case_ids_sha256": case_ids_sha256(case_ids),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"{path}: incomplete or inconsistent cohort run ({key})")
    if record.get("method_assets") != method_assets:
        raise ValueError(f"{path}: method assets differ from case provenance")
    try:
        marker_snr = float(record["target_snr_db"])
        result_snr = float(results[0]["target_snr_db"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid target SNR") from error
    if marker_snr != result_snr:
        raise ValueError(f"{path}: target SNR differs from results.csv")
    return record


def aggregate_training_replicates(
    replicate_root: str | Path,
    *,
    selection: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate cohort means across distinct checkpoints after strict coverage checks."""
    root = Path(replicate_root).resolve()
    if selection is not None and (
        Path(selection).name != selection or not selection.startswith("selection-")
    ):
        raise ValueError("selection must be one selection-<count>-<id> directory name")

    reference_cases: list[tuple[str, ...]] | None = None
    reference_groups: set[tuple[Any, ...]] | None = None
    per_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    replicate_rows: list[dict[str, Any]] = []
    sources = []
    method: str | None = None
    reference_contract: dict[str, Any] | None = None
    checkpoint_hashes = set()

    expected_replicates = 5
    for index in range(expected_replicates):
        replicate = f"s{index}"
        relative = Path(f"replicate-{replicate}")
        if selection is not None:
            relative /= selection
        run_root = root / relative
        results_path = run_root / "results.csv"
        summary_path = run_root / "summary.csv"
        results = _read_csv(results_path)
        summary = _read_csv(summary_path)

        model_ids = {str(row.get("model_id", "")) for row in results}
        methods = {str(row.get("method", "")) for row in results}
        if len(model_ids) != 1 or len(methods) != 1:
            raise ValueError(f"{results_path}: expected exactly one model and method")
        model_id = next(iter(model_ids))
        run_method = next(iter(methods))
        family = METHOD_FAMILIES.get(run_method)
        if family is None:
            raise ValueError(f"{results_path}: {run_method!r} is not a checkpoint-based method")
        if model_id != f"{family}-{replicate}":
            raise ValueError(f"{results_path}: expected model_id {family}-{replicate!s}")
        if method is None:
            method = run_method
        elif run_method != method:
            raise ValueError("replicate methods differ")

        checkpoint_sha256, contract, method_assets = _checkpoint_provenance(
            run_root,
            results,
            method=run_method,
            model_id=model_id,
        )
        _verify_complete_run(
            run_root,
            results,
            method=run_method,
            model_id=model_id,
            method_assets=method_assets,
        )
        if checkpoint_sha256 in checkpoint_hashes:
            raise ValueError(f"{results_path}: checkpoint is reused by another replicate")
        checkpoint_hashes.add(checkpoint_sha256)
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(f"{results_path}: evaluation contract differs between replicates")

        case_identity = [
            tuple(str(row.get(field, "")) for field in _CASE_IDENTITY_FIELDS) for row in results
        ]
        if len(set(case_identity)) != len(case_identity):
            raise ValueError(f"{results_path}: duplicate case identity")
        if reference_cases is None:
            reference_cases = case_identity
        elif case_identity != reference_cases:
            raise ValueError(f"{results_path}: case coverage or measurement identity differs")

        verified = _summary_with_verified_means(results, summary, summary_path)
        groups = {_group_key(row) for row in verified}
        if reference_groups is None:
            reference_groups = groups
        elif groups != reference_groups:
            raise ValueError(f"{summary_path}: cohort groups differ between replicates")

        for row in verified:
            key = _group_key(row)
            per_group.setdefault(key, []).append(row)
            output = {
                "replicate": replicate,
                "model_id": model_id,
                **dict(zip(FINAL_COLUMNS[:6], key, strict=True)),
            }
            for metric in METRICS:
                output[f"{metric}_mean"] = float(row[f"{metric}_mean"])
                output[f"{metric}_case_sd"] = float(row[f"{metric}_std"])
            replicate_rows.append(output)
        sources.append(
            {
                "replicate": replicate,
                "model_id": model_id,
                "checkpoint_sha256": checkpoint_sha256,
                "results_path": (relative / "results.csv").as_posix(),
                "results_sha256": _sha256(results_path),
                "summary_path": (relative / "summary.csv").as_posix(),
                "summary_sha256": _sha256(summary_path),
                "cohort_run_path": (relative / COHORT_RUN_FILENAME).as_posix(),
                "cohort_run_sha256": _sha256(run_root / COHORT_RUN_FILENAME),
            }
        )

    final_rows = []
    for key, rows in sorted(per_group.items(), key=lambda item: tuple(map(str, item[0]))):
        if len(rows) != expected_replicates:
            raise ValueError(f"cohort group {key!r} does not cover every replicate")
        output = {
            **dict(zip(FINAL_COLUMNS[:6], key, strict=True)),
            "replicate_count": expected_replicates,
        }
        for metric in METRICS:
            values = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=np.float64)
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_training_sd"] = float(np.std(values, ddof=1))
        final_rows.append(output)

    assert reference_cases is not None
    assert reference_contract is not None
    case_ids = [case[0] for case in reference_cases]
    group_records = [dict(zip(FINAL_COLUMNS[:6], key, strict=True)) for key in sorted(per_group)]
    manifest = {
        "schema_name": "opdis_ps.training_replicate_aggregation",
        "schema_version": 1,
        "method": method,
        "replicate_count": expected_replicates,
        "selection": selection,
        "case_count": len(case_ids),
        "case_ids_sha256": case_ids_sha256(case_ids),
        "evaluation_contract": reference_contract,
        "cohort_groups": group_records,
        "aggregation": {
            "unit": "complete-cohort mean per independent training checkpoint",
            "center": "arithmetic mean across five checkpoint-specific cohort means",
            "uncertainty": "sample standard deviation across five checkpoint-specific cohort means",
            "ddof": 1,
        },
        "sources": sources,
    }
    return final_rows, replicate_rows, manifest


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_training_replicate_outputs(
    output_root: str | Path,
    final_rows: list[dict[str, Any]],
    replicate_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Install all aggregation outputs together and never replace different results."""
    output = Path(output_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        _write_csv(temporary / "summary.csv", final_rows, FINAL_COLUMNS)
        _write_csv(temporary / "replicates.csv", replicate_rows, REPLICATE_COLUMNS)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            names = ("summary.csv", "replicates.csv", "manifest.json")
            if all((output / name).is_file() for name in names) and all(
                (output / name).read_bytes() == (temporary / name).read_bytes() for name in names
            ):
                shutil.rmtree(temporary)
                return
            raise FileExistsError(f"{output}: refusing to replace different aggregation outputs")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def aggregation_id(manifest: dict[str, Any]) -> str:
    """Return a stable output ID for one exact set of replicate sources."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "aggregation-" + hashlib.sha256(payload).hexdigest()[:12]

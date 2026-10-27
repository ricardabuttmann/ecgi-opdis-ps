from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from opdis_ps.cohort import (
    CASE_COLUMNS,
    COHORT_RUN_FILENAME,
    METRICS,
    SUMMARY_COLUMNS,
    aggregate_results,
    case_ids_sha256,
)
from opdis_ps.replicate_aggregation import (
    aggregate_training_replicates,
    write_training_replicate_outputs,
)


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_replicate(
    root: Path,
    index: int,
    *,
    case_suffix: str = "",
    corrupt=False,
    checkpoint_index: int | None = None,
) -> None:
    value = float(index + 1)
    rows = []
    for case_index in range(2):
        rows.append(
            {
                "case_id": f"case-{case_index}{case_suffix}",
                "geometry_id": f"geometry-{case_index}",
                "operator_id": f"geometry-{case_index}-nominal",
                "population": "healthy",
                "block_id": "",
                "partition": "test",
                "method": "opdis-ps",
                "model_id": f"prior-s{index}",
                "forward_spec": "conductivity",
                "target_snr_db": 40.0,
                "realized_snr_db": 40.01,
                **dict.fromkeys(METRICS, value),
                "runtime_seconds": 1.0,
            }
        )
    run_root = root / f"replicate-s{index}"
    _write_csv(run_root / "results.csv", rows, CASE_COLUMNS)
    summaries = aggregate_results(rows)
    if corrupt:
        summaries[0]["relative_error_global_mean"] = 99.0
    _write_csv(run_root / "summary.csv", summaries, SUMMARY_COLUMNS)
    checkpoint = f"sha256:checkpoint-{index if checkpoint_index is None else checkpoint_index}"
    method_assets = {
        "checkpoint": checkpoint,
        "config": "sha256:config",
        "implementation": "sha256:implementation",
        "runtime_contract": "sha256:runtime-contract",
        "likelihood_artifact": "sha256:artifact",
        "model_id": f"prior-s{index}",
        "model_package_manifest": "sha256:model-package",
    }
    for row in rows:
        path = run_root / "cases" / row["case_id"] / "opdis-ps" / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    **row,
                    "cohort_schema_version": 1,
                    "device": "cpu",
                    "runtime_environment": {
                        "python": "3.11.0",
                        "numpy": "1.26.0",
                        "scipy": "1.11.0",
                        "torch": "2.1.0",
                    },
                    "model_sha256": checkpoint,
                    "model_package": "paper-pretrained-models-v1",
                    "model_package_manifest_sha256": "sha256:model-package",
                    "package_id": "paper-evaluation-v1",
                    "package_release_version": "1.0.0",
                    "package_manifest_sha256": "sha256:dataset",
                    "method_assets": method_assets,
                }
            ),
            encoding="utf-8",
        )
    (run_root / COHORT_RUN_FILENAME).write_text(
        json.dumps(
            {
                "schema_name": "opdis_ps.cohort_run",
                "schema_version": 1,
                "status": "complete",
                "package_id": "paper-evaluation-v1",
                "package_release_version": "1.0.0",
                "package_manifest_sha256": "sha256:dataset",
                "method": "opdis-ps",
                "model_id": f"prior-s{index}",
                "method_assets": method_assets,
                "forward_spec": "conductivity",
                "target_snr_db": 40.0,
                "selected_case_count": len(rows),
                "completed_case_count": len(rows),
                "case_ids_sha256": case_ids_sha256(row["case_id"] for row in rows),
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_training_replicates_reports_run_level_sample_sd(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index)

    final_rows, replicate_rows, manifest = aggregate_training_replicates(tmp_path)

    assert len(final_rows) == 1
    summary = final_rows[0]
    assert summary["case_count"] == 2
    assert summary["replicate_count"] == 5
    assert summary["relative_error_global_mean"] == 3.0
    assert np.isclose(summary["relative_error_global_training_sd"], np.sqrt(2.5))
    assert [row["model_id"] for row in replicate_rows] == [f"prior-s{i}" for i in range(5)]
    assert manifest["method"] == "opdis-ps"
    assert len(manifest["sources"]) == 5
    assert manifest["case_count"] == 2
    assert manifest["evaluation_contract"]["package_manifest_sha256"] == "sha256:dataset"
    assert manifest["aggregation"]["ddof"] == 1


def test_aggregate_training_replicates_rejects_different_case_coverage(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index, case_suffix="-different" if index == 4 else "")

    with pytest.raises(ValueError, match="case coverage"):
        aggregate_training_replicates(tmp_path)


def test_aggregate_training_replicates_verifies_each_summary(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index, corrupt=index == 3)

    with pytest.raises(ValueError, match="does not match results.csv"):
        aggregate_training_replicates(tmp_path)


def test_aggregate_training_replicates_requires_every_checkpoint(tmp_path: Path):
    for index in range(4):
        _write_replicate(tmp_path, index)

    with pytest.raises(FileNotFoundError, match="replicate-s4"):
        aggregate_training_replicates(tmp_path)


def test_aggregate_training_replicates_rejects_reused_checkpoint(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index, checkpoint_index=0 if index == 4 else index)

    with pytest.raises(ValueError, match="checkpoint is reused"):
        aggregate_training_replicates(tmp_path)


def test_aggregate_training_replicates_rejects_incomplete_run(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index)
    path = tmp_path / "replicate-s3" / COHORT_RUN_FILENAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "running"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete or inconsistent"):
        aggregate_training_replicates(tmp_path)


def test_aggregate_training_replicates_verifies_results_against_case_records(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path, index)
    path = tmp_path / "replicate-s2" / "results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["relative_error_global"] = "0.01"
    _write_csv(path, rows, CASE_COLUMNS)
    _write_csv(
        tmp_path / "replicate-s2" / "summary.csv",
        aggregate_results(rows),
        SUMMARY_COLUMNS,
    )

    with pytest.raises(ValueError, match="differs from results.csv"):
        aggregate_training_replicates(tmp_path)


def test_aggregation_outputs_are_idempotent_but_not_overwritten(tmp_path: Path):
    for index in range(5):
        _write_replicate(tmp_path / "inputs", index)
    final_rows, replicate_rows, manifest = aggregate_training_replicates(tmp_path / "inputs")
    output = tmp_path / "aggregation"

    write_training_replicate_outputs(output, final_rows, replicate_rows, manifest)
    write_training_replicate_outputs(output, final_rows, replicate_rows, manifest)

    changed = [dict(final_rows[0])]
    changed[0]["relative_error_global_mean"] = 99.0
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_training_replicate_outputs(output, changed, replicate_rows, manifest)

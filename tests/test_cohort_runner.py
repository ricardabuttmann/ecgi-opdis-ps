from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from opdis_ps.cohort import METRICS, aggregate_results, method_assets, run_cohort, select_case_ids


class _Dataset:
    package_id = "paper-evaluation-v1"
    release_version = "1.0.0"
    manifest_sha256 = "sha256:package"
    cases = {
        "healthy-1": {
            "population": "healthy",
            "geometry_id": "geometry-healthy-1",
            "block_id": None,
            "partition": "test",
            "measurement_specs": ["nominal", "conductivity"],
        },
        "lbbb-1": {
            "population": "pathology",
            "geometry_id": "geometry-lbbb-1",
            "block_id": "test_lbbb_mild",
            "partition": "test",
            "measurement_specs": ["conductivity"],
        },
        "rbbb-1": {
            "population": "pathology",
            "geometry_id": "geometry-rbbb-1",
            "block_id": "test_rbbb_mild",
            "partition": "test",
            "measurement_specs": ["conductivity"],
        },
    }

    def load_case(self, case_id: str, *, forward_spec: str, snr_db: float | None):
        return type(
            "Case",
            (),
            {
                "case_id": case_id,
                "geometry_id": f"geometry-{case_id}",
                "forward_spec": forward_spec,
                "operator_discrepancy": "lung-conductivity",
                "noise_kind": "awgn_iid",
                "noise_seed": 7,
                "target_snr_db": float("inf") if snr_db is None else snr_db,
            },
        )()


def _record(method: str, case, output_root: Path, config_root: Path, **kwargs) -> dict:
    del config_root
    del kwargs
    value = 1.0 if case.case_id == "healthy-1" else 3.0
    record = {
        "method": method,
        "case_id": case.case_id,
        "geometry_id": case.geometry_id,
        "operator_id": f"{case.geometry_id}-nominal",
        "forward_spec": case.forward_spec,
        "operator_discrepancy": case.operator_discrepancy,
        "noise_kind": case.noise_kind,
        "noise_seed": case.noise_seed,
        "target_snr_db": case.target_snr_db,
        "realized_snr_db": case.target_snr_db,
        "runtime_seconds": 0.5,
        **dict.fromkeys(METRICS, value),
    }
    method_output = output_root / method
    method_output.mkdir(parents=True, exist_ok=True)
    (method_output / "metrics.json").write_text(json.dumps(record), encoding="utf-8")
    np.savez(method_output / "reconstruction.npz", reconstruction=np.zeros((1, 1)))
    return record


def test_selection_uses_public_population_block_and_measurement_metadata():
    dataset = _Dataset()
    assert select_case_ids(dataset, cohort="healthy", forward_spec="nominal") == ["healthy-1"]
    assert select_case_ids(dataset, cohort="pathology", forward_spec="conductivity") == [
        "lbbb-1",
        "rbbb-1",
    ]
    assert select_case_ids(dataset, cohort="test_lbbb_mild", forward_spec="conductivity") == [
        "lbbb-1"
    ]
    assert select_case_ids(
        dataset,
        cohort="all",
        forward_spec="conductivity",
        case_ids=["rbbb-1"],
        limit=1,
    ) == ["rbbb-1"]


def test_cohort_run_writes_tables_and_resumes(tmp_path: Path, monkeypatch):
    calls = []
    assets = {"config": "sha256:config", "implementation": "sha256:code"}

    def counted(*args, **kwargs):
        calls.append(args[1].case_id)
        return _record(*args, **kwargs)

    monkeypatch.setattr("opdis_ps.cohort.run_loaded_case", counted)
    monkeypatch.setattr("opdis_ps.cohort.method_assets", lambda *args, **kwargs: dict(assets))
    dataset = _Dataset()
    options = dict(
        dataset=dataset,
        case_ids=["healthy-1", "lbbb-1"],
        method="tikhonov",
        forward_spec="conductivity",
        snr_db=40.0,
        output_root=tmp_path,
        config_root=tmp_path,
    )
    rows, summaries = run_cohort(**options)
    assert calls == ["healthy-1", "lbbb-1"]
    assert len(rows) == len(summaries) == 2
    assert {row["metric_protocol"] for row in rows} == {"paper-2026-tied-front-mean-v1"}

    with (tmp_path / "results.csv").open(newline="") as handle:
        result_rows = list(csv.DictReader(handle))
    with (tmp_path / "summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert [row["case_id"] for row in result_rows] == ["healthy-1", "lbbb-1"]
    assert {row["population"] for row in summary_rows} == {"healthy", "pathology"}

    run_cohort(**options)
    assert calls == ["healthy-1", "lbbb-1"]

    assets["implementation"] = "sha256:changed-code"
    with pytest.raises(ValueError, match="different.*identity"):
        run_cohort(**options)
    run_cohort(**options, force=True)
    assert calls == ["healthy-1", "lbbb-1", "healthy-1", "lbbb-1"]


def test_corrupt_completed_record_requires_explicit_force(tmp_path: Path, monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1].case_id)
        return _record(*args, **kwargs)

    monkeypatch.setattr("opdis_ps.cohort.run_loaded_case", counted)
    monkeypatch.setattr("opdis_ps.cohort.method_assets", lambda *args, **kwargs: {})
    options = dict(
        dataset=_Dataset(),
        case_ids=["healthy-1"],
        method="tikhonov",
        forward_spec="conductivity",
        snr_db=40.0,
        output_root=tmp_path,
        config_root=tmp_path,
    )
    run_cohort(**options)
    path = tmp_path / "cases" / "healthy-1" / "tikhonov" / "metrics.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["target_snr_db"] = "corrupt"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="different or invalid identity"):
        run_cohort(**options)
    run_cohort(**options, force=True)
    assert calls == ["healthy-1", "healthy-1"]

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different or invalid identity"):
        run_cohort(**options)
    run_cohort(**options, force=True)
    assert calls == ["healthy-1", "healthy-1", "healthy-1"]


def test_method_assets_bind_public_implementation(tmp_path: Path):
    config_root = tmp_path / "configs" / "paper"
    config_root.mkdir(parents=True)
    (config_root / "tikhonov.toml").write_text(
        'method = "tikhonov"\n[settings]\norder = 1\n', encoding="utf-8"
    )
    implementation = tmp_path / "src" / "package" / "model.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("VALUE = 1\n", encoding="utf-8")
    runtime_files = {
        "scripts/run_cohort.py": "print('cohort')\n",
        "pyproject.toml": "[project]\nname = 'test'\n",
        "requirements-tested.txt": "numpy==1\n",
    }
    for relative, content in runtime_files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    first = method_assets("tikhonov", forward_spec="conductivity", config_root=config_root)

    implementation.write_text("VALUE = 2\n", encoding="utf-8")
    second = method_assets("tikhonov", forward_spec="conductivity", config_root=config_root)

    assert first["config"] == second["config"]
    assert first["implementation"] != second["implementation"]


def test_aggregate_reports_sample_standard_deviation():
    rows = []
    for value in (1.0, 3.0):
        rows.append(
            {
                "population": "healthy",
                "block_id": "",
                "method": "tikhonov",
                "forward_spec": "conductivity",
                "target_snr_db": 40.0,
                **dict.fromkeys(METRICS, value),
            }
        )
    summary = aggregate_results(rows)[0]
    assert summary["case_count"] == 2
    assert summary["model_id"] == ""
    assert summary["relative_error_global_mean"] == 2.0
    assert np.isclose(summary["relative_error_global_std"], np.sqrt(2.0))


def test_single_case_aggregate_reports_undefined_case_standard_deviation():
    row = {
        "population": "healthy",
        "block_id": "",
        "method": "tikhonov",
        "forward_spec": "conductivity",
        "target_snr_db": 40.0,
        **dict.fromkeys(METRICS, 1.0),
    }
    summary = aggregate_results([row])[0]
    assert np.isnan(summary["relative_error_global_std"])


def test_aggregate_keeps_model_replicates_separate():
    rows = []
    for model_id, value in (("prior-s0", 1.0), ("prior-s1", 3.0)):
        rows.append(
            {
                "population": "healthy",
                "block_id": "",
                "method": "opdis-ps",
                "model_id": model_id,
                "forward_spec": "conductivity",
                "target_snr_db": 40.0,
                **dict.fromkeys(METRICS, value),
            }
        )
    summaries = aggregate_results(rows)
    assert [row["model_id"] for row in summaries] == ["prior-s0", "prior-s1"]
    assert [row["case_count"] for row in summaries] == [1, 1]

"""Portable prepared-case validation and openCARP signal extraction tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opdis_ps.igb import extract_hsp
from opdis_ps.simulation import load_simulation_config, run_prepared_simulation
from opdis_ps.simulation_case import load_prepared_simulation_case


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_vtx(path: Path, indices: list[int], domain: str = "extra") -> None:
    path.write_text(
        f"{len(indices)}\n{domain}\n" + "\n".join(str(value) for value in indices) + "\n"
    )


def _prepared_case(root: Path) -> Path:
    root.mkdir()
    files = {
        "mesh_points": "combined_volume.pts",
        "mesh_elements": "combined_volume.elem",
        "mesh_fibers": "combined_volume.lon",
        "parameter_file": "combined_parameter.par",
        "ground_vertices": "ground.vtx",
        "epicardial_vertices": "epi.vtx",
        "stimulus_lv": "stim_lv.vtx",
    }
    (root / files["mesh_points"]).write_text("4\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n")
    (root / files["mesh_elements"]).write_text("1\nTt 0 1 2 3 101\n")
    (root / files["mesh_fibers"]).write_text("1\n1 0 0\n")
    (root / files["parameter_file"]).write_text("num_gregions = 1\n")
    _write_vtx(root / files["ground_vertices"], [0])
    _write_vtx(root / files["epicardial_vertices"], [1, 2])
    _write_vtx(root / files["stimulus_lv"], [1], domain="intra")
    records = {
        role: {
            "path": name,
            "size_bytes": (root / name).stat().st_size,
            "sha256": _digest(root / name),
        }
        for role, name in files.items()
    }
    manifest = {
        "schema_version": 1,
        "case_id": "simulation-example-001",
        "seed": 1,
        "files": records,
        "stimuli": [{"name": "lv", "file_role": "stimulus_lv", "start_ms": 10.0}],
        "expected": {
            "volume_nodes": 4,
            "volume_elements": 1,
            "epicardial_nodes": 2,
            "time_frames": 3,
        },
    }
    (root / "case_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root


def _config(path: Path) -> Path:
    path.write_text(
        """[simulation]
tend_ms = 450.0
spacedt_ms = 5.0
timedt_ms = 5.0
stim_strength = 200.0
stim_duration_ms = 2.0
bidomain_mode = 2
"""
    )
    return path


def test_prepared_case_validates_all_declared_files(tmp_path):
    root = _prepared_case(tmp_path / "case")

    case = load_prepared_simulation_case(root)

    assert case.case_id == "simulation-example-001"
    assert case.seed == 1
    assert case.expected["volume_nodes"] == 4
    assert case.path("mesh_elements").name == "combined_volume.elem"


def test_prepared_case_rejects_changed_bytes(tmp_path):
    root = _prepared_case(tmp_path / "case")
    (root / "ground.vtx").write_text("1\nextra\n3\n")

    with pytest.raises(ValueError, match="SHA-256"):
        load_prepared_simulation_case(root)


def test_validate_only_requires_the_frozen_seed_without_opencarp(tmp_path):
    root = _prepared_case(tmp_path / "case")
    config = _config(tmp_path / "simulation.toml")

    summary = run_prepared_simulation(
        case_dir=root,
        config_path=config,
        output_dir=tmp_path / "unused",
        ranks=8,
        seed=1,
        validate_only=True,
    )

    assert summary["verified_files"] == 7
    assert summary["stimuli"] == 1
    assert summary["carputils_api"] == "v11"
    assert "carputils_api" not in load_simulation_config(config)
    assert not (tmp_path / "unused").exists()


def test_simulation_cli_reports_wrong_seed_without_traceback(tmp_path):
    root = _prepared_case(tmp_path / "case")
    config = _config(tmp_path / "simulation.toml")
    script = Path(__file__).parents[1] / "scripts" / "run_simulation_example.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case-dir",
            str(root),
            "--config",
            str(config),
            "--seed",
            "2",
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "error: seed 2 does not match the prepared physiology seed 1" in result.stderr
    assert "Traceback" not in result.stderr


def test_simulation_cli_explains_how_to_obtain_a_missing_bundle(tmp_path):
    config = _config(tmp_path / "simulation.toml")
    script = Path(__file__).parents[1] / "scripts" / "run_simulation_example.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--case-dir",
            str(tmp_path / "artifacts" / "simulation-example-v1"),
            "--config",
            str(config),
            "--seed",
            "1",
            "--validate-only",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "prepared simulation bundle not found" in result.stderr
    assert "extract it under artifacts/ as described in the README" in result.stderr
    assert "Traceback" not in result.stderr


def test_igb_extraction_produces_average_referenced_hsp(tmp_path):
    root = _prepared_case(tmp_path / "case")
    values = np.arange(12, dtype="<f4").reshape(3, 4)
    header = b"x:4 y:1 z:1 t:3 type:float dt:5.0\x0c"
    (tmp_path / "phie.igb").write_bytes(header.ljust(1024, b"\x00") + values.tobytes())

    hsp, metadata = extract_hsp(
        tmp_path / "phie.igb",
        root / "epi.vtx",
    )

    np.testing.assert_allclose(hsp.mean(axis=0), 0.0, atol=1e-7)
    assert hsp.shape == (2, 3)
    assert metadata["dt"] == "5.0"

"""Public prepared-case simulation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from opdis_ps.igb import extract_hsp
from opdis_ps.simulation_case import load_prepared_simulation_case


def load_simulation_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"simulation"} or not isinstance(config["simulation"], dict):
        raise ValueError(f"{path}: expected exactly one [simulation] table")
    settings = config["simulation"]
    required = {
        "tend_ms",
        "spacedt_ms",
        "timedt_ms",
        "stim_strength",
        "stim_duration_ms",
        "bidomain_mode",
    }
    if set(settings) != required:
        raise ValueError(f"{path}: [simulation] must contain exactly {sorted(required)}")
    for name in required - {"bidomain_mode"}:
        value = settings[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"simulation.{name} must be finite and positive")
    if settings["bidomain_mode"] != 2:
        raise ValueError("the released protocol requires pseudo-bidomain mode 2")
    return settings


def _sha256_array_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repository root")


def run_prepared_simulation(
    *,
    case_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    ranks: int,
    seed: int,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Validate, simulate, and extract one epicardial HSP example."""
    if ranks < 1:
        raise ValueError("ranks must be positive")
    case = load_prepared_simulation_case(case_dir)
    settings = load_simulation_config(config_path)
    if seed != case.seed:
        raise ValueError(
            f"seed {seed} does not match the prepared physiology seed {case.seed} in the manifest"
        )
    summary = {
        "case_id": case.case_id,
        "seed": case.seed,
        "carputils_api": "v11",
        "verified_files": len(case.files),
        "stimuli": len(case.stimuli),
        "expected": {
            key: case.expected[key]
            for key in ("volume_nodes", "volume_elements", "epicardial_nodes", "time_frames")
        },
    }
    if validate_only:
        return summary

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing simulation output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output / "raw"
    runner = repository_root() / "simulations" / "run_prepared_hsp_v11.py"
    command = [
        sys.executable,
        str(runner),
        "--case-dir",
        str(case.root),
        "--config",
        str(Path(config_path).resolve()),
        "--output",
        str(raw_output),
        "--seed",
        str(seed),
        "--np",
        str(ranks),
    ]
    subprocess.run(command, check=True)

    hsp, header = extract_hsp(
        raw_output / "phie.igb",
        case.path("epicardial_vertices"),
    )
    if hsp.shape != (case.expected["epicardial_nodes"], case.expected["time_frames"]):
        raise ValueError(f"unexpected HSP shape {hsp.shape}")
    output.mkdir(parents=True, exist_ok=True)
    hsp_path = output / "hsp.npy"
    np.save(hsp_path, hsp)
    summary.update(
        {
            "hsp_shape": list(hsp.shape),
            "hsp_reference": "average over epicardial nodes at every time frame",
            "igb_dt": float(header.get("dt", 0.0)),
            "hsp_sha256": _sha256_array_file(hsp_path),
        }
    )
    (output / "simulation_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary

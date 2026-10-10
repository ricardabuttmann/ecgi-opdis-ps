"""One public adapter for every example reconstruction method."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from opdis_ps.config import load_method_config
from opdis_ps.data import ExampleCase, load_example_case
from opdis_ps.metrics import evaluate_reconstruction
from opdis_ps.public_models import (
    run_ddrm,
    run_dps,
    run_palette,
    run_spatiotemporal_tikhonov,
    run_tikhonov,
    run_total_variation,
    run_transformer,
)

IMPLEMENTED_METHODS = (
    "tikhonov",
    "st-tikhonov",
    "tv",
    "transformer",
    "palette",
    "dps",
    "ddrm",
    "opdis-ps",
    "opdis-ps-marginal",
    "opdis-ps-marginal-isotropic",
)


def run_loaded_case(
    method: str,
    case: ExampleCase,
    output_root: str | Path,
    config_root: str | Path,
) -> dict[str, Any]:
    """Run one method on either a legacy example or an indexed dataset case."""
    if method not in IMPLEMENTED_METHODS:
        raise ValueError(f"unknown method {method!r}; available: {IMPLEMENTED_METHODS}")
    config = load_method_config(config_root, method)
    settings = config.get("settings", {})
    started = time.perf_counter()

    config_root = Path(config_root).resolve()
    if config_root.name != "paper" or config_root.parent.name != "configs":
        raise ValueError("config_root must point to the repository's configs/paper directory")
    repo_root = config_root.parents[1]
    if method == "tikhonov":
        result = run_tikhonov(case, settings)
    elif method == "tv":
        result = run_total_variation(case, settings)
    elif method == "st-tikhonov":
        result = run_spatiotemporal_tikhonov(case, settings)
    elif method == "transformer":
        result = run_transformer(case, settings, repo_root)
    elif method == "palette":
        result = run_palette(case, settings, repo_root)
    elif method == "dps":
        result = run_dps(case, settings, repo_root)
    else:
        result = run_ddrm(case, settings, repo_root)

    runtime_seconds = time.perf_counter() - started
    metrics = evaluate_reconstruction(
        result.reconstruction,
        case.truth,
        case.points_mm,
        case.dt_ms,
    )
    record: dict[str, Any] = {
        "method": method,
        "case_id": case.case_id,
        "geometry_id": case.geometry_id,
        "operator_id": f"{case.geometry_id}-nominal" if case.geometry_id else "",
        "forward_spec": case.forward_spec,
        "operator_discrepancy": case.operator_discrepancy,
        "noise_kind": case.noise_kind,
        "noise_seed": case.noise_seed,
        "target_snr_db": case.target_snr_db,
        "realized_snr_db": case.realized_snr_db,
        **metrics,
        "runtime_seconds": runtime_seconds,
        **result.diagnostics,
    }
    method_output = Path(output_root) / method
    method_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        method_output / "reconstruction.npz",
        reconstruction=result.reconstruction.astype(np.float32),
        truth=case.truth.astype(np.float32),
        measurement=case.measurement.astype(np.float32),
    )
    (method_output / "metrics.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def run_method(
    method: str,
    case_path: str | Path,
    output_root: str | Path,
    config_root: str | Path,
) -> dict[str, Any]:
    """Compatibility wrapper for the bundled self-contained NPZ examples."""
    return run_loaded_case(method, load_example_case(case_path), output_root, config_root)

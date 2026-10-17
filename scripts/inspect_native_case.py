#!/usr/bin/env python3
"""Inspect one native ECGI case and optionally synthesize its compatible nominal BSP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from opdis_ps.native_dataset import NativeDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--include-torso-geometry", action="store_true")
    parser.add_argument(
        "--bsp-output",
        type=Path,
        help="load the native operator and save A_native @ hsp as a NumPy file",
    )
    args = parser.parse_args()

    dataset = NativeDataset(args.dataset)
    case = dataset.load_case(
        args.case_id,
        include_operator=args.bsp_output is not None,
        include_torso_geometry=args.include_torso_geometry,
    )
    report = {
        "case_id": case.case_id,
        "geometry_id": case.geometry_id,
        "heart_id": case.heart_id,
        "torso_id": case.torso_id,
        "partition": case.partition,
        "population": case.population,
        "hsp_shape": list(case.hsp.shape),
        "heart_point_count": len(case.heart.points_mm),
        "heart_face_count": len(case.heart.faces),
        "operator_id": case.operator_id,
    }
    if case.torso is not None:
        report["torso_point_count"] = len(case.torso.points_mm)
        report["torso_face_count"] = len(case.torso.faces)
    if args.bsp_output is not None:
        bsp = case.synthesize_bsp()
        args.bsp_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.bsp_output, bsp, allow_pickle=False)
        report["bsp_shape"] = list(bsp.shape)
        report["bsp_output"] = str(args.bsp_output)
        report["max_abs_electrode_mean_mv"] = float(
            np.max(np.abs(np.asarray(bsp, dtype=np.float64).mean(axis=0)))
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

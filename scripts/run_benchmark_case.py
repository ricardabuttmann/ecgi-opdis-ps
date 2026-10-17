#!/usr/bin/env python3
"""Run one released method on one indexed paper benchmark case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opdis_ps.dataset import FORWARD_SPECS, PaperDataset
from opdis_ps.pretrained_models import METHOD_FAMILIES
from opdis_ps.runner import IMPLEMENTED_METHODS, run_loaded_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--forward-spec", choices=FORWARD_SPECS, required=True)
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--method", choices=IMPLEMENTED_METHODS, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--models", type=Path)
    parser.add_argument("--replicate", type=int, choices=range(5), default=0)
    args = parser.parse_args()

    if args.method in METHOD_FAMILIES and args.models is None:
        parser.error(f"{args.method} requires --models pointing to paper-pretrained-models-v1")

    dataset = PaperDataset(args.dataset, geometry_root=args.geometry_root)
    case = dataset.load_case(
        args.case_id,
        forward_spec=args.forward_spec,
        snr_db=None if args.clean else args.snr_db,
    )
    if args.method.startswith("opdis-ps") and args.forward_spec == "nominal":
        parser.error("OpDis-PS requires a released discrepancy measurement")
    config_root = Path(__file__).resolve().parents[1] / "configs" / "paper"
    record = run_loaded_case(
        args.method,
        case,
        args.output / args.case_id,
        config_root,
        models_root=args.models,
        replicate=args.replicate,
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

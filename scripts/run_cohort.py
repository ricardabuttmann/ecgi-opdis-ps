#!/usr/bin/env python3
"""Run or resume one method over a public paper-dataset cohort."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from opdis_ps.cohort import run_cohort, select_case_ids
from opdis_ps.dataset import FORWARD_SPECS, PaperDataset
from opdis_ps.pretrained_models import METHOD_FAMILIES
from opdis_ps.runner import IMPLEMENTED_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument(
        "--cohort",
        default="all",
        help="all, healthy, pathology, or an exact public block_id from cases.jsonl",
    )
    parser.add_argument("--case-id", action="append", default=[], help="repeat to select case IDs")
    parser.add_argument("--forward-spec", choices=FORWARD_SPECS, required=True)
    parser.add_argument("--snr-db", type=float, default=40.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--method", choices=IMPLEMENTED_METHODS, required=True)
    parser.add_argument("--limit", type=int, help="run only the first N selected cases")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="recompute completed cases")
    parser.add_argument("--models", type=Path)
    parser.add_argument("--replicate", type=int, choices=range(5), default=0)
    args = parser.parse_args()

    if args.method in METHOD_FAMILIES and args.models is None:
        parser.error(f"{args.method} requires --models pointing to paper-pretrained-models-v1")

    if args.method.startswith("opdis-ps") and args.forward_spec == "nominal":
        parser.error("OpDis-PS requires a released discrepancy measurement")
    dataset = PaperDataset(args.dataset, geometry_root=args.geometry_root)
    case_ids = select_case_ids(
        dataset,
        cohort=args.cohort,
        forward_spec=args.forward_spec,
        case_ids=args.case_id,
        limit=args.limit,
    )
    snr_db = None if args.clean else args.snr_db
    snr_label = "clean" if snr_db is None else f"snr-{snr_db:g}db"
    default_output = (
        Path("outputs/cohort") / args.cohort / args.forward_spec / snr_label / args.method
    )
    if args.method in METHOD_FAMILIES:
        default_output /= f"replicate-s{args.replicate}"
    if args.case_id or args.limit is not None:
        selection_digest = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()[:12]
        default_output /= f"selection-{len(case_ids)}-{selection_digest}"
    output = args.output or default_output
    config_root = Path(__file__).resolve().parents[1] / "configs" / "paper"
    rows, summaries = run_cohort(
        dataset,
        case_ids=case_ids,
        method=args.method,
        forward_spec=args.forward_spec,
        snr_db=snr_db,
        output_root=output,
        config_root=config_root,
        models_root=args.models,
        replicate=args.replicate,
        force=args.force,
        on_progress=lambda message: print(message, flush=True),
    )
    print(f"completed cases: {len(rows)}")
    print(f"aggregate groups: {len(summaries)}")
    print(f"per-case results: {(output / 'results.csv').resolve()}")
    print(f"aggregate results: {(output / 'summary.csv').resolve()}")


if __name__ == "__main__":
    main()

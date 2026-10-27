#!/usr/bin/env python3
"""Combine five checkpoint-specific cohort summaries into paper-level mean ± SD."""

from __future__ import annotations

import argparse
from pathlib import Path

from opdis_ps.replicate_aggregation import (
    aggregate_training_replicates,
    aggregation_id,
    write_training_replicate_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "replicate_root",
        type=Path,
        help="method directory containing replicate-s0 through replicate-s4",
    )
    parser.add_argument(
        "--selection",
        help="optional shared selection-<count>-<id> subdirectory for smoke-test runs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    final_rows, replicate_rows, manifest = aggregate_training_replicates(
        args.replicate_root,
        selection=args.selection,
    )
    if args.output is not None:
        output = args.output
    else:
        output = args.replicate_root / "training-replicates" / aggregation_id(manifest)
    write_training_replicate_outputs(output, final_rows, replicate_rows, manifest)
    print(f"validated replicates: {manifest['replicate_count']}")
    print(f"aggregate groups: {len(final_rows)}")
    print(f"paper summary: {(output / 'summary.csv').resolve()}")
    print(f"run-level values: {(output / 'replicates.csv').resolve()}")
    print(f"source manifest: {(output / 'manifest.json').resolve()}")


if __name__ == "__main__":
    main()

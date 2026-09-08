#!/usr/bin/env python3
"""Validate and run one prepared openCARP simulation case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from opdis_ps.simulation import run_prepared_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/paper/simulation-example.toml")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/simulation-example"))
    parser.add_argument("--np", type=int, default=8, dest="ranks")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="verify the complete bundle without importing or launching openCARP",
    )
    args = parser.parse_args()
    try:
        summary = run_prepared_simulation(
            case_dir=args.case_dir,
            config_path=args.config,
            output_dir=args.output,
            ranks=args.ranks,
            seed=args.seed,
            validate_only=args.validate_only,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

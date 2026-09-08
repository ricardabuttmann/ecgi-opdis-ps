#!/usr/bin/env python3
"""Run every connected public method and open one combined result table."""

from __future__ import annotations

import argparse
from pathlib import Path

from opdis_ps.data import PUBLIC_EXAMPLES, public_example_path
from opdis_ps.results import refresh_results_table
from opdis_ps.runner import IMPLEMENTED_METHODS, run_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all connected ECGI example methods.")
    parser.add_argument("--case", choices=PUBLIC_EXAMPLES, default=PUBLIC_EXAMPLES[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-root", type=Path, default=Path("configs/paper"))
    parser.add_argument("--no-open", action="store_true", help="do not open results.html")
    args = parser.parse_args()

    case_path = public_example_path(args.case)
    output = args.output or Path("outputs") / args.case
    for method in IMPLEMENTED_METHODS:
        print(f"starting {method} on {args.case}", flush=True)
        record = run_method(method, case_path, output, args.config_root)
        print(f"completed {record['method']} on {record['case_id']}", flush=True)
    csv_path, html_path = refresh_results_table(output, open_browser=not args.no_open)
    print(f"results CSV: {csv_path.resolve()}")
    print(f"results table: {html_path.resolve()}")


if __name__ == "__main__":
    main()

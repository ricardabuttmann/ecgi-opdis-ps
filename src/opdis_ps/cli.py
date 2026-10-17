"""Command-line entry point shared by all released methods."""

from __future__ import annotations

import argparse
from pathlib import Path

from opdis_ps.data import PUBLIC_EXAMPLES, public_example_path
from opdis_ps.pretrained_models import METHOD_FAMILIES
from opdis_ps.results import refresh_results_table
from opdis_ps.runner import IMPLEMENTED_METHODS, run_method


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one released ECGI method on an example case.")
    parser.add_argument("--method", choices=IMPLEMENTED_METHODS, required=True)
    parser.add_argument("--case", choices=PUBLIC_EXAMPLES, default=PUBLIC_EXAMPLES[0])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config-root", type=Path, default=Path("configs/paper"))
    parser.add_argument(
        "--models", type=Path, help="extracted paper-pretrained-models-v1 directory"
    )
    parser.add_argument("--replicate", type=int, choices=range(5), default=0)
    parser.add_argument("--no-open", action="store_true", help="do not open results.html")
    args = parser.parse_args()

    if args.method in METHOD_FAMILIES and args.models is None:
        parser.error(f"{args.method} requires --models pointing to paper-pretrained-models-v1")

    case_path = public_example_path(args.case)
    output = args.output or Path("outputs") / args.case
    record = run_method(
        args.method,
        case_path,
        output,
        args.config_root,
        models_root=args.models,
        replicate=args.replicate,
    )
    _, html_path = refresh_results_table(output, open_browser=not args.no_open)
    print(f"completed {record['method']} on {record['case_id']}")
    print(f"results table: {html_path.resolve()}")


if __name__ == "__main__":
    main()

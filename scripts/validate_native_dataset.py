#!/usr/bin/env python3
"""Validate the structure and optionally every checksum of ``native-ecgi-v1``."""

from __future__ import annotations

import argparse
import json

from opdis_ps.dataset import validate_package_checksums
from opdis_ps.native_dataset import NativeDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="hash every installed file; this reads roughly 161 GiB for the complete package",
    )
    args = parser.parse_args()

    dataset = NativeDataset(args.dataset)
    report = dataset.validate_structure()
    if args.checksums:
        report["checksums"] = validate_package_checksums(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

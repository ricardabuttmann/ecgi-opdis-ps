#!/usr/bin/env python3
"""Validate one extracted paper dataset package."""

from __future__ import annotations

import argparse
import json

from opdis_ps.dataset import PaperDataset, validate_package_checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--geometry-root")
    args = parser.parse_args()

    checksum_report = validate_package_checksums(args.dataset)
    dataset = PaperDataset(args.dataset, geometry_root=args.geometry_root)
    report = {
        "status": "ok",
        "package_id": dataset.package_id,
        "case_count": len(dataset.cases),
        "operator_count": len(dataset.operators),
        "measurement_specs": sorted(dataset.measurements),
        "checksums": checksum_report,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

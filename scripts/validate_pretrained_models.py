#!/usr/bin/env python3
"""Validate every file and inference checkpoint in paper-pretrained-models-v1."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from ecgi.models.diffusion_common.checkpoint import INFERENCE_CHECKPOINT_FORMAT
from opdis_ps.pretrained_models import FAMILY_WEIGHTS, PretrainedModels, sha256_file


def checksum_rows(root: Path) -> dict[str, str]:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"missing package checksum file: {checksum_path}")
    rows: dict[str, str] = {}
    for number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or not relative or relative in rows:
            raise ValueError(f"SHA256SUMS:{number}: invalid or duplicate record")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            raise ValueError(f"SHA256SUMS:{number}: unsafe or missing path {relative!r}")
        actual = sha256_file(path).removeprefix("sha256:")
        if actual != digest:
            raise ValueError(f"{relative}: checksum mismatch")
        rows[relative] = digest
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if set(rows) != expected:
        raise ValueError("SHA256SUMS does not exactly cover all package files")
    return rows


def validate(root: Path) -> dict[str, object]:
    root = root.resolve()
    package = PretrainedModels(root)
    checksum_rows(root)
    expected_ids = {f"{family}-s{index}" for family in FAMILY_WEIGHTS for index in range(5)}
    if set(package.model_ids) != expected_ids:
        raise ValueError("model family/replicate coverage is incomplete")
    model_bytes = 0
    for family, weights_kind in FAMILY_WEIGHTS.items():
        for replicate in range(5):
            model = package.resolve(family, replicate)
            payload = torch.load(model.path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{model.model_id}: checkpoint is not a mapping")
            if payload.get("checkpoint_format") != INFERENCE_CHECKPOINT_FORMAT:
                raise ValueError(f"{model.model_id}: unsupported checkpoint format")
            if payload.get("weights_kind") != weights_kind:
                raise ValueError(f"{model.model_id}: checkpoint weight selection mismatch")
            if not isinstance(payload.get("model"), Mapping) or not payload["model"]:
                raise ValueError(f"{model.model_id}: empty model state")
            if "optimizer" in payload or "ema" in payload:
                raise ValueError(f"{model.model_id}: training-only state remains")
            model_bytes += model.size_bytes
    return {
        "status": "ok",
        "package_id": package.manifest["package_id"],
        "release_version": package.manifest.get("release_version"),
        "model_count": len(expected_ids),
        "model_size_bytes": model_bytes,
        "manifest_sha256": package.manifest_sha256,
        "checksums_sha256": sha256_file(root / "SHA256SUMS"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.package), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

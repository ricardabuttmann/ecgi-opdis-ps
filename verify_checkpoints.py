"""Verify the compact checkpoint files after GitHub or anonymous-proxy transport."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy  # noqa: F401  # Import first for Conda OpenMP-runtime compatibility.
import torch

EXPECTED = {
    "palette-s0.pt": {
        "sha256": "02c5fc5c2dfbc74f0760294b9c6951e5c7cbac713805bd6f5fbcd5e23ec48cb6",
        "size_bytes": 60_886_983,
        "weights_kind": "ema",
    },
    "prior-s0.pt": {
        "sha256": "802c0465810976c044f795818f6d714dce3cd2070de98458e25e24bc7c09a62d",
        "size_bytes": 59_608_139,
        "weights_kind": "ema",
    },
    "transformer-s0.pt": {
        "sha256": "cf89c3ff3b5bf3c1dab1319ed9b0696826b2beaa4a49a1615a0bfbb0e1861f34",
        "size_bytes": 45_801_242,
        "weights_kind": "raw",
    },
}
REQUIRED_KEYS = {
    "checkpoint_format",
    "weights_kind",
    "model",
    "step",
    "scale",
    "x_clip",
    "model_config",
    "meta",
    "source_checkpoint",
}
FORBIDDEN_PAYLOAD_KEYS = {"optimizer", "ema"}
FORBIDDEN_TEXT = ("/home/", "/Users/")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from strings(key)
            yield from strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from strings(child)


def verify(path: Path, expected: dict[str, Any]) -> None:
    actual_size = path.stat().st_size
    actual_sha256 = sha256(path)
    if actual_size != expected["size_bytes"]:
        raise ValueError(f"{path}: size {actual_size} != {expected['size_bytes']}")
    if actual_sha256 != expected["sha256"]:
        raise ValueError(f"{path}: SHA-256 {actual_sha256} != {expected['sha256']}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = REQUIRED_KEYS - set(checkpoint)
    forbidden = FORBIDDEN_PAYLOAD_KEYS & set(checkpoint)
    if missing:
        raise ValueError(f"{path}: missing required keys {sorted(missing)}")
    if forbidden:
        raise ValueError(f"{path}: contains training-only keys {sorted(forbidden)}")
    if checkpoint["checkpoint_format"] != "ecgi-inference-v1":
        raise ValueError(f"{path}: unsupported checkpoint format")
    if checkpoint["weights_kind"] != expected["weights_kind"]:
        raise ValueError(f"{path}: unexpected weight kind {checkpoint['weights_kind']!r}")
    if not isinstance(checkpoint["model"], Mapping) or not checkpoint["model"]:
        raise ValueError(f"{path}: empty model state")

    public_contract = {
        "source_checkpoint": checkpoint["source_checkpoint"],
        "model_config": checkpoint["model_config"],
        "meta": checkpoint["meta"],
    }
    leaked = [text for text in strings(public_contract) if any(x in text for x in FORBIDDEN_TEXT)]
    if leaked:
        raise ValueError(f"{path}: possible private identifiers in public metadata: {leaked}")

    print(
        f"OK {path.name}: {actual_size / 1024**2:.1f} MiB, "
        f"weights={checkpoint['weights_kind']}, sha256={actual_sha256}"
    )


def main() -> None:
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    for filename, expected in EXPECTED.items():
        verify(checkpoint_dir / filename, expected)
    print("All compact checkpoints verified.")


if __name__ == "__main__":
    main()

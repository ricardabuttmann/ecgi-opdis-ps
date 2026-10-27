"""Validation and loading of the compact inference checkpoints in this release."""

from __future__ import annotations

from pathlib import Path

import torch

INFERENCE_CHECKPOINT_FORMAT = "ecgi-inference-v1"
CHECKPOINT_WEIGHT_KINDS = ("ema", "raw")


def read_inference_checkpoint(
    path: str | Path,
    *,
    expected_weights: str,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Load and validate one compact checkpoint distributed with this release."""
    if expected_weights not in CHECKPOINT_WEIGHT_KINDS:
        raise ValueError(
            f"weights must be one of {CHECKPOINT_WEIGHT_KINDS}, got {expected_weights!r}"
        )
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    checkpoint_format = checkpoint.get("checkpoint_format")
    if checkpoint_format != INFERENCE_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} has unsupported checkpoint_format {checkpoint_format!r}")
    stored_weights = checkpoint.get("weights_kind")
    if stored_weights not in CHECKPOINT_WEIGHT_KINDS:
        raise ValueError(f"{path} has invalid weights_kind {stored_weights!r}")
    if stored_weights != expected_weights:
        raise ValueError(
            f"{path} contains {stored_weights!r} weights, "
            f"not requested {expected_weights!r} weights"
        )
    if "model" not in checkpoint:
        raise ValueError(f"{path} does not contain model weights")
    return checkpoint

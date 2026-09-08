"""Shared projection used by the released diffusion samplers."""

from __future__ import annotations

import torch


def average_reference_grid(x: torch.Tensor) -> torch.Tensor:
    """Subtract the spatial mean at each time frame."""
    return x - x.mean(dim=(-2, -1), keepdim=True)

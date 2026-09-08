"""Diffusion-timestep embeddings for the released 3D denoisers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(tau: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding of the diffusion step tau, shape [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device).float() / half)
    args = tau.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))  # constant zero-pad of an embedding channel, not a spatial axis
    return emb


class TimeEmbedding(nn.Module):
    """tau [B] -> sinusoidal embedding -> 2-layer MLP -> [B, hidden], fed into every ResBlock."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_embedding(tau, self.dim))

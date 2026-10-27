"""Standard temporal Transformer for direct BSP-to-HSP regression.

The cardiac time samples are the tokens. Each token contains all electrodes at one time point,
and a linear head predicts all fixed-template HSP values at that same time point. The model does
not receive the forward operator, a spatial basis, or hand-crafted spatial features.

Reference:
    Vaswani et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_encoding(length: int, d_model: int) -> torch.Tensor:
    """Return the fixed sinusoidal position encoding from Vaswani et al. as ``[length, d]``."""
    if length < 1:
        raise ValueError(f"length must be positive, got {length}")
    if d_model < 2:
        raise ValueError(f"d_model must be at least 2, got {d_model}")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(length, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding


class TemporalTransformer(nn.Module):
    """Encoder-only Transformer mapping ``[B,T,M_bsp]`` directly to ``[B,T,N_hsp]``."""

    def __init__(
        self,
        *,
        n_electrodes: int,
        n_hsp: int,
        max_time: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_electrodes < 1 or n_hsp < 1:
            raise ValueError("n_electrodes and n_hsp must be positive")
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.n_electrodes = n_electrodes
        self.n_hsp = n_hsp
        self.max_time = max_time
        self.input_projection = nn.Linear(n_electrodes, d_model)
        self.register_buffer(
            "position_encoding",
            sinusoidal_encoding(max_time, d_model),
            persistent=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(d_model, n_hsp)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize matrix parameters independently with the standard Xavier scheme."""
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, bsp: torch.Tensor) -> torch.Tensor:
        """Predict normalized HSP from normalized BSP.

        Args:
            bsp: ``[batch, time, electrodes]``.
        Returns:
            ``[batch, time, template_points]``.
        """
        if bsp.ndim != 3:
            raise ValueError(f"expected BSP [batch,time,electrodes], got {tuple(bsp.shape)}")
        if bsp.shape[-1] != self.n_electrodes:
            raise ValueError(f"expected {self.n_electrodes} electrodes, got {bsp.shape[-1]}")
        if bsp.shape[1] > self.max_time:
            raise ValueError(
                f"sequence has {bsp.shape[1]} time samples, checkpoint supports {self.max_time}"
            )
        hidden = self.input_projection(bsp)
        hidden = hidden + self.position_encoding[: bsp.shape[1]].to(dtype=hidden.dtype)
        return self.output_projection(self.encoder(hidden))

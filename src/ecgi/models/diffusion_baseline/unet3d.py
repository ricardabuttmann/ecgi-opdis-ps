"""3D U-Net backbone for epsilon or v-prediction on the padded UVC canvas.

Structural reference: the ResBlock (GroupNorm -> SiLU -> conv -> add time-embedding ->
GroupNorm -> SiLU -> dropout -> conv -> residual) follows OpenAI's guided-diffusion (Dhariwal &
Nichol 2021, MIT-licensed) -- the canonical DDPM U-Net pattern. Adapted to 3D with the
axis-specific boundary conditions from padding.py (every plain nn.Conv3d uses padding=0 and is
preceded by conv_pad(), never Conv3d's own single-mode padding).

Shape table (base_channels=32, matches the baseline plan; B = batch):
    Input                  [B,   5, 96, 32, 60]
    Initial Conv           [B,  32, 96, 32, 60]
    Level 0 (2 ResBlocks)  [B,  32, 96, 32, 60]  -> Downsample (2,2,2)
    Level 1 (2 ResBlocks)  [B,  64, 48, 16, 30]  -> Downsample (2,2,2)
    Level 2 (2 ResBlocks)  [B, 128, 24,  8, 15]  -> Downsample (2,2,1)
    Bottleneck (2 ResBlocks) [B, 256, 12,  4, 15]
    ... symmetric decoder with skip concatenation ...
    Output                [B,   1, 96, 32, 60]

This module does NOT call canvas_pad/canvas_crop itself: it only ever sees the already-padded
[96, 32, 60] canvas and returns a tensor of the same spatial shape. The caller (diffusion.py)
owns padding the input and cropping the output back to [91, 30, 60].

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ecgi.models.diffusion_baseline.padding import conv_pad
from ecgi.models.diffusion_common.embeddings import TimeEmbedding


class ResBlock3D(nn.Module):
    """GroupNorm-SiLU-conv, add time embedding, GroupNorm-SiLU-dropout-conv, residual."""

    def __init__(
        self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=0)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=0)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(conv_pad(F.silu(self.norm1(x))))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None, None]
        h = self.conv2(conv_pad(self.dropout(F.silu(self.norm2(h)))))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    """Strided conv (kernel=3, our conv_pad, stride=given per-axis factor); also changes channels.

    kernel=3, pad=1, stride=s halves an even axis exactly (floor((N-1)/2)+1 == N/2 for even N) and
    leaves an axis with stride=1 unchanged -- both cases occur here (Phi stays at stride=1 in the
    last downsample, (2,2,1)).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(conv_pad(x))


class Upsample3D(nn.Module):
    """Nearest-neighbour upsample (periodicity-safe: just repeats samples, no new boundary
    content) followed by a conv_pad'd conv to smooth and change channels."""

    def __init__(self, in_ch: int, out_ch: int, stride: tuple[int, int, int]):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(conv_pad(self.upsample(x)))


class UNet3D(nn.Module):
    """Diffusion predictor whose target parameterization is selected by its wrapper."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        c0, c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        time_dim = base_channels * 4

        self.time_embed = TimeEmbedding(dim=base_channels, hidden=time_dim)
        self.init_conv = nn.Conv3d(in_channels, c0, kernel_size=3, padding=0)

        self.enc0 = nn.ModuleList([ResBlock3D(c0, c0, time_dim, dropout=dropout) for _ in range(2)])
        self.down0 = Downsample3D(c0, c1, stride=(2, 2, 2))
        self.enc1 = nn.ModuleList([ResBlock3D(c1, c1, time_dim, dropout=dropout) for _ in range(2)])
        self.down1 = Downsample3D(c1, c2, stride=(2, 2, 2))
        self.enc2 = nn.ModuleList([ResBlock3D(c2, c2, time_dim, dropout=dropout) for _ in range(2)])
        self.down2 = Downsample3D(c2, c3, stride=(2, 2, 1))

        self.bottleneck = nn.ModuleList(
            [ResBlock3D(c3, c3, time_dim, dropout=dropout) for _ in range(2)]
        )

        self.up2 = Upsample3D(c3, c2, stride=(2, 2, 1))
        self.dec2 = nn.ModuleList(
            [
                ResBlock3D(c2 * 2, c2, time_dim, dropout=dropout),
                ResBlock3D(c2, c2, time_dim, dropout=dropout),
            ]
        )
        self.up1 = Upsample3D(c2, c1, stride=(2, 2, 2))
        self.dec1 = nn.ModuleList(
            [
                ResBlock3D(c1 * 2, c1, time_dim, dropout=dropout),
                ResBlock3D(c1, c1, time_dim, dropout=dropout),
            ]
        )
        self.up0 = Upsample3D(c1, c0, stride=(2, 2, 2))
        self.dec0 = nn.ModuleList(
            [
                ResBlock3D(c0 * 2, c0, time_dim, dropout=dropout),
                ResBlock3D(c0, c0, time_dim, dropout=dropout),
            ]
        )

        self.out_norm = nn.GroupNorm(8, c0)
        self.out_conv = nn.Conv3d(c0, out_channels, kernel_size=3, padding=0)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(tau)

        h = self.init_conv(conv_pad(x))
        for block in self.enc0:
            h = block(h, t_emb)
        skip0 = h
        h = self.down0(h)

        for block in self.enc1:
            h = block(h, t_emb)
        skip1 = h
        h = self.down1(h)

        for block in self.enc2:
            h = block(h, t_emb)
        skip2 = h
        h = self.down2(h)

        for block in self.bottleneck:
            h = block(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        for block in self.dec2:
            h = block(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, skip1], dim=1)
        for block in self.dec1:
            h = block(h, t_emb)

        h = self.up0(h)
        h = torch.cat([h, skip0], dim=1)
        for block in self.dec0:
            h = block(h, t_emb)

        h = self.out_conv(conv_pad(F.silu(self.out_norm(h))))
        return h

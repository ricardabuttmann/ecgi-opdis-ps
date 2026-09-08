"""Palette-style conditional VP-DDPM on the fixed UVC template.

The epsilon or v predictor takes one additional fixed conditioning channel (the sixth U-Net input)
at every denoising step. The released Palette checkpoint uses ``bsp_linear_projection`` conditioning:
a learned shared linear layer maps the body-surface potentials into the heart-template grid, placing
the observation in the reconstruction's own coordinate system (paper Appendix B.1.2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ecgi.models.diffusion_baseline.diffusion import GaussianDiffusion, build_coord_channels
from ecgi.models.diffusion_baseline.padding import canvas_crop, canvas_pad
from ecgi.models.diffusion_baseline.unet3d import UNet3D

__all__ = [
    "BspConditionalGaussianDiffusion",
    "build_conditional_diffusion",
    "build_coord_channels",
]

BSP_LINEAR_PROJECTION = "bsp_linear_projection"


class BspConditionEncoder(nn.Module):
    """Learn one shared instantaneous BSP-to-template projection.

    The adapter deliberately has no temporal attention or hidden nonlinear inverse model.  It
    only makes the non-image measurement compatible with the Palette-style grid conditioner; the
    existing 3D U-Net remains responsible for temporal and spatial modelling.
    """

    def __init__(self, *, n_electrodes: int, n_z: int, n_phi: int):
        super().__init__()
        if min(n_electrodes, n_z, n_phi) < 1:
            raise ValueError("BSP condition dimensions must be positive")
        self.n_electrodes = n_electrodes
        self.n_z = n_z
        self.n_phi = n_phi
        self.projection = nn.Linear(n_electrodes, n_z * n_phi)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, bsp: torch.Tensor) -> torch.Tensor:
        """Map ``[B,T,E]`` normalized BSP to ``[B,1,T,Z,Phi]`` condition grids."""
        if bsp.ndim != 3 or bsp.shape[-1] != self.n_electrodes:
            raise ValueError(
                f"expected BSP [batch,time,{self.n_electrodes}], got {tuple(bsp.shape)}"
            )
        grid = self.projection(bsp).reshape(bsp.shape[0], bsp.shape[1], self.n_z, self.n_phi)
        grid = grid - grid.mean(dim=(-2, -1), keepdim=True)
        return grid.unsqueeze(1)


class ConditionalGaussianDiffusion(GaussianDiffusion, ABC):
    """VP-DDPM noise/v predictor conditioned on one target-space reconstruction channel."""

    def __init__(
        self,
        *,
        T: int = 1000,
        t_canvas: int = 96,
        z_canvas: int = 32,
        base_channels: int = 32,
        parameterization: str = "eps",
    ):
        super().__init__(
            unet=UNet3D(in_channels=6, base_channels=base_channels),
            T=T,
            t_canvas=t_canvas,
            z_canvas=z_canvas,
            parameterization=parameterization,
        )

    @abstractmethod
    def prepare_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """Map the released condition representation to ``[batch,1,time,z,phi]``."""
        raise NotImplementedError

    def predict_noise(
        self,
        x_tau: torch.Tensor,
        coords: torch.Tensor,
        condition: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon from noisy HSP, coordinates, and the fixed condition."""
        if condition.shape != x_tau.shape:
            raise ValueError(
                f"condition shape {tuple(condition.shape)} != x_tau shape {tuple(x_tau.shape)}"
            )
        model_in = torch.cat([x_tau, condition, coords], dim=1)
        padded, offsets = canvas_pad(model_in, self.t_canvas, self.z_canvas)
        raw = canvas_crop(self.unet(padded, tau), offsets)
        if self.parameterization == "eps":
            return raw
        a = self.sqrt_alpha_bars[tau].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[tau].view(-1, 1, 1, 1, 1)
        return b * x_tau + a * raw

class BspConditionalGaussianDiffusion(ConditionalGaussianDiffusion):
    """Conditional VP-DDPM whose learned adapter consumes raw normalized BSP."""

    def __init__(
        self,
        *,
        n_electrodes: int,
        n_z: int = 30,
        n_phi: int = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition_encoder = BspConditionEncoder(
            n_electrodes=n_electrodes,
            n_z=n_z,
            n_phi=n_phi,
        )

    def prepare_condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(condition)


def build_conditional_diffusion(
    conditioning: str,
    *,
    T: int = 1000,
    t_canvas: int = 96,
    z_canvas: int = 32,
    base_channels: int = 32,
    parameterization: str = "eps",
    n_electrodes: int | None = None,
    n_z: int = 30,
    n_phi: int = 60,
) -> ConditionalGaussianDiffusion:
    """Construct the checkpoint-declared conditioning architecture."""
    common = {
        "T": T,
        "t_canvas": t_canvas,
        "z_canvas": z_canvas,
        "base_channels": base_channels,
        "parameterization": parameterization,
    }
    if conditioning != BSP_LINEAR_PROJECTION:
        raise ValueError(
            f"the released Palette checkpoint requires conditioning={BSP_LINEAR_PROJECTION!r}, "
            f"got {conditioning!r}"
        )
    if n_electrodes is None:
        raise ValueError("BSP conditioning requires n_electrodes")
    return BspConditionalGaussianDiffusion(
        n_electrodes=n_electrodes,
        n_z=n_z,
        n_phi=n_phi,
        **common,
    )

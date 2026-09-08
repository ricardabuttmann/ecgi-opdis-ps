"""Variance-preserving DDPM wrapper supporting epsilon and v parameterizations.

Released checkpoints declare their parameterization explicitly. For v-prediction, the raw network
output is converted to epsilon before it is passed to the common samplers.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from ecgi.mesh_operators.template import Template
from ecgi.models.diffusion_baseline.padding import canvas_crop, canvas_pad
from ecgi.models.diffusion_baseline.unet3d import UNet3D
from ecgi.models.diffusion_common.schedule import make_schedule


def build_coord_channels(
    template: Template, n_time: int = 91, dt_ms: float = 5.0, t_start_ms: float = 0.0
) -> torch.Tensor:
    """Fixed [4, n_time, n_z, n_phi] positional channels: Z_ea, sin(phi), cos(phi), t_norm.

    Z_ea/phi come from the template (constant across time, broadcast); t_norm = 2t/T_end - 1
    (constant across space, broadcast) with T_end = t_start_ms + (n_time - 1) * dt_ms, so it
    spans exactly [-1, 1] over the recording regardless of dt_ms/n_time.
    """
    z_grid = torch.from_numpy(template.as_matrix(template.z_ea)).float()  # [n_z, n_phi]
    phi_grid = torch.from_numpy(template.as_matrix(template.phi)).float()  # [n_z, n_phi]
    sin_grid, cos_grid = torch.sin(phi_grid), torch.cos(phi_grid)

    t_end_ms = t_start_ms + (n_time - 1) * dt_ms
    t_ms = t_start_ms + torch.arange(n_time).float() * dt_ms
    t_norm = 2.0 * (t_ms - t_start_ms) / (t_end_ms - t_start_ms) - 1.0  # [n_time], spans [-1, 1]

    n_z, n_phi = template.n_z, template.n_phi
    z_ch = z_grid.expand(n_time, n_z, n_phi)
    sin_ch = sin_grid.expand(n_time, n_z, n_phi)
    cos_ch = cos_grid.expand(n_time, n_z, n_phi)
    t_ch = t_norm.view(n_time, 1, 1).expand(n_time, n_z, n_phi)
    return torch.stack([z_ch, sin_ch, cos_ch, t_ch], dim=0)  # [4, n_time, n_z, n_phi]


PARAMETERIZATIONS = ("eps", "v")


def resolve_parameterization(model_config: dict | None, *, source: str = "") -> str:
    """Read an explicitly declared epsilon or v parameterization.

    Sampling a v-trained checkpoint with eps-only math fails *silently*: it produces a finite,
    plausible-looking reconstruction that is quantitatively wrong. The compact release format
    therefore requires the value explicitly.
    """
    where = f" ({source})" if source else ""
    if not isinstance(model_config, dict):
        raise ValueError(f"checkpoint{where} has no model_config mapping")
    if "parameterization" not in model_config:
        raise ValueError(
            f"checkpoint model_config{where} does not declare 'parameterization'"
        )
    value = model_config["parameterization"]
    if value not in PARAMETERIZATIONS:
        raise ValueError(
            f"checkpoint model_config{where} declares unknown parameterization {value!r}; "
            f"expected one of {PARAMETERIZATIONS}"
        )
    return value


def load_diffusion_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    weights: str,
) -> tuple[GaussianDiffusion, float, float]:
    """Load a prior that satisfies the compact release checkpoint contract."""
    from ecgi.models.diffusion_common.checkpoint import read_inference_checkpoint

    path = Path(checkpoint_path)
    checkpoint = read_inference_checkpoint(path, expected_weights=weights)
    model_config = checkpoint.get("model_config")
    parameterization = resolve_parameterization(model_config, source=str(path))
    if model_config.get("field_whitening") is not None:
        raise ValueError(f"{path} is not the unwhitened prior distributed with this release")
    try:
        timesteps = int(model_config["T"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has no valid diffusion horizon T") from error
    if timesteps < 1:
        raise ValueError(f"{path} has invalid diffusion horizon T={timesteps}")

    model = GaussianDiffusion(T=timesteps, parameterization=parameterization).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    try:
        scale = float(checkpoint["scale"])
        x_clip = float(checkpoint["x_clip"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has no valid scale and x_clip") from error
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path} has invalid scale={scale}")
    if not math.isfinite(x_clip) or x_clip <= 0.0:
        raise ValueError(f"{path} has invalid x_clip={x_clip}")
    return model, scale, x_clip


class GaussianDiffusion(nn.Module):
    """VP-DDPM wrapper around a UNet3D noise/v predictor.

    ``parameterization`` selects what the network regresses:

    * ``"eps"`` -- the noise, the original DDPM target.
    * ``"v"``   -- the velocity ``v = sqrt(abar) * eps - sqrt(1-abar) * x0`` (Salimans & Ho 2022).
      At the high-noise end (abar -> 0) the eps target is nearly a copy of the input, so the
      network gets almost no gradient toward the clean signal there -- which is exactly where the
      sample's global amplitude is decided. The v target at that end is ~= -x0, so v-prediction
      forces the clean structure to be predicted and is the principled fix for a prior whose
      amplitude is systematically off.

    ``predict_noise`` returns eps in BOTH parameterizations (converting v -> eps internally), so
    every sampler/DPS/DDRM consumer is unchanged as long as it rebuilds the model with the same
    ``parameterization`` the checkpoint was trained under.

    """

    def __init__(
        self,
        unet: UNet3D | None = None,
        T: int = 1000,
        t_canvas: int = 96,
        z_canvas: int = 32,
        parameterization: str = "eps",
    ):
        super().__init__()
        if parameterization not in PARAMETERIZATIONS:
            raise ValueError(
                f"parameterization must be one of {PARAMETERIZATIONS}, got {parameterization!r}"
            )
        self.unet = unet if unet is not None else UNet3D()
        self.T, self.t_canvas, self.z_canvas = T, t_canvas, z_canvas
        self.parameterization = parameterization
        sched = make_schedule(T=T)
        # Registered as buffers (not plain attributes) so .to(device)/.cuda() carries them along.
        self.register_buffer("sqrt_alpha_bars", torch.from_numpy(sched.sqrt_alpha_bars).float())
        self.register_buffer(
            "sqrt_one_minus_alpha_bars", torch.from_numpy(sched.sqrt_one_minus_alpha_bars).float()
        )

    def _net_output(
        self, x_tau: torch.Tensor, coords: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Raw cropped network output (eps_hat or v_hat, per ``parameterization``).

        x_tau: [B,1,91,30,60], coords: [B,4,91,30,60] (fixed Z_ea/sin/cos/t_norm channels).
        Concatenates into the network's 5-channel input, canvas-pads (the U-Net needs sizes
        divisible by 8 for 3 downsample stages), runs the network, crops back to the real domain.
        """
        model_in = torch.cat([x_tau, coords], dim=1)  # [B,5,91,30,60]
        padded, offsets = canvas_pad(model_in, self.t_canvas, self.z_canvas)
        out = self.unet(padded, tau)  # [B,1,96,32,60]
        return canvas_crop(out, offsets)  # [B,1,91,30,60]

    def predict_noise(
        self, x_tau: torch.Tensor, coords: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Predicted eps, regardless of parameterization (v is converted here).

        eps = sqrt(1-abar) * x_tau + sqrt(abar) * v is the inverse of the v definition given
        x_tau, so every downstream sampler keeps its eps-based math untouched.
        """
        out = self._net_output(x_tau, coords, tau)
        if self.parameterization == "eps":
            return out
        a = self.sqrt_alpha_bars[tau].view(-1, 1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[tau].view(-1, 1, 1, 1, 1)
        return b * x_tau + a * out

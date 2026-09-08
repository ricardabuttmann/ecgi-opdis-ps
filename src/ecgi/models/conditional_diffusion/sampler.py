"""Conditional DDIM sampling with one fixed prepared condition."""

from __future__ import annotations

import torch

from ecgi.models.conditional_diffusion.diffusion import ConditionalGaussianDiffusion
from ecgi.models.diffusion_baseline.sampler import average_reference_grid


@torch.no_grad()
def conditional_ddim_sample(
    diffusion: ConditionalGaussianDiffusion,
    coords: torch.Tensor,
    condition: torch.Tensor,
    *,
    n_steps: int = 50,
    eta: float = 0.0,
    scale: float = 1.0,
    x0_clip: float | None = None,
    temperature: float = 1.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``p(x | condition)`` and return physical units as ``[B,1,T,Z,Phi]``."""
    device = device or next(diffusion.parameters()).device
    diffusion.eval()
    condition = diffusion.prepare_condition(condition.to(device))
    batch, _, n_time, n_z, n_phi = condition.shape
    coords = coords.to(device).expand(batch, -1, -1, -1, -1)
    alpha_bars = diffusion.sqrt_alpha_bars.to(device) ** 2
    taus = torch.linspace(diffusion.T - 1, 0, n_steps, device=device).round().long()
    taus = torch.unique_consecutive(taus)

    x = temperature * torch.randn(
        batch,
        1,
        n_time,
        n_z,
        n_phi,
        device=device,
        generator=generator,
    )
    x = average_reference_grid(x)
    for i, tau in enumerate(taus):
        eps_hat = diffusion.predict_noise(x, coords, condition, tau.expand(batch))
        a_t = alpha_bars[tau]
        x0_hat = (x - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt()
        if x0_clip is not None:
            x0_hat = x0_hat.clamp(-x0_clip, x0_clip)
        x0_hat = average_reference_grid(x0_hat)

        a_prev = alpha_bars[taus[i + 1]] if i + 1 < len(taus) else torch.ones((), device=device)
        if eta > 0.0 and i + 1 < len(taus):
            sigma = eta * (((1 - a_prev) / (1 - a_t)) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
        else:
            sigma = torch.zeros((), device=device)
        x = a_prev.sqrt() * x0_hat + (1.0 - a_prev - sigma**2).clamp(min=0).sqrt() * eps_hat
        if sigma > 0:
            step_noise = torch.randn(
                x.shape,
                dtype=x.dtype,
                device=x.device,
                generator=generator,
            )
            x = x + temperature * sigma * step_noise
        x = average_reference_grid(x)
    return x * scale

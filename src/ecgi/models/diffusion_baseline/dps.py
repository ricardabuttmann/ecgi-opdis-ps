"""Diffusion Posterior Sampling for the linear ECGI observation model.

Each DDIM step uses the Tweedie estimate ``x0_hat`` and descends the measurement residual
``||H x0_hat - y||`` through the denoiser. The constant spatial mode remains projected out because
the average-referenced forward operator cannot observe it.
"""

from __future__ import annotations

import torch

from ecgi.models.diffusion_baseline.diffusion import GaussianDiffusion
from ecgi.models.diffusion_baseline.sampler import average_reference_grid


def _apply_H(x0_grid: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """H x0 per time frame. x0_grid: [B,1,T,Z,phi]; H: [m, Z*phi]. Returns [B,T,m].

    The (Z, phi) flatten is row-major, identical to Template.as_matrix's inverse (`.ravel()`) and
    thus to the column order of forward_A -- this is the one place the grid and the operator meet.
    """
    B, _, T, Z, P = x0_grid.shape
    flat = x0_grid[:, 0].reshape(B, T, Z * P)  # [B,T,Z*phi], row-major over (Z,phi)
    return torch.einsum("mn,btn->btm", H, flat)  # [B,T,m]


def _apply_HT(r: torch.Tensor, H: torch.Tensor, Z: int, P: int) -> torch.Tensor:
    """H^T r, reshaped back to a grid. r: [B,T,m]; returns [B,1,T,Z,phi]."""
    B, T, _ = r.shape
    g = torch.einsum("mn,btm->btn", H, r)  # [B,T,Z*phi]
    return g.reshape(B, 1, T, Z, P)


def dps_sample(
    diffusion: GaussianDiffusion,
    coords: torch.Tensor,
    H: torch.Tensor,
    y: torch.Tensor,
    *,
    batch: int = 1,
    n_steps: int = 50,
    eta: float = 0.0,
    zeta: float = 1.0,
    project_mean_zero: bool = True,
    scale: float = 1.0,
    x0_clip: float | None = None,
    temperature: float = 1.0,
    res_eps: float = 1e-8,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DPS posterior sampling. Returns [batch, 1, T, Z, phi], denormalized by `scale`.

    H: [m, Z*phi] forward operator (torch, same units/reference as the data). y: [m, T] measurement
    in NORMALIZED units (divide the physical measurement by `scale` before calling, so H x0_hat --
    x0_hat being normalized -- and y live in the same space). zeta: guidance strength (0 recovers
    the unconditional ddim_sample bit-for-bit). eta: DDIM stochasticity (0 = deterministic).

    x0_clip: same Tweedie-x0 clamp as ddim_sample (essential, not optional -- see sampler.py). DPS
    inherits the identical reverse chain, so it inherits the t=T-1 1/sqrt(alpha_bar) explosion and
    needs the same bound. clamp is differentiable (grad 1 inside, 0 outside), so the guidance
    gradient correctly ignores components already pinned to the bound.

    temperature: same low-temperature knob as ddim_sample -- it scales the initial latent x_T (and
    any eta>0 increment), so temperature<1 starts every sample nearer the noise prior's mode and
    shrinks the prior's spread. Here it only touches the PRIOR terms; the measurement-guidance step
    (zeta * J^T H^T r) is untouched, so lowering temperature shifts the prior/likelihood balance
    toward the data -- a second lever alongside zeta against an over-dispersed prior. Selected on
    validation via evaluate_prior; default 1.0 = unchanged.

    The released paper configuration uses the original DPS update: differentiate the unweighted
    residual norm and apply the residual-normalized step ``zeta / ||r||`` at every noise level.
    """
    device = device or next(diffusion.parameters()).device
    diffusion.eval()
    guided = zeta != 0.0
    T = diffusion.T
    alpha_bars = diffusion.sqrt_alpha_bars.to(device) ** 2  # [T]

    taus = torch.linspace(T - 1, 0, n_steps, device=device).round().long()
    taus = torch.unique_consecutive(taus)

    _, _, Tt, Z, Phi = coords.shape
    H = H.to(device)
    y_bt = y.to(device).t().unsqueeze(0)  # [m,T] -> [1,T,m], broadcast over batch
    x = temperature * torch.randn(batch, 1, Tt, Z, Phi, device=device, generator=generator)
    if project_mean_zero:
        x = average_reference_grid(x)
    coords = coords.to(device).expand(batch, -1, -1, -1, -1)

    for i in range(len(taus)):
        tau = taus[i]
        a_t = alpha_bars[tau]

        if guided:
            # Guidance needs d x0_hat / d x, so this stretch runs WITH autograd (unlike ddim_sample).
            x = x.detach().requires_grad_(True)
            eps_hat = diffusion.predict_noise(x, coords, tau.expand(batch))
            x0_hat = (x - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt()
            if x0_clip is not None:
                x0_hat = x0_hat.clamp(-x0_clip, x0_clip)
            if project_mean_zero:
                x0_hat = average_reference_grid(x0_hat)

            r = _apply_H(x0_hat, H) - y_bt  # [B,T,m], data residual at x0_hat
            g_grid = _apply_HT(r, H, Z, Phi)  # H^T r, analytic (no autograd through H)
            grad_x = torch.autograd.grad(x0_hat, x, grad_outputs=g_grid)[0]  # J^T H^T r
            rnorm = r.reshape(batch, -1).norm(dim=1)  # [B], per-sample misfit magnitude

            x, eps_hat, x0_hat = x.detach(), eps_hat.detach(), x0_hat.detach()
        else:
            with torch.no_grad():
                eps_hat = diffusion.predict_noise(x, coords, tau.expand(batch))
                x0_hat = (x - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt()
                if x0_clip is not None:
                    x0_hat = x0_hat.clamp(-x0_clip, x0_clip)
                if project_mean_zero:
                    x0_hat = average_reference_grid(x0_hat)

        with torch.no_grad():
            a_prev = alpha_bars[taus[i + 1]] if i + 1 < len(taus) else torch.ones((), device=device)
            if eta > 0.0 and i + 1 < len(taus):
                sigma = eta * (((1 - a_prev) / (1 - a_t)) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
            else:
                sigma = torch.zeros((), device=device)

            dir_xt = (1.0 - a_prev - sigma**2).clamp(min=0).sqrt() * eps_hat
            x = a_prev.sqrt() * x0_hat + dir_xt
            if sigma > 0:
                x = x + temperature * sigma * torch.randn_like(x)
            if guided:
                step = (zeta / (rnorm + res_eps)).view(batch, 1, 1, 1, 1)
                x = x - step * grad_x
            if project_mean_zero:
                x = average_reference_grid(x)

    return x * scale

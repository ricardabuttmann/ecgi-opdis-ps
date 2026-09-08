"""DDRM posterior sampling for the linear ECGI observation model.

Implements the spectral DDRM update of Kawar et al. (NeurIPS 2022, arXiv:2201.11793) for a per-case
operator in reduced Helmert coordinates. The reference implementation is MIT licensed.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from ecgi.forward.operator_spectrum import (
    operator_spectrum,
)
from ecgi.models.diffusion_baseline.diffusion import GaussianDiffusion
from ecgi.models.diffusion_baseline.sampler import average_reference_grid

# Reference defaults from the DDRM repository's own example commands: `--eta 0.85 --etaB 1`.
# `eta` feeds both the observed-but-clean branch (A) and the unobserved branch (C); `eta_b` is the
# blend weight of branch B, where 1 means a pure overwrite from the data.
DEFAULT_ETA = 0.85
DEFAULT_ETA_B = 1.0


@dataclass(frozen=True)
class _SpectralOperator:
    """Torch-side SVD bundle for one case, in reduced Helmert coordinates."""

    left: torch.Tensor  # U    [k, k]
    singular_values: torch.Tensor  # s    [k]
    right: torch.Tensor  # Vr   [k, n]
    resolved: torch.Tensor  # bool [k], singular values above the relative floor
    reduced_data: torch.Tensor  # y_bar [k, T] = U^T R^T y


def _prepare_operator(
    operator: np.ndarray,
    measurement: np.ndarray,
    *,
    relative_floor: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> _SpectralOperator:
    """Decompose `H` and project the measurement into spectral coordinates.

    The reduction to Helmert coordinates happens here and nowhere else: the average-referenced
    measurement covariance is singular in electrode space, so the whitening DDRM assumes only holds
    after `R^T`. `operator_spectrum` owns that step.
    """
    spectrum = operator_spectrum(operator)
    reduced_measurement = spectrum.reduced_basis.T @ np.asarray(measurement, dtype=np.float64)
    spectral_measurement = spectrum.left_vectors.T @ reduced_measurement  # [k, T]
    resolved = spectrum.resolved_mask(relative_floor)
    return _SpectralOperator(
        left=torch.as_tensor(spectrum.left_vectors, dtype=dtype, device=device),
        singular_values=torch.as_tensor(spectrum.singular_values, dtype=dtype, device=device),
        right=torch.as_tensor(spectrum.right_vectors, dtype=dtype, device=device),
        resolved=torch.as_tensor(resolved, dtype=torch.bool, device=device),
        reduced_data=torch.as_tensor(spectral_measurement, dtype=dtype, device=device),
    )


def _to_spectral(grid: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """[B,1,T,Z,phi] -> [B,T,k]. The (Z, phi) flatten is row-major, matching forward_A's columns."""
    B, _, T, Z, P = grid.shape
    return torch.einsum("kn,btn->btk", right, grid[:, 0].reshape(B, T, Z * P))


def _from_spectral(coefficients: torch.Tensor, right: torch.Tensor, Z: int, P: int) -> torch.Tensor:
    """[B,T,k] -> [B,1,T,Z,phi], the adjoint of `_to_spectral`."""
    B, T, _ = coefficients.shape
    return torch.einsum("kn,btk->btn", right, coefficients).reshape(B, 1, T, Z, P)


def spectral_branches(
    x0_spectral: torch.Tensor,
    branch_c: torch.Tensor,
    *,
    singular_values: torch.Tensor,
    inverse_singular_values: torch.Tensor,
    resolved: torch.Tensor,
    reduced_data: torch.Tensor,
    sigma_y: float,
    sigma_next: float,
    eta: float,
    eta_b: float,
    noise: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Return the new spectral coefficients, choosing per coordinate between branches A, B and C.

    Separated from the chain because this is where the method actually lives and where a wrong
    constant would be invisible end to end: with an untrained or merely imperfect network the final
    denoise step masks any error here, so the branch rule has to be tested on its own.

    `branch_c` is what the ordinary full-space step already produced for these coordinates; it is
    kept for every unresolved direction.
    """
    std = eta * sigma_next
    tilde = float(np.sqrt(max(sigma_next**2 - std**2, 0.0)))
    noise_a, noise_b = noise

    # A -- the chain is cleaner than this coordinate's measurement, so the data only corrects.
    # sigma_y = 0 makes the residual meaningless, but branch A is then never selected.
    safe_sigma_y = sigma_y if sigma_y > 0.0 else 1.0
    residual = (reduced_data - singular_values * x0_spectral) / safe_sigma_y
    branch_a = x0_spectral + tilde * residual + std * noise_a

    # B -- the chain is noisier, so take the data and restore the marginal to sigma_next.
    variance_b = sigma_next**2 - (eta_b * sigma_y * inverse_singular_values) ** 2
    branch_b = (
        eta_b * reduced_data * inverse_singular_values
        + (1.0 - eta_b) * x0_spectral
        + variance_b.clamp(min=0.0).sqrt() * noise_b
    )

    # The boundary `s_i * sigma_next == sigma_y` belongs to branch B (the data). It is reached
    # exactly at the final DDNM step, where sigma_next and sigma_y are both zero: only `>=` writes
    # `y_bar / s` there, whereas a strict `>` falls through to branch A's denoiser estimate and
    # silently drops data consistency in the last step.
    take_data = resolved & (singular_values * sigma_next >= sigma_y)
    chosen = torch.where(take_data, branch_b, branch_a)
    return torch.where(resolved, chosen, branch_c)


def initial_state(
    noise_grid: torch.Tensor,
    spectral: _SpectralOperator,
    *,
    inverse_singular_values: torch.Tensor,
    sigma_y: float,
    sigma_top: float,
) -> torch.Tensor:
    """Build `x_T` from the measurement plus exactly enough noise to reach the top noise level.

    Coordinates the measurement can already say something about (`s_i sigma_top > sigma_y`) start at
    `y_bar_i / s_i` and are topped up with `sqrt(sigma_top^2 - (sigma_y/s_i)^2)`; everything else
    starts as plain noise at `sigma_top`. The whole thing is then divided by `sigma_top`, which
    converts the `x_mod` construction back to the ordinary VP state the chain expects.

    The scaling here is easy to get wrong in a way nothing downstream reveals: `noise_grid` enters
    the spectral coordinates already carrying a factor `sigma_top`, so the per-coordinate correction
    is `(residual/sigma_top - 1)`, not `(residual - sigma_top)`. With `sigma_top` around `2e4` the
    latter is off by four orders of magnitude and still produces a finite, plausible-looking sample.
    """
    _, _, _, Z, P = noise_grid.shape
    s = spectral.singular_values
    informative = spectral.resolved & (s * sigma_top > sigma_y)
    y_bar = spectral.reduced_data.t().unsqueeze(0)

    data_start = torch.zeros_like(_to_spectral(noise_grid, spectral.right))
    data_start[..., informative] = (y_bar * inverse_singular_values)[..., informative]
    residual_sigma = torch.full_like(s, sigma_top)
    residual_sigma[informative] = (
        (sigma_top**2 - (sigma_y * inverse_singular_values[informative]) ** 2).clamp(min=0.0).sqrt()
    )

    scaled_noise = sigma_top * noise_grid
    spectral_noise = _to_spectral(scaled_noise, spectral.right)
    correction = data_start + (residual_sigma / sigma_top - 1.0) * spectral_noise
    return (scaled_noise + _from_spectral(correction, spectral.right, Z, P)) / sigma_top


def ddrm_sample(
    diffusion: GaussianDiffusion,
    coords: torch.Tensor,
    spectral: _SpectralOperator,
    *,
    sigma_y: float,
    batch: int = 1,
    n_steps: int = 20,
    eta: float = DEFAULT_ETA,
    eta_b: float = DEFAULT_ETA_B,
    project_mean_zero: bool = True,
    scale: float = 1.0,
    x0_clip: float | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DDRM posterior sampling. Returns [batch, 1, T, Z, phi], denormalized by `scale`.

    `sigma_y` is the per-coordinate measurement noise scale in **reduced Helmert coordinates and in
    the prior's normalized units** -- i.e. `measurement_sigma(noise_rms, m) / scale`. Passing the
    raw recorded RMS understates it by `sqrt(m/(m-1))`; passing physical units breaks the comparison
    against `sigma_t` entirely. `sigma_y = 0` selects the DDNM limit.

    `eta` is the stochasticity of branches A and C, `eta_b` the data weight of branch B. The
    reference defaults are 0.85 and 1.0.
    """
    device = device or next(diffusion.parameters()).device
    diffusion.eval()
    if sigma_y < 0 or not np.isfinite(sigma_y):
        raise ValueError(f"sigma_y must be non-negative and finite, got {sigma_y!r}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must lie in [0, 1], got {eta}")
    if not 0.0 <= eta_b <= 1.0:
        raise ValueError(f"eta_b must lie in [0, 1], got {eta_b}")

    T_total = diffusion.T
    alpha_bars = diffusion.sqrt_alpha_bars.to(device) ** 2
    _, _, n_time, Z, P = coords.shape
    coords = coords.to(device).expand(batch, -1, -1, -1, -1)

    taus = torch.linspace(T_total - 1, 0, n_steps, device=device).round().long()
    taus = torch.unique_consecutive(taus)

    s = spectral.singular_values
    resolved = spectral.resolved
    y_bar = spectral.reduced_data.t().unsqueeze(0)  # [k,T] -> [1,T,k], broadcast over batch

    # Reciprocal singular values, floored. Only `resolved` coordinates are ever divided by, and the
    # branch conditions below additionally keep the division bounded -- see the design note.
    inverse_s = torch.zeros_like(s)
    inverse_s[resolved] = 1.0 / s[resolved]

    def noise_like(reference: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            reference.shape, device=device, dtype=reference.dtype, generator=generator
        )

    # --- initialization: data where the measurement can already say something, noise elsewhere ---
    sigma_top = float(((1.0 - alpha_bars[taus[0]]) / alpha_bars[taus[0]]).sqrt())
    noise_grid = noise_like(torch.empty(batch, 1, n_time, Z, P, device=device, dtype=s.dtype))
    x = initial_state(
        noise_grid,
        spectral,
        inverse_singular_values=inverse_s,
        sigma_y=sigma_y,
        sigma_top=sigma_top,
    )
    if project_mean_zero:
        x = average_reference_grid(x)

    for index in range(len(taus)):
        tau = taus[index]
        a_t = alpha_bars[tau]
        a_prev = (
            alpha_bars[taus[index + 1]] if index + 1 < len(taus) else torch.ones((), device=device)
        )

        with torch.no_grad():
            eps_hat = diffusion.predict_noise(x, coords, tau.expand(batch))
            x0 = (x - (1.0 - a_t).sqrt() * eps_hat) / a_t.sqrt()
            if x0_clip is not None:
                x0 = x0.clamp(-x0_clip, x0_clip)
            if project_mean_zero:
                x0 = average_reference_grid(x0)

            sigma_next = float(((1.0 - a_prev) / a_prev).clamp(min=0.0).sqrt())

            # Branch C in the full state space: the ordinary step, applied everywhere. The observed
            # directions are corrected afterwards, so V is never needed.
            std_c = eta * sigma_next
            tilde_c = float(np.sqrt(max(sigma_next**2 - std_c**2, 0.0)))
            x_mod = x0 + tilde_c * eps_hat + std_c * noise_like(x0)

            # Note there is deliberately no `sigma_next > 0` guard: at the final step sigma_next is
            # zero, and that is exactly when branch B must write the data value with no noise added.
            # Skipping the correction there would discard data consistency in the last step.
            if s.numel() > 0:
                c0 = _to_spectral(x0, spectral.right)  # x0 in spectral coordinates
                c_step = _to_spectral(x_mod, spectral.right)  # what branch C put there
                c_new = spectral_branches(
                    c0,
                    c_step,
                    singular_values=s,
                    inverse_singular_values=inverse_s,
                    resolved=resolved,
                    reduced_data=y_bar,
                    sigma_y=sigma_y,
                    sigma_next=sigma_next,
                    eta=eta,
                    eta_b=eta_b,
                    noise=(noise_like(c0), noise_like(c0)),
                )
                x_mod = x_mod + _from_spectral(c_new - c_step, spectral.right, Z, P)

            x = a_prev.sqrt() * x_mod
            if project_mean_zero:
                x = average_reference_grid(x)

    return x * scale

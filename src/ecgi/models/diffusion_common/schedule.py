"""VP-DDPM noise schedule: cosine alpha_bar(tau) and the lookup tables built from it.

Pure numpy, no torch dependency -- this is precomputed once (1000 entries) and converted to
torch tensors by whichever module builds the network (diffusion.py), so it stays trivially
testable without a GPU/conda env. Cosine schedule per Nichol & Dhariwal 2021 ("Improved
Denoising Diffusion Probabilistic Models"), eq. 17: it keeps SNR from collapsing too fast near
tau=0 compared to the original linear-beta schedule, which matters more the smaller/cleaner the
signal (our HSP amplitude is O(1) after normalization, not 8-bit pixel noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Schedule:
    """Precomputed VP-DDPM lookup tables, length T, indexed by diffusion step tau in [0, T)."""

    betas: np.ndarray  # beta_tau
    alphas: np.ndarray  # 1 - beta_tau
    alpha_bars: np.ndarray  # prod_{s<=tau} alpha_s  (a.k.a. alpha_bar_tau)
    sqrt_alpha_bars: np.ndarray  # sqrt(alpha_bar_tau), scales x_0 in q(x_tau | x_0)
    sqrt_one_minus_alpha_bars: np.ndarray  # sqrt(1 - alpha_bar_tau), scales the noise eps

    @property
    def T(self) -> int:
        return self.betas.shape[0]


def cosine_alpha_bar(t: np.ndarray, T: int, s: float = 0.008) -> np.ndarray:
    """Cosine alpha_bar schedule (Nichol & Dhariwal eq. 17), t in [0, T] (continuous OK).

    f(t) = cos((t/T + s) / (1 + s) * pi/2)^2, alpha_bar(t) = f(t) / f(0). Monotonically
    decreasing 1 -> 0; s=0.008 keeps beta_0 away from the numerically-unstable "too small a
    first step" the raw cosine has at t=0.
    """
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    return f / f[0]


def make_schedule(T: int = 1000, s: float = 0.008, beta_max: float = 0.999) -> Schedule:
    """Build the T-step cosine schedule as betas_tau = 1 - alpha_bar_tau / alpha_bar_{tau-1},
    clipped to beta_max (the discretization of a continuous alpha_bar can otherwise blow up
    beta near tau=T, matching the reference implementation's clamp).
    """
    alpha_bar_full = cosine_alpha_bar(np.arange(T + 1, dtype=np.float64), T, s)  # T+1 points
    betas = np.clip(1.0 - alpha_bar_full[1:] / alpha_bar_full[:-1], a_min=0.0, a_max=beta_max)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return Schedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=np.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=np.sqrt(1.0 - alpha_bars),
    )

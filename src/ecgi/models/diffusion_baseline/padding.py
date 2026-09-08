"""Axis-specific padding for the 3D U-Net: [B, C, T, Z, Phi] with THREE different boundary
conditions, so a plain `nn.Conv3d(padding=1)` (one mode for all axes) does not work.

  T (time)     replicate   hold the edge frame -- no periodicity and no meaningful mirror
                           either side of a recording's start/end.
  Z (Z_ea)     replicate   NOT reflect: apex/basis have no physical mirror symmetry, so
                           replicate (hold the edge value) is the honest choice there.
  Phi (angle)  circular    Phi = -pi === +pi is the same meridian: true periodic boundary.
                           Fixed, no mode choice -- this is a physical fact, not a hyperparameter.

Two use sites:
  canvas_pad/canvas_crop   ONCE, at the U-Net's input/output: [T,Z,Phi] = [91,30,60] has no
                           common power-of-two factor for 3 downsample stages, so this pads to
                           a canvas [96,32,60] that halves cleanly, then crops back after.
  conv_pad                 before EVERY Conv3d(kernel=3, padding=0) inside the network, so the
                           kernel sees the correct 3 boundary conditions at every resolution
                           level, not just once at the outer edge.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Axis positions in [B, C, T, Z, Phi] (negative = counted from the end, so this stays correct
# regardless of how many leading dims -- B, C -- precede them).
_T_DIM, _Z_DIM, _PHI_DIM = -3, -2, -1


def _pad_axis(x: torch.Tensor, dim: int, left: int, right: int, mode: str) -> torch.Tensor:
    """Pad ONE axis with a non-constant mode, whatever the tensor's total rank.

    torch's non-constant F.pad (reflect/replicate/circular) hard-codes the pad-tuple length to
    the input rank (5D input -> exactly 6 values, ALL three trailing dims, one shared mode) --
    there is no way to target a single axis of a 5D tensor directly, which is what we need since
    T/Z/Phi each have a different mode. Fix: move `dim` to the end, collapse every other dim into
    one flat axis (a 3D (N, 1, W) tensor is always valid for a length-2 pad), pad, restore shape.
    """
    if left == 0 and right == 0:
        return x
    x = x.movedim(dim, -1)
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    flat = F.pad(flat, (left, right), mode=mode)
    x = flat.reshape(*shape[:-1], shape[-1] + left + right)
    return x.movedim(-1, dim)


def pad_time(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Replicate-pad the T axis: hold the edge frame, no periodicity/mirror symmetry in time."""
    return _pad_axis(x, _T_DIM, left, right, mode="replicate")


def pad_z(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Replicate-pad the Z axis: apex/basis have no mirror symmetry, so hold the edge value."""
    return _pad_axis(x, _Z_DIM, left, right, mode="replicate")


def pad_phi(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Circular-pad the Phi axis: Phi = -pi === +pi, a true periodic boundary."""
    return _pad_axis(x, _PHI_DIM, left, right, mode="circular")


def pad3d(
    x: torch.Tensor,
    t: tuple[int, int] = (0, 0),
    z: tuple[int, int] = (0, 0),
    phi: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Apply all three axis-specific paddings in sequence (order irrelevant: disjoint dims)."""
    x = pad_time(x, *t)
    x = pad_z(x, *z)
    x = pad_phi(x, *phi)
    return x


def conv_pad(x: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Pad by `k` on every side of every axis -- the ±1 halo a 3x3x3, padding=0 Conv3d needs.

    Use this immediately before every Conv3d(kernel_size=3, padding=0) in the network instead
    of the conv's own (single-mode) padding argument.
    """
    return pad3d(x, t=(k, k), z=(k, k), phi=(k, k))


def _split_pad(size: int, target: int) -> tuple[int, int]:
    """How much to add on (left, right) to grow `size` up to `target`, biased +1 to the right."""
    extra = target - size
    if extra < 0:
        raise ValueError(f"target {target} < size {size}")
    left = extra // 2
    return left, extra - left


def canvas_pad(
    x: torch.Tensor, t_target: int, z_target: int
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad [.., T, Z, Phi] up to [.., t_target, z_target, Phi] once, at the network boundary.

    Phi is never canvas-padded: its size must already be a multiple of every downsample factor
    (60 = 2*2*15, matches the planned (2,2,2),(2,2,2),(2,2,1) stages), so only its per-conv
    circular boundary (conv_pad) is needed, not a resize.
    Returns the padded tensor and (t_left, t_right, z_left, z_right) for canvas_crop to invert.
    """
    t_left, t_right = _split_pad(x.shape[_T_DIM], t_target)
    z_left, z_right = _split_pad(x.shape[_Z_DIM], z_target)
    return pad3d(x, t=(t_left, t_right), z=(z_left, z_right)), (t_left, t_right, z_left, z_right)


def canvas_crop(x: torch.Tensor, offsets: tuple[int, int, int, int]) -> torch.Tensor:
    """Invert canvas_pad: crop back to the original [T, Z] size using the saved offsets."""
    t_left, t_right, z_left, z_right = offsets
    t_slice = slice(t_left, x.shape[_T_DIM] - t_right if t_right else None)
    z_slice = slice(z_left, x.shape[_Z_DIM] - z_right if z_right else None)
    return x[..., t_slice, z_slice, :]

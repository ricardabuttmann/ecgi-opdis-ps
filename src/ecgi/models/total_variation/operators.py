"""Coordinate-aware spatial total variation on the regular ``(Z_ea, phi)`` template.

The sparse operator stacks forward differences in the open apico-basal direction and the
periodic rotational direction. Each derivative is multiplied by the coordinate-cell area, so
the mixed L2,1 norm approximates

    integral sqrt((d x / d Z_ea)^2 + (d x / d phi)^2) d Z_ea d phi.

This is a template-coordinate regularizer, not a physical surface-gradient discretization: the
heart-independent template does not carry a patient-specific three-dimensional metric tensor.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ecgi.mesh_operators.template import Template


def _uniform_step(values: np.ndarray, name: str) -> float:
    """Return and validate the positive spacing of one uniformly sampled coordinate."""
    unique = np.unique(values)
    if unique.size < 2:
        raise ValueError(f"{name} requires at least two distinct coordinates")
    steps = np.diff(unique)
    step = float(steps[0])
    if step <= 0.0 or not np.allclose(steps, step, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} coordinates must be uniformly spaced")
    return step


def template_tv_gradient(template: Template) -> sp.csr_matrix:
    """Build ``K = [K_z; K_phi]`` for area-weighted forward differences.

    ``K_z`` is zero on the last Z row (open boundary). ``K_phi`` wraps the last phi column to
    the first. For ``X`` shaped ``(n_nodes, n_time)``, reshape ``K @ X`` to
    ``(2, n_nodes, n_time)`` before applying the isotropic L2 norm to axis 0.
    """
    if template.size != template.n_z * template.n_phi:
        raise ValueError("template size does not match n_z * n_phi")
    dz = _uniform_step(template.z_ea, "Z_ea")
    dphi = _uniform_step(template.phi, "phi")
    n_z, n_phi, n_nodes = template.n_z, template.n_phi, template.size
    nodes = np.arange(n_nodes)
    iz, iphi = np.divmod(nodes, n_phi)

    # Cell-area * derivative: dz*dphi * (x[z+1] - x[z])/dz = dphi * difference.
    valid_z = iz < n_z - 1
    z_rows = nodes[valid_z]
    z_next = z_rows + n_phi
    K_z = sp.coo_matrix(
        (
            np.concatenate([-np.full(z_rows.size, dphi), np.full(z_rows.size, dphi)]),
            (np.concatenate([z_rows, z_rows]), np.concatenate([z_rows, z_next])),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()

    # Cell-area * derivative in phi; the forward neighbour wraps at +pi -> -pi.
    phi_next = iz * n_phi + (iphi + 1) % n_phi
    K_phi = sp.coo_matrix(
        (
            np.concatenate([-np.full(n_nodes, dz), np.full(n_nodes, dz)]),
            (np.concatenate([nodes, nodes]), np.concatenate([nodes, phi_next])),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    return sp.vstack([K_z, K_phi], format="csr")


def isotropic_total_variation(values: np.ndarray, gradient: sp.spmatrix) -> float:
    """Return the spatial isotropic TV summed over nodes and, if present, time."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("values must have shape (n_nodes,) or (n_nodes, n_time)")
    n_nodes = x.shape[0]
    if gradient.shape != (2 * n_nodes, n_nodes):
        raise ValueError(f"gradient shape must be {(2 * n_nodes, n_nodes)}, got {gradient.shape}")
    components = np.asarray(gradient @ x).reshape(2, n_nodes, x.shape[1])
    return float(np.sum(np.linalg.norm(components, axis=0)))

"""Path-free tensor-grid template compatible with the production sampler interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOUNDARY_EDGE = "edge"
BOUNDARY_CELL = "cell"


@dataclass(frozen=True)
class Template:
    z_ea: np.ndarray
    phi: np.ndarray
    n_z: int
    n_phi: int
    chart_id: str = "harmonic_ea_v1"

    @property
    def boundary(self) -> str:
        return BOUNDARY_CELL

    @property
    def size(self) -> int:
        return int(self.z_ea.size)

    def as_matrix(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values).reshape(self.n_z, self.n_phi)


def build_template(
    n_z: int = 30,
    n_phi: int = 60,
    z_cap: float = 0.03,
    chart: str | None = None,
) -> Template:
    """Build the cell-centred harmonic equal-area lattice used by released checkpoints.

    ``z_ea`` is the equal-area harmonic-chart coordinate, so uniformly spaced cell centres
    ``(arange(n_z) + 0.5) / n_z`` already fall on equal-area apico-basal bands. ``z_cap`` is
    accepted only for call-signature compatibility and is unused here: the apical-cap and
    basal-collar removal is baked into the upstream ``z_ea`` chart definition, not re-applied by
    this reconstruction helper.
    """
    del z_cap
    z = (np.arange(n_z) + 0.5) / n_z
    phi = -np.pi + 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    return Template(zz.ravel(), pp.ravel(), n_z, n_phi, chart or "harmonic_ea_v1")

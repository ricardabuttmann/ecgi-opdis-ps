"""Reconstruction metrics x_hat vs x_true (phi_e), pipeline-agnostic pure functions.

Inputs are plain arrays [n_epi, n_time] (no case/geometry objects). Spatial metrics compare across
nodes at each time step; temporal metrics compare across time at each node.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricResult:
    """Aggregated scalar plus the raw per-slice values it was averaged from.

    `values` lets eval/report.py build distributions/plots (e.g. RE(t) over the QRS
    complex) without metrics.py having to know about cross-case aggregation.
    """

    mean: float
    values: np.ndarray


def _pearson_cc(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    """Pearson correlation between a and b, reduced along `axis`.

    A slice with (near-)zero variance in either input -- e.g. a uniform pre-/post-activation
    frame, which average referencing sends to exactly zero -- has an undefined correlation and
    is returned as nan (not a spurious 0), so the nan-aware means in spatial_cc/temporal_cc drop
    it instead of letting it drag the average toward zero. Threshold is relative to the largest
    slice denominator.
    """
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a**2).sum(axis=axis) * (b**2).sum(axis=axis))
    cc = np.full_like(num, np.nan)
    valid = den > den.max() * 1e-12
    cc[valid] = num[valid] / den[valid]
    return cc


def _safe_nanmean(v: np.ndarray) -> float:
    """Mean over the finite entries; nan if there are none (every slice was degenerate)."""
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


def spatial_cc(x_hat: np.ndarray, x_true: np.ndarray) -> MetricResult:
    """Pearson CC per time step, across nodes -> spatial pattern match at each instant."""
    cc = _pearson_cc(x_hat, x_true, axis=0)  # (n_time,)
    return MetricResult(mean=_safe_nanmean(cc), values=cc)


def temporal_cc(x_hat: np.ndarray, x_true: np.ndarray) -> MetricResult:
    """Pearson CC per node, across time -> waveform shape match at each node."""
    cc = _pearson_cc(x_hat, x_true, axis=1)  # (n_epi,)
    return MetricResult(mean=_safe_nanmean(cc), values=cc)


def relative_error_global(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Global (Frobenius) relative error ||x_hat - x_true||_F / ||x_true||_F over the WHOLE
    [n_epi, n_time] block -- a single amplitude- and pattern-sensitive scalar.

    The single global denominator is robust to near-isoelectric rest frames: those contribute
    negligibly to both numerator and denominator. This is the RE convention reported in the paper.
    """
    den = float(np.linalg.norm(x_true))
    return float(np.linalg.norm(x_hat - x_true) / den) if den > 0 else float("nan")


def activation_time(x: np.ndarray, dt_ms: float) -> np.ndarray:
    """LAT per node (ms): time of steepest downstroke, argmax(-dx/dt) along the time axis.

    dt_ms is a required parameter (not read from any metadata file) to keep this pure and
    pipeline-agnostic; the caller owns the sample's actual time sampling.
    """
    dphidt = np.gradient(x, dt_ms, axis=1)
    lat_idx = np.argmax(-dphidt, axis=1)
    return lat_idx * dt_ms


def activation_time_error(x_hat: np.ndarray, x_true: np.ndarray, dt_ms: float) -> MetricResult:
    """Absolute LAT error per node (ms)."""
    err = np.abs(activation_time(x_hat, dt_ms) - activation_time(x_true, dt_ms))
    return MetricResult(mean=_safe_nanmean(err), values=err)


def _spatial_clusters(points_mm: np.ndarray, link_mm: float) -> np.ndarray:
    """Connected-component labels: nodes closer than link_mm are in the same cluster.

    A radius graph on the points (edge iff Euclidean distance < link_mm) reduced to its
    connected components -- so a set of early-activation nodes splits into the spatially
    separated breakthrough regions it actually forms. Returns an (n,) integer label per point.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = points_mm.shape[0]
    if n == 0:
        return np.empty(0, dtype=int)
    pairs = cKDTree(points_mm).query_pairs(link_mm, output_type="ndarray")
    if len(pairs):
        graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    else:
        graph = csr_matrix((n, n))
    _, labels = connected_components(graph, directed=False)
    return labels


def _early_cluster_centroids(
    x: np.ndarray,
    points_mm: np.ndarray,
    dt_ms: float,
    amp_frac: float,
    window_ms: float,
    link_mm: float,
) -> np.ndarray:
    """Centroids (k, 3) of every spatially separated breakthrough sharing the earliest window.

    Shared core of the localization metric. Three guards turn the raw LAT field into a small set
    of trustworthy breakthrough sites, ordered by increasing LAT of their earliest node:
      - amplitude gate: a node counts only if its downstroke amplitude reaches amp_frac of the
        strongest node, dropping flat/non-activating nodes whose argmax(-dx/dt) lands on a
        meaningless early frame (the LAT=0, amplitude~0 artifact that wrecked a bare argmin).
      - earliest-window: keep gated nodes activating within window_ms of the earliest gated LAT.
      - spatial cluster (link_mm): connected components of the surviving nodes -> the distinct
        breakthrough regions. Two near-simultaneous fronts at different places stay SEPARATE
        centroids instead of being averaged into a midpoint that lies on neither.
    """
    amp = x.max(axis=1) - x.min(axis=1)
    valid = amp >= amp_frac * amp.max()
    lat = np.where(valid, activation_time(x, dt_ms), np.inf)
    early = np.flatnonzero(lat <= lat.min() + window_ms)
    labels = _spatial_clusters(points_mm[early], link_mm)
    n = int(labels.max()) + 1 if labels.size else 0
    order = np.argsort([lat[early][labels == k].min() for k in range(n)])  # earliest cluster first
    return np.array([points_mm[early][labels == k].mean(axis=0) for k in order]).reshape(-1, 3)


def earliest_breakthrough(
    x: np.ndarray,
    points_mm: np.ndarray,
    dt_ms: float,
    amp_frac: float = 0.3,
    window_ms: float = 5.0,
    link_mm: float = 12.0,
) -> np.ndarray:
    """3D centroid (mm) of the single EARLIEST genuine epicardial breakthrough region.

    The stim sites are endocardial/intramural and near-simultaneous, so the epicardium sees a
    SEQUENCE of transmural breakthroughs, not one focus per site; the earliest is the physically
    meaningful landmark (typically the thin-walled RV). Returns the first of
    _early_cluster_centroids -- used as the reconstruction's single estimated breakthrough site.
    """
    return _early_cluster_centroids(x, points_mm, dt_ms, amp_frac, window_ms, link_mm)[0]


def localization_error(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    points_mm: np.ndarray,
    dt_ms: float,
    amp_frac: float = 0.3,
    window_ms: float = 5.0,
    link_mm: float = 12.0,
) -> float:
    """Distance (mm) from x_hat's first breakthrough to the NEAREST of x_true's earliest breakthroughs.

    Localizes the FIRST epicardial breakthrough, not the five endocardial stim sites -- those are
    transmurally hidden and unrecoverable from ~200 surface electrodes; this answers "does the
    reconstruction's first breakthrough land on a true earliest breakthrough (typically the RV)".

    Asymmetric on purpose. x_hat contributes its single estimated site (earliest_breakthrough).
    x_true can legitimately have SEVERAL near-simultaneous earliest breakthroughs (fascicles firing
    together); when two of them tie in time, "the one true first" is a coin flip, so scoring x_hat
    against one fixed pick would inject up to ~40 mm of pure ambiguity. Taking the nearest of
    x_true's early breakthroughs removes that coin-flip -- hitting EITHER simultaneous front counts
    -- while a genuinely misplaced x_hat (true has one breakthrough, x_hat put its first elsewhere)
    is still fully penalized. This is the standard pacing-site-localization convention (estimate ->
    nearest true site).

    The true sites are derived from x_true itself (no simulation metadata), so this stays a pure
    (x_hat, x_true, ...) function. `points_mm` must be node coordinates in millimeters, in the SAME
    node order as x_hat/x_true; combined-mesh .pts are micrometers, so convert (points_um / 1000)
    first. Returns a single float, not a MetricResult: there is no natural per-time or per-node
    slice to report alongside the aggregate for one case.
    """
    site_hat = earliest_breakthrough(x_hat, points_mm, dt_ms, amp_frac, window_ms, link_mm)
    true_sites = _early_cluster_centroids(x_true, points_mm, dt_ms, amp_frac, window_ms, link_mm)
    return float(np.linalg.norm(true_sites - site_hat, axis=1).min())

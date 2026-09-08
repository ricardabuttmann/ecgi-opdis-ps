"""Small, dependency-light readers for openCARP signal extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_vtx(path: str | Path) -> np.ndarray:
    """Read an openCARP VTX node set and validate its count."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        try:
            count = int(handle.readline())
        except ValueError as error:
            raise ValueError(f"{path}: first line must be an integer count") from error
        domain = handle.readline().strip()
        if domain not in {"extra", "intra"}:
            raise ValueError(f"{path}: second line must be 'extra' or 'intra'")
        indices = np.loadtxt(handle, dtype=np.int64, ndmin=1)
    if indices.size != count:
        raise ValueError(f"{path}: header says {count} nodes but read {indices.size}")
    if np.any(indices < 0):
        raise ValueError(f"{path}: node indices must be non-negative")
    return indices


def read_igb(path: str | Path) -> tuple[np.ndarray, dict[str, str]]:
    """Read a scalar openCARP IGB file as ``[time, node]``."""
    path = Path(path)
    with path.open("rb") as handle:
        header_text = handle.read(1024).decode("ascii", errors="replace").split("\x0c", 1)[0]
        raw = handle.read()
    header = {}
    for token in header_text.replace("\r", " ").replace("\n", " ").split():
        if ":" in token:
            key, value = token.split(":", 1)
            header[key] = value
    try:
        n_nodes = int(header["x"]) * int(header.get("y", 1)) * int(header.get("z", 1))
        n_time = int(header["t"])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: invalid IGB dimensions") from error
    dtypes = {"float": "<f4", "double": "<f8", "short": "<i2", "int": "<i4"}
    data_type = header.get("type", "float")
    if data_type not in dtypes:
        raise ValueError(f"{path}: unsupported IGB type {data_type!r}")
    array = np.frombuffer(raw, dtype=dtypes[data_type])
    if array.size != n_nodes * n_time:
        raise ValueError(f"{path}: {array.size} values != x*t = {n_nodes}*{n_time}")
    return array.reshape(n_time, n_nodes), header


def extract_hsp(
    phie_path: str | Path,
    epicardial_vertices: str | Path,
) -> tuple[np.ndarray, dict[str, str]]:
    """Extract average-referenced epicardial HSP from one simulation."""
    phie, header = read_igb(phie_path)
    epicardial = read_vtx(epicardial_vertices)
    highest = int(epicardial.max())
    if highest >= phie.shape[1]:
        raise ValueError(
            f"surface node {highest} is out of range for {phie.shape[1]} simulated nodes"
        )
    hsp = phie[:, epicardial].T.astype(np.float32)
    hsp -= hsp.mean(axis=0, keepdims=True)
    if not np.isfinite(hsp).all():
        raise ValueError("extracted HSP contains non-finite values")
    return hsp, header

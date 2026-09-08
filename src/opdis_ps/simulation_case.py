"""Strict loading and verification of a prepared openCARP case bundle."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_FILE_ROLES = {
    "mesh_points",
    "mesh_elements",
    "mesh_fibers",
    "parameter_file",
    "ground_vertices",
    "epicardial_vertices",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FileRecord:
    """One path-independent file identity from the case manifest."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class StimulusRecord:
    """One already prepared stimulus site."""

    name: str
    file_role: str
    start_ms: float


@dataclass(frozen=True)
class PreparedSimulationCase:
    """Verified, immutable input bundle for one pseudo-bidomain run."""

    root: Path
    case_id: str
    seed: int
    files: dict[str, FileRecord]
    stimuli: tuple[StimulusRecord, ...]
    expected: dict[str, int]

    def path(self, role: str) -> Path:
        try:
            record = self.files[role]
        except KeyError as error:
            raise KeyError(f"case manifest has no file role {role!r}") from error
        return self.root / PurePosixPath(record.path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{field} must be a normalized path inside the case bundle")
    return value


def _file_record(value: Any, *, role: str) -> FileRecord:
    if not isinstance(value, dict) or set(value) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"files.{role} must contain exactly path, size_bytes, and sha256")
    path = _relative_path(value["path"], field=f"files.{role}.path")
    size_bytes = value["size_bytes"]
    digest = value["sha256"]
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 1:
        raise ValueError(f"files.{role}.size_bytes must be a positive integer")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"files.{role}.sha256 must be a lowercase SHA-256 digest")
    return FileRecord(path=path, size_bytes=size_bytes, sha256=digest)


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_prepared_simulation_case(
    case_dir: str | Path, *, verify_hashes: bool = True
) -> PreparedSimulationCase:
    """Load ``case_manifest.json`` and verify every declared input byte."""
    root = Path(case_dir).resolve()
    manifest_path = root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"prepared simulation bundle not found: {manifest_path}. Download the OSF bundle and "
            "extract it under artifacts/ as described in the README before running this command"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("case_manifest.json must use schema_version 1")
    case_id = manifest.get("case_id")
    if not isinstance(case_id, str) or not case_id or "/" in case_id or "\\" in case_id:
        raise ValueError("case_id must be a non-empty anonymous identifier")
    seed = _positive_int(manifest.get("seed"), field="seed")

    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict):
        raise ValueError("files must be an object keyed by semantic role")
    missing_roles = sorted(REQUIRED_FILE_ROLES - set(raw_files))
    if missing_roles:
        raise ValueError(f"case manifest is missing file roles: {missing_roles}")
    files = {role: _file_record(value, role=role) for role, value in raw_files.items()}
    paths = [record.path for record in files.values()]
    if len(paths) != len(set(paths)):
        raise ValueError("each manifest file role must refer to a distinct path")

    raw_stimuli = manifest.get("stimuli")
    if not isinstance(raw_stimuli, list) or not raw_stimuli:
        raise ValueError("stimuli must be a non-empty list")
    stimuli = []
    names = set()
    for index, value in enumerate(raw_stimuli):
        if not isinstance(value, dict) or set(value) != {"name", "file_role", "start_ms"}:
            raise ValueError(
                f"stimuli[{index}] must contain exactly name, file_role, and start_ms"
            )
        name = value["name"]
        role = value["file_role"]
        start_ms = value["start_ms"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"stimuli[{index}].name must be non-empty and unique")
        if not isinstance(role, str) or role not in files:
            raise ValueError(f"stimuli[{index}].file_role must name a declared file")
        if (
            not isinstance(start_ms, (int, float))
            or isinstance(start_ms, bool)
            or not math.isfinite(start_ms)
            or start_ms < 0
        ):
            raise ValueError(f"stimuli[{index}].start_ms must be finite and non-negative")
        names.add(name)
        stimuli.append(StimulusRecord(name=name, file_role=role, start_ms=float(start_ms)))

    raw_expected = manifest.get("expected")
    required_expected = {
        "volume_nodes",
        "volume_elements",
        "epicardial_nodes",
        "time_frames",
    }
    optional_expected = {"torso_nodes"}
    if (
        not isinstance(raw_expected, dict)
        or not required_expected.issubset(raw_expected)
        or not set(raw_expected).issubset(required_expected | optional_expected)
    ):
        raise ValueError(
            f"expected must contain {sorted(required_expected)} and may contain torso_nodes"
        )
    expected = {
        key: _positive_int(value, field=f"expected.{key}")
        for key, value in raw_expected.items()
    }
    if "torso_nodes" in expected and "torso_vertices" not in files:
        raise ValueError("files.torso_vertices is required when expected.torso_nodes is declared")

    for role, record in files.items():
        path = (root / PurePosixPath(record.path)).resolve()
        if root not in path.parents:
            raise ValueError(f"files.{role}.path resolves outside the case bundle")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record.size_bytes:
            raise ValueError(
                f"{record.path}: size {path.stat().st_size} != manifest {record.size_bytes}"
            )
        if verify_hashes:
            actual = sha256_file(path)
            if actual != record.sha256:
                raise ValueError(f"{record.path}: SHA-256 {actual} != manifest {record.sha256}")

    case = PreparedSimulationCase(
        root=root,
        case_id=case_id,
        seed=seed,
        files=files,
        stimuli=tuple(stimuli),
        expected=expected,
    )
    _verify_open_carp_headers(case)
    return case


def _first_integer(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        first = handle.readline().strip()
    try:
        return int(first)
    except ValueError as error:
        raise ValueError(f"{path}: first line must be an integer count") from error


def _verify_open_carp_headers(case: PreparedSimulationCase) -> None:
    observed = {
        "volume_nodes": _first_integer(case.path("mesh_points")),
        "volume_elements": _first_integer(case.path("mesh_elements")),
        "epicardial_nodes": _first_integer(case.path("epicardial_vertices")),
    }
    if "torso_vertices" in case.files:
        if "torso_nodes" not in case.expected:
            raise ValueError("expected.torso_nodes is required when torso_vertices is declared")
        observed["torso_nodes"] = _first_integer(case.path("torso_vertices"))
    for field, value in observed.items():
        if value != case.expected[field]:
            raise ValueError(
                f"{field}: file header {value} != manifest expected {case.expected[field]}"
            )

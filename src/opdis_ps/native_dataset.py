"""Indexed, representation-preserving access to ``native-ecgi-v1``.

The loader resolves public identities and slices the released NumPy shards.  It never projects a
native signal to the paper grid or another template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opdis_ps.dataset import _read_json, _read_jsonl, _sha256


@dataclass(frozen=True)
class NativeHeartSurface:
    heart_id: str
    points_mm: np.ndarray
    faces: np.ndarray
    original_node_id: np.ndarray
    has_incident_face: np.ndarray


@dataclass(frozen=True)
class NativeTorsoSurface:
    torso_id: str
    points_mm: np.ndarray
    faces: np.ndarray


@dataclass(frozen=True)
class NativePlacement:
    geometry_id: str
    heart_to_torso: np.ndarray
    electrode_xyz_mm: np.ndarray
    electrode_slot_id: np.ndarray


@dataclass(frozen=True)
class NativeCase:
    case_id: str
    geometry_id: str
    heart_id: str
    torso_id: str
    partition: str
    population: str
    hsp: np.ndarray
    time_ms: np.ndarray
    heart: NativeHeartSurface
    operator_id: str | None = None
    operator: np.ndarray | None = None
    placement: NativePlacement | None = None
    torso: NativeTorsoSurface | None = None

    @property
    def heart_points_mm(self) -> np.ndarray:
        return self.heart.points_mm

    @property
    def heart_faces(self) -> np.ndarray:
        return self.heart.faces

    def synthesize_bsp(self) -> np.ndarray:
        """Return the exactly compatible nominal measurement ``A_native @ hsp``."""

        if self.operator is None:
            raise ValueError("load the case with include_operator=True before synthesizing BSP")
        return self.operator @ self.hsp


def _rows_by_id(path: Path, identity: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    indexed = {str(row[identity]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"{path}: duplicate {identity}")
    return indexed


def _safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{root}: unsafe or missing payload {relative!r}")
    return path


class _IndexedFiles:
    def __init__(self, directory: Path, identity: str) -> None:
        self.directory = directory
        self.identity = identity
        manifest = _read_json(directory / "manifest.json")
        index_record = manifest.get("index", {})
        index_path = _safe_file(directory, str(index_record.get("path", "index.jsonl")))
        if _sha256(index_path) != index_record.get("sha256"):
            raise ValueError(f"{index_path}: digest differs from component manifest")
        self.rows = _rows_by_id(index_path, identity)

    def payload(self, identity: str) -> tuple[Path, dict[str, Any]]:
        try:
            row = self.rows[identity]
        except KeyError as exc:
            raise KeyError(f"{self.directory}: unknown {self.identity} {identity!r}") from exc
        path = _safe_file(self.directory, str(row["path"]))
        if path.stat().st_size != int(row["size_bytes"]) or _sha256(path) != row["sha256"]:
            raise ValueError(f"{path}: payload differs from component index")
        return path, row

    def complete(self) -> bool:
        return all((self.directory / str(row["path"])).is_file() for row in self.rows.values())


class _ConcatenatedShards:
    def __init__(self, directory: Path, identity: str, axis: int) -> None:
        self.directory = directory
        self.identity = identity
        self.axis = axis
        self.manifest = _read_json(directory / "manifest.json")
        index_record = self.manifest.get("index", {})
        index_path = _safe_file(directory, str(index_record.get("path", "index.jsonl")))
        if _sha256(index_path) != index_record.get("sha256"):
            raise ValueError(f"{index_path}: digest differs from component manifest")
        self.rows = _rows_by_id(index_path, identity)
        self._arrays: dict[str, np.ndarray] = {}

    def get(self, identity: str) -> np.ndarray:
        try:
            row = self.rows[identity]
        except KeyError as exc:
            raise KeyError(f"{self.directory}: unknown {self.identity} {identity!r}") from exc
        shard_name = str(row["shard"])
        if shard_name not in self._arrays:
            path = _safe_file(self.directory, shard_name)
            self._arrays[shard_name] = np.load(path, mmap_mode="r", allow_pickle=False)
        values = self._arrays[shard_name]
        if self.axis == 0:
            result = values[int(row["row_start"]) : int(row["row_stop"])]
        else:
            result = values[:, int(row["column_start"]) : int(row["column_stop"])]
        expected = tuple(int(value) for value in row["shape"])
        if result.shape != expected or result.dtype != np.dtype("<f4"):
            raise ValueError(
                f"{self.directory / shard_name}: {identity} has {result.shape}/{result.dtype}, "
                f"expected {expected}/float32"
            )
        return np.asarray(result)

    def complete(self) -> bool:
        records = self.manifest.get("shards", [])
        return bool(records) and all(
            (self.directory / str(record["path"])).is_file() for record in records
        )


class NativeDataset:
    """Resolve native signals, surfaces, placements and optional operators by public ID."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("package_id") != "native-ecgi-v1":
            raise ValueError(f"{manifest_path}: expected package_id 'native-ecgi-v1'")
        if self.manifest.get("direct_equation") != "bsp = A_native @ hsp":
            raise ValueError(f"{manifest_path}: unexpected native measurement equation")
        self.release_version = str(self.manifest.get("release_version", ""))
        self.manifest_sha256 = _sha256(manifest_path)

        self.cases = _rows_by_id(self.root / "cases.jsonl", "case_id")
        self.geometries = _rows_by_id(self.root / "geometries.jsonl", "geometry_id")
        self.hearts = _rows_by_id(self.root / "hearts.jsonl", "heart_id")
        self.torsos = _rows_by_id(self.root / "torsos.jsonl", "torso_id")
        self.operators = _rows_by_id(self.root / "operators.jsonl", "operator_id")
        self.operator_by_geometry: dict[str, str] = {}
        for operator_id, row in self.operators.items():
            geometry_id = str(row["geometry_id"])
            if geometry_id in self.operator_by_geometry:
                raise ValueError(f"operators.jsonl: duplicate operator for {geometry_id}")
            self.operator_by_geometry[geometry_id] = operator_id

        self._heart_files = self._optional_files("geometry/hearts", "heart_id")
        self._torso_files = self._optional_files("geometry/torsos", "torso_id")
        self._placement_files = self._optional_files("geometry/placements", "geometry_id")
        self._hsp = self._optional_shards("signals/hsp", "case_id", axis=0)
        self._native_operators = self._optional_shards(
            "operators/native", "operator_id", axis=1
        )
        self.validate_structure()

    def _optional_files(self, relative: str, identity: str) -> _IndexedFiles | None:
        directory = self.root / relative
        return _IndexedFiles(directory, identity) if (directory / "manifest.json").is_file() else None

    def _optional_shards(
        self, relative: str, identity: str, *, axis: int
    ) -> _ConcatenatedShards | None:
        directory = self.root / relative
        return (
            _ConcatenatedShards(directory, identity, axis)
            if (directory / "manifest.json").is_file()
            else None
        )

    @property
    def available_components(self) -> dict[str, bool]:
        heart_complete = self._heart_files is not None and self._heart_files.complete()
        torso_complete = (
            self._torso_files is not None
            and self._placement_files is not None
            and self._torso_files.complete()
            and self._placement_files.complete()
        )
        return {
            "metadata": True,
            "heart-surfaces": heart_complete,
            "torso-geometry": torso_complete,
            "native-hsp": self._hsp is not None and self._hsp.complete(),
            "operators-native": (
                self._native_operators is not None and self._native_operators.complete()
            ),
        }

    def validate_structure(self) -> dict[str, Any]:
        """Validate metadata joins and all indexes belonging to present components."""

        for case_id, case in self.cases.items():
            geometry_id = str(case["geometry_id"])
            geometry = self.geometries.get(geometry_id)
            if geometry is None:
                raise ValueError(f"{case_id}: unknown geometry_id {geometry_id!r}")
            for key in ("heart_id", "torso_id"):
                if str(case[key]) != str(geometry[key]):
                    raise ValueError(f"{case_id}: {key} differs from {geometry_id}")
        for geometry_id, geometry in self.geometries.items():
            if str(geometry["heart_id"]) not in self.hearts:
                raise ValueError(f"{geometry_id}: unknown heart_id")
            if str(geometry["torso_id"]) not in self.torsos:
                raise ValueError(f"{geometry_id}: unknown torso_id")
            if geometry_id not in self.operator_by_geometry:
                raise ValueError(f"{geometry_id}: no released native operator metadata")
        for operator_id, operator in self.operators.items():
            geometry_id = str(operator["geometry_id"])
            geometry = self.geometries.get(geometry_id)
            if geometry is None:
                raise ValueError(f"{operator_id}: unknown geometry_id {geometry_id!r}")
            for key in ("heart_id", "torso_id"):
                if str(operator[key]) != str(geometry[key]):
                    raise ValueError(f"{operator_id}: {key} differs from {geometry_id}")
            if operator.get("forward_spec") != "nominal":
                raise ValueError(f"{operator_id}: expected nominal native operator")
        expected = {
            self._heart_files: set(self.hearts),
            self._torso_files: set(self.torsos),
            self._placement_files: set(self.geometries),
            self._hsp: set(self.cases),
            self._native_operators: set(self.operators),
        }
        for component, identities in expected.items():
            if component is not None and set(component.rows) != identities:
                raise ValueError(
                    f"{component.directory}: component index does not exactly cover metadata"
                )
        return {
            "status": "ok",
            "package_id": "native-ecgi-v1",
            "case_count": len(self.cases),
            "geometry_count": len(self.geometries),
            "heart_count": len(self.hearts),
            "torso_count": len(self.torsos),
            "operator_count": len(self.operators),
            "components": self.available_components,
        }

    @staticmethod
    def _require(component: Any, name: str) -> Any:
        if component is None:
            raise FileNotFoundError(f"native-ecgi-v1 component {name!r} is not installed")
        return component

    def load_heart(self, heart_id: str) -> NativeHeartSurface:
        files = self._require(self._heart_files, "heart-surfaces")
        path, _ = files.payload(heart_id)
        with np.load(path, allow_pickle=False) as arrays:
            result = NativeHeartSurface(
                heart_id=heart_id,
                points_mm=np.asarray(arrays["points"], dtype=np.float32),
                faces=np.asarray(arrays["faces"], dtype=np.int32),
                original_node_id=np.asarray(arrays["original_node_id"], dtype=np.int64),
                has_incident_face=np.asarray(arrays["has_incident_face"], dtype=bool),
            )
        if result.points_mm.ndim != 2 or result.points_mm.shape[1] != 3:
            raise ValueError(f"{path}: invalid heart points")
        if result.faces.ndim != 2 or result.faces.shape[1] != 3:
            raise ValueError(f"{path}: invalid heart faces")
        if result.original_node_id.shape != (len(result.points_mm),):
            raise ValueError(f"{path}: original_node_id does not match points")
        if result.has_incident_face.shape != (len(result.points_mm),):
            raise ValueError(f"{path}: has_incident_face does not match points")
        return result

    def load_torso(self, torso_id: str) -> NativeTorsoSurface:
        files = self._require(self._torso_files, "torso-geometry")
        path, _ = files.payload(torso_id)
        with np.load(path, allow_pickle=False) as arrays:
            result = NativeTorsoSurface(
                torso_id=torso_id,
                points_mm=np.asarray(arrays["points"], dtype=np.float32),
                faces=np.asarray(arrays["faces"], dtype=np.int32),
            )
        if result.points_mm.ndim != 2 or result.points_mm.shape[1] != 3:
            raise ValueError(f"{path}: invalid torso points")
        if result.faces.ndim != 2 or result.faces.shape[1] != 3:
            raise ValueError(f"{path}: invalid torso faces")
        return result

    def load_placement(self, geometry_id: str) -> NativePlacement:
        files = self._require(self._placement_files, "torso-geometry")
        path, _ = files.payload(geometry_id)
        with np.load(path, allow_pickle=False) as arrays:
            result = NativePlacement(
                geometry_id=geometry_id,
                heart_to_torso=np.asarray(arrays["heart_to_torso"], dtype=np.float64),
                electrode_xyz_mm=np.asarray(arrays["electrode_xyz"], dtype=np.float32),
                electrode_slot_id=np.asarray(arrays["electrode_slot_id"], dtype=np.int32),
            )
        if result.heart_to_torso.shape != (4, 4):
            raise ValueError(f"{path}: heart_to_torso must have shape (4, 4)")
        if result.electrode_xyz_mm.shape != (176, 3):
            raise ValueError(f"{path}: electrode_xyz must have shape (176, 3)")
        if result.electrode_slot_id.shape != (176,):
            raise ValueError(f"{path}: electrode_slot_id must have shape (176,)")
        return result

    def load_hsp(self, case_id: str) -> np.ndarray:
        field = self._require(self._hsp, "native-hsp")
        return field.get(case_id)

    def load_operator(self, operator_id: str) -> np.ndarray:
        field = self._require(self._native_operators, "operators-native")
        return field.get(operator_id)

    def load_case(
        self,
        case_id: str,
        *,
        include_operator: bool = False,
        include_torso_geometry: bool = False,
    ) -> NativeCase:
        try:
            metadata = self.cases[case_id]
        except KeyError as exc:
            raise KeyError(f"unknown case_id {case_id!r}") from exc
        geometry_id = str(metadata["geometry_id"])
        heart_id = str(metadata["heart_id"])
        torso_id = str(metadata["torso_id"])
        hsp = self.load_hsp(case_id)
        heart = self.load_heart(heart_id)
        if hsp.shape != (len(heart.points_mm), 91):
            raise ValueError(f"{case_id}: HSP shape does not match released heart surface")

        operator_id = self.operator_by_geometry[geometry_id] if include_operator else None
        operator = self.load_operator(operator_id) if operator_id is not None else None
        if operator is not None and operator.shape != (176, hsp.shape[0]):
            raise ValueError(f"{case_id}: operator shape does not match native HSP")

        placement = (
            self.load_placement(geometry_id)
            if include_operator or include_torso_geometry
            else None
        )
        torso = self.load_torso(torso_id) if include_torso_geometry else None
        dt_ms = float(self._require(self._hsp, "native-hsp").manifest["dt_ms"])
        return NativeCase(
            case_id=case_id,
            geometry_id=geometry_id,
            heart_id=heart_id,
            torso_id=torso_id,
            partition=str(metadata["partition"]),
            population=str(metadata["population"]),
            hsp=hsp,
            time_ms=np.arange(hsp.shape[1], dtype=np.float64) * dt_ms,
            heart=heart,
            operator_id=operator_id,
            operator=operator,
            placement=placement,
            torso=torso,
        )

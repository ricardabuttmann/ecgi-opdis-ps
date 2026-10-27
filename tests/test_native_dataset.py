from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from opdis_ps.native_dataset import NativeDataset


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _file_component(directory: Path, identity: str, row: dict) -> None:
    index = directory / "index.jsonl"
    _jsonl(index, [row])
    _json(
        directory / "manifest.json",
        {
            "schema_version": 1,
            "record_count": 1,
            "index": {"path": index.name, "sha256": _sha256(index)},
        },
    )


def _native_package(root: Path) -> tuple[np.ndarray, np.ndarray]:
    root.mkdir()
    case_id = "healthy-test-p0001-r00"
    geometry_id = "geometry-0001"
    heart_id = "heart-0001"
    torso_id = "torso-001"
    operator_id = "geometry-0001-native"
    _json(
        root / "manifest.json",
        {
            "package_id": "native-ecgi-v1",
            "release_version": "test",
            "direct_equation": "bsp = A_native @ hsp",
        },
    )
    _jsonl(
        root / "cases.jsonl",
        [
            {
                "case_id": case_id,
                "geometry_id": geometry_id,
                "heart_id": heart_id,
                "torso_id": torso_id,
                "partition": "test",
                "population": "healthy",
            }
        ],
    )
    _jsonl(
        root / "geometries.jsonl",
        [
            {
                "geometry_id": geometry_id,
                "heart_id": heart_id,
                "torso_id": torso_id,
                "electrode_count": 176,
            }
        ],
    )
    _jsonl(root / "hearts.jsonl", [{"heart_id": heart_id}])
    _jsonl(root / "torsos.jsonl", [{"torso_id": torso_id}])
    _jsonl(
        root / "torso-ct-sources.jsonl",
        [
            {
                "torso_id": torso_id,
                "source_dataset": {
                    "name": "Dataset with segmentations of 117 important anatomical structures in 1228 CT images",
                    "version": "2.0.1",
                    "doi": "10.5281/zenodo.10047292",
                },
                "source_subject_id": "s0574",
                "ct_relative_path": "s0574/ct.nii.gz",
                "segmentations_relative_path": "s0574/segmentations",
                "released_coordinate_system": "source CT physical LPS coordinates in millimetres",
            }
        ],
    )
    _jsonl(
        root / "lung-adjustments.jsonl",
        [
            {
                "schema_version": 1,
                "geometry_id": geometry_id,
                "method": "radial push away from the placed epicardial surface bounding-box center",
                "radial_origin_mm": [1.0, 2.0, 3.0],
                "coordinate_system": "source CT physical LPS coordinates in millimetres",
                "lung_left": {
                    "radial_push_mm": 3.0,
                    "epicardial_vertices_inside_before": 2,
                    "surface_triangle_intersections_before": 1,
                    "relative_radial_deformation_pct": 2.0,
                },
                "lung_right": {
                    "radial_push_mm": 0.0,
                    "epicardial_vertices_inside_before": 0,
                    "surface_triangle_intersections_before": 0,
                    "relative_radial_deformation_pct": 0.0,
                },
                "clearance_sufficient": True,
            }
        ],
    )
    _jsonl(
        root / "operators.jsonl",
        [
            {
                "operator_id": operator_id,
                "geometry_id": geometry_id,
                "heart_id": heart_id,
                "torso_id": torso_id,
                "forward_spec": "nominal",
            }
        ],
    )

    hearts = root / "geometry/hearts"
    torsos = root / "geometry/torsos"
    placements = root / "geometry/placements"
    lungs = root / "geometry/lungs"
    for directory in (hearts, torsos, placements, lungs):
        directory.mkdir(parents=True)
    heart_file = hearts / f"{heart_id}.npz"
    np.savez_compressed(
        heart_file,
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4"),
        faces=np.asarray([[0, 1, 2]], dtype="<i4"),
        original_node_id=np.asarray([5, 6, 7], dtype="<i8"),
        has_incident_face=np.ones(3, dtype=bool),
    )
    _file_component(
        hearts,
        "heart_id",
        {
            "heart_id": heart_id,
            "path": heart_file.name,
            "size_bytes": heart_file.stat().st_size,
            "sha256": _sha256(heart_file),
        },
    )
    torso_file = torsos / f"{torso_id}.npz"
    np.savez_compressed(
        torso_file,
        points=np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype="<f4"),
        faces=np.asarray([[0, 1, 2]], dtype="<i4"),
    )
    _file_component(
        torsos,
        "torso_id",
        {
            "torso_id": torso_id,
            "path": torso_file.name,
            "size_bytes": torso_file.stat().st_size,
            "sha256": _sha256(torso_file),
        },
    )
    lung_file = lungs / f"{torso_id}.npz"
    lung_left_points = np.asarray([[2, 2, 3], [1, 3, 3], [1, 2, 4]], dtype="<f8")
    lung_right_points = np.asarray([[0, 2, 3], [1, 1, 3], [1, 2, 2]], dtype="<f8")
    np.savez_compressed(
        lung_file,
        lung_left_points=lung_left_points,
        lung_left_faces=np.asarray([[0, 1, 2]], dtype="<i4"),
        lung_right_points=lung_right_points,
        lung_right_faces=np.asarray([[0, 1, 2]], dtype="<i4"),
    )
    _file_component(
        lungs,
        "torso_id",
        {
            "torso_id": torso_id,
            "path": lung_file.name,
            "size_bytes": lung_file.stat().st_size,
            "sha256": _sha256(lung_file),
            "coordinate_system": "source CT physical LPS coordinates in millimetres",
            "lung_left": {"point_count": 3, "face_count": 1},
            "lung_right": {"point_count": 3, "face_count": 1},
        },
    )
    placement_file = placements / f"{geometry_id}.npz"
    np.savez_compressed(
        placement_file,
        heart_to_torso=np.eye(4, dtype="<f8"),
        electrode_xyz=np.zeros((176, 3), dtype="<f4"),
        electrode_slot_id=np.arange(176, dtype="<i4"),
    )
    _file_component(
        placements,
        "geometry_id",
        {
            "geometry_id": geometry_id,
            "path": placement_file.name,
            "size_bytes": placement_file.stat().st_size,
            "sha256": _sha256(placement_file),
        },
    )

    hsp = np.arange(3 * 91, dtype="<f4").reshape(3, 91)
    hsp -= hsp.mean(axis=0, keepdims=True)
    hsp_dir = root / "signals/hsp"
    hsp_dir.mkdir(parents=True)
    hsp_file = hsp_dir / "hsp-00000.npy"
    np.save(hsp_file, hsp, allow_pickle=False)
    hsp_index = hsp_dir / "index.jsonl"
    _jsonl(
        hsp_index,
        [
            {
                "case_id": case_id,
                "geometry_id": geometry_id,
                "heart_id": heart_id,
                "shard": hsp_file.name,
                "row_start": 0,
                "row_stop": 3,
                "shape": [3, 91],
            }
        ],
    )
    _json(
        hsp_dir / "manifest.json",
        {
            "schema_version": 1,
            "dt_ms": 5.0,
            "index": {"path": hsp_index.name, "sha256": _sha256(hsp_index)},
            "shards": [{"path": hsp_file.name}],
        },
    )

    operator = np.arange(176 * 3, dtype=np.float64).reshape(176, 3)
    operator -= operator.mean(axis=0, keepdims=True)
    operator = operator.astype("<f4")
    operator_dir = root / "operators/native"
    operator_dir.mkdir(parents=True)
    operator_file = operator_dir / "operators-00000.npy"
    np.save(operator_file, operator, allow_pickle=False)
    operator_index = operator_dir / "index.jsonl"
    _jsonl(
        operator_index,
        [
            {
                "operator_id": operator_id,
                "geometry_id": geometry_id,
                "shard": operator_file.name,
                "column_start": 0,
                "column_stop": 3,
                "shape": [176, 3],
            }
        ],
    )
    _json(
        operator_dir / "manifest.json",
        {
            "schema_version": 1,
            "index": {"path": operator_index.name, "sha256": _sha256(operator_index)},
            "shards": [{"path": operator_file.name}],
        },
    )
    return hsp, operator


def test_native_dataset_loads_joined_case_and_synthesizes_bsp(tmp_path: Path) -> None:
    expected_hsp, expected_operator = _native_package(tmp_path / "native-ecgi-v1")
    dataset = NativeDataset(tmp_path / "native-ecgi-v1")

    assert all(dataset.available_components.values())
    case = dataset.load_case(
        "healthy-test-p0001-r00",
        include_operator=True,
        include_torso_geometry=True,
    )

    np.testing.assert_array_equal(case.hsp, expected_hsp)
    np.testing.assert_array_equal(case.operator, expected_operator)
    np.testing.assert_array_equal(case.synthesize_bsp(), expected_operator @ expected_hsp)
    assert case.heart_points_mm.shape == (3, 3)
    assert case.heart_faces.shape == (1, 3)
    assert case.placement is not None
    assert case.torso is not None
    assert case.torso_source.source_subject_id == "s0574"
    assert case.torso_source.ct_relative_path == "s0574/ct.nii.gz"
    assert case.torso_source.source_dataset_doi == "10.5281/zenodo.10047292"
    assert case.lung_adjustment.lung_left.radial_push_mm == 3.0
    assert case.lung_adjustment.lung_right.radial_push_mm == 0.0
    np.testing.assert_array_equal(case.lung_adjustment.radial_origin_mm, [1.0, 2.0, 3.0])
    assert case.lung_adjustment.clearance_sufficient is True
    lungs = dataset.load_lungs(case.torso_id)
    adjusted = dataset.load_adjusted_lungs(case.geometry_id)
    np.testing.assert_array_equal(lungs.lung_left.points_mm[0], [2.0, 2.0, 3.0])
    np.testing.assert_array_equal(adjusted.lung_left.points_mm[0], [5.0, 2.0, 3.0])
    np.testing.assert_array_equal(adjusted.lung_right.points_mm, lungs.lung_right.points_mm)
    np.testing.assert_array_equal(adjusted.lung_left.faces, lungs.lung_left.faces)
    np.testing.assert_array_equal(case.time_ms, np.arange(91) * 5.0)


def test_native_dataset_allows_signal_only_installation(tmp_path: Path) -> None:
    root = tmp_path / "native-ecgi-v1"
    _native_package(root)
    shutil.rmtree(root / "operators/native")

    dataset = NativeDataset(root)
    assert not dataset.available_components["operators-native"]
    case = dataset.load_case("healthy-test-p0001-r00")
    assert case.operator is None
    with pytest.raises(FileNotFoundError, match="operators-native"):
        dataset.load_case("healthy-test-p0001-r00", include_operator=True)


def test_native_dataset_rejects_mismatched_case_geometry(tmp_path: Path) -> None:
    root = tmp_path / "native-ecgi-v1"
    _native_package(root)
    row = json.loads((root / "cases.jsonl").read_text())
    row["heart_id"] = "heart-does-not-match"
    _jsonl(root / "cases.jsonl", [row])

    with pytest.raises(ValueError, match="heart_id differs"):
        NativeDataset(root)

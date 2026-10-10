from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from opdis_ps.dataset import PaperDataset, add_paper_noise, validate_package_checksums


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _field(
    root: Path, *, identity: str, identity_value: str, name: str, values: np.ndarray
) -> None:
    root.mkdir(parents=True)
    shard = root / f"{name}-00000.npy"
    np.save(shard, values[None].astype(np.float32), allow_pickle=False)
    index = root / "index.jsonl"
    _jsonl(index, [{identity: identity_value, "shard": shard.name, "offset": 0}])
    _json(
        root / "manifest.json",
        {
            "index": {
                "path": index.name,
                "size_bytes": index.stat().st_size,
                "sha256": _sha256(index),
            }
        },
    )


def _package(tmp_path: Path) -> tuple[Path, np.ndarray]:
    root = tmp_path / "paper-evaluation-v1"
    root.mkdir()
    case_id = "healthy-test-p0001-r00"
    geometry_id = "geometry-0001"
    rng = np.random.default_rng(4)
    forward = rng.normal(size=(3, 4))
    forward -= forward.mean(axis=0, keepdims=True)
    truth = rng.normal(size=(4, 5))
    clean = forward @ truth

    _json(root / "manifest.json", {"package_id": "paper-evaluation-v1"})
    _jsonl(
        root / "cases.jsonl",
        [
            {
                "case_id": case_id,
                "geometry_id": geometry_id,
                "measurement_specs": ["nominal", "conductivity", "subcut_fat_mild"],
            }
        ],
    )
    _jsonl(
        root / "noise.jsonl",
        [
            {
                "case_id": case_id,
                "kind": "awgn_iid",
                "base_seed": 20_260_722,
                "noise_seed": 12345,
            }
        ],
    )
    _jsonl(
        root / "operators.jsonl",
        [{"operator_id": f"{geometry_id}-nominal", "geometry_id": geometry_id}],
    )
    _field(root / "hsp", identity="case_id", identity_value=case_id, name="truth", values=truth)
    for spec in ("nominal", "conductivity", "subcut_fat_mild"):
        _field(
            root / "measurements" / spec.replace("_", "-"),
            identity="case_id",
            identity_value=case_id,
            name="measurement",
            values=clean,
        )
    _field(
        root / "operators" / "nominal",
        identity="operator_id",
        identity_value=f"{geometry_id}-nominal",
        name="operator",
        values=forward,
    )

    geometry = root / "geometry" / "template-surface.npz"
    geometry.parent.mkdir()
    laplacian = sp.eye(4, format="csr")
    np.savez(
        geometry,
        points_mm=np.zeros((4, 3)),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        template_z=np.asarray([0.25, 0.25, 0.75, 0.75]),
        template_phi=np.asarray([-np.pi, 0.0, -np.pi, 0.0]),
        template_shape=np.asarray([2, 2]),
        time_ms=np.arange(5) * 5.0,
        grid_weights=np.full(4, 0.25),
        laplacian_data=laplacian.data,
        laplacian_indices=laplacian.indices,
        laplacian_indptr=laplacian.indptr,
        laplacian_shape=np.asarray(laplacian.shape),
        lambda_min=np.asarray(1.0),
    )
    _json(
        root / "geometry" / "manifest.json",
        {"file": {"path": geometry.name, "sha256": _sha256(geometry)}},
    )

    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{_sha256(path).removeprefix('sha256:')}  {path.relative_to(root)}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")
    return root, clean


def test_indexed_loader_joins_public_arrays_and_frozen_noise(tmp_path: Path) -> None:
    root, clean = _package(tmp_path)
    dataset = PaperDataset(root)

    first = dataset.load_case("healthy-test-p0001-r00", forward_spec="conductivity", snr_db=40.0)
    second = dataset.load_case("healthy-test-p0001-r00", forward_spec="conductivity", snr_db=40.0)

    np.testing.assert_allclose(first.measurement_clean, clean.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(first.measurement, second.measurement)
    np.testing.assert_allclose(first.measurement.mean(axis=0), 0.0, atol=3e-8)
    assert first.forward_spec == "conductivity"
    assert first.operator_discrepancy == "lung-conductivity"
    assert first.noise_seed == 12345
    assert first.target_snr_db == 40.0
    assert np.isfinite(first.realized_snr_db)


def test_checksum_validation_rejects_tampering(tmp_path: Path) -> None:
    root, _ = _package(tmp_path)
    assert validate_package_checksums(root)["status"] == "ok"

    (root / "cases.jsonl").write_text("tampered\n", encoding="utf-8")
    try:
        validate_package_checksums(root)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("tampered package passed checksum validation")


def test_clean_condition_does_not_modify_measurement() -> None:
    clean = np.arange(15, dtype=float).reshape(3, 5)
    actual, realized, norm = add_paper_noise(clean, snr_db=None, noise_seed=7)
    np.testing.assert_array_equal(actual, clean)
    assert realized is None
    assert norm == 0.0

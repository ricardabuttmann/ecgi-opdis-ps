from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from opdis_ps.pretrained_models import PretrainedModels, normalize_replicate, resolve_method_model


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "paper-pretrained-models-v1"
    checkpoint = root / "prior" / "prior-s2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"compact-checkpoint")
    digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema_name": "ecgi.paper_pretrained_models",
        "schema_version": 1,
        "package_id": "paper-pretrained-models-v1",
        "models": [
            {
                "model_id": "prior-s2",
                "family": "prior",
                "replicate": "s2",
                "weights_kind": "ema",
                "file": {
                    "path": "prior/prior-s2.pt",
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": digest,
                },
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_resolves_requested_family_and_replicate(tmp_path: Path):
    root = _package(tmp_path)
    package = PretrainedModels(root)
    model = package.resolve("prior", 2)
    assert model.model_id == "prior-s2"
    assert model.path == root / "prior" / "prior-s2.pt"
    assert package.for_method("opdis-ps", "s2") == model
    assert package.for_method("tikhonov", 2) is None


def test_rejects_corrupt_checkpoint(tmp_path: Path):
    root = _package(tmp_path)
    (root / "prior" / "prior-s2.pt").write_bytes(b"changed-checkpoint")
    with pytest.raises(ValueError, match="size differs|SHA-256 differs"):
        PretrainedModels(root).resolve("prior", 2)


def test_learned_method_requires_external_package():
    with pytest.raises(ValueError, match="requires --models"):
        resolve_method_model("transformer", None)
    assert resolve_method_model("tikhonov", None) is None


@pytest.mark.parametrize("value, expected", [(0, "s0"), (4, "s4"), ("s3", "s3")])
def test_normalize_replicate(value, expected):
    assert normalize_replicate(value) == expected


@pytest.mark.parametrize("value", [-1, 5, "0", "s5"])
def test_rejects_unknown_replicate(value):
    with pytest.raises(ValueError, match="replicate"):
        normalize_replicate(value)

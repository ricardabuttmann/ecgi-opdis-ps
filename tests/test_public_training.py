from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ecgi.models.diffusion_common.checkpoint import read_inference_checkpoint
from ecgi.pipeline.public_training import TrainingPairs


ENTRY = {"prior": "train_diffusion", "transformer": "train_transformer", "palette": "train_conditional_diffusion"}


def _field(root: Path, ids: list[str], data: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "part.npy", data, allow_pickle=False)
    index = root / "index.jsonl"
    index.write_text("".join(json.dumps({"case_id": case_id, "shard": "part.npy", "offset": i}) + "\n" for i, case_id in enumerate(ids)))
    (root / "manifest.json").write_text(json.dumps({"index": {"path": "index.jsonl", "sha256": "sha256:" + hashlib.sha256(index.read_bytes()).hexdigest()}}))


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "paper-training-v1"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"package_id": "paper-training-v1"}))
    ids = ["healthy-train-p0001-r00", "healthy-val-p0001-r00"]
    (root / "cases.jsonl").write_text("".join(json.dumps({"case_id": case_id, "partition": split}) + "\n" for case_id, split in zip(ids, ("train", "val"), strict=True)))
    rng = np.random.default_rng(7)
    hsp = rng.normal(size=(2, 64, 8)).astype(np.float32)
    hsp -= hsp.mean(axis=1, keepdims=True)
    bsp = rng.normal(size=(2, 4, 8)).astype(np.float32)
    bsp -= bsp.mean(axis=1, keepdims=True)
    _field(root / "hsp", ids, hsp)
    _field(root / "measurements" / "nominal", ids, bsp)
    return root


def test_pairs_join_by_case_id(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    ids = ["healthy-val-p0001-r00", "healthy-train-p0001-r00"]
    bsp_dir = root / "measurements" / "nominal"
    arrays = np.load(bsp_dir / "part.npy").copy()[::-1]
    for path in bsp_dir.iterdir():
        path.unlink()
    _field(bsp_dir, ids, arrays)
    data = {"train_cases": 1, "val_cases": 1, "n_time": 8, "n_z": 8, "n_phi": 8,
            "n_electrodes": 4, "hsp_scale": 1.0, "bsp_scale": 1.0}
    pairs = TrainingPairs(root, "train", data, paired=True)
    _, actual = pairs[0]
    np.testing.assert_allclose(actual.numpy(), arrays[1].T)


def _config(path: Path, family: str) -> None:
    model = (
        'T = 8\nt_canvas = 8\nz_canvas = 8\nbase_channels = 8\nparameterization = "v"\n'
        if family != "transformer"
        else 'd_model = 8\nn_heads = 2\nn_layers = 1\ndim_feedforward = 16\ndropout = 0.0\n'
    )
    path.write_text(
        f'family = "{family}"\n[data]\ntrain_cases = 1\nval_cases = 1\nn_time = 8\nn_z = 8\nn_phi = 8\nn_electrodes = 4\nchart_id = "harmonic_ea_v1"\nhsp_scale = 1.0\nbsp_scale = 1.0\nx_clip = 4.0\n'
        f'[model]\n{model}[training]\nseed = 3\nvalidation_seed = 4\nepochs = 1\nbatch_size = 1\nlr = 0.0001\nweight_decay = 0.0\nema = 0.9\ngrad_clip = 1.0\nnoise_augment = true\nnoise_seed = 5\nsnr_db_low = 10.0\nsnr_db_high = 40.0\nclean_prob = 0.2\n'
    )


@pytest.mark.parametrize("family", list(ENTRY))
def test_training_entry_one_step(tmp_path: Path, family: str) -> None:
    dataset = _fixture(tmp_path)
    config = tmp_path / "config.toml"
    _config(config, family)
    output = tmp_path / "result"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"), "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    subprocess.run(
        [sys.executable, "-m", f"ecgi.pipeline.{ENTRY[family]}", "--dataset-dir", str(dataset), "--config", str(config), "--output-dir", str(output), "--device", "cpu", "--max-steps", "1"],
        check=True, capture_output=True, text=True, env=env, timeout=120,
    )
    checkpoint = read_inference_checkpoint(
        output / "best.pt", expected_weights="raw" if family == "transformer" else "ema"
    )
    assert checkpoint["checkpoint_format"] == "ecgi-inference-v1"
    assert checkpoint["weights_kind"] == ("raw" if family == "transformer" else "ema")
    assert checkpoint["step"] == 1
    assert len((output / "history.jsonl").read_text().splitlines()) == 1
    assert (output / "latest.pt").is_file()

"""Portable training loops over the extracted paper-training-v1 shard package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ecgi.mesh_operators.template import build_template
from ecgi.models.conditional_diffusion.diffusion import build_conditional_diffusion
from ecgi.models.diffusion_baseline.diffusion import GaussianDiffusion, build_coord_channels
from ecgi.models.diffusion_baseline.unet3d import UNet3D
from ecgi.models.transformer.network import TemporalTransformer
from opdis_ps.dataset import ShardField


class TrainingPairs(Dataset):
    """Join HSP and clean nominal BSP by public case ID, never shard order."""

    def __init__(self, root: Path, split: str, data: dict, *, paired: bool):
        self.root = root
        self.split = split
        rows = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines() if line]
        self.ids = sorted(str(row["case_id"]) for row in rows if row.get("partition") == split)
        expected = int(data[f"{split}_cases"])
        if len(self.ids) != expected or len(set(self.ids)) != len(self.ids):
            raise ValueError(f"{split}: expected {expected} unique cases, found {len(self.ids)}")
        self.hsp = ShardField(root / "hsp", "case_id")
        self.bsp = ShardField(root / "measurements" / "nominal", "case_id") if paired else None
        if not set(self.ids).issubset(self.hsp.rows):
            raise ValueError(f"{split}: missing HSP shard rows")
        if paired and not set(self.ids).issubset(self.bsp.rows):
            raise ValueError(f"{split}: missing nominal BSP shard rows")
        self.n_time, self.n_z, self.n_phi = (int(data[key]) for key in ("n_time", "n_z", "n_phi"))
        self.n_electrodes = int(data["n_electrodes"])
        self.hsp_scale = _positive(data["hsp_scale"], "hsp_scale")
        self.bsp_scale = _positive(data["bsp_scale"], "bsp_scale")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        case_id = self.ids[index]
        hsp = self.hsp.get(case_id)
        expected_hsp = (self.n_z * self.n_phi, self.n_time)
        if hsp.shape != expected_hsp or not np.isfinite(hsp).all():
            raise ValueError(f"{case_id}: invalid HSP shape or values {hsp.shape}; expected {expected_hsp}")
        x = torch.from_numpy(np.ascontiguousarray(hsp.T.reshape(self.n_time, self.n_z, self.n_phi))) / self.hsp_scale
        x = x.unsqueeze(0)
        if self.bsp is None:
            return x
        bsp = self.bsp.get(case_id)
        expected_bsp = (self.n_electrodes, self.n_time)
        if bsp.shape != expected_bsp or not np.isfinite(bsp).all():
            raise ValueError(f"{case_id}: invalid BSP shape or values {bsp.shape}; expected {expected_bsp}")
        b = torch.from_numpy(np.ascontiguousarray(bsp.T)) / self.bsp_scale
        return x, b


def _positive(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _model(family: str, data: dict, spec: dict):
    time, nz, nphi = (int(data[key]) for key in ("n_time", "n_z", "n_phi"))
    electrodes = int(data["n_electrodes"])
    if family == "prior":
        config = {"T": int(spec["T"]), "t_canvas": int(spec["t_canvas"]),
                  "z_canvas": int(spec["z_canvas"]), "in_channels": 5,
                  "base_channels": int(spec["base_channels"]),
                  "parameterization": str(spec["parameterization"]),
                  "min_snr_gamma": None, "field_whitening": None}
        model = GaussianDiffusion(
            unet=UNet3D(base_channels=config["base_channels"]), T=config["T"],
            t_canvas=config["t_canvas"], z_canvas=config["z_canvas"],
            parameterization=config["parameterization"])
    elif family == "transformer":
        config = {"n_electrodes": electrodes, "n_hsp": nz * nphi, "max_time": time,
                  **{k: spec[k] for k in ("d_model", "n_heads", "n_layers", "dim_feedforward", "dropout")}}
        model = TemporalTransformer(**config)
    else:
        config = {"T": int(spec["T"]), "t_canvas": int(spec["t_canvas"]),
                  "z_canvas": int(spec["z_canvas"]), "in_channels": 6,
                  "base_channels": int(spec["base_channels"]),
                  "conditioning": "bsp_linear_projection", "parameterization": str(spec["parameterization"]),
                  "n_electrodes": electrodes, "n_z": nz, "n_phi": nphi}
        model = build_conditional_diffusion(
            config["conditioning"], T=config["T"], t_canvas=config["t_canvas"],
            z_canvas=config["z_canvas"], base_channels=config["base_channels"],
            parameterization=config["parameterization"], n_electrodes=electrodes,
            n_z=nz, n_phi=nphi)
    return model, config


def _noise_augment(bsp: torch.Tensor, spec: dict, generator: torch.Generator) -> torch.Tensor:
    """Average-referenced iid Gaussian noise with uniform per-sample SNR in dB."""
    batch, time, electrodes = bsp.shape
    noise = torch.randn(bsp.shape, device=bsp.device, generator=generator)
    noise -= noise.mean(dim=-1, keepdim=True)
    snr = float(spec["snr_db_low"]) + (float(spec["snr_db_high"]) - float(spec["snr_db_low"])) * torch.rand(batch, 1, 1, device=bsp.device, generator=generator)
    gain = torch.linalg.vector_norm(bsp, dim=(1, 2), keepdim=True) * 10.0 ** (-snr / 20.0) / math.sqrt(time * (electrodes - 1))
    clean = torch.rand(batch, 1, 1, device=bsp.device, generator=generator) < float(spec["clean_prob"])
    return bsp + noise * torch.where(clean, 0.0, gain)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for target, source in zip(self.shadow.parameters(), model.parameters(), strict=True):
            target.lerp_(source, 1 - self.decay)
        for target, source in zip(self.shadow.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def _validation(model, loader, family: str, coords, device: torch.device, seed: int) -> float:
    model.eval()
    # A fixed local seed makes diffusion checkpoint selection comparable across epochs.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        total = 0.0
        with torch.no_grad():
            for item in loader:
                if family == "prior":
                    x = item.to(device)
                    loss = model.training_loss(x, coords.expand(x.shape[0], -1, -1, -1, -1))
                else:
                    x, bsp = (part.to(device) for part in item)
                    if family == "transformer":
                        loss = (model(bsp) - x[:, 0].flatten(start_dim=2)).square().mean()
                    else:
                        loss = model.training_loss(x, coords.expand(x.shape[0], -1, -1, -1, -1), bsp)
                total += float(loss) * x.shape[0]
    return total / len(loader.dataset)


def train(family: str, dataset_dir: Path, config_path: Path, output_dir: Path, *, device_name: str = "auto", max_steps: int = 0) -> Path:
    """Train one public paper model family; write inference-compatible selected weights."""
    dataset_dir, config_path, output_dir = (path.resolve() for path in (dataset_dir, config_path, output_dir))
    config_bytes = config_path.read_bytes()
    cfg = tomllib.loads(config_bytes.decode("utf-8"))
    if cfg.get("family") != family:
        raise ValueError(f"{config_path}: expected family={family!r}")
    data, spec, run = (cfg[name] for name in ("data", "model", "training"))
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if manifest.get("package_id") != "paper-training-v1":
        raise ValueError(f"{dataset_dir}: expected paper-training-v1 package")
    paired = family != "prior"
    train_set = TrainingPairs(dataset_dir, "train", data, paired=paired)
    val_set = TrainingPairs(dataset_dir, "val", data, paired=paired)
    # Fail on shape errors before creating any output.
    train_set[0]
    val_set[0]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
    seed = int(run["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model, model_config = _model(family, data, spec)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_positive(run["lr"], "lr"), weight_decay=float(run["weight_decay"]))
    ema = EMA(model, float(run["ema"])) if family != "transformer" else None
    template = build_template(int(data["n_z"]), int(data["n_phi"]), chart=str(data["chart_id"]))
    coords = build_coord_channels(template, int(data["n_time"])).unsqueeze(0).to(device) if family != "transformer" else None
    train_loader = DataLoader(train_set, batch_size=int(run["batch_size"]), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), num_workers=0)
    val_loader = DataLoader(val_set, batch_size=int(run["batch_size"]), num_workers=0)
    noise_generator = torch.Generator(device=device).manual_seed(int(run.get("noise_seed", seed + 1)))
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "config.toml").write_bytes(config_bytes)
    provenance = {"family": family, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "dataset_manifest_sha256": hashlib.sha256((dataset_dir / "manifest.json").read_bytes()).hexdigest(),
                  "train_cases": len(train_set), "val_cases": len(val_set), "seed": seed,
                  "chart_id": str(data["chart_id"]), "selection": "validation_loss"}
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    best = float("inf")
    stale_epochs = 0
    step = 0
    with (output_dir / "history.jsonl").open("w") as log:
        for epoch in range(int(run["epochs"])):
            model.train()
            losses = []
            for item in train_loader:
                if family == "prior":
                    x = item.to(device)
                    loss = model.training_loss(x, coords.expand(x.shape[0], -1, -1, -1, -1))
                else:
                    x, bsp = (part.to(device) for part in item)
                    if family == "transformer":
                        if bool(run.get("noise_augment", False)):
                            bsp = _noise_augment(bsp, run, noise_generator)
                        loss = (model(bsp) - x[:, 0].flatten(start_dim=2)).square().mean()
                    else:
                        loss = model.training_loss(x, coords.expand(x.shape[0], -1, -1, -1, -1), bsp)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(run["grad_clip"]))
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                losses.append(float(loss.detach()))
                step += 1
                if max_steps and step >= max_steps:
                    break
            selected = ema.shadow if ema is not None else model
            val_loss = _validation(selected, val_loader, family, coords, device, int(run["validation_seed"]))
            record = {"epoch": epoch, "step": step, "train_loss": sum(losses) / len(losses), "val_loss": val_loss}
            log.write(json.dumps(record) + "\n")
            log.flush()
            meta = {"bsp_scale": float(data["bsp_scale"]) if paired else None,
                    "training": provenance, "epoch": epoch, "val_loss": val_loss}
            checkpoint = {"checkpoint_format": "ecgi-inference-v1", "weights_kind": "raw" if ema is None else "ema",
                          "model": selected.state_dict(), "step": step, "scale": float(data["hsp_scale"]),
                          "x_clip": float(data["x_clip"]) if family != "transformer" else None,
                          "model_config": model_config, "meta": meta}
            torch.save(checkpoint, output_dir / "latest.pt")
            if val_loss < best:
                best = val_loss
                stale_epochs = 0
                torch.save(checkpoint, output_dir / "best.pt")
            else:
                stale_epochs += 1
            print(f"[{family}] epoch={epoch} step={step} train={record['train_loss']:.6g} val={val_loss:.6g}", flush=True)
            if max_steps and step >= max_steps:
                break
            if family == "transformer" and stale_epochs >= int(run["patience"]):
                break
    return output_dir / "best.pt"


def main(family: str) -> None:
    parser = argparse.ArgumentParser(description=f"Train the public {family} model")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="extracted paper-training-v1 directory")
    parser.add_argument("--config", type=Path, required=True, help="frozen TOML training configuration")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for checkpoints and history")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-steps", type=int, default=0, help="small smoke run; zero trains all epochs")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    train(family, args.dataset_dir, args.config, args.output_dir, device_name=args.device, max_steps=args.max_steps)

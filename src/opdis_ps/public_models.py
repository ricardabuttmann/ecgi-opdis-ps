"""Thin public adapters around the released production model implementations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from opdis_ps.data import ExampleCase


@dataclass(frozen=True)
class PublicModelResult:
    reconstruction: np.ndarray
    diagnostics: dict[str, Any]


def _device(requested: str, *, allow_mps: bool):
    import torch

    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not allow_mps:
            raise ValueError(
                "MPS is not enabled for stochastic diffusion examples because their seeded "
                "production path supports CPU and CUDA only"
            )
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint(path: Path, *, expected_weights: str) -> dict[str, Any]:
    from ecgi.models.diffusion_common.checkpoint import read_inference_checkpoint

    return read_inference_checkpoint(path, expected_weights=expected_weights)


def _generator(device: Any, seed: int):
    """Match the production wrappers' case-local RNG behavior."""
    import torch

    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"seeded sampling is not supported on {device.type!r}")
    return torch.Generator(device=device).manual_seed(seed)


def _likelihood_artifact(repo_root: Path, case: ExampleCase, filename: str) -> Path:
    path = repo_root / "artifacts" / case.operator_discrepancy / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {filename} for example {case.operator_discrepancy!r}: {path}"
        )
    return path


def _template(case: ExampleCase):
    from ecgi.mesh_operators.template import Template

    n_z, n_phi = case.template_shape
    return Template(case.template_z, case.template_phi, n_z, n_phi)


def run_tikhonov(case: ExampleCase, settings: dict[str, Any]) -> PublicModelResult:
    from ecgi.models.tikhonov.tikhonov import Tikhonov

    model = Tikhonov(
        order=int(settings["order"]),
        method=str(settings["selection"]),
        lambda_range=tuple(float(value) for value in settings["lambda_range"]),
        rank_tol=float(settings["rank_tol"]),
    )
    reconstruction = model.reconstruct(
        SimpleNamespace(b=case.measurement, forward_A=case.forward),
        SimpleNamespace(laplacian=case.laplacian, lambda_min=case.lambda_min),
    )
    assert model.last_result_ is not None
    return PublicModelResult(
        reconstruction,
        {
            "selected_lambda": float(model.last_result_.lambda_),
            "retained_rank": int(model.last_result_.retained_rank),
        },
    )


def run_spatiotemporal_tikhonov(
    case: ExampleCase, settings: dict[str, Any]
) -> PublicModelResult:
    from ecgi.models.tikhonov_st.tikhonov_st import TikhonovSpatioTemporal

    model = TikhonovSpatioTemporal(
        order=int(settings["spatial_order"]),
        order_t=int(settings["temporal_order"]),
        rho=float(settings["rho"]),
        lambda_range=tuple(float(value) for value in settings["lambda_range"]),
        rank_tol=float(settings["rank_tol"]),
    )
    reconstruction = model.reconstruct(
        SimpleNamespace(b=case.measurement, forward_A=case.forward),
        SimpleNamespace(laplacian=case.laplacian, lambda_min=case.lambda_min),
    )
    assert model.last_result_ is not None
    return PublicModelResult(
        reconstruction,
        {
            "selected_lambda": float(model.last_result_.lambda_),
            "retained_rank": int(model.last_result_.retained_ranks.sum()),
        },
    )


def run_total_variation(case: ExampleCase, settings: dict[str, Any]) -> PublicModelResult:
    from ecgi.mesh_operators.template import build_template
    from ecgi.models.total_variation import TotalVariation

    model = TotalVariation(
        alpha=float(settings["alpha"]),
        epsilon=float(settings["epsilon"]),
        max_iterations=int(settings["max_iterations"]),
        tolerance=float(settings["tolerance"]),
    )
    n_z, n_phi = case.template_shape
    reconstruction = model.reconstruct(
        SimpleNamespace(b=case.measurement, forward_A=case.forward),
        SimpleNamespace(template=build_template(n_z=n_z, n_phi=n_phi)),
    )
    assert model.last_result_ is not None
    return PublicModelResult(
        reconstruction,
        {
            "alpha": model.alpha,
            "epsilon": model.epsilon,
            "iterations": int(model.last_result_.iterations),
            "converged": bool(model.last_result_.converged),
            "relative_change": float(model.last_result_.relative_change),
            "objective": float(model.last_result_.objective),
        },
    )


def run_transformer(
    case: ExampleCase, settings: dict[str, Any], checkpoint_path: Path
) -> PublicModelResult:
    import torch

    from ecgi.data.dataset import average_reference
    from ecgi.models.transformer.network import TemporalTransformer

    checkpoint = _checkpoint(checkpoint_path, expected_weights=str(settings["weights"]))
    model = TemporalTransformer(**checkpoint["model_config"])
    device = _device(str(settings.get("device", "auto")), allow_mps=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    bsp_scale = float(checkpoint["meta"]["bsp_scale"])
    hsp_scale = float(checkpoint["scale"])
    batch = torch.as_tensor(
        np.ascontiguousarray(case.measurement.T / bsp_scale),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        prediction = model(batch)[0].cpu().numpy() * hsp_scale
    return PublicModelResult(
        average_reference(prediction.T),
        {"device": str(device), "checkpoint_weights": str(settings["weights"])},
    )


def run_palette(
    case: ExampleCase, settings: dict[str, Any], checkpoint_path: Path
) -> PublicModelResult:
    import torch

    from ecgi.models.conditional_diffusion.diffusion import build_conditional_diffusion
    from ecgi.models.conditional_diffusion.sampler import conditional_ddim_sample
    from ecgi.models.diffusion_baseline.diffusion import build_coord_channels

    checkpoint = _checkpoint(checkpoint_path, expected_weights=str(settings["weights"]))
    config = checkpoint["model_config"]
    model = build_conditional_diffusion(
        str(config["conditioning"]),
        T=int(config["T"]),
        t_canvas=int(config["t_canvas"]),
        z_canvas=int(config["z_canvas"]),
        base_channels=int(config["base_channels"]),
        parameterization=str(config["parameterization"]),
        n_electrodes=int(config["n_electrodes"]),
        n_z=int(config["n_z"]),
        n_phi=int(config["n_phi"]),
    )
    device = _device(str(settings.get("device", "auto")), allow_mps=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    scale = float(checkpoint["scale"])
    condition = (
        torch.from_numpy(case.measurement.T.astype(np.float32)).unsqueeze(0)
        / float(checkpoint["meta"]["bsp_scale"])
    )
    coords = build_coord_channels(_template(case), case.measurement.shape[1]).unsqueeze(0)
    sample_seed = int(settings["sample_seed"])
    generator = _generator(device, sample_seed)
    draws = []
    for _ in range(int(settings["point_samples"])):
        sample = conditional_ddim_sample(
            model,
            coords,
            condition,
            n_steps=int(settings["steps"]),
            eta=float(settings["eta"]),
            scale=scale,
            x0_clip=checkpoint.get("x_clip"),
            temperature=float(settings["temperature"]),
            device=device,
            generator=generator,
        )
        draws.append(sample[0, 0].reshape(case.measurement.shape[1], -1).T.cpu().numpy())
    return PublicModelResult(
        np.mean(draws, axis=0),
        {
            "device": str(device),
            "checkpoint_weights": str(settings["weights"]),
            "point_samples": len(draws),
            "sample_seed": sample_seed,
        },
    )


def run_dps(
    case: ExampleCase, settings: dict[str, Any], checkpoint_path: Path
) -> PublicModelResult:
    import torch

    from ecgi.models.diffusion_baseline.diffusion import build_coord_channels, load_diffusion_model
    from ecgi.models.diffusion_baseline.dps import dps_sample

    device = _device(str(settings.get("device", "auto")), allow_mps=False)
    diffusion, scale, x_clip = load_diffusion_model(
        checkpoint_path,
        device,
        weights=str(settings["weights"]),
    )
    coords = build_coord_channels(_template(case), case.measurement.shape[1]).unsqueeze(0)
    forward = torch.from_numpy(np.ascontiguousarray(case.forward, dtype=np.float32))
    measurement = torch.from_numpy(np.ascontiguousarray(case.measurement, dtype=np.float32)) / scale
    sample_seed = int(settings["sample_seed"])
    generator = _generator(device, sample_seed)
    draws = []
    for _ in range(int(settings["point_samples"])):
        sample = dps_sample(
            diffusion,
            coords,
            forward,
            measurement,
            batch=1,
            n_steps=int(settings["steps"]),
            eta=float(settings["eta"]),
            zeta=float(settings["zeta"]),
            scale=float(scale),
            x0_clip=x_clip,
            temperature=float(settings["temperature"]),
            device=device,
            generator=generator,
        )
        draws.append(sample[0, 0].reshape(case.measurement.shape[1], -1).T.cpu().numpy())
    return PublicModelResult(
        np.mean(draws, axis=0),
        {
            "device": str(device),
            "checkpoint_weights": str(settings["weights"]),
            "point_samples": len(draws),
            "sample_seed": sample_seed,
        },
    )


def run_ddrm(
    case: ExampleCase,
    settings: dict[str, Any],
    repo_root: Path,
    checkpoint_path: Path,
) -> PublicModelResult:
    from ecgi.forward.measurement_whitening import (
        ConditionalApproximationError,
        MeasurementWhitening,
        conditional_corrected_measurement_operator,
        whitened_measurement_operator,
    )
    from ecgi.forward.operator_spectrum import measurement_sigma
    from ecgi.models.diffusion_baseline.ddrm import _prepare_operator, ddrm_sample
    from ecgi.models.diffusion_baseline.diffusion import (
        build_coord_channels,
        load_diffusion_model,
    )

    device = _device(str(settings.get("device", "auto")), allow_mps=False)
    diffusion, scale, x_clip = load_diffusion_model(
        checkpoint_path,
        device,
        weights=str(settings["weights"]),
    )
    operator = np.ascontiguousarray(case.forward, dtype=np.float64)
    measurement = np.ascontiguousarray(case.measurement, dtype=np.float64)
    sigma_epsilon = measurement_sigma(
        case.noise_norm / math.sqrt(case.measurement.size),
        case.measurement.shape[0],
    )
    mode = str(settings["mode"])
    if mode == "measurement_only":
        sigma_y = sigma_epsilon / scale
    elif mode == "measurement_plus_discrepancy_full":
        covariance = MeasurementWhitening.from_npz(
            _likelihood_artifact(repo_root, case, str(settings["discrepancy"])),
            ridge=float(settings.get("discrepancy_ridge", 0.0)),
        )
        operator, measurement = whitened_measurement_operator(
            operator, measurement, covariance, sigma_epsilon
        )
        sigma_y = 1.0 / scale
    elif mode == "measurement_plus_discrepancy_conditional":
        conditional = ConditionalApproximationError.from_npz(
            _likelihood_artifact(repo_root, case, str(settings["conditional_artifact"])),
            ridge=float(settings.get("discrepancy_ridge", 0.0)),
        )
        operator, measurement = conditional_corrected_measurement_operator(
            operator, measurement, conditional
        )
        operator, measurement = whitened_measurement_operator(
            operator,
            measurement,
            conditional.residual_whitening,
            sigma_epsilon,
        )
        sigma_y = 1.0 / scale
    else:
        raise ValueError(f"unsupported public DDRM mode {mode!r}")

    spectral = _prepare_operator(
        operator,
        measurement / scale,
        relative_floor=float(settings.get("relative_floor", 1e-8)),
        device=device,
    )
    coords = build_coord_channels(_template(case), case.measurement.shape[1]).unsqueeze(0)
    sample_seed = int(settings["sample_seed"])
    generator = _generator(device, sample_seed)
    draws = []
    n_samples = int(settings["point_samples"])
    batch_size = min(n_samples, 4)
    for offset in range(0, n_samples, batch_size):
        this_batch = min(batch_size, n_samples - offset)
        sample = ddrm_sample(
            diffusion,
            coords,
            spectral,
            sigma_y=sigma_y,
            batch=this_batch,
            n_steps=int(settings["steps"]),
            eta=float(settings["eta"]),
            eta_b=float(settings["eta_b"]),
            scale=float(scale),
            x0_clip=x_clip,
            device=device,
            generator=generator,
        )
        draws.append(
            sample[:, 0]
            .reshape(this_batch, case.measurement.shape[1], -1)
            .transpose(1, 2)
            .cpu()
            .numpy()
        )
    samples = np.concatenate(draws, axis=0)
    return PublicModelResult(
        samples.mean(axis=0),
        {
            "device": str(device),
            "checkpoint_weights": str(settings["weights"]),
            "point_samples": int(samples.shape[0]),
            "sample_seed": sample_seed,
            "resolved_rank": int(spectral.resolved.sum().item()),
            "sigma_y": float(sigma_y),
        },
    )

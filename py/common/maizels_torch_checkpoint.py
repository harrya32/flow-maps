"""OTP-FM/MFM/SF2M inference loaders for cell-data checkpoint evaluation."""

from __future__ import annotations

from argparse import Namespace
from collections import OrderedDict
from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import torch

from . import maizels


REPO_ROOT = Path(__file__).resolve().parents[2]
MFM_ROOT = REPO_ROOT / "metric-flow-matching"
if str(MFM_ROOT) not in sys.path:
    sys.path.insert(0, str(MFM_ROOT))


@dataclass
class TorchCheckpoint:
    sampler: object
    config: Dict
    dimension: int
    metadata: Dict


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    return device


def _config_dict(value) -> Dict:
    if isinstance(value, (Namespace, SimpleNamespace)):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint configuration must be a mapping.")
    for key in ("args", "config"):
        if key in value and isinstance(value[key], (Mapping, Namespace, SimpleNamespace)):
            return _config_dict(value[key])
    # W&B's saved config.yaml wraps each argument in a {value: ...} mapping.
    return {
        str(key): item["value"] if isinstance(item, Mapping) and "value" in item else item
        for key, item in value.items() if not str(key).startswith("_")
    }


def _read_config(path: Path) -> Dict:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    else:
        import yaml

        value = yaml.safe_load(path.read_text())
    return _config_dict(value)


def _checkpoint_config(backend: str, path: Path, payload: Dict, config_path=None):
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        return _read_config(config_path), str(config_path)
    embedded = payload.get("config") or payload.get("hyper_parameters")
    if embedded:
        return _config_dict(embedded), "embedded checkpoint configuration"
    names = ("experiment.json",) if backend == "otpfm" else (
        "sweep_config.yaml", "config.yaml", "hparams.yaml", "config.json"
    )
    for directory in list(path.parents)[:7]:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return _read_config(candidate), str(candidate)
        if directory == REPO_ROOT:
            break
    expected = "the run's experiment.json" if backend == "otpfm" else "the training YAML"
    raise FileNotFoundError(
        f"No saved {backend} configuration found for {path}; pass --config_path with {expected}."
    )


def maizels_config(
    config: Dict,
    backend: str,
    *,
    dataset_location: str = "",
    schedule: Optional[str] = None,
    time_mode: Optional[str] = None,
    hparam_val_times=None,
):
    """Reconstruct the saved marginal clock without constructing training data."""
    from mfm.dataloaders.maizels_data import SCHEDULES, _timepoint_values

    dataset_kind = config.get("dataset_name") if backend == "otpfm" else config.get("data_type")
    if dataset_kind not in (None, "maizels"):
        raise ValueError(f"Expected a Maizels configuration, got {dataset_kind!r}.")
    default_schedule = "d3_d3p8_d8" if backend == "otpfm" else "d3_d8"
    selected_schedule = schedule or config.get("maizels_schedule") or default_schedule
    if selected_schedule not in SCHEDULES:
        raise ValueError(f"Unknown Maizels schedule {selected_schedule!r}.")
    selected_clock = time_mode or config.get("maizels_time_mode") or config.get("time_mode") or "real_time"
    if selected_clock not in ("real_time", "equal_time"):
        raise ValueError(f"Unknown Maizels model clock {selected_clock!r}.")
    retained = tuple(SCHEDULES[selected_schedule])
    saved_retained = config.get("retained_days")
    if schedule is None and saved_retained is not None and tuple(saved_retained) != retained:
        raise ValueError("Saved retained days disagree with the Maizels schedule.")
    saved_map = config.get("time_map")
    values = (
        [float(saved_map[day]) for day in maizels.TIMEPOINTS]
        if saved_map is not None and schedule is None and time_mode is None
        else _timepoint_values(retained, selected_clock)
    )
    if hparam_val_times is None:
        hparam_val_times = config.get(
            "maizels_hparam_val_times", config.get("hparam_val_times", ("D3.4", "D6"))
        )
    location = dataset_location or config.get("dataset_path") or config.get("maizels_dataset_path") or config.get("dataset_location") or str(Path.cwd())
    return SimpleNamespace(problem=SimpleNamespace(
        dataset_location=str(location),
        maizels_schedule=selected_schedule,
        maizels_time_mode=selected_clock,
        retained_timepoints=retained,
        timepoint_order=maizels.TIMEPOINTS,
        timepoint_values=values,
        hparam_val_times=hparam_val_times,
    ))


def _otpfm_checkpoint(path, payload, config, config_source, device, batch_size, ema):
    from otpfm_baseline.model import OTPFMSampler
    from experiments.train import create_model

    if "model_state_dict" not in payload:
        raise ValueError(f"Expected an OTP-FM model_state_dict in {path}.")
    dimension = int(config.get("pca_dim", 50))
    # Inference uses only the velocity networks, so no training-time potentials
    # or OT couplings need to be reconstructed. The solver is inactive here.
    inference_config = {**config, "solver_type": config.get("solver_type") or "direct"}
    model = create_model(inference_config, dimension, OrderedDict(), str(device))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    use_ema = (
        bool(ema) if ema is not None else (
            config["evaluation_weights"] == "ema" if "evaluation_weights" in config
            else float(config.get("ema_decay", 0.99)) > 0.0
        )
    )
    if use_ema and model.flownet_ema is None:
        raise ValueError("The OTP-FM checkpoint has no EMA velocity network.")
    scaler = None
    if config.get("normalize", True):
        mean, scale = config.get("normalization_mean"), config.get("normalization_scale")
        if mean is None or scale is None:
            raise ValueError(
                "OTP-FM normalization is missing; use the run's experiment.json with "
                "normalization_mean and normalization_scale."
            )
        mean = np.asarray(mean, dtype=np.float32)
        scale = float(scale)
        if mean.size != dimension or not np.isfinite(mean).all() or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid saved OTP-FM normalization statistics.")
        scaler = SimpleNamespace(mean_=mean.reshape(1, dimension), scale_=scale)
    model.eval()
    return TorchCheckpoint(
        OTPFMSampler(model, scaler, batch_size=batch_size, ema=use_ema),
        config, dimension,
        {
            "backend": "otpfm", "config_path": config_source,
            "parameters": "EMA" if use_ema else "instantaneous",
            "dimension": dimension, "global_step": payload.get("global_step"),
            "normalization_mean": None if scaler is None else scaler.mean_.tolist(),
            "normalization_scale": None if scaler is None else scaler.scale_,
        },
    )


class MFMSampler:
    """Batched explicit Euler integration of a saved MFM velocity field."""

    def __init__(self, model, dimension: int, device: torch.device, batch_size: int):
        self.model = model
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def sample(self, source, *, start_time, end_time, n_steps, sampler="euler"):
        if sampler != "euler":
            raise ValueError("Metric flow matching supports Euler sampling.")
        if n_steps <= 0 or self.batch_size <= 0 or end_time < start_time:
            raise ValueError("Euler sampling requires positive step/batch counts and end >= start.")
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or not len(source) or source.shape[1] != self.dimension:
            raise ValueError("MFM source population has the wrong shape or is empty.")
        dt = (float(end_time) - float(start_time)) / int(n_steps)
        pieces = []
        for left in range(0, len(source), self.batch_size):
            x = torch.as_tensor(source[left:left + self.batch_size], device=self.device)
            for step in range(int(n_steps)):
                t = torch.full(
                    (len(x), 1), float(start_time) + step * dt,
                    dtype=x.dtype, device=self.device,
                )
                x = x + dt * self.model(t, x)
            pieces.append(x.cpu().numpy())
        prediction = np.concatenate(pieces)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Non-finite MFM Euler predictions.")
        return prediction


class SF2MSampler:
    """Euler--Maruyama with the saved probability-flow velocity and score.

    Generate each step's Brownian increment for the whole population before
    batching network inference. This preserves the random stream when the
    inference batch size changes and leaves the caller's source cells intact.
    """

    def __init__(self, velocity, score, dimension, device, batch_size, sigma):
        self.velocity = velocity
        self.score = score
        self.dimension = dimension
        self.device = device
        self.batch_size = batch_size
        self.sigma = float(sigma)
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("SF2M diffusion sigma must be finite and positive.")

    @torch.inference_mode()
    def sample(
        self, source, *, start_time, end_time, n_steps,
        sampler="euler_maruyama", seed=2701,
    ):
        if sampler != "euler_maruyama":
            raise ValueError("SF2M supports Euler--Maruyama sampling.")
        start_time, end_time = float(start_time), float(end_time)
        if (
            n_steps <= 0 or self.batch_size <= 0
            or not np.isfinite([start_time, end_time]).all()
            or not 0.0 <= start_time <= end_time <= 1.0
        ):
            raise ValueError("SF2M sampling requires positive steps/batches and 0 <= start <= end <= 1.")
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or not len(source) or source.shape[1] != self.dimension:
            raise ValueError("SF2M source population has the wrong shape or is empty.")
        if not np.isfinite(source).all():
            raise ValueError("SF2M source population must be finite.")
        if end_time == start_time:
            return source.copy()
        # as_tensor shares the NumPy buffer on CPU; clone before in-place updates.
        state = torch.as_tensor(source, device=self.device).clone()
        dt = (end_time - start_time) / int(n_steps)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for step in range(int(n_steps)):
            noise = torch.randn(
                state.shape, generator=generator, dtype=state.dtype, device="cpu",
            ).to(self.device)
            for left in range(0, len(state), self.batch_size):
                right = left + self.batch_size
                x = state[left:right]
                t = torch.full(
                    (len(x), 1), start_time + step * dt,
                    dtype=x.dtype, device=self.device,
                )
                drift = self.velocity(t, x) + 0.5 * self.sigma**2 * self.score(t, x)
                state[left:right] = x + dt * drift + self.sigma * np.sqrt(dt) * noise[left:right]
            if not torch.isfinite(state).all():
                raise FloatingPointError(f"Non-finite SF2M predictions at sampling step {step + 1}.")
        return state.cpu().numpy()


def _velocity_network(state, prefix, config, device, ema, label):
    """Restore one VelocityNet, including its own EMA shadow parameters."""
    from mfm.networks.flow_networks.mlp import VelocityNet
    from mfm.flow_matchers.ema import EMA

    flow_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not flow_state:
        raise ValueError(f"No {prefix.rstrip('.')} weights found; select the flow_model checkpoint.")
    has_ema = any(key.startswith("shadow_params.") for key in flow_state)
    raw_state = (
        {key[len("model."):]: value for key, value in flow_state.items() if key.startswith("model.")}
        if has_ema else flow_state
    )
    linear_weights = sorted(
        (int(match.group(1)), value)
        for key, value in raw_state.items()
        if (match := re.fullmatch(r"model\.(\d+)\.weight", key)) and value.ndim == 2
    )
    if not linear_weights:
        raise ValueError(f"The {label} checkpoint does not contain a supported VelocityNet MLP.")
    dimension = int(linear_weights[-1][1].shape[0])
    hidden_dims = [int(weight.shape[0]) for _, weight in linear_weights[:-1]]
    if int(config.get("dim", dimension)) != dimension or list(config.get("hidden_dims_flow", hidden_dims)) != hidden_dims:
        raise ValueError(
            f"{label} training config does not match checkpoint architecture: "
            f"dim={dimension}, hidden_dims_flow={hidden_dims}."
        )
    model = VelocityNet(
        dim=dimension, hidden_dims=hidden_dims,
        activation=config.get("activation_flow", "selu"), batch_norm=False,
    )
    use_ema = has_ema if ema is None else bool(ema)
    if use_ema and not has_ema:
        raise ValueError(f"The {label} {prefix.rstrip('.')} checkpoint has no EMA weights.")
    if use_ema:
        model = EMA(model, decay=float(config.get("ema_decay") or 0.999))
        model.load_state_dict(flow_state, strict=True)
    else:
        model.load_state_dict(raw_state, strict=True)
    model = model.to(device)
    model.eval()
    return model, dimension, hidden_dims, use_ema


def _mfm_checkpoint(path, payload, config, config_source, device, batch_size, ema):
    state = payload.get("state_dict", {})
    if config.get("sf2m") or any(key.startswith("score_net.") for key in state):
        raise ValueError("SF2M checkpoints require --backend sf2m; this backend is for MFM.")
    model, dimension, hidden_dims, use_ema = _velocity_network(
        state, "flow_net.", config, device, ema, "MFM",
    )
    return TorchCheckpoint(
        MFMSampler(model, dimension, device, batch_size), config, dimension,
        {
            "backend": "mfm", "config_path": config_source,
            "parameters": "EMA" if use_ema else "instantaneous",
            "dimension": dimension, "hidden_dims_flow": hidden_dims,
            "activation_flow": config.get("activation_flow", "selu"),
            "global_step": payload.get("global_step"),
        },
    )


def _sf2m_checkpoint(path, payload, config, config_source, device, batch_size, ema):
    if config.get("mfm") or config.get("sf2m") is False:
        raise ValueError("The SF2M backend requires an SF2M checkpoint (sf2m=true, mfm=false).")
    if "sf2m_sigma" not in config:
        raise ValueError("Saved sf2m_sigma is required to reconstruct SF2M diffusion.")
    state = payload.get("state_dict", {})
    velocity, dimension, hidden_dims, flow_ema = _velocity_network(
        state, "flow_net.", config, device, ema, "SF2M",
    )
    score, score_dimension, score_hidden_dims, score_ema = _velocity_network(
        state, "score_net.", config, device, ema, "SF2M",
    )
    if score_dimension != dimension:
        raise ValueError("SF2M velocity and score output dimensions differ.")
    if flow_ema != score_ema:
        raise ValueError("SF2M velocity and score have inconsistent EMA weights; use --no-ema for both.")
    sampler = SF2MSampler(
        velocity, score, dimension, device, batch_size, config["sf2m_sigma"],
    )
    return TorchCheckpoint(sampler, config, dimension, {
        "backend": "sf2m", "config_path": config_source,
        "parameters": "EMA" if flow_ema else "instantaneous",
        "dimension": dimension, "hidden_dims_flow": hidden_dims,
        "hidden_dims_score": score_hidden_dims,
        "activation_flow": config.get("activation_flow", "selu"),
        "sf2m_sigma": sampler.sigma, "global_step": payload.get("global_step"),
    })


def checkpoints_from_sweep(root, backend, seeds=None):
    """Find validation-selected checkpoints recorded by the MFM/SF2M runner."""
    root = Path(root).expanduser().resolve()
    selected_seeds = None if seeds is None else set(seeds)
    checkpoints = {}
    with (root / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("status") != "complete" or row.get("method") != backend:
            continue
        seed = int(row["seed"])
        if selected_seeds is not None and seed not in selected_seeds:
            continue
        if seed in checkpoints:
            raise ValueError(f"Multiple completed {backend} configurations for seed {seed}; use explicit --checkpoint paths.")
        metrics_path = Path(row["metrics_json"]).expanduser()
        metrics = json.loads(metrics_path.read_text())
        checkpoint_path = metrics.get("final_eval/best_checkpoint_path")
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        else:
            candidates = sorted(
                path for path in Path(row["run_output_folder"]).rglob("*.ckpt")
                if path.parent.name == "flow_model" and path.name != "last.ckpt"
            )
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify one best {backend} checkpoint for seed {seed}; use --checkpoint.")
            checkpoint_path = candidates[0].resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Recorded best checkpoint is missing: {checkpoint_path}")
        checkpoints[seed] = (backend, seed, checkpoint_path)
    if selected_seeds is not None and selected_seeds != set(checkpoints):
        raise ValueError(f"No completed {backend} runs for seeds {sorted(selected_seeds - set(checkpoints))}.")
    if not checkpoints:
        raise ValueError(f"No completed {backend} checkpoints in {root / 'results.csv'}.")
    return [checkpoints[seed] for seed in sorted(checkpoints)]


def load_checkpoint(
    backend: str,
    path: Path,
    *,
    device: torch.device,
    batch_size: int = 1024,
    config_path=None,
    ema: Optional[bool] = None,
) -> TorchCheckpoint:
    if backend not in ("otpfm", "mfm", "sf2m"):
        raise ValueError(f"Unsupported checkpoint backend {backend!r}.")
    if batch_size <= 0:
        raise ValueError("Sampling batch size must be positive.")
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a checkpoint mapping in {path}.")
    config, source = _checkpoint_config(backend, path, payload, config_path)
    loader = {"otpfm": _otpfm_checkpoint, "mfm": _mfm_checkpoint, "sf2m": _sf2m_checkpoint}[backend]
    return loader(path, payload, config, source, device, batch_size, ema)

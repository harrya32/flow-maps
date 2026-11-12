"""Restore native deterministic CITE/Multi models and their training scale."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path

import flax.serialization
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from . import cite_multi, flow_map
from .cite_multi_checkpoint_eval import heldout_day
from .deterministic_checkpoint import pushforward


def training_config(
    *,
    seed,
    dataset_location="",
    dataset_name=None,
    omitted_day=None,
    time_mode=None,
    config_path=None,
    cfg_path="configs.cite_multi_pca100",
    slurm_id=None,
):
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text())
        else:
            import yaml

            value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(
                "Native --config_path must contain a full configuration mapping."
            )
        # Accept the full nested config exported to JSON or saved by W&B.
        value = {
            key: item["value"] if isinstance(item, dict) and "value" in item else item
            for key, item in value.items()
            if not key.startswith("_")
        }
        if not {"problem", "network", "training"}.issubset(value):
            raise ValueError(
                "Native --config_path must contain the full problem/network/training config."
            )
        cfg = ml_collections.ConfigDict(value)
        source = str(path)
        if (
            dataset_name is not None
            and cite_multi.canonical_dataset_name(cfg.problem.dataset_name)
            != dataset_name
        ):
            raise ValueError(
                "--dataset_name disagrees with the saved training dataset."
            )
        if (
            omitted_day is not None
            and heldout_day(cfg.problem.heldout_timepoint) != omitted_day
        ):
            raise ValueError("--heldout_day disagrees with the saved training fold.")
    else:
        if slurm_id is None or dataset_name is None or omitted_day is None:
            raise ValueError(
                "Native evaluation requires --slurm_id, --dataset_name and --heldout_day, or a full --config_path."
            )
        factory = importlib.import_module(cfg_path).get_config
        supported = inspect.signature(factory).parameters
        optional = {
            "seed": seed,
            "dataset_name": dataset_name,
            "heldout_day": omitted_day,
            "cite_multi_time_mode": time_mode,
        }
        cfg = factory(
            slurm_id,
            dataset_location,
            "",
            **{
                key: value
                for key, value in optional.items()
                if key in supported and value is not None
            },
        )
        source = f"{cfg_path} (slurm_id={slurm_id})"
    if cfg.problem.target != "cite_multi_pca100":
        raise ValueError("The native training config must target cite_multi_pca100.")
    cfg.training.seed = int(seed)
    cfg.problem.dataset_location = str(
        cite_multi.resolve_dataset_path(
            dataset_location or cfg.problem.dataset_location,
            cfg.problem.dataset_name,
        )
    )
    if time_mode is not None:
        cfg.problem.cite_multi_time_mode = time_mode
        cfg.problem.maizels_time_mode = time_mode
        cfg.problem.timepoint_order = list(cite_multi.TIMEPOINTS)
        cfg.problem.timepoint_values = list(
            cite_multi.normalized_time_map(time_mode).values()
        )
    return cfg, source


def training_rescale(cfg, override=None):
    """Reproduce setup_target's scalar std, including pair sampling/filtering."""
    if override is not None:
        scale, source = float(override), "--rescale"
    elif cfg.problem.gaussian_scale != "adaptive":
        scale, source = 1.0, "fixed training scale"
    elif cite_multi.uses_minibatch_ot_config(cfg):
        pools, _ = cite_multi.make_minibatch_ot_training_pools(cfg)
        weighted_mean = weighted_second_moment = total_weight = 0.0
        for interval in pools["intervals"]:
            for role in ("source_time", "target_time"):
                x = np.asarray(
                    pools["timepoints"][interval[role]]["x"], dtype=np.float64
                )
                weight = float(interval["nominal_pairs"]) * x.shape[1]
                weighted_mean += weight * float(np.mean(x))
                weighted_second_moment += weight * float(np.mean(x * x))
                total_weight += weight
        mean = weighted_mean / total_weight
        scale = float(
            np.sqrt(max(weighted_second_moment / total_weight - mean * mean, 0.0))
        )
        source = "retained training pools"
    else:
        print("Reconstructing the native training-pair scale", flush=True)
        pairs, _ = cite_multi.make_pair_pool(cfg)
        scale = float(np.std(np.concatenate([pairs["x0"], pairs["x1"]], axis=0)))
        source = "reproduced training pairs"
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The native training scale must be finite and positive.")
    cfg.network.rescale = scale
    return scale, source


def resolve_device(value):
    if value == "auto":
        return jax.devices()[0]
    name, _, index = value.partition(":")
    if name not in ("cpu", "cuda", "gpu"):
        raise ValueError("Native --device supports auto, cpu, cuda, or cuda:INDEX.")
    backend = "gpu" if name in ("cuda", "gpu") else "cpu"
    devices = jax.devices(backend)
    selected = int(index or 0)
    if not 0 <= selected < len(devices):
        raise ValueError(f"Unavailable JAX device {value!r}.")
    return devices[selected]


@dataclass
class NativeCheckpoint:
    net: object
    params: object
    dimension: int
    device: object
    batch_size: int
    metadata: dict

    def sample(self, source, *, start_time, end_time, n_steps, sampler="flowmap"):
        if sampler not in ("flowmap", "euler"):
            raise ValueError("Native deterministic sampling supports flowmap or euler.")
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or not len(source) or source.shape[1] != self.dimension:
            raise ValueError(
                "The source population does not match the checkpoint dimension."
            )
        with jax.default_device(self.device):
            return pushforward(
                self.net,
                self.params,
                source,
                start_time=start_time,
                end_time=end_time,
                n_steps=n_steps,
                batch_size=self.batch_size,
                euler=sampler == "euler",
            )


def load_checkpoint(path, cfg, *, device, batch_size=1024, ema_fac=None, rescale=None):
    scale, scale_source = training_rescale(cfg, rescale)
    dimension = int(cfg.problem.d)
    if dimension != 100 or int(cfg.network.output_dim) != dimension:
        raise ValueError("CITE/Multi native checkpoints must use 100 PCA dimensions.")
    payload = flax.serialization.msgpack_restore(Path(path).read_bytes())
    if not isinstance(payload, dict) or "params" not in payload:
        raise ValueError(
            "Expected a native deterministic training-state .pkl checkpoint."
        )
    saved = payload["params"]
    if ema_fac is not None:
        available = payload.get("ema_params", {})
        matches = [key for key in available if float(key) == float(ema_fac)]
        if not matches:
            raise ValueError(
                f"EMA decay {ema_fac} is absent from this checkpoint; available: {list(available)}."
            )
        saved = available[matches[0]]
    with jax.default_device(device):
        net, template, _ = flow_map.initialize_flow_map(
            cfg.network,
            jnp.zeros(dimension, dtype=jnp.float32),
            jax.random.PRNGKey(0),
        )
        params = flax.serialization.from_state_dict(template, saved)
        expected_leaves, expected_tree = jax.tree_util.tree_flatten(template)
        leaves, tree = jax.tree_util.tree_flatten(params)
        if tree != expected_tree or any(
            np.shape(a) != np.shape(b) for a, b in zip(expected_leaves, leaves)
        ):
            raise ValueError(
                "The native training config does not match the checkpoint architecture."
            )
        params = jax.device_put(params, device)
    return NativeCheckpoint(
        net,
        params,
        dimension,
        device,
        batch_size,
        {
            "backend": "flowmaps",
            "dimension": dimension,
            "parameters": "instantaneous" if ema_fac is None else f"EMA {ema_fac}",
            "global_step": int(np.asarray(payload.get("step", 0))),
            "rescale": scale,
            "rescale_source": scale_source,
        },
    )

"""
Nicholas M. Boffi
10/5/25

Code for basic wandb visualization and logging.
"""

import functools
import os
import signal
import sys
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import seaborn as sns
import wandb
from flax.serialization import to_bytes
from jax.flatten_util import ravel_pytree
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, fid_utils, flow_map, state_utils

Parameters = Dict[str, Dict]


def is_lowd_problem(cfg: config_dict.ConfigDict) -> bool:
    """Returns True for two-dimensional non-image datasets."""
    return getattr(cfg.problem, "image_dims", None) is None and cfg.problem.d == 2


def is_image_problem(cfg: config_dict.ConfigDict) -> bool:
    """Returns True for image-shaped problems."""
    return getattr(cfg.problem, "image_dims", None) is not None


def finite_lowd_limits(
    points: np.ndarray, pad_frac: float = 0.1, default_lim: float = 2.0
) -> Tuple[list, list]:
    """Compute robust 2D plot limits, ignoring non-finite values."""
    finite_mask = np.isfinite(points).all(axis=1)
    finite_points = points[finite_mask]
    if finite_points.shape[0] == 0:
        return [-default_lim, default_lim], [-default_lim, default_lim]

    mins = finite_points.min(axis=0)
    maxs = finite_points.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-3)
    pads = pad_frac * ranges
    xlim = [mins[0] - pads[0], maxs[0] + pads[0]]
    ylim = [mins[1] - pads[1], maxs[1] + pads[1]]
    return xlim, ylim


def extract_x1_from_batch(batch):
    """Extract target samples from either plain or paired low-dimensional batches."""
    if isinstance(batch, dict) and "x1" in batch:
        return batch["x1"]
    return batch


def _forbidden_box_bounds(cfg: config_dict.ConfigDict):
    """Return configured forbidden-box bounds as (xmin, xmax, ymin, ymax)."""
    box_cfg = getattr(getattr(cfg, "logging", None), "forbidden_box", None)
    if box_cfg is None:
        box_cfg = getattr(getattr(cfg, "problem", None), "forbidden_box", None)
    if box_cfg is None or not bool(getattr(box_cfg, "enabled", True)):
        return None

    xlim = getattr(box_cfg, "xlim", None)
    ylim = getattr(box_cfg, "ylim", None)
    if xlim is None:
        xlim = getattr(box_cfg, "x", None)
    if ylim is None:
        ylim = getattr(box_cfg, "y", None)
    if xlim is None or ylim is None:
        return None

    xmin, xmax = [float(value) for value in xlim]
    ymin, ymax = [float(value) for value in ylim]
    return xmin, xmax, ymin, ymax


def _points_in_forbidden_box(points: jnp.ndarray, bounds) -> jnp.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )


def _np_points_in_forbidden_box(points: np.ndarray, bounds) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )


def _constraint_box_bounds(cfg: config_dict.ConfigDict):
    constraint_cfg = getattr(cfg, "constraints", None)
    if constraint_cfg is not None:
        xlim = getattr(constraint_cfg, "xlim", None)
        ylim = getattr(constraint_cfg, "ylim", None)
        if xlim is None:
            xlim = getattr(constraint_cfg, "box_xlim", None)
        if ylim is None:
            ylim = getattr(constraint_cfg, "box_ylim", None)
        if xlim is not None and ylim is not None:
            xmin, xmax = [float(value) for value in xlim]
            ymin, ymax = [float(value) for value in ylim]
            return xmin, xmax, ymin, ymax

    return _forbidden_box_bounds(cfg)


def _box_signed_distance(points: jnp.ndarray, bounds) -> jnp.ndarray:
    xmin, xmax, ymin, ymax = bounds
    center = jnp.asarray([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=points.dtype)
    half_size = jnp.asarray(
        [(xmax - xmin) / 2.0, (ymax - ymin) / 2.0], dtype=points.dtype
    )
    q = jnp.abs(points - center) - half_size
    outside_q = jnp.maximum(q, 0.0)
    eps = jnp.asarray(1e-12, dtype=points.dtype)
    outside = jnp.sqrt(jnp.sum(outside_q * outside_q, axis=-1) + eps) - jnp.sqrt(eps)
    inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def _draw_forbidden_box(ax, cfg: config_dict.ConfigDict, *, label: bool = False) -> None:
    bounds = _forbidden_box_bounds(cfg)
    if bounds is None:
        return

    xmin, xmax, ymin, ymax = bounds
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="none",
            edgecolor="crimson",
            linewidth=1.4,
            linestyle="--",
            alpha=0.95,
            zorder=20,
            label="forbidden B" if label else None,
        )
    )


def _flow_map_batch(
    apply_fn,
    params: Dict,
    s: float,
    t: float,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate X_{s,t} on a batch of low-dimensional points."""
    return jax.vmap(
        lambda x, lbl: apply_fn(
            params,
            s,
            t,
            x,
            label=lbl,
            train=False,
            calc_weight=False,
            return_X_and_phi=False,
        )
    )(xs, labels)


def _one_step_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate the direct 1-step path t -> X_{0,t}(x0)."""
    paths = np.zeros((x0s.shape[0], len(times), x0s.shape[-1]), dtype=np.float32)
    paths[:, 0, :] = np.asarray(x0s)
    for idx, tt in enumerate(times[1:], start=1):
        xt = _flow_map_batch(apply_fn, params, 0.0, float(tt), x0s, labels)
        paths[:, idx, :] = np.asarray(xt)
    return paths


def _multi_step_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Track every node in an N-step flow-map rollout."""
    return np.asarray(
        flow_map.batch_sample_trajectory(apply_fn, params, x0s, n_steps, labels),
        dtype=np.float32,
    )


def _vector_field_batch(
    apply_fn,
    params: Dict,
    t: float,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """Evaluate the instantaneous vector field b_t on a batch of points."""
    return jax.vmap(
        lambda x, lbl: apply_fn(
            params,
            t,
            x,
            label=lbl,
            train=False,
            calc_weight=False,
            method="calc_b",
        )
    )(xs, labels)


def _euler_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Track every node in an Euler rollout of dx/dt = b_t(x)."""
    ts = jnp.linspace(0.0, 1.0, n_steps + 1)

    def step(x, idx):
        t0 = ts[idx]
        dt = ts[idx + 1] - ts[idx]
        x_next = x + dt * _vector_field_batch(apply_fn, params, t0, x, labels)
        return x_next, x_next

    _, states = jax.lax.scan(step, x0s, jnp.arange(n_steps))
    paths = jnp.concatenate([x0s[None, ...], states], axis=0)
    return np.asarray(jnp.swapaxes(paths, 0, 1), dtype=np.float32)


def _trajectory_segments(paths: np.ndarray) -> np.ndarray:
    """Convert paths with shape (N, T, 2) to LineCollection segments."""
    segments = np.stack([paths[:, :-1, :], paths[:, 1:, :]], axis=2)
    return segments.reshape((-1, 2, paths.shape[-1]))


def _draw_trajectory_paths(
    ax,
    paths: np.ndarray,
    x0s: np.ndarray,
    x1s: np.ndarray,
    *,
    title: str,
    xlim: list,
    ylim: list,
    fontsize: float,
    cfg: config_dict.ConfigDict = None,
) -> None:
    """Draw trajectory lines with dots at all evaluated path nodes."""
    path_points = paths.reshape((-1, paths.shape[-1]))
    ax.scatter(
        x0s[:, 0], x0s[:, 1], s=0.2, alpha=0.2, marker="o", c="gray", label="base"
    )
    ax.scatter(
        x1s[:, 0], x1s[:, 1], s=0.2, alpha=0.2, marker="o", c="C0", label="target"
    )
    ax.add_collection(
        LineCollection(
            _trajectory_segments(paths),
            colors="black",
            linewidths=0.25,
            alpha=0.2,
        )
    )
    ax.scatter(
        path_points[:, 0],
        path_points[:, 1],
        s=2.0,
        alpha=0.25,
        marker="o",
        c="black",
        linewidths=0,
        label="trajectory",
    )
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
    ax.tick_params(axis="both", labelsize=fontsize)
    if cfg is not None:
        _draw_forbidden_box(ax, cfg, label=True)


def _rollout_forbidden_box_metrics(
    cfg: config_dict.ConfigDict, rollout_paths: list, rollout_name: str
) -> Dict[str, float]:
    """Measure box occupancy on the exact nodes plotted for rollout paths."""
    bounds = _forbidden_box_bounds(cfg)
    if bounds is None:
        return {}

    metrics = {}
    for step, paths in rollout_paths:
        path_points = np.asarray(paths).reshape((-1, paths.shape[-1]))
        inside = _np_points_in_forbidden_box(path_points, bounds)
        total = int(inside.size)
        pct = 100.0 * int(np.sum(inside)) / max(total, 1)
        prefix = f"forbidden_box/{rollout_name}_{step}"
        metrics[f"{prefix}_point_pct"] = pct

    return metrics


def get_params_for_sampling(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    param_type: str = "visual",
) -> jnp.ndarray:
    """
    Get the appropriate parameters for sampling (visualization or FID).

    Args:
        cfg: Configuration
        train_state: Current training state
        param_type: Either "visual" or "fid" to select the right config parameter

    Returns:
        Parameters to use for sampling (unreplicated for single-device use)
    """
    # Determine which config parameter to check based on param_type
    if param_type == "visual":
        config_param = "visual_ema_factor"
    elif param_type == "fid":
        config_param = "fid_ema_factor"
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

    # Select which parameters to use
    if (
        hasattr(cfg.logging, config_param)
        and getattr(cfg.logging, config_param) is not None
    ):
        ema_factor = getattr(cfg.logging, config_param)
        # Use EMA parameters with specified factor
        if ema_factor in train_state.ema_params:
            params = train_state.ema_params[ema_factor]
        else:
            print(
                f"Warning: EMA factor {ema_factor} not found in ema_params. Using instantaneous params."
            )
            params = train_state.params
    else:
        # Use instantaneous parameters (default)
        params = train_state.params

    # Visual uses unreplicated params, FID uses replicated params for pmap
    if param_type == "visual":
        return dist_utils.safe_unreplicate(cfg, params)
    else:
        return params


def compute_fid_on_the_fly(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
    n_samples: int = 10000,
    batch_size: int = 256,
    n_steps_flow: int = 8,
) -> Tuple[float, jnp.ndarray]:
    """
    Compute FID on the fly during training using distributed sampling.

    Args:
        cfg: Configuration
        statics: Static arguments containing dataset stats
        train_state: Current training state
        prng_key: Random key
        n_samples: Number of samples to generate (default 10,000)
        batch_size: Batch size for sampling (will be split across devices)
        n_steps_flow: Number of steps for flow map models

    Returns:
        Tuple of FID score and updated PRNG key
    """
    # Check if FID reference statistics are available
    if not hasattr(cfg.logging, "fid_stats_path"):
        print(
            "Warning: No FID reference statistics path configured. Set cfg.logging.fid_stats_path"
        )
        return jnp.nan, prng_key

    # Load reference statistics
    try:
        fid_stats = np.load(cfg.logging.fid_stats_path)
        mu_real, sigma_real = fid_stats["mu"], fid_stats["sigma"]
    except FileNotFoundError:
        print(f"Warning: FID stats file not found at {cfg.logging.fid_stats_path}")
        return jnp.nan, prng_key
    except Exception as e:
        print(f"Warning: Error loading FID stats: {e}")
        return jnp.nan, prng_key

    # Get number of devices and adjust batch size
    per_device_batch_size = batch_size // cfg.training.ndevices
    if batch_size % cfg.training.ndevices != 0:
        per_device_batch_size += 1
        batch_size = per_device_batch_size * cfg.training.ndevices

    # Use flow map steps
    n_steps = n_steps_flow

    # Get pre-initialized FID network from statics
    if statics.inception_fn is None:
        print("Warning: Inception network not initialized. FID computation disabled.")
        return jnp.nan, prng_key
    inception_fn = statics.inception_fn

    # Get pmap sampler function based on number of devices
    if cfg.training.ndevices == 1:
        sampler = flow_map.batch_sample
    else:
        sampler = flow_map.pmap_batch_sample

    # Initialize statistics for Welford's online algorithm
    n_seen = 0
    mu_gen = None
    M2_gen = None

    # Generate samples in batches
    n_full_batches = n_samples // batch_size
    remainder = n_samples % batch_size

    for batch_idx in range(n_full_batches + (1 if remainder > 0 else 0)):
        # Determine current batch size
        if batch_idx == n_full_batches and remainder > 0:
            current_batch_size = remainder
            current_per_device_batch = (
                remainder + cfg.training.ndevices - 1
            ) // cfg.training.ndevices
            padded_batch_size = current_per_device_batch * cfg.training.ndevices
        else:
            current_batch_size = batch_size
            current_per_device_batch = per_device_batch_size
            padded_batch_size = batch_size

        # Generate noise and reshape for pmap
        prng_key, sample_key = jax.random.split(prng_key)
        x0_full = statics.sample_rho0(padded_batch_size, sample_key)

        if cfg.training.ndevices > 1:
            x0_batched = x0_full.reshape(
                cfg.training.ndevices, current_per_device_batch, *cfg.problem.image_dims
            )
        else:
            x0_batched = x0_full

        # Handle labels for conditional generation
        if cfg.training.conditional:
            if cfg.training.class_dropout > 0:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes + 1, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
            else:
                labels = jnp.array(
                    np.random.choice(cfg.problem.num_classes, padded_batch_size)
                ).reshape(cfg.training.ndevices, current_per_device_batch)
        else:
            labels = None

        # Get parameters for FID sampling
        params_for_fid = get_params_for_sampling(cfg, train_state, param_type="fid")

        # Sample images across devices
        imgs_batched = sampler(
            train_state.apply_fn,
            params_for_fid,
            x0_batched,
            n_steps,
            labels,
        )

        # Flatten from devices and clip
        imgs = imgs_batched.reshape(padded_batch_size, *cfg.problem.image_dims)
        imgs = jnp.clip(imgs, -1, 1)

        # Only keep the actual samples we need
        imgs = imgs[:current_batch_size]

        # Convert from NCHW to NHWC for Inception
        imgs = imgs.transpose(0, 2, 3, 1)

        # Extract Inception features (no need for pmap here since we're back to single batch)
        features = fid_utils.resize_and_incept(imgs, inception_fn)
        features = np.asarray(np.squeeze(features))

        # Update running statistics using Welford's method
        batch_mean = features.mean(0)
        batch_cov = (
            np.cov(features, rowvar=False)
            if features.shape[0] > 1
            else np.zeros((features.shape[1], features.shape[1]))
        )

        n_seen += current_batch_size

        if mu_gen is None:
            mu_gen = batch_mean
            M2_gen = (
                batch_cov * (current_batch_size - 1)
                if current_batch_size > 1
                else np.zeros_like(batch_cov)
            )
        else:
            delta = batch_mean - mu_gen
            mu_gen += delta * current_batch_size / n_seen
            M2_gen += (
                batch_cov * (current_batch_size - 1)
                + np.outer(delta, delta)
                * (n_seen - current_batch_size)
                * current_batch_size
                / n_seen
            )

    # Compute final covariance and FID
    sigma_gen = M2_gen / (n_seen - 1)
    fid_score = fid_utils.fid_from_stats(mu_gen, sigma_gen, mu_real, sigma_real)

    return float(fid_score), prng_key


def _save_ckpt_on_signal(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState
) -> None:
    save_state(train_state, cfg)
    sys.exit(0)


def register_signal_handlers(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
) -> None:
    """Drop a checkpoint on SIGTERM or SIGINT."""
    handler = functools.partial(_save_ckpt_on_signal, cfg, train_state)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def save_state(
    train_state: state_utils.EMATrainState,
    cfg: config_dict.ConfigDict,
) -> None:
    """Save flax training state."""
    output_folder = cfg.logging.output_folder if cfg.logging.output_folder else "."
    os.makedirs(output_folder, exist_ok=True)
    ckpt_idx = dist_utils.safe_index(cfg, train_state.step) // cfg.logging.save_freq
    ckpt_path = f"{output_folder}/{cfg.logging.output_name}_{ckpt_idx}.pkl"

    with open(ckpt_path, "wb") as f:
        state = jax.device_get(dist_utils.safe_unreplicate(cfg, train_state))
        f.write(to_bytes(state))


@jax.jit
def compute_grad_norm(grads: Dict) -> float:
    """Computes the norm of the gradient, where the gradient is input
    as an hk.Params object (treated as a PyTree)."""
    flat_params = ravel_pytree(grads)[0]
    return jnp.linalg.norm(flat_params)


def _covariance(x: jnp.ndarray) -> jnp.ndarray:
    x_centered = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.maximum(x.shape[0] - 1, 1)
    return (x_centered.T @ x_centered) / denom


def _maybe_clip_constraint_state(
    x: jnp.ndarray, cfg: config_dict.ConfigDict
) -> jnp.ndarray:
    x_clip = float(getattr(cfg.constraints, "x_clip", 0.0))
    clip_mode = getattr(cfg.constraints, "x_clip_mode", "hard")
    if x_clip > 0:
        if clip_mode == "hard":
            return jnp.clip(x, -x_clip, x_clip)
        elif clip_mode == "tanh":
            return x_clip * jnp.tanh(x / x_clip)
        else:
            raise ValueError(f"Unknown constraints.x_clip_mode: {clip_mode}")
    return x


def _select_kde_observations(
    x0: jnp.ndarray, x1: jnp.ndarray, cfg: config_dict.ConfigDict
) -> jnp.ndarray:
    source = getattr(cfg.constraints, "kde_obs", "target")
    if source == "target":
        obs = x1
    elif source == "base_target":
        obs = jnp.concatenate([x0, x1], axis=0)
    else:
        raise ValueError(f"Unknown constraints.kde_obs: {source}")

    max_points = int(getattr(cfg.constraints, "kde_max_points", 0))
    if max_points > 0:
        obs = obs[:max_points]
    return obs


def _kde_log_density(
    x: jnp.ndarray, obs: jnp.ndarray, bandwidth: float
) -> jnp.ndarray:
    d = x.shape[-1]
    m = obs.shape[0]
    h2 = bandwidth * bandwidth

    diffs = x[:, None, :] - obs[None, :, :]
    sqdist = jnp.sum(diffs * diffs, axis=-1)
    log_kernels = -0.5 * sqdist / h2

    log_norm = jnp.log(float(m)) + d * jnp.log(bandwidth) + 0.5 * d * jnp.log(
        2.0 * jnp.pi
    )
    return jsp.special.logsumexp(log_kernels, axis=1) - log_norm


def _rbf_kernel_mixture(
    x: jnp.ndarray, y: jnp.ndarray, bandwidths: jnp.ndarray
) -> jnp.ndarray:
    diffs = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diffs * diffs, axis=-1)

    bws = jnp.maximum(jnp.asarray(bandwidths, dtype=x.dtype), 1e-6)
    scales = 2.0 * (bws[:, None, None] ** 2)
    kernels = jnp.exp(-sqdist[None, :, :] / scales)
    return jnp.mean(kernels, axis=0)


def _weighted_kernel_mean(
    kernel: jnp.ndarray,
    x_weights: jnp.ndarray,
    y_weights: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    weights = x_weights[:, None] * y_weights[None, :]
    denom = jnp.maximum(jnp.sum(x_weights) * jnp.sum(y_weights), eps)
    return jnp.sum(kernel * weights) / denom


def _weighted_mmd(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_weights: jnp.ndarray,
    y_weights: jnp.ndarray,
    bandwidths: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    k_xx = _rbf_kernel_mixture(x, x, bandwidths)
    k_yy = _rbf_kernel_mixture(y, y, bandwidths)
    k_xy = _rbf_kernel_mixture(x, y, bandwidths)

    mmd2 = (
        _weighted_kernel_mean(k_xx, x_weights, x_weights, eps)
        + _weighted_kernel_mean(k_yy, y_weights, y_weights, eps)
        - 2.0 * _weighted_kernel_mean(k_xy, x_weights, y_weights, eps)
    )
    return jnp.maximum(mmd2, 0.0)


def _endpoint_matching_mmd(
    cfg: config_dict.ConfigDict,
    x0: jnp.ndarray,
    x1_hat: jnp.ndarray,
    x1: jnp.ndarray,
) -> jnp.ndarray:
    endpoint_cfg = cfg.training.endpoint_matching
    bandwidths = jnp.asarray(
        getattr(endpoint_cfg, "bandwidths", [0.25, 0.5, 1.0, 2.0, 4.0]),
        dtype=x1.dtype,
    )
    eps = float(getattr(endpoint_cfg, "eps", 1e-6))
    all_weights = jnp.ones((x0.shape[0],), dtype=x1.dtype)

    if not bool(getattr(endpoint_cfg, "branch_conditional", False)):
        return _weighted_mmd(x1_hat, x1, all_weights, all_weights, bandwidths, eps)

    branch_axis = int(getattr(endpoint_cfg, "branch_axis", 1))
    branch_threshold = float(getattr(endpoint_cfg, "branch_threshold", 0.0))
    branch_weights = (x0[:, branch_axis] >= branch_threshold).astype(x1.dtype)

    loss = 0.0
    normalizer = 0.0
    min_branch_mass = float(getattr(endpoint_cfg, "min_branch_mass", 1.0))
    for weights in [branch_weights, 1.0 - branch_weights]:
        branch_present = (jnp.sum(weights) >= min_branch_mass).astype(x1.dtype)
        branch_mmd = _weighted_mmd(x1_hat, x1, weights, weights, bandwidths, eps)
        loss += branch_present * branch_mmd
        normalizer += branch_present

    return loss / jnp.maximum(normalizer, 1.0)


def _path_positions(
    params: Dict,
    apply_fn,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    if label is None:
        return jax.vmap(
            lambda x, tt: apply_fn(
                params,
                0.0,
                tt,
                x,
                label=None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, tau)
    else:
        return jax.vmap(
            lambda x, lbl, tt: apply_fn(
                params,
                0.0,
                tt,
                x,
                label=lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, label, tau)


def _map_between_positions(
    params: Dict,
    apply_fn,
    s: jnp.ndarray,
    t: jnp.ndarray,
    x: jnp.ndarray,
    label: jnp.ndarray,
) -> jnp.ndarray:
    """Compute X_{s_i,t_i}(x_i) for each sample i."""
    if label is None:
        return jax.vmap(
            lambda ss, tt, xi: apply_fn(
                params,
                ss,
                tt,
                xi,
                label=None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(s, t, x)
    else:
        return jax.vmap(
            lambda ss, tt, xi, lbl: apply_fn(
                params,
                ss,
                tt,
                xi,
                label=lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(s, t, x, label)


def compute_constraint_metrics(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
    statics: state_utils.StaticArgs = None,
) -> Dict[str, float]:
    """Compute configured trajectory-constraint errors for logging."""
    if not hasattr(cfg, "constraints") or not cfg.constraints.enabled:
        return {}

    # unpack the full loss args and unreplicate
    data_args = loss_fn_args[1:]
    (
        x0batch,
        x1batch,
        label_batch,
        sbatch,
        tbatch,
        _,
        _,
        _,
        constraint_scale_batch,
        stage2_scale_batch,
    ) = dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)
    constraint_scale = jnp.mean(jnp.squeeze(constraint_scale_batch))
    stage2_scale = jnp.mean(jnp.squeeze(stage2_scale_batch))
    if bool(getattr(cfg.constraints, "stage2_only", False)):
        constraint_scale = constraint_scale * stage2_scale
    if label_batch is not None:
        label_batch = jnp.squeeze(label_batch)

    ctype = cfg.constraints.type

    if ctype == "mid_moment":
        params = dist_utils.safe_unreplicate(cfg, train_state.params)
        t_star = float(cfg.constraints.time)
        tau = jnp.full((x0batch.shape[0],), t_star, dtype=x0batch.dtype)
        x_tstar = _path_positions(params, train_state.apply_fn, x0batch, label_batch, tau)
        x_tstar = _maybe_clip_constraint_state(x_tstar, cfg)

        target_mean = jnp.asarray(cfg.constraints.target_mean, dtype=x_tstar.dtype)
        target_cov = jnp.asarray(cfg.constraints.target_cov, dtype=x_tstar.dtype)

        mean_vec = jnp.mean(x_tstar, axis=0)
        cov_mat = _covariance(x_tstar)
        mean_mse = jnp.mean((mean_vec - target_mean) ** 2)
        cov_mse = jnp.mean((cov_mat - target_cov) ** 2)
        weighted = constraint_scale * cfg.constraints.weight * (
            cfg.constraints.lambda_mean * mean_mse
            + cfg.constraints.lambda_cov * cov_mse
        )

        return {
            "constraint/mid_mean_mse": mean_mse,
            "constraint/mid_cov_mse": cov_mse,
            "constraint/mid_total": weighted,
            "constraint/mid_mean_x": mean_vec[0],
            "constraint/mid_mean_y": mean_vec[1],
            "constraint/anneal_scale": constraint_scale,
            "stage/two_stage_scale": stage2_scale,
        }

    if ctype == "kde_path":
        params = dist_utils.safe_unreplicate(cfg, train_state.params)
        tau = jnp.clip(tbatch, 0.0, 1.0)
        x_tau = _path_positions(params, train_state.apply_fn, x0batch, label_batch, tau)
        x_tau = _maybe_clip_constraint_state(x_tau, cfg)

        obs = _select_kde_observations(x0batch, x1batch, cfg)
        bandwidth = float(getattr(cfg.constraints, "kde_bandwidth", 0.25))
        logp = _kde_log_density(x_tau, obs, bandwidth)

        penalty_type = getattr(cfg.constraints, "kde_penalty", "hinge")
        if penalty_type == "hinge":
            logp_floor = float(getattr(cfg.constraints, "kde_logp_floor", -2.0))
            penalties = jax.nn.relu(logp_floor - logp) ** 2
        elif penalty_type == "nll":
            penalties = -logp
        else:
            raise ValueError(f"Unknown constraints.kde_penalty: {penalty_type}")

        penalty_clip = float(getattr(cfg.constraints, "kde_penalty_clip", 0.0))
        if penalty_clip > 0:
            penalties = jnp.minimum(penalties, penalty_clip)

        lambda_kde = float(getattr(cfg.constraints, "lambda_kde", 1.0))
        penalty_mean = jnp.mean(penalties)
        weighted = constraint_scale * cfg.constraints.weight * lambda_kde * penalty_mean

        return {
            "constraint/kde_logp_mean": jnp.mean(logp),
            "constraint/kde_penalty_mean": penalty_mean,
            "constraint/kde_total": weighted,
            "constraint/kde_tau_mean": jnp.mean(tau),
            "constraint/anneal_scale": constraint_scale,
            "stage/two_stage_scale": stage2_scale,
        }

    if ctype == "box_path":
        return {}

    return {}


def compute_endpoint_matching_metrics(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> Dict[str, float]:
    """Compute endpoint distribution matching errors for X_{0,1}(x0)."""
    endpoint_cfg = getattr(cfg.training, "endpoint_matching", None)
    if endpoint_cfg is None or not getattr(endpoint_cfg, "enabled", False):
        return {}

    data_args = loss_fn_args[1:]
    (
        x0batch,
        x1batch,
        label_batch,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    if label_batch is not None:
        label_batch = jnp.squeeze(label_batch)

    params = dist_utils.safe_unreplicate(cfg, train_state.params)
    tau = jnp.ones((x0batch.shape[0],), dtype=x0batch.dtype)
    x1_hat = _path_positions(
        params,
        train_state.apply_fn,
        x0batch,
        label_batch,
        tau,
    )
    endpoint_mmd = _endpoint_matching_mmd(cfg, x0batch, x1_hat, x1batch)
    endpoint_weight = float(getattr(endpoint_cfg, "weight", 1.0))

    return {
        "endpoint_matching/mmd": endpoint_mmd,
        "endpoint_matching/weighted": endpoint_weight * endpoint_mmd,
    }


def compute_forbidden_box_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> Dict[str, float]:
    """Log learned and analytical path rates inside a configured box."""
    bounds = _forbidden_box_bounds(cfg)
    if bounds is None:
        return {}

    data_args = loss_fn_args[1:]
    (
        x0batch,
        x1batch,
        label_batch,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    if label_batch is not None:
        label_batch = jnp.squeeze(label_batch)

    box_cfg = getattr(cfg.logging, "forbidden_box", None)
    params = dist_utils.safe_unreplicate(cfg, train_state.params)

    metrics = {}
    times = getattr(box_cfg, "times", None)
    if times is not None:
        learned_any = jnp.zeros((x0batch.shape[0],), dtype=bool)
        interp_any = jnp.zeros((x0batch.shape[0],), dtype=bool)
        for time_value in times:
            tau_i = jnp.full((x0batch.shape[0],), float(time_value), dtype=x0batch.dtype)
            learned_i = _path_positions(
                params,
                train_state.apply_fn,
                x0batch,
                label_batch,
                tau_i,
            )
            interp_i = statics.interp.batch_calc_It(tau_i, x0batch, x1batch, label_batch)
            learned_any = learned_any | _points_in_forbidden_box(learned_i, bounds)
            interp_any = interp_any | _points_in_forbidden_box(interp_i, bounds)

        learned_any_f = learned_any.astype(jnp.float32)
        interp_any_f = interp_any.astype(jnp.float32)
        metrics.update(
            {
                "forbidden_box/learned_path_rate": jnp.mean(learned_any_f),
                "forbidden_box/interpolant_path_rate": jnp.mean(interp_any_f),
            }
        )

    return metrics


def log_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    grads: jnp.ndarray,
    loss_value: float,
    loss_fn_args: Tuple,
    prng_key: jnp.ndarray,
    step_time: float,
) -> jnp.ndarray:
    """Log some metrics to wandb, make a figure, and checkpoint the parameters."""

    grads = dist_utils.safe_unreplicate(cfg, grads)
    loss_value = dist_utils.safe_index(cfg, jnp.array(loss_value))
    step = dist_utils.safe_index(cfg, train_state.step)
    learning_rate = statics.schedule(step)

    # Standard metrics
    metrics = {
        f"loss": loss_value,
        f"grad": compute_grad_norm(grads),
        f"learning_rate": learning_rate,
        f"step_time": step_time,
    }

    # Log constraint errors if configured.
    try:
        metrics.update(
            compute_constraint_metrics(cfg, train_state, loss_fn_args, statics=statics)
        )
    except Exception as e:
        print(f"Warning: Constraint metric computation failed: {e}")

    # Log endpoint distribution matching errors if configured.
    try:
        metrics.update(
            compute_endpoint_matching_metrics(cfg, train_state, loss_fn_args)
        )
    except Exception as e:
        print(f"Warning: Endpoint matching metric computation failed: {e}")

    # Log forbidden-box diagnostics if configured. These metrics evaluate the
    # learned flow map at several path times, so keep them off the per-step path.
    try:
        box_cfg = getattr(cfg.logging, "forbidden_box", None)
        box_freq = int(getattr(box_cfg, "freq", getattr(cfg.logging, "visual_freq", 1)))
        if box_freq > 0 and (int(step) % box_freq) == 0:
            metrics.update(
                compute_forbidden_box_metrics(cfg, statics, train_state, loss_fn_args)
            )
    except Exception as e:
        print(f"Warning: Forbidden-box metric computation failed: {e}")

    # Log stage-2 schedule scale when present.
    try:
        data_args = loss_fn_args[1:]
        (_, _, _, _, _, _, _, _, _, stage2_scale_batch) = (
            dist_utils.unreplicate_loss_fn_args(cfg, data_args)
        )
        metrics["stage/two_stage_scale"] = jnp.mean(jnp.squeeze(stage2_scale_batch))
    except Exception:
        pass

    # Compute FID on-the-fly if enabled and at the right frequency
    if (
        hasattr(cfg.logging, "fid_freq")
        and cfg.logging.fid_freq > 0
        and (step % cfg.logging.fid_freq) == 0
        and step > 0
    ):
        try:
            # Get step counts configuration - can be a list or single value
            steps_config = getattr(cfg.logging, "fid_n_steps_flow", 8)

            # Convert to list if single value
            if isinstance(steps_config, (list, tuple)):
                n_steps_list = list(steps_config)
            else:
                n_steps_list = [steps_config]

            # Compute FID for each step count
            for n_steps in n_steps_list:
                fid_score, prng_key = compute_fid_on_the_fly(
                    cfg,
                    statics,
                    train_state,
                    prng_key,
                    n_samples=getattr(cfg.logging, "fid_n_samples", 10000),
                    batch_size=getattr(cfg.logging, "fid_batch_size", 256),
                    n_steps_flow=n_steps,
                )
                # Log with step-specific key
                metrics[f"fid_{n_steps}_steps"] = fid_score
        except Exception as e:
            print(f"Warning: FID computation failed: {e}")

    wandb.log(metrics)

    if (
        hasattr(cfg.logging, "visual_freq")
        and cfg.logging.visual_freq > 0
        and (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.visual_freq) == 0
    ):
        if is_lowd_problem(cfg):
            prng_key = make_lowd_plot(cfg, statics, train_state, prng_key)
        elif is_image_problem(cfg):
            prng_key = make_image_plot(cfg, statics, train_state, prng_key)

        make_loss_fn_args_plot(cfg, statics, train_state, loss_fn_args)

    if (dist_utils.safe_index(cfg, train_state.step) % cfg.logging.save_freq) == 0:
        save_state(train_state, cfg)

    return prng_key


def make_lowd_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> None:
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 5, 10, 25]
    titles = ["base and target"] + [rf"${step}$-step" for step in steps]

    ## extract target samples
    plot_x1s = extract_x1_from_batch(next(statics.ds))[: cfg.logging.plot_bs]

    ## draw multi-step samples from the model
    x0s = statics.sample_rho0(cfg.logging.plot_bs, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), cfg.logging.plot_bs, cfg.problem.d))
    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            -jnp.ones(cfg.logging.plot_bs),
        )

    # Track full direct and multi-step trajectories for a subset of particles.
    line_bs = min(cfg.logging.plot_bs, getattr(cfg.logging, "line_plot_bs", 1000))
    n_line_times = max(2, int(getattr(cfg.logging, "line_plot_n_times", 20)))
    line_times = np.linspace(0.0, 1.0, n_line_times)
    x0_line = x0s[:line_bs]
    x1_line = plot_x1s[:line_bs]
    labels_line = -jnp.ones(line_bs)
    one_step_line_paths = _one_step_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_line,
        labels_line,
        line_times,
    )

    multi_step_counts = getattr(cfg.logging, "multi_step_line_steps", steps)
    if isinstance(multi_step_counts, int):
        multi_step_counts = [multi_step_counts]
    multi_step_counts = [max(1, int(step)) for step in multi_step_counts]
    multi_line_bs = min(
        line_bs,
        int(getattr(cfg.logging, "multi_step_line_plot_bs", min(line_bs, 500))),
    )
    x0_multi_line = x0s[:multi_line_bs]
    x1_multi_line = plot_x1s[:multi_line_bs]
    labels_multi_line = -jnp.ones(multi_line_bs)
    multi_step_line_paths = [
        (
            step,
            _multi_step_paths(
                train_state.apply_fn,
                params_for_visual,
                x0_multi_line,
                labels_multi_line,
                step,
            ),
        )
        for step in multi_step_counts
    ]
    euler_step_counts = getattr(cfg.logging, "euler_line_steps", [5, 10, 25, 100])
    if isinstance(euler_step_counts, int):
        euler_step_counts = [euler_step_counts]
    euler_step_counts = [max(1, int(step)) for step in euler_step_counts]
    euler_line_paths = [
        (
            step,
            _euler_paths(
                train_state.apply_fn,
                params_for_visual,
                x0_multi_line,
                labels_multi_line,
                step,
            ),
        )
        for step in euler_step_counts
    ]

    # determine plotting limits from observed and generated samples
    all_points = np.concatenate(
        [
            np.asarray(x0s),
            np.asarray(plot_x1s),
            np.asarray(xhats).reshape((-1, cfg.problem.d)),
            one_step_line_paths.reshape((-1, cfg.problem.d)),
            *[
                paths.reshape((-1, cfg.problem.d))
                for _, paths in multi_step_line_paths
            ],
            *[
                paths.reshape((-1, cfg.problem.d))
                for _, paths in euler_line_paths
            ],
        ],
        axis=0,
    )
    xlim, ylim = finite_lowd_limits(all_points)

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax in axs.ravel():
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[jj]
        ax.set_title(title, fontsize=fontsize)

        if jj == 0:
            ax.scatter(x0s[:, 0], x0s[:, 1], s=0.1, alpha=0.5, marker="o", c="black")
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )
        else:
            ax.scatter(
                plot_x1s[:, 0], plot_x1s[:, 1], s=0.1, alpha=0.5, marker="o", c="C0"
            )

            ax.scatter(
                xhats[jj - 1, :, 0],
                xhats[jj - 1, :, 1],
                s=0.1,
                alpha=0.5,
                marker="o",
                c="black",
            )
        _draw_forbidden_box(ax, cfg)

    wandb.log({"samples": wandb.Image(fig)})

    # Visualize direct trajectory slices X_{0,t}(x0) for fixed times.
    traj_bs = min(cfg.logging.plot_bs, getattr(cfg.logging, "traj_plot_bs", 2000))
    x0_traj = x0s[:traj_bs]
    x1_traj = plot_x1s[:traj_bs]
    labels = -jnp.ones(traj_bs)
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]

    traj_fig, traj_axs = plt.subplots(
        nrows=1,
        ncols=len(time_points),
        figsize=(fw * len(time_points), fh),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for tt, ax in zip(time_points, traj_axs.ravel()):
        if tt == 0.0:
            xt = x0_traj
        else:
            xt = jax.vmap(
                lambda x, lbl: train_state.apply_fn(
                    params_for_visual,
                    0.0,
                    tt,
                    x,
                    label=lbl,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(x0_traj, labels)

        ax.scatter(x1_traj[:, 0], x1_traj[:, 1], s=0.3, alpha=0.5, marker="o", c="C0")
        ax.scatter(xt[:, 0], xt[:, 1], s=0.3, alpha=0.5, marker="o", c="black")
        ax.set_title(rf"$t={tt:.2f}$", fontsize=fontsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)
        _draw_forbidden_box(ax, cfg)

    wandb.log({"trajectory_times": wandb.Image(traj_fig)})

    line_fig, line_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    _draw_trajectory_paths(
        line_ax,
        one_step_line_paths,
        np.asarray(x0_line),
        np.asarray(x1_line),
        title=f"1-step trajectory X_{{0,t}}(x), {n_line_times} times",
        xlim=xlim,
        ylim=ylim,
        fontsize=fontsize,
        cfg=cfg,
    )
    line_ax.legend(loc="upper right", fontsize=10, markerscale=6, frameon=True)
    wandb.log({"trajectory_1step_lines": wandb.Image(line_fig)})

    multi_fig, multi_axs = plt.subplots(
        nrows=1,
        ncols=len(multi_step_line_paths),
        figsize=(fw * len(multi_step_line_paths), fh),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (step, paths) in zip(multi_axs.ravel(), multi_step_line_paths):
        _draw_trajectory_paths(
            ax,
            paths,
            np.asarray(x0_multi_line),
            np.asarray(x1_multi_line),
            title=f"{step}-step rollout",
            xlim=xlim,
            ylim=ylim,
            fontsize=fontsize,
            cfg=cfg,
        )
    multi_axs.ravel()[0].legend(
        loc="upper right", fontsize=10, markerscale=6, frameon=True
    )

    multistep_box_metrics = _rollout_forbidden_box_metrics(
        cfg, multi_step_line_paths, "multistep"
    )
    wandb.log(
        {
            "trajectory_multistep_lines": wandb.Image(multi_fig),
            **multistep_box_metrics,
        }
    )

    euler_fig, euler_axs = plt.subplots(
        nrows=1,
        ncols=len(euler_line_paths),
        figsize=(fw * len(euler_line_paths), fh),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (step, paths) in zip(euler_axs.ravel(), euler_line_paths):
        _draw_trajectory_paths(
            ax,
            paths,
            np.asarray(x0_multi_line),
            np.asarray(x1_multi_line),
            title=f"{step}-step Euler",
            xlim=xlim,
            ylim=ylim,
            fontsize=fontsize,
            cfg=cfg,
        )
    euler_axs.ravel()[0].legend(
        loc="upper right", fontsize=10, markerscale=6, frameon=True
    )
    euler_box_metrics = _rollout_forbidden_box_metrics(
        cfg, euler_line_paths, "euler"
    )
    wandb.log(
        {
            "trajectory_euler_lines": wandb.Image(euler_fig),
            **euler_box_metrics,
        }
    )
    return prng_key


def make_image_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> None:
    """Make a plot of the generated images."""
    # Use flow map batch sampler for single-device visualization
    batch_sample = flow_map.batch_sample

    # Get parameters for visualization (already unreplicated)
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 1, 1
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 4, 8, 16]

    titles = [rf"{step}-step" for step in steps]

    ## draw multi-step samples from the model
    n_images = 16
    x0s = statics.sample_rho0(n_images, prng_key)
    prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), n_images, *cfg.problem.image_dims))

    ## set up conditioning information
    if cfg.training.conditional:
        if cfg.training.class_dropout > 0:
            assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
            labels = jnp.array(np.random.choice(cfg.problem.num_classes + 1, n_images))
        else:
            labels = jnp.array(np.random.choice(cfg.problem.num_classes, n_images))
        prng_key = jax.random.split(prng_key)[0]
    else:
        labels = None

    for kk, step in enumerate(steps):
        xhats[kk] = batch_sample(
            train_state.apply_fn,
            params_for_visual,
            x0s,
            step,
            labels,
        )

    # transpose (S, N, C, H, W) -> (S, N, H, W, C)
    xhats = xhats.transpose(0, 1, 3, 4, 2)

    ## make the image grids
    nrows = 2 if n_images > 8 else 1
    ncols = n_images // nrows

    for kk, title in enumerate(titles):
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(fw * ncols, fh * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axs = axs.reshape((nrows, ncols))

        fig.suptitle(title, fontsize=fontsize)

        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")

        ## visualize the generated images
        for ii in range(nrows):
            for jj in range(ncols):
                index = ii * ncols + jj
                image = datasets.unnormalize_image(xhats[kk, index])
                axs[ii, jj].imshow(image)

        wandb.log({titles[kk]: wandb.Image(fig)})

    return prng_key


def make_loss_fn_args_plot(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
) -> None:
    """Make a plot of the loss function arguments."""
    # unpack the full loss arguments
    data_args = loss_fn_args[1:]
    (x0batch, x1batch, label_batch, sbatch, tbatch, _, _, _, _, _) = (
        dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    )

    # remove pmap reshaping
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    if label_batch is not None:
        label_batch = jnp.squeeze(label_batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = statics.interp.batch_calc_It(tbatch, x0batch, x1batch, label_batch)
    lowd_problem = is_lowd_problem(cfg)

    ## set up plot array
    if lowd_problem:
        titles = [r"$x_0$", r"$x_1$", r"$x_t$", r"$(s, t)$"]
        all_points = np.concatenate(
            [np.asarray(x0batch), np.asarray(x1batch), np.asarray(xtbatch)],
            axis=0,
        )
        xlim, ylim = finite_lowd_limits(all_points)
    else:
        titles = [r"$(s, t)$"]

    ## construct the figure
    nrows = 1
    ncols = len(titles)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fw * ncols, fh * nrows),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )

    for kk, ax in enumerate(axs.ravel()):
        if kk == (len(titles) - 1):
            ax.set_xlim([-0.1, 1.1])
            ax.set_ylim([-0.1, 1.1])
        else:
            if lowd_problem:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

        ax.set_aspect("equal")
        ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
        ax.tick_params(axis="both", labelsize=fontsize)

    # do the plotting
    for jj in range(ncols):
        title = titles[jj]
        ax = axs[0, jj]
        ax.set_title(title, fontsize=fontsize)

        if lowd_problem:
            if jj == 0:
                ax.scatter(x0batch[:, 0], x0batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 1:
                ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 2:
                ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
            elif jj == 3:
                ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")
            if jj < 3:
                _draw_forbidden_box(ax, cfg)
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})

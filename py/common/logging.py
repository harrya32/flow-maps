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


def compute_constraint_metrics(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    loss_fn_args: Tuple,
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
        _,
        tbatch,
        _,
        _,
        _,
        constraint_scale_batch,
        stage2_scale_batch,
    ) = dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    tbatch = jnp.squeeze(tbatch)
    constraint_scale = jnp.mean(jnp.squeeze(constraint_scale_batch))
    stage2_scale = jnp.mean(jnp.squeeze(stage2_scale_batch))
    if bool(getattr(cfg.constraints, "stage2_only", False)):
        constraint_scale = constraint_scale * stage2_scale
    if label_batch is not None:
        label_batch = jnp.squeeze(label_batch)

    params = dist_utils.safe_unreplicate(cfg, train_state.params)
    ctype = cfg.constraints.type

    if ctype == "mid_moment":
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

    return {}


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
        metrics.update(compute_constraint_metrics(cfg, train_state, loss_fn_args))
    except Exception as e:
        print(f"Warning: Constraint metric computation failed: {e}")

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
    plot_x1s = next(statics.ds)[: cfg.logging.plot_bs]

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

    # determine plotting limits from observed and generated samples
    all_points = np.concatenate(
        [np.asarray(x0s), np.asarray(plot_x1s), np.asarray(xhats).reshape((-1, cfg.problem.d))],
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

    wandb.log({"trajectory_times": wandb.Image(traj_fig)})

    # Visualize full trajectory lines for a subset of particles.
    line_bs = min(cfg.logging.plot_bs, getattr(cfg.logging, "line_plot_bs", 1000))
    n_line_times = max(2, int(getattr(cfg.logging, "line_plot_n_times", 20)))
    line_times = np.linspace(0.0, 1.0, n_line_times)
    x0_line = x0s[:line_bs]
    labels_line = -jnp.ones(line_bs)

    line_paths = np.zeros((line_bs, n_line_times, cfg.problem.d))
    line_paths[:, 0, :] = np.asarray(x0_line)
    for idx, tt in enumerate(line_times[1:], start=1):
        xt = jax.vmap(
            lambda x, lbl: train_state.apply_fn(
                params_for_visual,
                0.0,
                float(tt),
                x,
                label=lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0_line, labels_line)
        line_paths[:, idx, :] = np.asarray(xt)

    # Build line segments for efficient rendering.
    line_segments = np.stack([line_paths[:, :-1, :], line_paths[:, 1:, :]], axis=2)
    line_segments = line_segments.reshape((-1, 2, cfg.problem.d))

    line_fig, line_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    line_ax.scatter(
        x0s[:, 0], x0s[:, 1], s=0.2, alpha=0.25, marker="o", c="gray", label="base"
    )
    line_ax.scatter(
        plot_x1s[:, 0], plot_x1s[:, 1], s=0.2, alpha=0.25, marker="o", c="C0", label="target"
    )
    line_ax.add_collection(
        LineCollection(line_segments, colors="black", linewidths=0.25, alpha=0.2)
    )
    line_ax.set_title(f"{line_bs} trajectories, {n_line_times} times", fontsize=fontsize)
    line_ax.set_xlim(xlim)
    line_ax.set_ylim(ylim)
    line_ax.set_aspect("equal")
    line_ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
    line_ax.tick_params(axis="both", labelsize=fontsize)
    line_ax.legend(loc="upper right", fontsize=10, markerscale=6, frameon=True)

    wandb.log({"trajectory_lines": wandb.Image(line_fig)})
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
    (x0batch, x1batch, _, sbatch, tbatch, _, _, _, _, _) = (
        dist_utils.unreplicate_loss_fn_args(cfg, data_args)
    )

    # remove pmap reshaping
    x0batch = jnp.squeeze(x0batch)
    x1batch = jnp.squeeze(x1batch)
    sbatch = jnp.squeeze(sbatch)
    tbatch = jnp.squeeze(tbatch)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    # compute xts
    xtbatch = statics.interp.batch_calc_It(tbatch, x0batch, x1batch)
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
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})

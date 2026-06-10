"""
Nicholas M. Boffi
10/5/25

Loss functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import loss_args

Parameters = Dict[str, Dict]


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.mean(batched_outputs)

    return wrapper


def has_constraint(cfg: config_dict.ConfigDict) -> bool:
    return hasattr(cfg, "constraints") and getattr(cfg.constraints, "enabled", False)


def has_two_stage(cfg: config_dict.ConfigDict) -> bool:
    return hasattr(cfg.training, "two_stage") and getattr(
        cfg.training.two_stage, "enabled", False
    )


def _covariance(x: jnp.ndarray) -> jnp.ndarray:
    """Compute sample covariance matrix for a batch of vectors."""
    x_centered = x - jnp.mean(x, axis=0, keepdims=True)
    denom = jnp.maximum(x.shape[0] - 1, 1)
    return (x_centered.T @ x_centered) / denom


def _maybe_clip_constraint_state(
    x: jnp.ndarray, cfg: config_dict.ConfigDict
) -> jnp.ndarray:
    """Optionally clip X_{0,t*}(x0) when evaluating constraints."""
    x_clip = float(getattr(cfg.constraints, "x_clip", 0.0))
    clip_mode = getattr(cfg.constraints, "x_clip_mode", "hard")
    if x_clip > 0:
        return _maybe_clip(x, x_clip, clip_mode=clip_mode)
    return x


def _maybe_clip(x: jnp.ndarray, x_clip: float, clip_mode: str = "hard") -> jnp.ndarray:
    """Optionally clip/squash an array by an absolute value."""
    if x_clip > 0:
        if clip_mode == "hard":
            return jnp.clip(x, -x_clip, x_clip)
        elif clip_mode == "tanh":
            return x_clip * jnp.tanh(x / x_clip)
        else:
            raise ValueError(f"Unknown clip_mode: {clip_mode}")
    return x


def _select_kde_observations(
    x0: jnp.ndarray, x1: jnp.ndarray, cfg: config_dict.ConfigDict
) -> jnp.ndarray:
    """Select observed points used for KDE in path constraints."""
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
    """Evaluate isotropic Gaussian KDE log-density at query points."""
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
    """Compute a multi-bandwidth RBF kernel matrix."""
    diffs = x[:, None, :] - y[None, :, :]
    sqdist = jnp.sum(diffs * diffs, axis=-1)

    bws = jnp.maximum(jnp.asarray(bandwidths, dtype=x.dtype), 1e-6)
    scales = 2.0 * (bws[:, None, None] ** 2)
    kernels = jnp.exp(-sqdist[None, :, :] / scales)
    return jnp.mean(kernels, axis=0)


def _path_positions(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    tau: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
) -> jnp.ndarray:
    """Compute X_{0,tau_i}(x0_i) for each sample i."""
    if label is None:
        return jax.vmap(
            lambda x, tt: X.apply(
                params,
                0.0,
                tt,
                x,
                None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, tau)
    else:
        return jax.vmap(
            lambda x, lbl, tt: X.apply(
                params,
                0.0,
                tt,
                x,
                lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, label, tau)


def mid_moment_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Constraint on mean/covariance of the pushforward at time t*."""
    t_star = float(cfg.constraints.time)

    if label is None:
        x_tstar = jax.vmap(
            lambda x: X.apply(
                params,
                0.0,
                t_star,
                x,
                None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0)
    else:
        x_tstar = jax.vmap(
            lambda x, lbl: X.apply(
                params,
                0.0,
                t_star,
                x,
                lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, label)

    x_tstar = _maybe_clip_constraint_state(x_tstar, cfg)
    target_mean = jnp.asarray(cfg.constraints.target_mean, dtype=x_tstar.dtype)
    target_cov = jnp.asarray(cfg.constraints.target_cov, dtype=x_tstar.dtype)

    mean_mse = jnp.mean((jnp.mean(x_tstar, axis=0) - target_mean) ** 2)
    cov_mse = jnp.mean((_covariance(x_tstar) - target_cov) ** 2)

    loss = cfg.constraints.weight * (
        cfg.constraints.lambda_mean * mean_mse
        + cfg.constraints.lambda_cov * cov_mse
    )
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


def kde_path_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Penalize path points that leave high-density observed regions via KDE."""
    tau = jnp.clip(t, 0.0, 1.0)
    x_tau = _path_positions(params, x0, label, tau, X=X)
    x_tau = _maybe_clip_constraint_state(x_tau, cfg)

    obs = _select_kde_observations(x0, x1, cfg)
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
    loss = cfg.constraints.weight * lambda_kde * jnp.mean(penalties)
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


def endpoint_mmd_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """MMD between model pushforward X_{0,1}(x0) and target batch x1."""
    two_stage_cfg = cfg.training.two_stage
    tau = jnp.ones((x0.shape[0],), dtype=x0.dtype)
    x1_hat = _path_positions(params, x0, label, tau, X=X)

    x_clip = float(getattr(two_stage_cfg, "endpoint_mmd_x_clip", 0.0))
    clip_mode = getattr(two_stage_cfg, "endpoint_mmd_clip_mode", "hard")
    x1_hat = _maybe_clip(x1_hat, x_clip, clip_mode=clip_mode)
    x1 = _maybe_clip(x1, x_clip, clip_mode=clip_mode)

    if bool(getattr(two_stage_cfg, "endpoint_mmd_normalize", False)):
        eps = float(getattr(two_stage_cfg, "endpoint_mmd_normalize_eps", 1e-3))
        center = jnp.mean(x1, axis=0, keepdims=True)
        scale = jnp.std(x1, axis=0, keepdims=True)
        scale = jnp.maximum(scale, eps)
        x1_hat = (x1_hat - center) / scale
        x1 = (x1 - center) / scale

    bandwidths = jnp.asarray(
        getattr(two_stage_cfg, "endpoint_mmd_bandwidths", [0.05, 0.1, 0.2, 0.4]),
        dtype=x1.dtype,
    )

    k_xx = _rbf_kernel_mixture(x1_hat, x1_hat, bandwidths)
    k_yy = _rbf_kernel_mixture(x1, x1, bandwidths)
    k_xy = _rbf_kernel_mixture(x1_hat, x1, bandwidths)

    k_xx_term = jnp.mean(k_xx)
    if bool(getattr(two_stage_cfg, "endpoint_mmd_detach_self", False)):
        k_xx_term = jax.lax.stop_gradient(k_xx_term)

    k_xx_weight = float(getattr(two_stage_cfg, "endpoint_mmd_self_weight", 1.0))
    cross_weight = float(getattr(two_stage_cfg, "endpoint_mmd_cross_weight", 2.0))

    mmd2 = k_xx_weight * k_xx_term + jnp.mean(k_yy) - cross_weight * jnp.mean(k_xy)
    mmd2 = jnp.maximum(mmd2, 0.0)
    return jnp.nan_to_num(mmd2, nan=0.0, posinf=1e6, neginf=1e6)


def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Compute the diagonal (interpolant) term of the loss."""

    # compute interpolant and the target
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)

    # compute the weighted loss
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)

    # Diagonal uses s=t
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def psd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    psd_type: str,
    stopgrad_type: str,
) -> float:
    """Compute the PSD (Progressive Self-Distillation) term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the full jump
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )

    # break it down into two jumps
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )

        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )

        X_ut, phi_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)

    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # Compute the distillation loss
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )

    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            t,
            Xst_Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")
    error = b_eval - dt_Xst
    lsd_loss = jnp.sum(error**2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st


def esd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the ESD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the derivative with respect to the first time
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )

    # stopgrad everything to avoid backpropagating through the UNet spatial Jacobian
    if stopgrad_type == "full":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.lax.stop_gradient(
            jax.jvp(
                lambda x: X.apply(
                    teacher_params, s, t, x, label, train=False, rngs=rng
                ),
                primals=(Is,),
                tangents=(b_eval,),
            )
        )

    # stopgrad the b, so it's like EMD
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    # pure residual minimization -- no stopgrad
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    esd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * esd_loss + weight_st


def setup_loss(
    cfg: config_dict.ConfigDict, net: flow_map.FlowMap, interp: interpolant.Interpolant
) -> Callable:
    """Setup the loss function."""

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")

    # Pure diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def diagonal_only_loss(params, x0, x1, label, t, rng):
        return diagonal_term(
            params,
            x0,
            x1,
            label,
            t,
            rng,
            interp=interp,
            X=net,
        )

    # Pure off-diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
    def offdiagonal_only_loss(
        params, teacher_params, x0, x1, label, s, t, u, h, dropout_keys
    ):
        rng = {"dropout": dropout_keys}

        if cfg.training.loss_type == "psd":
            return psd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                u,
                h,
                rng,
                interp=interp,
                X=net,
                psd_type=cfg.training.psd_type,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        elif cfg.training.loss_type == "lsd":
            return lsd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        elif cfg.training.loss_type == "esd":
            return esd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        else:
            raise ValueError(f"Unknown loss_type: {cfg.training.loss_type}")

    def loss(
        params,
        teacher_params,
        x0,
        x1,
        label,
        s,
        t,
        u,
        h,
        dropout_keys,
        constraint_scale_batch,
        stage2_scale_batch,
    ):
        """Split batch into diagonal and off-diagonal portions."""
        total_bs = x0.shape[0]
        diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)
        stage2_scale = jnp.mean(stage2_scale_batch)

        if has_two_stage(cfg):
            two_stage_cfg = cfg.training.two_stage
            diag_weight_stage1 = float(getattr(two_stage_cfg, "diag_weight_stage1", 1.0))
            diag_weight_stage2 = float(getattr(two_stage_cfg, "diag_weight_stage2", 0.0))
            offdiag_weight_stage1 = float(
                getattr(two_stage_cfg, "offdiag_weight_stage1", 1.0)
            )
            offdiag_weight_stage2 = float(
                getattr(two_stage_cfg, "offdiag_weight_stage2", 1.0)
            )
        else:
            diag_weight_stage1 = 1.0
            diag_weight_stage2 = 1.0
            offdiag_weight_stage1 = 1.0
            offdiag_weight_stage2 = 1.0

        diag_weight = (1.0 - stage2_scale) * diag_weight_stage1 + (
            stage2_scale * diag_weight_stage2
        )
        offdiag_weight = (1.0 - stage2_scale) * offdiag_weight_stage1 + (
            stage2_scale * offdiag_weight_stage2
        )

        weighted_base_loss = 0.0
        base_normalizer = 0.0

        # Compute diagonal loss on first portion
        if diag_bs > 0:
            label_diag = None if label is None else label[:diag_bs]
            diag_loss = diagonal_only_loss(
                params,
                x0[:diag_bs],
                x1[:diag_bs],
                label_diag,
                t[:diag_bs],
                dropout_keys[:diag_bs],
            )
            weighted_base_loss += diag_weight * diag_loss * diag_bs
            base_normalizer += diag_weight * diag_bs

        # Compute off-diagonal loss on second portion
        if offdiag_bs > 0:
            label_offdiag = None if label is None else label[diag_bs:]
            u_offdiag = None if u is None else u[diag_bs:]
            h_offdiag = None if h is None else h[diag_bs:]

            offdiag_loss = offdiagonal_only_loss(
                params,
                teacher_params,
                x0[diag_bs:],
                x1[diag_bs:],
                label_offdiag,
                s[diag_bs:],
                t[diag_bs:],
                u_offdiag,
                h_offdiag,
                dropout_keys[diag_bs:],
            )
            weighted_base_loss += offdiag_weight * offdiag_loss * offdiag_bs
            base_normalizer += offdiag_weight * offdiag_bs

        # Normalize by effective weighted batch size.
        base_normalizer = jnp.maximum(base_normalizer, 1.0)
        total_loss = weighted_base_loss / base_normalizer

        # Stage-2 endpoint distribution matching.
        if has_two_stage(cfg):
            endpoint_weight = float(
                getattr(cfg.training.two_stage, "endpoint_mmd_weight", 0.0)
            )
            if endpoint_weight > 0:
                total_loss += (
                    stage2_scale
                    * endpoint_weight
                    * endpoint_mmd_constraint(
                        params,
                        x0,
                        x1,
                        label,
                        X=net,
                        cfg=cfg,
                    )
                )

        # Optional trajectory constraint terms
        if has_constraint(cfg):
            constraint_scale = jnp.mean(constraint_scale_batch)
            if bool(getattr(cfg.constraints, "stage2_only", False)):
                constraint_scale = constraint_scale * stage2_scale
            if cfg.constraints.type == "mid_moment":
                total_loss += constraint_scale * mid_moment_constraint(
                    params,
                    x0,
                    label,
                    X=net,
                    cfg=cfg,
                )
            elif cfg.constraints.type == "kde_path":
                total_loss += constraint_scale * kde_path_constraint(
                    params,
                    x0,
                    x1,
                    label,
                    t,
                    X=net,
                    cfg=cfg,
                )
            else:
                raise ValueError(f"Unknown constraint type: {cfg.constraints.type}")

        return total_loss

    return loss

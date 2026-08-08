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
from . import maizels

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


def has_endpoint_matching(cfg: config_dict.ConfigDict) -> bool:
    return hasattr(cfg.training, "endpoint_matching") and getattr(
        cfg.training.endpoint_matching, "enabled", False
    )


def uses_box_loss_points(cfg: config_dict.ConfigDict) -> bool:
    return (
        has_constraint(cfg)
        and getattr(cfg.constraints, "type", None) == "box_path"
        and getattr(cfg.constraints, "box_path_mode", "x0_t") == "loss_points"
    )


def uses_flow_map_box_loss_points(cfg: config_dict.ConfigDict) -> bool:
    return uses_box_loss_points(cfg) and _box_constraint_mode(cfg) == "flow_map"


def uses_maizels_loss_points(cfg: config_dict.ConfigDict) -> bool:
    return (
        has_constraint(cfg)
        and getattr(cfg.constraints, "type", None) == "maizels_lineage_path"
        and getattr(cfg.constraints, "path_mode", "flowmap") == "loss_points"
    )


def _box_constraint_mode(cfg: config_dict.ConfigDict) -> str:
    """Select how box path constraints are evaluated at off-diagonal points."""
    mode = getattr(cfg.constraints, "constraint_mode", "flow_map")
    if mode not in ("flow_map", "flow_matching"):
        raise ValueError(
            "constraints.constraint_mode must be 'flow_map' or 'flow_matching'"
        )
    return mode


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


def _box_bounds_from_cfg(
    cfg: config_dict.ConfigDict,
) -> Tuple[float, float, float, float]:
    """Read axis-aligned forbidden-box bounds from the constraint config."""
    constraint_cfg = cfg.constraints
    xlim = getattr(constraint_cfg, "xlim", None)
    ylim = getattr(constraint_cfg, "ylim", None)
    if xlim is None:
        xlim = getattr(constraint_cfg, "box_xlim", None)
    if ylim is None:
        ylim = getattr(constraint_cfg, "box_ylim", None)
    if xlim is None or ylim is None:
        raise ValueError(
            "box_path constraints require constraints.xlim and constraints.ylim"
        )

    xmin, xmax = [float(value) for value in xlim]
    ymin, ymax = [float(value) for value in ylim]
    return xmin, xmax, ymin, ymax


def _box_signed_distance(
    x: jnp.ndarray, bounds: Tuple[float, float, float, float]
) -> jnp.ndarray:
    """Signed distance to an axis-aligned box, positive outside and negative inside."""
    xmin, xmax, ymin, ymax = bounds
    center = jnp.asarray([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0], dtype=x.dtype)
    half_size = jnp.asarray(
        [(xmax - xmin) / 2.0, (ymax - ymin) / 2.0], dtype=x.dtype
    )
    q = jnp.abs(x - center) - half_size
    outside_q = jnp.maximum(q, 0.0)
    eps = jnp.asarray(1e-12, dtype=x.dtype)
    outside = jnp.sqrt(jnp.sum(outside_q * outside_q, axis=-1) + eps) - jnp.sqrt(eps)
    inside = jnp.minimum(jnp.maximum(q[..., 0], q[..., 1]), 0.0)
    return outside + inside


def _box_path_penalty(x: jnp.ndarray, cfg: config_dict.ConfigDict) -> jnp.ndarray:
    bounds = _box_bounds_from_cfg(cfg)
    signed_distance = _box_signed_distance(x, bounds)
    margin = float(getattr(cfg.constraints, "margin", 0.0))
    penalties = jax.nn.relu(margin - signed_distance) ** 2

    penalty_clip = float(getattr(cfg.constraints, "box_penalty_clip", 0.0))
    if penalty_clip > 0:
        penalties = jnp.minimum(penalties, penalty_clip)

    lambda_box = float(getattr(cfg.constraints, "lambda_box", 1.0))
    return cfg.constraints.weight * lambda_box * jnp.mean(penalties)


def _dive_gate_cfg_value(
    cfg: config_dict.ConfigDict,
    name: str,
    default,
):
    """Read dive-gate geometry from constraints, logging, then problem config."""
    constraint_cfg = cfg.constraints
    logging_cfg = getattr(getattr(cfg, "logging", None), "dive_gate", None)
    problem_cfg = cfg.problem

    if hasattr(constraint_cfg, name):
        return getattr(constraint_cfg, name)
    if logging_cfg is not None and hasattr(logging_cfg, name):
        return getattr(logging_cfg, name)
    if hasattr(problem_cfg, name):
        return getattr(problem_cfg, name)
    return default


def _dive_gate_geometry(
    cfg: config_dict.ConfigDict,
    dtype,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    depth = float(getattr(cfg.problem, "dive_gate_depth", 0.85))
    pre_checkpoint_center = jnp.asarray(
        _dive_gate_cfg_value(cfg, "pre_checkpoint_center", [-0.9, 0.0]),
        dtype=dtype,
    )
    pre_checkpoint_radii = jnp.asarray(
        _dive_gate_cfg_value(cfg, "pre_checkpoint_radii", [0.35, 0.24]),
        dtype=dtype,
    )
    gate_center = jnp.asarray(
        _dive_gate_cfg_value(cfg, "gate_center", [0.0, -depth]),
        dtype=dtype,
    )
    gate_radii = jnp.asarray(
        _dive_gate_cfg_value(cfg, "gate_radii", [0.45, 0.30]),
        dtype=dtype,
    )
    checkpoint_center = jnp.asarray(
        _dive_gate_cfg_value(cfg, "checkpoint_center", [0.9, 0.0]),
        dtype=dtype,
    )
    checkpoint_radii = jnp.asarray(
        _dive_gate_cfg_value(cfg, "checkpoint_radii", [0.35, 0.24]),
        dtype=dtype,
    )
    eps = jnp.asarray(1e-6, dtype=dtype)
    return (
        pre_checkpoint_center,
        jnp.maximum(pre_checkpoint_radii, eps),
        gate_center,
        jnp.maximum(gate_radii, eps),
        checkpoint_center,
        jnp.maximum(checkpoint_radii, eps),
    )


def _dive_gate_path_mode(cfg: config_dict.ConfigDict) -> str:
    mode = getattr(
        cfg.constraints,
        "path_mode",
        getattr(cfg.constraints, "constraint_mode", "flow_map"),
    )
    if mode == "auto":
        return "flow_map"

    if mode in ("euler", "flow_matching", "velocity"):
        return "euler"
    if mode in ("flow_map", "direct"):
        return "flow_map"

    raise ValueError(
        "constraints.path_mode must be one of 'auto', 'euler', or 'flow_map'"
    )


def _dive_gate_path_times(
    cfg: config_dict.ConfigDict,
    dtype,
) -> jnp.ndarray:
    path_times = getattr(cfg.constraints, "path_times", None)
    if path_times is None:
        n_times = int(getattr(cfg.constraints, "path_n_times", 101))
        if n_times < 2:
            raise ValueError("constraints.path_n_times must be >= 2")
        return jnp.linspace(0.0, 1.0, n_times, dtype=dtype)

    path_times = jnp.asarray(path_times, dtype=dtype)
    if path_times.shape[0] < 2:
        raise ValueError("constraints.path_times must contain at least two times")
    return path_times


def _soft_ellipse_indicator(
    x: jnp.ndarray,
    center: jnp.ndarray,
    radii: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    scaled = (x - center) / radii
    normalized_sqdist = jnp.sum(scaled * scaled, axis=-1)
    return jax.nn.sigmoid((1.0 - normalized_sqdist) / temperature)


def _dive_gate_soft_terms(
    paths: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> Tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    """Soft temporal-logic terms for: hit A, B, C, and in that order."""
    (
        pre_checkpoint_center,
        pre_checkpoint_radii,
        gate_center,
        gate_radii,
        checkpoint_center,
        checkpoint_radii,
    ) = _dive_gate_geometry(cfg, paths.dtype)
    temperature = float(getattr(cfg.constraints, "indicator_temperature", 0.08))
    eps = jnp.asarray(float(getattr(cfg.constraints, "eps", 1e-6)), dtype=paths.dtype)

    p_a = _soft_ellipse_indicator(
        paths,
        pre_checkpoint_center,
        pre_checkpoint_radii,
        temperature,
    )
    p_b = _soft_ellipse_indicator(paths, gate_center, gate_radii, temperature)
    p_c = _soft_ellipse_indicator(
        paths,
        checkpoint_center,
        checkpoint_radii,
        temperature,
    )

    no_a_prefix_inclusive = jnp.cumprod(jnp.clip(1.0 - p_a, 0.0, 1.0), axis=1)
    no_a_before = jnp.concatenate(
        [jnp.ones_like(no_a_prefix_inclusive[:, :1]), no_a_prefix_inclusive[:, :-1]],
        axis=1,
    )
    no_b_prefix_inclusive = jnp.cumprod(jnp.clip(1.0 - p_b, 0.0, 1.0), axis=1)
    no_b_before = jnp.concatenate(
        [jnp.ones_like(no_b_prefix_inclusive[:, :1]), no_b_prefix_inclusive[:, :-1]],
        axis=1,
    )

    miss_a_prob = no_a_prefix_inclusive[:, -1]
    miss_b_prob = no_b_prefix_inclusive[:, -1]
    miss_c_prob = jnp.prod(jnp.clip(1.0 - p_c, 0.0, 1.0), axis=1)
    bad_b_before_a_prob = 1.0 - jnp.prod(
        jnp.clip(1.0 - p_b * no_a_before, 0.0, 1.0),
        axis=1,
    )
    bad_c_before_b_prob = 1.0 - jnp.prod(
        jnp.clip(1.0 - p_c * no_b_before, 0.0, 1.0),
        axis=1,
    )
    hit_a_prob = jnp.clip(1.0 - miss_a_prob, eps, 1.0)
    hit_b_prob = jnp.clip(1.0 - miss_b_prob, eps, 1.0)
    hit_c_prob = jnp.clip(1.0 - miss_c_prob, eps, 1.0)

    hit_loss_type = getattr(cfg.constraints, "hit_loss", "miss")
    if hit_loss_type == "miss":
        hit_a_loss = jnp.mean(miss_a_prob)
        hit_b_loss = jnp.mean(miss_b_prob)
        hit_c_loss = jnp.mean(miss_c_prob)
    elif hit_loss_type == "nll":
        hit_a_loss = jnp.mean(-jnp.log(hit_a_prob))
        hit_b_loss = jnp.mean(-jnp.log(hit_b_prob))
        hit_c_loss = jnp.mean(-jnp.log(hit_c_prob))
    else:
        raise ValueError("constraints.hit_loss must be 'miss' or 'nll'")

    order_loss = jnp.mean(bad_b_before_a_prob + bad_c_before_b_prob)
    return (
        hit_a_loss,
        hit_b_loss,
        hit_c_loss,
        order_loss,
        jnp.mean(hit_a_prob),
        jnp.mean(hit_b_prob),
        jnp.mean(hit_c_prob),
        jnp.mean(bad_b_before_a_prob),
        jnp.mean(bad_c_before_b_prob),
    )


def _flow_map_path_grid(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    times = _dive_gate_path_times(cfg, x0.dtype)

    if label is None:
        return jax.vmap(
            lambda x: jax.vmap(
                lambda tau: X.apply(
                    params,
                    0.0,
                    tau,
                    x,
                    None,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(times)
        )(x0)

    return jax.vmap(
        lambda x, lbl: jax.vmap(
            lambda tau: X.apply(
                params,
                0.0,
                tau,
                x,
                lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(times)
    )(x0, label)


def _velocity_batch(
    params: Parameters,
    t: jnp.ndarray,
    x: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
) -> jnp.ndarray:
    if label is None:
        return jax.vmap(
            lambda xi: X.apply(
                params,
                t,
                xi,
                None,
                train=False,
                method="calc_b",
            )
        )(x)

    return jax.vmap(
        lambda xi, lbl: X.apply(
            params,
            t,
            xi,
            lbl,
            train=False,
            method="calc_b",
        )
    )(x, label)


def _euler_velocity_paths(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    n_steps = int(getattr(cfg.constraints, "euler_steps", 100))
    if n_steps < 1:
        raise ValueError("constraints.euler_steps must be >= 1")

    times = jnp.linspace(0.0, 1.0, n_steps + 1, dtype=x0.dtype)

    def step(x, idx):
        t0 = times[idx]
        dt = times[idx + 1] - times[idx]
        x_next = x + dt * _velocity_batch(params, t0, x, label, X=X)
        return x_next, x_next

    _, states = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return jnp.concatenate([x0[:, None, :], jnp.swapaxes(states, 0, 1)], axis=1)


def _dive_gate_constraint_paths(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    mode = _dive_gate_path_mode(cfg)
    if mode == "flow_map":
        return _flow_map_path_grid(params, x0, label, X=X, cfg=cfg)
    return _euler_velocity_paths(params, x0, label, X=X, cfg=cfg)


def dive_gate_path_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Penalize learned paths that miss B or reach C before B."""
    constraint_bs = _flow_matching_constraint_batch_size(cfg, x0.shape[0])
    x0 = x0[:constraint_bs]
    if label is not None:
        label = label[:constraint_bs]

    paths = _dive_gate_constraint_paths(params, x0, label, X=X, cfg=cfg)
    (
        hit_a_loss,
        hit_b_loss,
        hit_c_loss,
        order_loss,
        _,
        _,
        _,
        _,
        _,
    ) = _dive_gate_soft_terms(paths, cfg)

    lambda_hit = float(getattr(cfg.constraints, "lambda_hit", 1.0))
    lambda_hit_a = float(getattr(cfg.constraints, "lambda_hit_a", lambda_hit))
    lambda_hit_b = float(getattr(cfg.constraints, "lambda_hit_b", lambda_hit))
    lambda_hit_c = float(getattr(cfg.constraints, "lambda_hit_c", 1.0))
    lambda_order = float(getattr(cfg.constraints, "lambda_order", 1.0))
    loss = cfg.constraints.weight * (
        lambda_hit_a * hit_a_loss
        + lambda_hit_b * hit_b_loss
        + lambda_hit_c * hit_c_loss
        + lambda_order * order_loss
    )
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


def _maizels_path_mode(cfg: config_dict.ConfigDict) -> str:
    mode = getattr(cfg.constraints, "path_mode", "flowmap")
    if mode in ("auto", "flowmap", "flow_map_sampling", "sampling", "multistep"):
        return "flowmap"
    if mode in ("direct", "one_step", "flow_map"):
        return "direct"
    if mode in ("euler", "flow_matching", "velocity"):
        return "euler"
    raise ValueError(
        "constraints.path_mode must be one of 'flowmap', 'direct', or 'euler' "
        f"for maizels_lineage_path, got {mode!r}."
    )


def _maizels_path_times(cfg: config_dict.ConfigDict, dtype) -> jnp.ndarray:
    path_times = getattr(cfg.constraints, "path_times", None)
    if path_times is not None:
        path_times = jnp.asarray(path_times, dtype=dtype)
        if path_times.shape[0] < 1:
            raise ValueError("constraints.path_times must contain at least one time")
        return path_times

    n_times = int(getattr(cfg.constraints, "path_n_times", 10))
    if n_times < 1:
        raise ValueError("constraints.path_n_times must be >= 1")
    return jnp.linspace(0.0, 1.0, n_times + 1, dtype=dtype)[1:]


def _flow_map_step_batch(
    params: Parameters,
    s: jnp.ndarray,
    t: jnp.ndarray,
    x: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
) -> jnp.ndarray:
    if label is None:
        return jax.vmap(
            lambda xi: X.apply(
                params,
                s,
                t,
                xi,
                None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x)

    return jax.vmap(
        lambda xi, lbl: X.apply(
            params,
            s,
            t,
            xi,
            lbl,
            train=False,
            calc_weight=False,
            return_X_and_phi=False,
        )
    )(x, label)


def _maizels_direct_path_grid(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    times = _maizels_path_times(cfg, x0.dtype)

    if label is None:
        return jax.vmap(
            lambda x: jax.vmap(
                lambda tau: X.apply(
                    params,
                    0.0,
                    tau,
                    x,
                    None,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(times)
        )(x0)

    return jax.vmap(
        lambda x, lbl: jax.vmap(
            lambda tau: X.apply(
                params,
                0.0,
                tau,
                x,
                lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(times)
    )(x0, label)


def _maizels_flowmap_sampling_paths(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    times = _maizels_path_times(cfg, x0.dtype)

    def step(carry, t_next):
        x, t_prev = carry
        x_next = _flow_map_step_batch(params, t_prev, t_next, x, label, X=X)
        return (x_next, t_next), x_next

    (_, _), states = jax.lax.scan(
        step,
        (x0, jnp.asarray(0.0, dtype=x0.dtype)),
        times,
    )
    return jnp.swapaxes(states, 0, 1)


def _maizels_euler_paths(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    n_steps = int(getattr(cfg.constraints, "euler_steps", 25))
    if n_steps < 1:
        raise ValueError("constraints.euler_steps must be >= 1")
    times = jnp.linspace(0.0, 1.0, n_steps + 1, dtype=x0.dtype)

    def step(x, idx):
        t0 = times[idx]
        dt = times[idx + 1] - times[idx]
        x_next = x + dt * _velocity_batch(params, t0, x, label, X=X)
        return x_next, x_next

    _, states = jax.lax.scan(step, x0, jnp.arange(n_steps))
    return jnp.swapaxes(states, 0, 1)


def _maizels_lineage_constraint_paths(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    mode = _maizels_path_mode(cfg)
    if mode == "flowmap":
        return _maizels_flowmap_sampling_paths(params, x0, label, X=X, cfg=cfg)
    if mode == "direct":
        return _maizels_direct_path_grid(params, x0, label, X=X, cfg=cfg)
    return _maizels_euler_paths(params, x0, label, X=X, cfg=cfg)


def _setup_maizels_lineage_classifier(cfg: config_dict.ConfigDict):
    classifier_path = getattr(cfg.problem, "classifier_path", maizels.DEFAULT_CLASSIFIER)
    params, class_names, scaler_mean, scaler_scale = maizels.load_jax_classifier_params(
        classifier_path
    )
    return {
        "params": params,
        "class_names": class_names,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "invalid_transition": jnp.asarray(
            maizels.lineage_invalid_transition_matrix(class_names),
            dtype=jnp.float32,
        ),
        "canonical_to_classifier": jnp.asarray(
            maizels.classifier_index_lookup(class_names),
            dtype=jnp.int32,
        ),
    }


def _maizels_lineage_terms(
    paths: jnp.ndarray,
    label: jnp.ndarray,
    classifier,
    cfg: config_dict.ConfigDict,
) -> Dict[str, jnp.ndarray]:
    if label is None or label.ndim != 2 or label.shape[1] < 2:
        raise ValueError(
            "maizels_lineage_path constraints require label[:, 0:2] to contain "
            "source and target cell-type ids."
        )

    flat = paths.reshape((-1, paths.shape[-1]))
    logits = maizels.jax_classifier_logits(
        classifier["params"],
        classifier["scaler_mean"],
        classifier["scaler_scale"],
        flat,
    )
    temperature = float(getattr(cfg.constraints, "classifier_temperature", 1.0))
    probs = jax.nn.softmax(
        logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), 1e-6),
        axis=-1,
    ).reshape((paths.shape[0], paths.shape[1], -1))

    lambda_final = float(getattr(cfg.constraints, "lambda_final", 0.0))
    target_type_ids = label[:, 1] if lambda_final > 0.0 else None
    terms = maizels.lineage_soft_terms_from_probs(
        probs,
        label[:, 0],
        classifier["invalid_transition"],
        classifier["canonical_to_classifier"],
        target_type_ids=target_type_ids,
    )
    return terms


def maizels_lineage_loss_point_constraint(
    x_s: jnp.ndarray,
    x_t: jnp.ndarray,
    label: jnp.ndarray,
    *,
    cfg: config_dict.ConfigDict,
    classifier,
) -> float:
    """Penalize invalid classifier transitions on an already-computed X_{s,t}(I_s)."""
    if label is None or label.ndim != 1 or label.shape[0] < 2:
        raise ValueError(
            "maizels_lineage_path loss_points constraints require each label to "
            "contain source and target cell-type ids."
        )

    states = jnp.stack([x_s, x_t], axis=0)
    logits = maizels.jax_classifier_logits(
        classifier["params"],
        classifier["scaler_mean"],
        classifier["scaler_scale"],
        states,
    )
    temperature = float(getattr(cfg.constraints, "classifier_temperature", 1.0))
    probs = jax.nn.softmax(
        logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), 1e-6),
        axis=-1,
    )[None, :, :]

    lambda_final = float(getattr(cfg.constraints, "lambda_final", 0.0))
    target_type_ids = label[None, 1] if lambda_final > 0.0 else None
    terms = maizels.lineage_soft_terms_from_probs(
        probs,
        label[None, 0],
        classifier["invalid_transition"],
        classifier["canonical_to_classifier"],
        target_type_ids=target_type_ids,
    )

    lambda_start = float(getattr(cfg.constraints, "lambda_start", 0.0))
    lambda_transition = float(getattr(cfg.constraints, "lambda_transition", 1.0))
    loss = cfg.constraints.weight * (
        lambda_start * terms["start_invalid_loss"]
        + lambda_transition * terms["transition_invalid_loss"]
        + lambda_final * terms["final_invalid_loss"]
    )
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


def maizels_lineage_path_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
    classifier,
) -> float:
    """Penalize classifier-probability mass on invalid lineage transitions."""
    constraint_bs = _flow_matching_constraint_batch_size(cfg, x0.shape[0])
    x0 = x0[:constraint_bs]
    if label is not None:
        label = label[:constraint_bs]

    paths = _maizels_lineage_constraint_paths(params, x0, label, X=X, cfg=cfg)
    terms = _maizels_lineage_terms(paths, label, classifier, cfg)

    lambda_start = float(getattr(cfg.constraints, "lambda_start", 1.0))
    lambda_transition = float(getattr(cfg.constraints, "lambda_transition", 1.0))
    lambda_final = float(getattr(cfg.constraints, "lambda_final", 0.0))
    loss = cfg.constraints.weight * (
        lambda_start * terms["start_invalid_loss"]
        + lambda_transition * terms["transition_invalid_loss"]
        + lambda_final * terms["final_invalid_loss"]
    )
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


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


def _weighted_kernel_mean(
    kernel: jnp.ndarray,
    x_weights: jnp.ndarray,
    y_weights: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Weighted average of a kernel matrix with safe empty-mask handling."""
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
    """Biased weighted MMD^2 between two batches."""
    k_xx = _rbf_kernel_mixture(x, x, bandwidths)
    k_yy = _rbf_kernel_mixture(y, y, bandwidths)
    k_xy = _rbf_kernel_mixture(x, y, bandwidths)

    mmd2 = (
        _weighted_kernel_mean(k_xx, x_weights, x_weights, eps)
        + _weighted_kernel_mean(k_yy, y_weights, y_weights, eps)
        - 2.0 * _weighted_kernel_mean(k_xy, x_weights, y_weights, eps)
    )
    return jnp.maximum(mmd2, 0.0)


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


def box_path_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    label: jnp.ndarray,
    t: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Penalize learned one-shot path samples that enter a forbidden box."""
    tau = jnp.clip(t, 0.0, 1.0)
    x_tau = _path_positions(params, x0, label, tau, X=X)
    x_tau = _maybe_clip_constraint_state(x_tau, cfg)

    loss = _box_path_penalty(x_tau, cfg)
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


def _euler_flow_matching_endpoint(
    params: Parameters,
    x_s: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    """Integrate dx/dt=b_t(x) from s to t with differentiable Euler steps."""
    n_steps = int(getattr(cfg.constraints, "euler_steps", 25))
    if n_steps < 1:
        raise ValueError("constraints.euler_steps must be >= 1")

    dt = (t - s) / float(n_steps)

    def step(x, idx):
        tau = s + dt * idx.astype(x.dtype)
        b_tau = X.apply(
            params,
            tau,
            x,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
        x_next = x + dt * b_tau
        return x_next, None

    x_t, _ = jax.lax.scan(step, x_s, jnp.arange(n_steps))
    return x_t


def _sample_constraint_times(
    key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample upper-triangle times with the same law as off-diagonal flow maps."""
    times = jax.random.uniform(
        key,
        shape=(2,),
        minval=cfg.training.tmin,
        maxval=cfg.training.tmax,
    )
    return jnp.minimum(times[0], times[1]), jnp.maximum(times[0], times[1])


def _flow_matching_constraint_batch_size(
    cfg: config_dict.ConfigDict,
    total_bs: int,
) -> int:
    """Match the off-diagonal constraint batch size from a reference flow-map run."""
    constraint_bs = int(getattr(cfg.constraints, "constraint_batch_size", 0))
    if constraint_bs > 0:
        return min(total_bs, constraint_bs)

    reference_diag_fraction = getattr(
        cfg.constraints, "constraint_reference_diag_fraction", None
    )
    if reference_diag_fraction is not None:
        diag_bs = max(1, int(total_bs * float(reference_diag_fraction)))
        return max(1, total_bs - diag_bs)

    constraint_fraction = float(
        getattr(cfg.constraints, "constraint_batch_fraction", 1.0)
    )
    return max(1, min(total_bs, int(total_bs * constraint_fraction)))


def _box_loss_point_constraint(
    params: Parameters,
    x_s: jnp.ndarray,
    x_st: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Evaluate the configured off-diagonal box constraint at one sample."""
    mode = _box_constraint_mode(cfg)
    if mode == "flow_map":
        x_box = x_st
    elif mode == "flow_matching":
        x_box = _euler_flow_matching_endpoint(
            params,
            x_s,
            label,
            s,
            t,
            rng,
            X=X,
            cfg=cfg,
        )
    else:
        raise ValueError(f"Unknown constraints.constraint_mode: {mode}")

    return _box_path_penalty(x_box, cfg)


def flow_matching_box_path_constraint(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    dropout_keys: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Penalize Euler/BPTT flow-matching rollouts over sampled (s,t) pairs."""
    constraint_bs = _flow_matching_constraint_batch_size(cfg, x0.shape[0])
    x0 = x0[:constraint_bs]
    x1 = x1[:constraint_bs]
    if label is not None:
        label = label[:constraint_bs]
    dropout_keys = dropout_keys[:constraint_bs]

    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def batch_constraint(params, x0_i, x1_i, label_i, key_i):
        s_i, t_i = _sample_constraint_times(key_i, cfg)
        x_s = interp.calc_It(s_i, x0_i, x1_i, label_i)
        x_t = _euler_flow_matching_endpoint(
            params,
            x_s,
            label_i,
            s_i,
            t_i,
            {"dropout": key_i},
            X=X,
            cfg=cfg,
        )
        return _box_path_penalty(x_t, cfg)

    return jnp.nan_to_num(
        batch_constraint(params, x0, x1, label, dropout_keys),
        nan=0.0,
        posinf=1e6,
        neginf=1e6,
    )


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


def endpoint_matching_loss(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    *,
    X: flow_map.FlowMap,
    cfg: config_dict.ConfigDict,
) -> float:
    """Endpoint distribution matching for the direct map X_{0,1}(x0)."""
    endpoint_cfg = cfg.training.endpoint_matching
    tau = jnp.ones((x0.shape[0],), dtype=x0.dtype)
    x1_hat = _path_positions(params, x0, label, tau, X=X)

    x_clip = float(getattr(endpoint_cfg, "x_clip", 0.0))
    clip_mode = getattr(endpoint_cfg, "clip_mode", "hard")
    x1_hat = _maybe_clip(x1_hat, x_clip, clip_mode=clip_mode)
    x1 = _maybe_clip(x1, x_clip, clip_mode=clip_mode)

    if bool(getattr(endpoint_cfg, "normalize", False)):
        eps = float(getattr(endpoint_cfg, "normalize_eps", 1e-3))
        center = jnp.mean(x1, axis=0, keepdims=True)
        scale = jnp.maximum(jnp.std(x1, axis=0, keepdims=True), eps)
        x1_hat = (x1_hat - center) / scale
        x1 = (x1 - center) / scale

    bandwidths = jnp.asarray(
        getattr(endpoint_cfg, "bandwidths", [0.25, 0.5, 1.0, 2.0, 4.0]),
        dtype=x1.dtype,
    )
    eps = float(getattr(endpoint_cfg, "eps", 1e-6))
    all_weights = jnp.ones((x0.shape[0],), dtype=x1.dtype)

    if not bool(getattr(endpoint_cfg, "branch_conditional", False)):
        loss = _weighted_mmd(x1_hat, x1, all_weights, all_weights, bandwidths, eps)
        return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)

    branch_axis = int(getattr(endpoint_cfg, "branch_axis", 1))
    branch_threshold = float(getattr(endpoint_cfg, "branch_threshold", 0.0))
    branch_weights = (x0[:, branch_axis] >= branch_threshold).astype(x1.dtype)
    branch_weight_list = [branch_weights, 1.0 - branch_weights]

    loss = 0.0
    normalizer = 0.0
    min_branch_mass = float(getattr(endpoint_cfg, "min_branch_mass", 1.0))
    for weights in branch_weight_list:
        branch_present = (jnp.sum(weights) >= min_branch_mass).astype(x1.dtype)
        branch_mmd = _weighted_mmd(x1_hat, x1, weights, weights, bandwidths, eps)
        loss += branch_present * branch_mmd
        normalizer += branch_present

    loss = loss / jnp.maximum(normalizer, 1.0)
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)


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
    It = interp.calc_It(t, x0, x1, label)
    It_dot = interp.calc_It_dot(t, x0, x1, label)

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
    Is = interp.calc_It(s, x0, x1, label)

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
    cfg: config_dict.ConfigDict,
    constraint_scale: jnp.ndarray,
    constraint_weight_factor: jnp.ndarray,
    maizels_lineage_classifier=None,
) -> float:
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1, label)

    # Compute the distillation loss
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )
    box_loss = 0.0
    if uses_flow_map_box_loss_points(cfg):
        box_loss = (
            constraint_scale
            * constraint_weight_factor
            * _box_loss_point_constraint(
                params,
                Is,
                Xst_Is,
                label,
                s,
                t,
                rng,
                X=X,
                cfg=cfg,
            )
        )
    maizels_loss = 0.0
    if uses_maizels_loss_points(cfg):
        maizels_loss = (
            constraint_scale
            * constraint_weight_factor
            * maizels_lineage_loss_point_constraint(
                Is,
                Xst_Is,
                label,
                cfg=cfg,
                classifier=maizels_lineage_classifier,
            )
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
    return jnp.exp(-weight_st) * lsd_loss + weight_st + box_loss + maizels_loss


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
    Is = interp.calc_It(s, x0, x1, label)

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
    maizels_lineage_classifier = None
    if has_constraint(cfg) and cfg.constraints.type == "maizels_lineage_path":
        maizels_lineage_classifier = _setup_maizels_lineage_classifier(cfg)

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
    @functools.partial(
        jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0, None, None)
    )
    def offdiagonal_only_loss(
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
        constraint_scale,
        constraint_weight_factor,
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
                cfg=cfg,
                constraint_scale=constraint_scale,
                constraint_weight_factor=constraint_weight_factor,
                maizels_lineage_classifier=maizels_lineage_classifier,
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
        constraint_scale = jnp.mean(constraint_scale_batch)
        if has_constraint(cfg) and bool(getattr(cfg.constraints, "stage2_only", False)):
            constraint_scale = constraint_scale * stage2_scale

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
        expected_base_normalizer = jnp.maximum(
            diag_weight * diag_bs + offdiag_weight * offdiag_bs,
            1.0,
        )
        loss_points_constraint_factor = 0.0
        if offdiag_bs > 0 and (
            uses_flow_map_box_loss_points(cfg) or uses_maizels_loss_points(cfg)
        ):
            loss_points_constraint_factor = expected_base_normalizer / jnp.maximum(
                offdiag_weight * offdiag_bs,
                1e-6,
            )

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
                constraint_scale,
                loss_points_constraint_factor,
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

        # Optional endpoint distribution matching for X_{0,1}(x0) -> x1.
        if has_endpoint_matching(cfg):
            endpoint_cfg = cfg.training.endpoint_matching
            endpoint_weight = float(getattr(endpoint_cfg, "weight", 1.0))
            total_loss += endpoint_weight * endpoint_matching_loss(
                params,
                x0,
                x1,
                label,
                X=net,
                cfg=cfg,
            )

        # Optional trajectory constraint terms
        if has_constraint(cfg):
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
            elif cfg.constraints.type == "box_path":
                constraint_mode = _box_constraint_mode(cfg)
                box_path_mode = getattr(cfg.constraints, "box_path_mode", "x0_t")
                if constraint_mode == "flow_matching":
                    total_loss += constraint_scale * flow_matching_box_path_constraint(
                        params,
                        x0,
                        x1,
                        label,
                        dropout_keys,
                        interp=interp,
                        X=net,
                        cfg=cfg,
                    )
                elif box_path_mode == "x0_t":
                    total_loss += constraint_scale * box_path_constraint(
                        params,
                        x0,
                        label,
                        t,
                        X=net,
                        cfg=cfg,
                    )
                elif box_path_mode == "loss_points":
                    if cfg.training.loss_type != "lsd":
                        raise ValueError(
                            "constraints.box_path_mode='loss_points' requires LSD loss"
                        )
                else:
                    raise ValueError(
                        f"Unknown constraints.box_path_mode: {box_path_mode}"
                    )
            elif cfg.constraints.type == "dive_gate_path":
                total_loss += constraint_scale * dive_gate_path_constraint(
                    params,
                    x0,
                    label,
                    X=net,
                    cfg=cfg,
                )
            elif cfg.constraints.type == "maizels_lineage_path":
                if uses_maizels_loss_points(cfg):
                    if cfg.training.loss_type != "lsd":
                        raise ValueError(
                            "constraints.path_mode='loss_points' requires LSD loss"
                        )
                else:
                    total_loss += constraint_scale * maizels_lineage_path_constraint(
                        params,
                        x0,
                        label,
                        X=net,
                        cfg=cfg,
                        classifier=maizels_lineage_classifier,
                    )
            else:
                raise ValueError(f"Unknown constraint type: {cfg.constraints.type}")

        return total_loss

    return loss

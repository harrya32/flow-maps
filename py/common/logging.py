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
from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib import pyplot as plt
from ml_collections import config_dict

from . import datasets, dist_utils, fid_utils, flow_map, loss_args, maizels, state_utils

Parameters = Dict[str, Dict]
_MAIZELS_VALIDATION_CACHE = {}


def is_lowd_problem(cfg: config_dict.ConfigDict) -> bool:
    """Returns True for non-image datasets with plottable first two dimensions."""
    return getattr(cfg.problem, "image_dims", None) is None and cfg.problem.d >= 2


def is_image_problem(cfg: config_dict.ConfigDict) -> bool:
    """Returns True for image-shaped problems."""
    return getattr(cfg.problem, "image_dims", None) is not None


def is_diagonal_only_training(cfg: config_dict.ConfigDict) -> bool:
    """Return True when training only uses diagonal velocity-matching points."""
    diag_fraction = getattr(cfg.optimization, "diag_fraction", None)
    if diag_fraction is not None:
        return float(diag_fraction) >= 1.0

    diag_bs = getattr(cfg.optimization, "diag_bs", None)
    if diag_bs is not None:
        return int(diag_bs) >= int(cfg.optimization.bs)

    return False


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


def lowd_limits_for(
    cfg: config_dict.ConfigDict,
    *arrays,
    pad_frac: float = 0.1,
    default_lim: float = 2.0,
) -> Tuple[list, list]:
    """Compute 2D plot limits for one panel, including visible constraint regions."""
    pieces = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0 or arr.ndim == 0 or arr.shape[-1] < 2:
            continue
        pieces.append(arr.reshape((-1, arr.shape[-1]))[:, :2])

    region_points = _lowd_region_limit_points(cfg)
    if region_points.size > 0:
        pieces.append(region_points)

    if not pieces:
        return [-default_lim, default_lim], [-default_lim, default_lim]

    return finite_lowd_limits(
        np.concatenate(pieces, axis=0),
        pad_frac=pad_frac,
        default_lim=default_lim,
    )


def extract_lowd_batch_components(batch):
    """Extract paired low-dimensional batch fields when they are available."""
    if isinstance(batch, dict) and "x1" in batch:
        return batch.get("x0"), batch["x1"], batch.get("label")
    return None, batch, None


def extract_x1_from_batch(batch):
    """Extract target samples from either plain or paired low-dimensional batches."""
    return extract_lowd_batch_components(batch)[1]


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


def _np_segments_intersect_forbidden_box(
    starts: np.ndarray, ends: np.ndarray, bounds
) -> np.ndarray:
    """Return whether each 2D segment intersects an axis-aligned box."""
    xmin, xmax, ymin, ymax = bounds
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)

    d = ends - starts
    tmin = np.zeros(starts.shape[0], dtype=np.float64)
    tmax = np.ones(starts.shape[0], dtype=np.float64)
    valid = np.ones(starts.shape[0], dtype=bool)

    for axis, low, high in [(0, xmin, xmax), (1, ymin, ymax)]:
        p = starts[:, axis]
        v = d[:, axis]
        parallel = np.abs(v) < 1e-12
        valid &= (~parallel) | ((p >= low) & (p <= high))

        nonparallel = ~parallel
        t1 = np.zeros_like(tmin)
        t2 = np.zeros_like(tmin)
        t1[nonparallel] = (low - p[nonparallel]) / v[nonparallel]
        t2[nonparallel] = (high - p[nonparallel]) / v[nonparallel]
        axis_min = np.minimum(t1, t2)
        axis_max = np.maximum(t1, t2)

        tmin[nonparallel] = np.maximum(tmin[nonparallel], axis_min[nonparallel])
        tmax[nonparallel] = np.minimum(tmax[nonparallel], axis_max[nonparallel])

    return valid & (tmax >= tmin) & (tmax >= 0.0) & (tmin <= 1.0)


def _np_trajectory_forbidden_box_rate(paths: np.ndarray, bounds) -> float:
    """Fraction of polyline trajectories whose segments touch the box."""
    paths = np.asarray(paths)
    if paths.ndim != 3 or paths.shape[1] < 2:
        return 0.0

    starts = paths[:, :-1, :].reshape((-1, paths.shape[-1]))
    ends = paths[:, 1:, :].reshape((-1, paths.shape[-1]))
    intersects = _np_segments_intersect_forbidden_box(starts, ends, bounds)
    intersects = intersects.reshape((paths.shape[0], paths.shape[1] - 1))
    return float(np.mean(np.any(intersects, axis=1)))


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


def _matched_gates_spec(cfg: config_dict.ConfigDict):
    gate_cfg = getattr(getattr(cfg, "logging", None), "matched_gates", None)
    if gate_cfg is None or not bool(getattr(gate_cfg, "enabled", False)):
        return None

    problem_cfg = getattr(cfg, "problem", None)
    if problem_cfg is None:
        return None

    return {
        "source": np.asarray(
            getattr(problem_cfg, "matched_gates_source_mean", [0.0, -2.0]),
            dtype=np.float32,
        ),
        "midpoint_a": np.asarray(
            getattr(problem_cfg, "gate_midpoint_a", [-0.35, 0.0]),
            dtype=np.float32,
        ),
        "midpoint_b": np.asarray(
            getattr(problem_cfg, "gate_midpoint_b", [0.35, 0.0]),
            dtype=np.float32,
        ),
        "endpoint_a": np.asarray(
            getattr(problem_cfg, "gate_endpoint_a", [-0.45, 2.0]),
            dtype=np.float32,
        ),
        "endpoint_b": np.asarray(
            getattr(problem_cfg, "gate_endpoint_b", [0.45, 2.0]),
            dtype=np.float32,
        ),
        "source_radius": float(
            getattr(gate_cfg, "source_radius", getattr(problem_cfg, "source_radius", 0.18))
        ),
        "gate_radius": float(
            getattr(gate_cfg, "gate_radius", getattr(problem_cfg, "gate_radius", 0.18))
        ),
        "endpoint_radius": float(
            getattr(
                gate_cfg,
                "endpoint_radius",
                getattr(problem_cfg, "endpoint_radius", 0.22),
            )
        ),
        "forbid_wrong_midpoint_first": bool(
            getattr(gate_cfg, "forbid_wrong_midpoint_first", True)
        ),
    }


def _draw_matched_gate_regions(
    ax, cfg: config_dict.ConfigDict, *, label: bool = False
) -> None:
    spec = _matched_gates_spec(cfg)
    if spec is None:
        return

    region_style = dict(facecolor="none", linewidth=1.4, linestyle="--", alpha=0.95)
    ax.add_patch(
        Circle(
            spec["source"],
            spec["source_radius"],
            edgecolor="0.45",
            label="source" if label else None,
            **region_style,
        )
    )
    ax.add_patch(
        Circle(
            spec["midpoint_a"],
            spec["gate_radius"],
            edgecolor="C0",
            label="A gate" if label else None,
            **region_style,
        )
    )
    ax.add_patch(
        Circle(
            spec["midpoint_b"],
            spec["gate_radius"],
            edgecolor="C1",
            label="B gate" if label else None,
            **region_style,
        )
    )
    ax.add_patch(
        Circle(spec["endpoint_a"], spec["endpoint_radius"], edgecolor="C0", **region_style)
    )
    ax.add_patch(
        Circle(spec["endpoint_b"], spec["endpoint_radius"], edgecolor="C1", **region_style)
    )

    text_kwargs = dict(fontsize=9, ha="center", va="center", zorder=25)
    ax.text(*(spec["source"] + np.asarray([0.0, -0.28])), "S", color="0.30", **text_kwargs)
    ax.text(*(spec["midpoint_a"] + np.asarray([-0.23, 0.0])), "M_A", color="C0", **text_kwargs)
    ax.text(*(spec["midpoint_b"] + np.asarray([0.23, 0.0])), "M_B", color="C1", **text_kwargs)
    ax.text(*(spec["endpoint_a"] + np.asarray([-0.25, 0.0])), "E_A", color="C0", **text_kwargs)
    ax.text(*(spec["endpoint_b"] + np.asarray([0.25, 0.0])), "E_B", color="C1", **text_kwargs)


def _dive_gate_spec(cfg: config_dict.ConfigDict):
    gate_cfg = getattr(getattr(cfg, "logging", None), "dive_gate", None)
    if gate_cfg is None or not bool(getattr(gate_cfg, "enabled", False)):
        return None

    problem_cfg = getattr(cfg, "problem", None)
    if problem_cfg is None:
        return None

    depth = float(getattr(problem_cfg, "dive_gate_depth", 0.85))
    gate_center = np.asarray(
        getattr(gate_cfg, "gate_center", [0.0, -depth]),
        dtype=np.float32,
    )
    checkpoint_center = np.asarray(
        getattr(gate_cfg, "checkpoint_center", [0.9, 0.0]),
        dtype=np.float32,
    )
    pre_checkpoint_center = np.asarray(
        getattr(gate_cfg, "pre_checkpoint_center", [-0.9, 0.0]),
        dtype=np.float32,
    )

    return {
        "source": np.asarray(
            getattr(problem_cfg, "dive_gate_source_mean", [-3.0, 0.0]),
            dtype=np.float32,
        ),
        "target": np.asarray(
            getattr(problem_cfg, "dive_gate_target_mean", [3.0, 0.0]),
            dtype=np.float32,
        ),
        "pre_checkpoint_center": pre_checkpoint_center,
        "gate_center": gate_center,
        "checkpoint_center": checkpoint_center,
        "pre_checkpoint_radii": np.asarray(
            getattr(
                gate_cfg,
                "pre_checkpoint_radii",
                getattr(problem_cfg, "checkpoint_radii", [0.35, 0.24]),
            ),
            dtype=np.float32,
        ),
        "gate_radii": np.asarray(
            getattr(gate_cfg, "gate_radii", getattr(problem_cfg, "gate_radii", [0.42, 0.28])),
            dtype=np.float32,
        ),
        "checkpoint_radii": np.asarray(
            getattr(
                gate_cfg,
                "checkpoint_radii",
                getattr(problem_cfg, "checkpoint_radii", [0.32, 0.24]),
            ),
            dtype=np.float32,
        ),
        "require_gate_hit": bool(getattr(gate_cfg, "require_gate_hit", True)),
    }


def _draw_dive_gate_regions(
    ax, cfg: config_dict.ConfigDict, *, label: bool = False
) -> None:
    spec = _dive_gate_spec(cfg)
    if spec is None:
        return

    region_style = dict(facecolor="none", linewidth=1.4, linestyle="--", alpha=0.95)
    ax.add_patch(
        Ellipse(
            spec["pre_checkpoint_center"],
            2.0 * spec["pre_checkpoint_radii"][0],
            2.0 * spec["pre_checkpoint_radii"][1],
            edgecolor="C4",
            label="A checkpoint" if label else None,
            **region_style,
        )
    )
    ax.add_patch(
        Ellipse(
            spec["gate_center"],
            2.0 * spec["gate_radii"][0],
            2.0 * spec["gate_radii"][1],
            edgecolor="C3",
            label="B gate" if label else None,
            **region_style,
        )
    )
    ax.add_patch(
        Ellipse(
            spec["checkpoint_center"],
            2.0 * spec["checkpoint_radii"][0],
            2.0 * spec["checkpoint_radii"][1],
            edgecolor="C2",
            label="C checkpoint" if label else None,
            **region_style,
        )
    )

    text_kwargs = dict(fontsize=9, ha="center", va="center", zorder=25)
    ax.text(
        *(spec["pre_checkpoint_center"] + np.asarray([0.0, 0.28])),
        "A",
        color="C4",
        **text_kwargs,
    )
    ax.text(
        *(spec["gate_center"] + np.asarray([0.0, -0.34])),
        "B",
        color="C3",
        **text_kwargs,
    )
    ax.text(
        *(spec["checkpoint_center"] + np.asarray([0.0, 0.28])),
        "C",
        color="C2",
        **text_kwargs,
    )


def _region_limit_points(center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)
    return np.asarray(
        [
            center - radii,
            center + radii,
            center + np.asarray([radii[0], -radii[1]], dtype=np.float32),
            center + np.asarray([-radii[0], radii[1]], dtype=np.float32),
        ],
        dtype=np.float32,
    )


def _lowd_region_limit_points(cfg: config_dict.ConfigDict) -> np.ndarray:
    points = []

    matched = _matched_gates_spec(cfg)
    if matched is not None:
        for key, radius_key in [
            ("source", "source_radius"),
            ("midpoint_a", "gate_radius"),
            ("midpoint_b", "gate_radius"),
            ("endpoint_a", "endpoint_radius"),
            ("endpoint_b", "endpoint_radius"),
        ]:
            radius = float(matched[radius_key])
            points.append(
                _region_limit_points(
                    matched[key],
                    np.asarray([radius, radius], dtype=np.float32),
                )
            )

    dive = _dive_gate_spec(cfg)
    if dive is not None:
        points.append(
            _region_limit_points(
                dive["pre_checkpoint_center"],
                dive["pre_checkpoint_radii"],
            )
        )
        points.append(_region_limit_points(dive["gate_center"], dive["gate_radii"]))
        points.append(
            _region_limit_points(dive["checkpoint_center"], dive["checkpoint_radii"])
        )

    if not points:
        return np.zeros((0, 2), dtype=np.float32)

    return np.concatenate(points, axis=0)


def _draw_lowd_regions(ax, cfg: config_dict.ConfigDict, *, label: bool = False) -> None:
    _draw_forbidden_box(ax, cfg, label=label)
    _draw_matched_gate_regions(ax, cfg, label=label)
    _draw_dive_gate_regions(ax, cfg, label=label)


def _matched_gate_first_hit(paths: np.ndarray, center: np.ndarray, radius: float):
    distances = np.linalg.norm(paths - center[None, None, :], axis=-1)
    hit = distances <= radius
    exists = np.any(hit, axis=1)
    first = np.argmax(hit, axis=1)
    first = np.where(exists, first, paths.shape[1])
    return exists, first


def _matched_gate_predicted_endpoint_b(paths: np.ndarray, spec) -> np.ndarray:
    final = paths[:, -1, :]
    dist_a = np.linalg.norm(final - spec["endpoint_a"][None, :], axis=1)
    dist_b = np.linalg.norm(final - spec["endpoint_b"][None, :], axis=1)
    return dist_b < dist_a


def _matched_gate_violation_details(paths: np.ndarray, cfg: config_dict.ConfigDict):
    spec = _matched_gates_spec(cfg)
    if spec is None:
        return None

    paths = np.asarray(paths, dtype=np.float32)
    pred_b = _matched_gate_predicted_endpoint_b(paths, spec)
    hit_ma, first_ma = _matched_gate_first_hit(
        paths, spec["midpoint_a"], spec["gate_radius"]
    )
    hit_mb, first_mb = _matched_gate_first_hit(
        paths, spec["midpoint_b"], spec["gate_radius"]
    )
    hit_ea, first_ea = _matched_gate_first_hit(
        paths, spec["endpoint_a"], spec["endpoint_radius"]
    )
    hit_eb, first_eb = _matched_gate_first_hit(
        paths, spec["endpoint_b"], spec["endpoint_radius"]
    )

    valid_a = hit_ma & ((~hit_ea) | (first_ma < first_ea))
    valid_b = hit_mb & ((~hit_eb) | (first_mb < first_eb))
    wrong_first_a = hit_mb & (first_mb < first_ma)
    wrong_first_b = hit_ma & (first_ma < first_mb)

    if spec["forbid_wrong_midpoint_first"]:
        valid_a = valid_a & (~wrong_first_a)
        valid_b = valid_b & (~wrong_first_b)

    valid = np.where(pred_b, valid_b, valid_a)
    correct_midpoint_hit = np.where(pred_b, hit_mb, hit_ma)
    wrong_first = np.where(pred_b, wrong_first_b, wrong_first_a)

    return {
        "invalid": ~valid,
        "pred_b": pred_b,
        "correct_midpoint_hit": correct_midpoint_hit,
        "wrong_midpoint_first": wrong_first,
    }


def _matched_gate_violation_mask(paths: np.ndarray, cfg: config_dict.ConfigDict):
    details = _matched_gate_violation_details(paths, cfg)
    if details is None:
        return None
    return details["invalid"]


def _ellipse_first_hit(paths: np.ndarray, center: np.ndarray, radii: np.ndarray):
    """Return first sampled-segment intersection time with an axis-aligned ellipse."""
    paths = np.asarray(paths, dtype=np.float32)
    scaled = (paths - center[None, None, :]) / radii[None, None, :]

    if scaled.shape[1] < 2:
        hit = np.sum(scaled * scaled, axis=-1) <= 1.0
        exists = np.any(hit, axis=1)
        first = np.argmax(hit, axis=1).astype(np.float32)
        first = np.where(exists, first, float(paths.shape[1]))
        return exists, first

    starts = scaled[:, :-1, :]
    ends = scaled[:, 1:, :]
    delta = ends - starts

    a = np.sum(delta * delta, axis=-1)
    b = 2.0 * np.sum(starts * delta, axis=-1)
    c = np.sum(starts * starts, axis=-1) - 1.0
    discriminant = b * b - 4.0 * a * c

    inf = np.full_like(a, np.inf, dtype=np.float32)
    segment_u = np.where(c <= 0.0, 0.0, inf)

    nondegenerate = a > 1e-12
    has_roots = nondegenerate & (discriminant >= 0.0)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0.0))
    denom = np.where(nondegenerate, 2.0 * a, 1.0)
    root1 = (-b - sqrt_disc) / denom
    root2 = (-b + sqrt_disc) / denom

    root1_valid = has_roots & (root1 >= 0.0) & (root1 <= 1.0)
    root2_valid = has_roots & (root2 >= 0.0) & (root2 <= 1.0)
    segment_u = np.minimum(segment_u, np.where(root1_valid, root1, inf))
    segment_u = np.minimum(segment_u, np.where(root2_valid, root2, inf))

    segment_index = np.arange(starts.shape[1], dtype=np.float32)[None, :]
    first_times = np.min(segment_index + segment_u, axis=1)
    exists = np.isfinite(first_times)
    first = np.where(exists, first_times, float(paths.shape[1]))
    return exists, first


def _dive_gate_violation_details(paths: np.ndarray, cfg: config_dict.ConfigDict):
    spec = _dive_gate_spec(cfg)
    if spec is None:
        return None

    paths = np.asarray(paths, dtype=np.float32)
    hit_a, first_a = _ellipse_first_hit(
        paths,
        spec["pre_checkpoint_center"],
        spec["pre_checkpoint_radii"],
    )
    hit_b, first_b = _ellipse_first_hit(
        paths,
        spec["gate_center"],
        spec["gate_radii"],
    )
    hit_c, first_c = _ellipse_first_hit(
        paths,
        spec["checkpoint_center"],
        spec["checkpoint_radii"],
    )
    b_before_a = hit_b & ((~hit_a) | (first_b <= first_a))
    c_before_b = hit_c & ((~hit_b) | (first_c <= first_b))
    valid = hit_a & hit_b & hit_c & (first_a < first_b) & (first_b < first_c)
    invalid = ~valid

    return {
        "invalid": invalid,
        "hit_a": hit_a,
        "hit_b": hit_b,
        "hit_c": hit_c,
        "b_before_a": b_before_a,
        "c_before_b": c_before_b,
    }


def _dive_gate_violation_mask(paths: np.ndarray, cfg: config_dict.ConfigDict):
    details = _dive_gate_violation_details(paths, cfg)
    if details is None:
        return None
    return details["invalid"]


def _dive_gate_cfg_value(
    cfg: config_dict.ConfigDict,
    name: str,
    default,
):
    constraint_cfg = getattr(cfg, "constraints", None)
    logging_cfg = getattr(getattr(cfg, "logging", None), "dive_gate", None)
    problem_cfg = getattr(cfg, "problem", None)

    if constraint_cfg is not None and hasattr(constraint_cfg, name):
        return getattr(constraint_cfg, name)
    if logging_cfg is not None and hasattr(logging_cfg, name):
        return getattr(logging_cfg, name)
    if problem_cfg is not None and hasattr(problem_cfg, name):
        return getattr(problem_cfg, name)
    return default


def _dive_gate_path_mode(cfg: config_dict.ConfigDict) -> str:
    constraint_cfg = cfg.constraints
    mode = getattr(
        constraint_cfg,
        "path_mode",
        getattr(constraint_cfg, "constraint_mode", "flow_map"),
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


def _dive_gate_soft_geometry(
    cfg: config_dict.ConfigDict,
    dtype,
):
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


def _soft_ellipse_indicator(
    x: jnp.ndarray,
    center: jnp.ndarray,
    radii: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    scaled = (x - center) / radii
    normalized_sqdist = jnp.sum(scaled * scaled, axis=-1)
    return jax.nn.sigmoid((1.0 - normalized_sqdist) / temperature)


def _dive_gate_soft_terms(paths: jnp.ndarray, cfg: config_dict.ConfigDict):
    (
        pre_checkpoint_center,
        pre_checkpoint_radii,
        gate_center,
        gate_radii,
        checkpoint_center,
        checkpoint_radii,
    ) = _dive_gate_soft_geometry(cfg, paths.dtype)
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
    return {
        "hit_a_loss": hit_a_loss,
        "hit_b_loss": hit_b_loss,
        "hit_c_loss": hit_c_loss,
        "hit_loss": hit_a_loss + hit_b_loss + hit_c_loss,
        "order_loss": order_loss,
        "hit_a_prob": jnp.mean(hit_a_prob),
        "hit_b_prob": jnp.mean(hit_b_prob),
        "hit_c_prob": jnp.mean(hit_c_prob),
        "bad_b_before_a_prob": jnp.mean(bad_b_before_a_prob),
        "bad_c_before_b_prob": jnp.mean(bad_c_before_b_prob),
        "soft_a_occupancy": jnp.mean(p_a),
        "soft_b_occupancy": jnp.mean(p_b),
        "soft_c_occupancy": jnp.mean(p_c),
    }


def _trajectory_violation_mask(paths: np.ndarray, cfg: config_dict.ConfigDict):
    masks = []
    for mask_fn in [_matched_gate_violation_mask, _dive_gate_violation_mask]:
        mask = mask_fn(paths, cfg)
        if mask is not None:
            masks.append(mask)

    if not masks:
        return None

    invalid = masks[0].copy()
    for mask in masks[1:]:
        invalid = invalid | mask
    return invalid


def _rollout_matched_gate_metrics(
    cfg: config_dict.ConfigDict, rollout_paths: list, rollout_name: str
) -> Dict[str, float]:
    """Measure matched-gate rule adherence for rollout polylines."""
    metrics = {}
    if _matched_gates_spec(cfg) is None:
        return metrics

    for step, paths in rollout_paths:
        details = _matched_gate_violation_details(paths, cfg)
        if details is None:
            continue

        prefix = f"matched_gates/{rollout_name}_{step}"
        metrics[f"{prefix}_violation_pct"] = 100.0 * float(
            np.mean(details["invalid"])
        )
        metrics[f"{prefix}_correct_midpoint_hit_pct"] = 100.0 * float(
            np.mean(details["correct_midpoint_hit"])
        )
        metrics[f"{prefix}_wrong_midpoint_first_pct"] = 100.0 * float(
            np.mean(details["wrong_midpoint_first"])
        )
        metrics[f"{prefix}_pred_endpoint_b_pct"] = 100.0 * float(
            np.mean(details["pred_b"])
        )

    return metrics


def _rollout_dive_gate_metrics(
    cfg: config_dict.ConfigDict, rollout_paths: list, rollout_name: str
) -> Dict[str, float]:
    """Measure dive-gate rule adherence for rollout polylines."""
    metrics = {}
    if _dive_gate_spec(cfg) is None:
        return metrics

    for step, paths in rollout_paths:
        details = _dive_gate_violation_details(paths, cfg)
        if details is None:
            continue

        prefix = f"dive_gate/{rollout_name}_{step}"
        metrics[f"{prefix}_violation_pct"] = 100.0 * float(
            np.mean(details["invalid"])
        )
        metrics[f"{prefix}_hit_a_pct"] = 100.0 * float(np.mean(details["hit_a"]))
        metrics[f"{prefix}_hit_b_pct"] = 100.0 * float(np.mean(details["hit_b"]))
        metrics[f"{prefix}_hit_c_pct"] = 100.0 * float(np.mean(details["hit_c"]))
        metrics[f"{prefix}_b_before_a_pct"] = 100.0 * float(
            np.mean(details["b_before_a"])
        )
        metrics[f"{prefix}_c_before_b_pct"] = 100.0 * float(
            np.mean(details["c_before_b"])
        )

    return metrics


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


def _interpolant_paths(
    interp,
    x0s: jnp.ndarray,
    x1s: jnp.ndarray,
    labels: jnp.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate the configured ground-truth interpolant on a time grid."""
    x0s = jnp.asarray(x0s)
    x1s = jnp.asarray(x1s)
    labels = None if labels is None else jnp.asarray(labels)

    paths = np.zeros((x0s.shape[0], len(times), x0s.shape[-1]), dtype=np.float32)
    for idx, tt in enumerate(times):
        tau = jnp.full((x0s.shape[0],), float(tt), dtype=x0s.dtype)
        xt = interp.batch_calc_It(tau, x0s, x1s, labels)
        paths[:, idx, :] = np.asarray(xt, dtype=np.float32)
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


def _maizels_constraint_path_mode(cfg: config_dict.ConfigDict) -> str:
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


def _maizels_constraint_path_times(
    cfg: config_dict.ConfigDict,
    dtype,
) -> jnp.ndarray:
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


def _maizels_flow_map_step_batch(
    apply_fn,
    params: Dict,
    s: jnp.ndarray,
    t: jnp.ndarray,
    xs: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    if labels is None:
        return jax.vmap(
            lambda x: apply_fn(
                params,
                s,
                t,
                x,
                label=None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(xs)

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


def _maizels_direct_constraint_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    times = _maizels_constraint_path_times(cfg, x0s.dtype)
    if labels is None:
        return jax.vmap(
            lambda x: jax.vmap(
                lambda tau: apply_fn(
                    params,
                    0.0,
                    tau,
                    x,
                    label=None,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(times)
        )(x0s)

    return jax.vmap(
        lambda x, lbl: jax.vmap(
            lambda tau: apply_fn(
                params,
                0.0,
                tau,
                x,
                label=lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(times)
    )(x0s, labels)


def _maizels_flowmap_constraint_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    times = _maizels_constraint_path_times(cfg, x0s.dtype)

    def step(carry, t_next):
        xs, t_prev = carry
        xs_next = _maizels_flow_map_step_batch(
            apply_fn,
            params,
            t_prev,
            t_next,
            xs,
            labels,
        )
        return (xs_next, t_next), xs_next

    (_, _), states = jax.lax.scan(
        step,
        (x0s, jnp.asarray(0.0, dtype=x0s.dtype)),
        times,
    )
    return jnp.swapaxes(states, 0, 1)


def _maizels_euler_constraint_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    n_steps = int(getattr(cfg.constraints, "euler_steps", 25))
    if n_steps < 1:
        raise ValueError("constraints.euler_steps must be >= 1")
    if labels is None:
        labels = -jnp.ones((x0s.shape[0],), dtype=jnp.int32)
    return jnp.asarray(
        _euler_paths(apply_fn, params, x0s, labels, n_steps)[:, 1:, :],
        dtype=x0s.dtype,
    )


def _maizels_lineage_constraint_paths(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> jnp.ndarray:
    mode = _maizels_constraint_path_mode(cfg)
    if mode == "flowmap":
        return _maizels_flowmap_constraint_paths(apply_fn, params, x0s, labels, cfg)
    if mode == "direct":
        return _maizels_direct_constraint_paths(apply_fn, params, x0s, labels, cfg)
    return _maizels_euler_constraint_paths(apply_fn, params, x0s, labels, cfg)


def _maizels_lineage_constraint_terms(
    paths: jnp.ndarray,
    labels: jnp.ndarray,
    cfg: config_dict.ConfigDict,
) -> Dict[str, jnp.ndarray]:
    if labels is None or labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError(
            "maizels_lineage_path metrics require label[:, 0:2] to contain "
            "source and target cell-type ids."
        )

    classifier_path = getattr(cfg.problem, "classifier_path", maizels.DEFAULT_CLASSIFIER)
    classifier_params, class_names, scaler_mean, scaler_scale = (
        maizels.load_jax_classifier_params(classifier_path)
    )
    flat = paths.reshape((-1, paths.shape[-1]))
    logits = maizels.jax_classifier_logits(
        classifier_params,
        scaler_mean,
        scaler_scale,
        flat,
    )
    temperature = float(getattr(cfg.constraints, "classifier_temperature", 1.0))
    probs = jax.nn.softmax(
        logits / jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), 1e-6),
        axis=-1,
    ).reshape((paths.shape[0], paths.shape[1], -1))
    lineage_transition_mode = maizels.lineage_transition_mode_from_config(cfg)

    lambda_final = float(getattr(cfg.constraints, "lambda_final", 0.0))
    target_type_ids = labels[:, 1] if lambda_final > 0.0 else None
    return maizels.lineage_soft_terms_from_probs(
        probs,
        labels[:, 0],
        jnp.asarray(
            maizels.lineage_invalid_transition_matrix(
                class_names,
                transition_mode=lineage_transition_mode,
            ),
            dtype=probs.dtype,
        ),
        jnp.asarray(maizels.classifier_index_lookup(class_names), dtype=jnp.int32),
        target_type_ids=target_type_ids,
    )


def _trajectory_segments(paths: np.ndarray) -> np.ndarray:
    """Convert paths with shape (N, T, D) to PC1/PC2 line segments."""
    xy = paths[:, :, :2]
    segments = np.stack([xy[:, :-1, :], xy[:, 1:, :]], axis=2)
    return segments.reshape((-1, 2, 2))


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
    invalid_mask = _trajectory_violation_mask(paths, cfg) if cfg is not None else None
    ax.scatter(
        x0s[:, 0], x0s[:, 1], s=0.2, alpha=0.2, marker="o", c="gray", label="base"
    )
    ax.scatter(
        x1s[:, 0], x1s[:, 1], s=0.2, alpha=0.2, marker="o", c="C0", label="target"
    )
    if invalid_mask is None:
        ax.add_collection(
            LineCollection(
                _trajectory_segments(paths),
                colors="black",
                linewidths=0.25,
                alpha=0.2,
                label="trajectory",
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
    else:
        valid_paths = paths[~invalid_mask]
        invalid_paths = paths[invalid_mask]
        if valid_paths.shape[0] > 0:
            ax.add_collection(
                LineCollection(
                    _trajectory_segments(valid_paths),
                    colors="black",
                    linewidths=0.25,
                    alpha=0.18,
                    label="valid trajectory",
                )
            )
            valid_points = valid_paths.reshape((-1, valid_paths.shape[-1]))
            ax.scatter(
                valid_points[:, 0],
                valid_points[:, 1],
                s=1.8,
                alpha=0.20,
                marker="o",
                c="black",
                linewidths=0,
            )
        if invalid_paths.shape[0] > 0:
            ax.add_collection(
                LineCollection(
                    _trajectory_segments(invalid_paths),
                    colors="crimson",
                    linewidths=0.65,
                    alpha=0.72,
                    label="violating trajectory",
                )
            )
            invalid_points = invalid_paths.reshape((-1, invalid_paths.shape[-1]))
            ax.scatter(
                invalid_points[:, 0],
                invalid_points[:, 1],
                s=2.2,
                alpha=0.45,
                marker="o",
                c="crimson",
                linewidths=0,
            )
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
    ax.tick_params(axis="both", labelsize=fontsize)
    if cfg is not None:
        _draw_lowd_regions(ax, cfg, label=True)


def _draw_maizels_validity_paths(
    ax,
    paths: np.ndarray,
    x0s: np.ndarray,
    x1s: np.ndarray,
    valid_mask: np.ndarray,
    *,
    title: str,
    xlim: list,
    ylim: list,
    fontsize: float,
) -> None:
    """Draw Maizels PC1/PC2 paths, coloring classifier-invalid paths red."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    invalid_mask = ~valid_mask
    ax.scatter(
        x1s[:, 0],
        x1s[:, 1],
        s=3.0,
        alpha=0.10,
        marker="o",
        c="C0",
        linewidths=0,
        label="target",
    )
    if valid_mask.any():
        valid_paths = paths[valid_mask]
        ax.add_collection(
            LineCollection(
                _trajectory_segments(valid_paths[:, :, :2]),
                colors="black",
                linewidths=0.35,
                alpha=0.30,
                label="classifier-valid",
            )
        )
        ax.scatter(
            x0s[valid_mask, 0],
            x0s[valid_mask, 1],
            s=4.0,
            alpha=0.55,
            marker="o",
            c="black",
            linewidths=0,
        )
    if invalid_mask.any():
        invalid_paths = paths[invalid_mask]
        ax.add_collection(
            LineCollection(
                _trajectory_segments(invalid_paths[:, :, :2]),
                colors="crimson",
                linewidths=0.85,
                alpha=0.80,
                label="classifier-invalid",
            )
        )
        ax.scatter(
            x0s[invalid_mask, 0],
            x0s[invalid_mask, 1],
            s=5.0,
            alpha=0.85,
            marker="o",
            c="crimson",
            linewidths=0,
        )
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
    ax.tick_params(axis="both", labelsize=fontsize)


def _maizels_check_times(check_n: int, *, include_final: bool) -> np.ndarray:
    check_n = max(1, int(check_n))
    if include_final:
        return np.linspace(0.0, 1.0, check_n + 1, dtype=np.float32)[1:]
    return np.linspace(0.0, 1.0, check_n + 2, dtype=np.float32)[1:-1]


def _maizels_time_tag(timepoint: str) -> str:
    return str(timepoint).replace(".", "p").replace("/", "_")


def _maizels_intermediate_timepoints(
    data: Dict[str, np.ndarray],
    source_time: str,
    target_time: str,
    max_times: int,
) -> list:
    source_value = maizels.parse_timepoint(source_time)
    target_value = maizels.parse_timepoint(target_time)
    unique = sorted(
        {
            str(tp)
            for tp, value in zip(data["timepoints"], data["time_values"])
            if source_value < float(value) < target_value
        },
        key=maizels.parse_timepoint,
    )
    if max_times <= 0 or len(unique) <= max_times:
        return unique

    idx = np.linspace(0, len(unique) - 1, num=max_times, dtype=int)
    return [unique[ii] for ii in np.unique(idx)]


def _random_subset_np(
    x: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    replace_if_needed: bool = False,
) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("Cannot sample from an empty array.")
    replace = bool(replace_if_needed and n > x.shape[0])
    idx = rng.choice(x.shape[0], size=int(n), replace=replace)
    return x[idx]


def _np_sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_norm + y_norm - 2.0 * (x @ y.T), 0.0)


def _median_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    *,
    max_points: int = 512,
) -> float:
    z = np.concatenate([x, y], axis=0)
    if z.shape[0] > max_points:
        z = z[rng.choice(z.shape[0], size=max_points, replace=False)]
    sqdist = _np_sqdist(z, z)
    tri = sqdist[np.triu_indices(sqdist.shape[0], k=1)]
    tri = tri[tri > 1e-12]
    if tri.size == 0:
        return 1.0
    return float(np.sqrt(np.median(tri)))


def _rbf_mmd2_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bandwidths,
    rng: np.random.Generator,
    bandwidth_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")

    if bandwidths is None or len(bandwidths) == 0:
        base_bw = _median_bandwidth(x, y, rng)
        bws = np.asarray(bandwidth_multipliers, dtype=np.float64) * base_bw
    else:
        bws = np.asarray(bandwidths, dtype=np.float64)
    bws = np.maximum(bws, 1e-6)

    xx = _np_sqdist(x, x)
    yy = _np_sqdist(y, y)
    xy = _np_sqdist(x, y)
    mmd2 = 0.0
    for bw in bws:
        scale = 2.0 * bw * bw
        mmd2 += (
            float(np.exp(-xx / scale).mean())
            + float(np.exp(-yy / scale).mean())
            - 2.0 * float(np.exp(-xy / scale).mean())
        )
    return max(float(mmd2 / len(bws)), 0.0)


def _sliced_wasserstein_2_np(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_projections: int,
    rng: np.random.Generator,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    if n <= 0:
        return float("nan")
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], size=n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], size=n, replace=False)]

    directions = rng.normal(size=(int(n_projections), x.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    return float(np.sqrt(np.mean((x_proj - y_proj) ** 2)))


def _euler_terminal_at_time(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    tau: float,
    n_steps: int,
) -> np.ndarray:
    """Euler rollout of dx/dt=b_t(x) from t=0 to a requested tau."""
    n_steps = max(1, int(n_steps))
    ts = jnp.linspace(0.0, float(tau), n_steps + 1)

    def step(x, idx):
        t0 = ts[idx]
        dt = ts[idx + 1] - ts[idx]
        x_next = x + dt * _vector_field_batch(apply_fn, params, t0, x, labels)
        return x_next, None

    final, _ = jax.lax.scan(step, x0s, jnp.arange(n_steps))
    return np.asarray(final, dtype=np.float32)


def _flowmap_terminal_at_time(
    apply_fn,
    params: Dict,
    x0s: jnp.ndarray,
    labels: jnp.ndarray,
    tau: float,
    n_steps: int,
) -> np.ndarray:
    """Iteratively sample with learned maps X_{t_k,t_{k+1}} from 0 to tau."""
    n_steps = max(1, int(n_steps))
    ts = jnp.linspace(0.0, float(tau), n_steps + 1)

    def step(x, idx):
        x_next = _flow_map_batch(
            apply_fn,
            params,
            ts[idx],
            ts[idx + 1],
            x,
            labels,
        )
        return x_next, None

    final, _ = jax.lax.scan(step, x0s, jnp.arange(n_steps))
    return np.asarray(final, dtype=np.float32)


def _maizels_distribution_eval_split(
    cfg: config_dict.ConfigDict,
    maizels_cfg,
    points_per_time: int,
    dataset_location: str,
) -> str:
    mode = str(getattr(maizels_cfg, "distribution_eval_source_pool", "auto")).lower()
    aliases = {
        "holdout": "heldout",
        "held_out": "heldout",
        "training": "train",
    }
    mode = aliases.get(mode, mode)
    if mode in ("heldout", "train", "all"):
        return mode
    if mode != "auto":
        raise ValueError(
            "logging.maizels.distribution_eval_source_pool must be one of "
            "'auto', 'heldout', 'train', or 'all'."
        )

    splits = maizels.endpoint_pool_splits(cfg, dataset_location=dataset_location)
    heldout_n = min(int(splits["source_holdout_n"]), int(splits["target_holdout_n"]))
    if heldout_n >= int(points_per_time):
        return "heldout"

    train_n = min(int(splits["source_train_n"]), int(splits["target_train_n"]))
    if train_n > 0:
        return "train"
    return "all"


def _log_maizels_distribution_eval(
    cfg: config_dict.ConfigDict,
    train_state: state_utils.EMATrainState,
    params_for_visual: Dict,
) -> None:
    """Log distributional metrics against unseen intermediate-day populations."""
    maizels_cfg = getattr(cfg.logging, "maizels", None)
    if maizels_cfg is None or not bool(getattr(maizels_cfg, "enabled", False)):
        return
    if not bool(getattr(maizels_cfg, "distribution_eval_enabled", True)):
        return

    dataset_location = getattr(cfg.problem, "dataset_location", None)
    data = maizels.all_timepoint_data(dataset_location)
    source_time = getattr(cfg.problem, "source_time", "D3")
    target_time = getattr(cfg.problem, "target_time", "D8")
    source_value = maizels.parse_timepoint(source_time)
    target_value = maizels.parse_timepoint(target_time)
    denom = max(target_value - source_value, 1e-12)

    max_times = int(getattr(maizels_cfg, "distribution_eval_max_timepoints", 0))
    timepoints = _maizels_intermediate_timepoints(
        data, source_time, target_time, max_times
    )
    if not timepoints:
        return

    points_per_time = int(getattr(maizels_cfg, "distribution_eval_points_per_time", 512))
    points_per_time = max(1, points_per_time)
    plot_pair_mode = getattr(
        maizels_cfg,
        "pair_mode",
        getattr(cfg.problem, "maizels_pair_mode", "none"),
    )
    if plot_pair_mode == "same_as_training":
        plot_pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    seed = int(
        getattr(
            maizels_cfg,
            "distribution_eval_seed",
            int(getattr(getattr(cfg, "training", None), "seed", 0)) + 1701,
        )
    )
    rng = np.random.default_rng(seed)

    eval_split = _maizels_distribution_eval_split(
        cfg,
        maizels_cfg,
        points_per_time,
        dataset_location,
    )
    eval_pairs, _ = maizels.make_endpoint_split_pair_pool(
        cfg,
        points_per_time,
        split=eval_split,
        dataset_location=dataset_location,
        pair_mode=str(plot_pair_mode),
        seed=seed + 17,
    )
    x0_eval_all = eval_pairs["x0"].astype(np.float32)
    x1_eval_all = eval_pairs["x1"].astype(np.float32)
    labels_eval_all = jnp.asarray(eval_pairs["label"])

    n_proj = int(getattr(maizels_cfg, "distribution_eval_wasserstein_projections", 256))
    euler_n_steps = int(
        getattr(
            maizels_cfg,
            "distribution_eval_euler_n_steps",
            getattr(maizels_cfg, "euler_n_steps", 25),
        )
    )
    flowmap_n_steps = int(
        getattr(
            maizels_cfg,
            "distribution_eval_flowmap_n_steps",
            getattr(maizels_cfg, "euler_n_steps", 25),
        )
    )
    raw_bandwidths = getattr(maizels_cfg, "distribution_eval_mmd_bandwidths", None)
    if raw_bandwidths is not None:
        raw_bandwidths = list(raw_bandwidths)
    bandwidth_multipliers = tuple(
        float(x)
        for x in getattr(
            maizels_cfg,
            "distribution_eval_mmd_bandwidth_multipliers",
            [0.25, 0.5, 1.0, 2.0, 4.0],
        )
    )

    metrics = {}
    aggregate = {
        "direct_rbf_mmd2": [],
        "direct_sliced_w2": [],
        "flowmap_rbf_mmd2": [],
        "flowmap_sliced_w2": [],
        "euler_rbf_mmd2": [],
        "euler_sliced_w2": [],
        "linear_rbf_mmd2": [],
        "linear_sliced_w2": [],
    }

    for timepoint in timepoints:
        actual_all = data["x"][data["timepoints"] == timepoint].astype(np.float32)
        n_compare = min(points_per_time, actual_all.shape[0], x0_eval_all.shape[0])
        if n_compare <= 0:
            continue

        actual = _random_subset_np(
            actual_all,
            n_compare,
            rng,
            replace_if_needed=False,
        )
        x0_eval = x0_eval_all[:n_compare]
        x1_eval = x1_eval_all[:n_compare]
        labels_eval = labels_eval_all[:n_compare]
        tau = float(
            np.clip(
                (maizels.parse_timepoint(timepoint) - source_value) / denom,
                0.0,
                1.0,
            )
        )

        direct = np.asarray(
            _flow_map_batch(
                train_state.apply_fn,
                params_for_visual,
                0.0,
                tau,
                jnp.asarray(x0_eval, dtype=jnp.float32),
                labels_eval,
            ),
            dtype=np.float32,
        )
        flowmap_sample = _flowmap_terminal_at_time(
            train_state.apply_fn,
            params_for_visual,
            jnp.asarray(x0_eval, dtype=jnp.float32),
            labels_eval,
            tau,
            flowmap_n_steps,
        )
        euler = _euler_terminal_at_time(
            train_state.apply_fn,
            params_for_visual,
            jnp.asarray(x0_eval, dtype=jnp.float32),
            labels_eval,
            tau,
            euler_n_steps,
        )
        linear = ((1.0 - tau) * x0_eval + tau * x1_eval).astype(np.float32)

        tag = _maizels_time_tag(timepoint)
        for name, pred in [
            ("direct", direct),
            ("flowmap", flowmap_sample),
            ("euler", euler),
            ("linear", linear),
        ]:
            mmd2 = _rbf_mmd2_np(
                pred,
                actual,
                bandwidths=raw_bandwidths,
                rng=rng,
                bandwidth_multipliers=bandwidth_multipliers,
            )
            sw2 = _sliced_wasserstein_2_np(
                pred,
                actual,
                n_projections=n_proj,
                rng=rng,
            )
            metrics[f"distribution_eval/{tag}_{name}_rbf_mmd2"] = mmd2
            metrics[f"distribution_eval/{tag}_{name}_sliced_w2"] = sw2
            aggregate[f"{name}_rbf_mmd2"].append(mmd2)
            aggregate[f"{name}_sliced_w2"].append(sw2)

    for key, values in aggregate.items():
        if values:
            metrics[f"distribution_eval/{key}_mean"] = float(np.mean(values))

    if metrics:
        wandb.log(metrics)


def _log_maizels_trajectory_diagnostics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    params_for_visual: Dict,
    paired_x0s: jnp.ndarray,
    x1s: jnp.ndarray,
    labels: jnp.ndarray,
    *,
    fontsize: float,
) -> None:
    """Log classifier-validity plots on Maizels D3/D8 cells held out from training."""
    maizels_cfg = getattr(cfg.logging, "maizels", None)
    if maizels_cfg is None or not bool(getattr(maizels_cfg, "enabled", False)):
        return

    n_plot = int(getattr(maizels_cfg, "plot_bs", 128))
    if n_plot <= 0:
        return

    plot_pair_mode = getattr(
        maizels_cfg,
        "pair_mode",
        getattr(cfg.problem, "maizels_pair_mode", "none"),
    )
    if plot_pair_mode == "same_as_training":
        plot_pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    plot_seed = int(
        getattr(
            maizels_cfg,
            "plot_seed",
            int(getattr(getattr(cfg, "training", None), "seed", 0)) + 997,
        )
    )
    heldout_pairs, _ = maizels.make_heldout_pair_pool(
        cfg,
        n_plot,
        dataset_location=getattr(cfg.problem, "dataset_location", None),
        pair_mode=str(plot_pair_mode),
        seed=plot_seed,
    )

    x0_plot = jnp.asarray(heldout_pairs["x0"], dtype=jnp.float32)
    x1_plot = jnp.asarray(heldout_pairs["x1"], dtype=jnp.float32)
    labels_plot = jnp.asarray(heldout_pairs["label"])
    labels_np = np.asarray(labels_plot)
    start_type_ids = labels_np[:, 0].astype(np.int32)
    target_type_ids = labels_np[:, 1].astype(np.int32)

    path_n_times = max(2, int(getattr(maizels_cfg, "path_n_times", 25)))
    path_times = np.linspace(0.0, 1.0, path_n_times, dtype=np.float32)
    euler_n_steps = max(
        1, int(getattr(maizels_cfg, "euler_n_steps", path_n_times - 1))
    )
    flowmap_n_steps = max(
        1, int(getattr(maizels_cfg, "flowmap_n_steps", euler_n_steps))
    )
    check_n_times = max(1, int(getattr(maizels_cfg, "check_n_times", 5)))
    prob_threshold = float(getattr(maizels_cfg, "prob_threshold", 0.85))
    margin_threshold = float(getattr(maizels_cfg, "margin_threshold", 1.0))
    classifier_batch_size = int(getattr(maizels_cfg, "classifier_batch_size", 8192))
    classifier_path = getattr(cfg.problem, "classifier_path", maizels.DEFAULT_CLASSIFIER)
    lineage_transition_mode = getattr(
        maizels_cfg,
        "lineage_transition_mode",
        "same_as_problem",
    )
    if lineage_transition_mode in (None, "", "same_as_problem", "same_as_training"):
        lineage_transition_mode = getattr(
            cfg.problem,
            "lineage_transition_mode",
            "descendant",
        )
    lineage_transition_mode = maizels.resolve_lineage_transition_mode(
        lineage_transition_mode
    )

    direct_paths = _one_step_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        path_times,
    )
    flowmap_paths = _multi_step_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        flowmap_n_steps,
    )
    euler_paths = _euler_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        euler_n_steps,
    )
    gt_paths = _interpolant_paths(
        statics.interp,
        x0_plot,
        x1_plot,
        labels_plot,
        path_times,
    )

    direct_check_paths = _one_step_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        _maizels_check_times(check_n_times, include_final=True),
    )
    flowmap_check_paths = _multi_step_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        check_n_times,
    )[:, 1:, :]
    euler_check_paths = _euler_paths(
        train_state.apply_fn,
        params_for_visual,
        x0_plot,
        labels_plot,
        check_n_times,
    )[:, 1:, :]
    gt_check_paths = _interpolant_paths(
        statics.interp,
        x0_plot,
        x1_plot,
        labels_plot,
        _maizels_check_times(check_n_times, include_final=False),
    )

    direct_validity = maizels.check_paths_with_classifier(
        paths=direct_check_paths,
        start_type_ids=start_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=None,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
    )
    flowmap_validity = maizels.check_paths_with_classifier(
        paths=flowmap_check_paths,
        start_type_ids=start_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=None,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
    )
    euler_validity = maizels.check_paths_with_classifier(
        paths=euler_check_paths,
        start_type_ids=start_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=None,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
    )
    gt_validity = maizels.check_paths_with_classifier(
        paths=gt_check_paths,
        start_type_ids=start_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=target_type_ids,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
    )

    direct_valid = np.asarray(direct_validity["valid"], dtype=bool)
    flowmap_valid = np.asarray(flowmap_validity["valid"], dtype=bool)
    euler_valid = np.asarray(euler_validity["valid"], dtype=bool)
    gt_valid = np.asarray(gt_validity["valid"], dtype=bool)
    panel_xlim, panel_ylim = lowd_limits_for(
        cfg,
        x0_plot,
        x1_plot,
        direct_paths,
        flowmap_paths,
        euler_paths,
        gt_paths,
    )

    fig, axs = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(24, 5),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    _draw_maizels_validity_paths(
        axs[0],
        direct_paths,
        np.asarray(x0_plot),
        np.asarray(x1_plot),
        direct_valid,
        title="Held-out direct flow-map paths in PC1/PC2",
        xlim=panel_xlim,
        ylim=panel_ylim,
        fontsize=fontsize,
    )
    _draw_maizels_validity_paths(
        axs[1],
        flowmap_paths,
        np.asarray(x0_plot),
        np.asarray(x1_plot),
        flowmap_valid,
        title=f"Held-out flow-map sampling in PC1/PC2 ({flowmap_n_steps} steps)",
        xlim=panel_xlim,
        ylim=panel_ylim,
        fontsize=fontsize,
    )
    _draw_maizels_validity_paths(
        axs[2],
        euler_paths,
        np.asarray(x0_plot),
        np.asarray(x1_plot),
        euler_valid,
        title=f"Held-out Euler rollouts in PC1/PC2 ({euler_n_steps} steps)",
        xlim=panel_xlim,
        ylim=panel_ylim,
        fontsize=fontsize,
    )
    _draw_maizels_validity_paths(
        axs[3],
        gt_paths,
        np.asarray(x0_plot),
        np.asarray(x1_plot),
        gt_valid,
        title="Held-out D3/D8 interpolants in PC1/PC2",
        xlim=panel_xlim,
        ylim=panel_ylim,
        fontsize=fontsize,
    )
    axs[0].legend(loc="upper right", fontsize=9, markerscale=3, frameon=True)

    wandb.log(
        {
            "plots/maizels_classifier_validity_paths": wandb.Image(fig),
            "maizels/model_direct_invalid_trajectory_pct": 100.0
            * float(np.mean(~direct_valid)),
            "maizels/model_flowmap_invalid_trajectory_pct": 100.0
            * float(np.mean(~flowmap_valid)),
            "maizels/model_euler_invalid_trajectory_pct": 100.0
            * float(np.mean(~euler_valid)),
            "maizels/interpolant_invalid_trajectory_pct": 100.0
            * float(np.mean(~gt_valid)),
        }
    )


def _rollout_forbidden_box_metrics(
    cfg: config_dict.ConfigDict, rollout_paths: list, rollout_name: str
) -> Dict[str, float]:
    """Measure point and trajectory box occupancy for rollout polylines."""
    bounds = _forbidden_box_bounds(cfg)
    if bounds is None:
        return {}

    metrics = {}
    for step, paths in rollout_paths:
        path_points = np.asarray(paths).reshape((-1, paths.shape[-1]))
        inside = _np_points_in_forbidden_box(path_points, bounds)
        total = int(inside.size)
        pct = 100.0 * int(np.sum(inside)) / max(total, 1)
        trajectory_pct = 100.0 * _np_trajectory_forbidden_box_rate(paths, bounds)
        prefix = f"forbidden_box/{rollout_name}_{step}"
        metrics[f"{prefix}_point_pct"] = pct
        metrics[f"{prefix}_trajectory_pct"] = trajectory_pct

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
        }

    if ctype == "box_path":
        return {}

    if ctype == "dive_gate_path":
        params = dist_utils.safe_unreplicate(cfg, train_state.params)
        constraint_bs = int(getattr(cfg.constraints, "constraint_batch_size", 0))
        if constraint_bs <= 0:
            constraint_fraction = float(
                getattr(cfg.constraints, "constraint_batch_fraction", 1.0)
            )
            constraint_bs = max(1, int(x0batch.shape[0] * constraint_fraction))
        constraint_bs = min(x0batch.shape[0], constraint_bs)

        x0_constraint = x0batch[:constraint_bs]
        label_constraint = None if label_batch is None else label_batch[:constraint_bs]
        mode = _dive_gate_path_mode(cfg)

        if mode == "flow_map":
            times = _dive_gate_path_times(cfg, x0_constraint.dtype)
            if label_constraint is None:
                paths = jax.vmap(
                    lambda x: jax.vmap(
                        lambda tau: train_state.apply_fn(
                            params,
                            0.0,
                            tau,
                            x,
                            label=None,
                            train=False,
                            calc_weight=False,
                            return_X_and_phi=False,
                        )
                    )(times)
                )(x0_constraint)
            else:
                paths = jax.vmap(
                    lambda x, lbl: jax.vmap(
                        lambda tau: train_state.apply_fn(
                            params,
                            0.0,
                            tau,
                            x,
                            label=lbl,
                            train=False,
                            calc_weight=False,
                            return_X_and_phi=False,
                        )
                    )(times)
                )(
                    x0_constraint,
                    label_constraint,
                )
        else:
            euler_steps = int(getattr(cfg.constraints, "euler_steps", 100))
            labels_for_euler = label_constraint
            if labels_for_euler is None:
                labels_for_euler = -jnp.ones((x0_constraint.shape[0],))
            paths = jnp.asarray(
                _euler_paths(
                    train_state.apply_fn,
                    params,
                    x0_constraint,
                    labels_for_euler,
                    euler_steps,
                ),
                dtype=x0_constraint.dtype,
            )

        terms = _dive_gate_soft_terms(paths, cfg)
        lambda_hit = float(getattr(cfg.constraints, "lambda_hit", 1.0))
        lambda_hit_a = float(getattr(cfg.constraints, "lambda_hit_a", lambda_hit))
        lambda_hit_b = float(getattr(cfg.constraints, "lambda_hit_b", lambda_hit))
        lambda_hit_c = float(getattr(cfg.constraints, "lambda_hit_c", 1.0))
        lambda_order = float(getattr(cfg.constraints, "lambda_order", 1.0))
        weighted = constraint_scale * cfg.constraints.weight * (
            lambda_hit_a * terms["hit_a_loss"]
            + lambda_hit_b * terms["hit_b_loss"]
            + lambda_hit_c * terms["hit_c_loss"]
            + lambda_order * terms["order_loss"]
        )

        metrics = {
            "constraint/dive_gate_hit_a_loss": terms["hit_a_loss"],
            "constraint/dive_gate_hit_b_loss": terms["hit_b_loss"],
            "constraint/dive_gate_hit_c_loss": terms["hit_c_loss"],
            "constraint/dive_gate_hit_loss": terms["hit_loss"],
            "constraint/dive_gate_order_loss": terms["order_loss"],
            "constraint/dive_gate_total": weighted,
            "constraint/dive_gate_hit_a_prob": terms["hit_a_prob"],
            "constraint/dive_gate_hit_b_prob": terms["hit_b_prob"],
            "constraint/dive_gate_hit_c_prob": terms["hit_c_prob"],
            "constraint/dive_gate_b_before_a_prob": terms["bad_b_before_a_prob"],
            "constraint/dive_gate_c_before_b_prob": terms["bad_c_before_b_prob"],
            "constraint/dive_gate_soft_a_occupancy": terms["soft_a_occupancy"],
            "constraint/dive_gate_soft_b_occupancy": terms["soft_b_occupancy"],
            "constraint/dive_gate_soft_c_occupancy": terms["soft_c_occupancy"],
            "constraint/anneal_scale": constraint_scale,
        }

        hard_details = _dive_gate_violation_details(np.asarray(paths), cfg)
        if hard_details is not None:
            metrics.update(
                {
                    "constraint/dive_gate_hard_violation_pct": 100.0
                    * float(np.mean(hard_details["invalid"])),
                    "constraint/dive_gate_hard_hit_a_pct": 100.0
                    * float(np.mean(hard_details["hit_a"])),
                    "constraint/dive_gate_hard_hit_b_pct": 100.0
                    * float(np.mean(hard_details["hit_b"])),
                    "constraint/dive_gate_hard_b_before_a_pct": 100.0
                    * float(np.mean(hard_details["b_before_a"])),
                    "constraint/dive_gate_hard_c_before_b_pct": 100.0
                    * float(np.mean(hard_details["c_before_b"])),
                }
            )
        return metrics

    if ctype == "maizels_lineage_path":
        params = dist_utils.safe_unreplicate(cfg, train_state.params)
        constraint_bs = int(getattr(cfg.constraints, "constraint_batch_size", 0))
        if getattr(cfg.constraints, "path_mode", "flowmap") == "loss_points":
            if statics is None or getattr(statics, "interp", None) is None:
                raise ValueError(
                    "maizels_lineage_path loss_points metrics require statics.interp"
                )

            diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, x0batch.shape[0])
            if offdiag_bs <= 0:
                zero = jnp.asarray(0.0, dtype=x0batch.dtype)
                return {"constraint/lineage_total": zero}

            x0_available = x0batch[diag_bs:]
            x1_available = x1batch[diag_bs:]
            label_available = None if label_batch is None else label_batch[diag_bs:]
            s_available = sbatch[diag_bs:]
            t_available = tbatch[diag_bs:]
            if constraint_bs <= 0:
                constraint_fraction = float(
                    getattr(cfg.constraints, "constraint_batch_fraction", 1.0)
                )
                constraint_bs = max(1, int(x0_available.shape[0] * constraint_fraction))
            constraint_bs = min(x0_available.shape[0], constraint_bs)

            x0_constraint = x0_available[:constraint_bs]
            x1_constraint = x1_available[:constraint_bs]
            label_constraint = (
                None if label_available is None else label_available[:constraint_bs]
            )
            s_constraint = s_available[:constraint_bs]
            t_constraint = t_available[:constraint_bs]
            x_s = statics.interp.batch_calc_It(
                s_constraint,
                x0_constraint,
                x1_constraint,
                label_constraint,
            )
            x_t = _map_between_positions(
                params,
                train_state.apply_fn,
                s_constraint,
                t_constraint,
                x_s,
                label_constraint,
            )
            paths = jnp.stack([x_s, x_t], axis=1)
        else:
            if constraint_bs <= 0:
                constraint_fraction = float(
                    getattr(cfg.constraints, "constraint_batch_fraction", 1.0)
                )
                constraint_bs = max(1, int(x0batch.shape[0] * constraint_fraction))
            constraint_bs = min(x0batch.shape[0], constraint_bs)

            x0_constraint = x0batch[:constraint_bs]
            label_constraint = (
                None if label_batch is None else label_batch[:constraint_bs]
            )
            paths = _maizels_lineage_constraint_paths(
                train_state.apply_fn,
                params,
                x0_constraint,
                label_constraint,
                cfg,
            )
        terms = _maizels_lineage_constraint_terms(paths, label_constraint, cfg)

        lambda_start = float(getattr(cfg.constraints, "lambda_start", 1.0))
        lambda_transition = float(getattr(cfg.constraints, "lambda_transition", 1.0))
        lambda_final = float(getattr(cfg.constraints, "lambda_final", 0.0))
        weighted = constraint_scale * cfg.constraints.weight * (
            lambda_start * terms["start_invalid_loss"]
            + lambda_transition * terms["transition_invalid_loss"]
            + lambda_final * terms["final_invalid_loss"]
        )

        return {"constraint/lineage_total": weighted}

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
        learned_paths = []
        interp_paths = []
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
            learned_paths.append(np.asarray(learned_i, dtype=np.float32))
            interp_paths.append(np.asarray(interp_i, dtype=np.float32))

        learned_any_f = learned_any.astype(jnp.float32)
        interp_any_f = interp_any.astype(jnp.float32)
        learned_path_arr = np.stack(learned_paths, axis=1)
        interp_path_arr = np.stack(interp_paths, axis=1)
        learned_point_inside = _np_points_in_forbidden_box(
            learned_path_arr.reshape((-1, learned_path_arr.shape[-1])), bounds
        )
        interp_point_inside = _np_points_in_forbidden_box(
            interp_path_arr.reshape((-1, interp_path_arr.shape[-1])), bounds
        )
        metrics.update(
            {
                "forbidden_box/learned_path_rate": jnp.mean(learned_any_f),
                "forbidden_box/interpolant_path_rate": jnp.mean(interp_any_f),
                "forbidden_box/learned_point_pct": 100.0
                * float(np.mean(learned_point_inside)),
                "forbidden_box/interpolant_point_pct": 100.0
                * float(np.mean(interp_point_inside)),
                "forbidden_box/learned_trajectory_pct": 100.0
                * _np_trajectory_forbidden_box_rate(learned_path_arr, bounds),
                "forbidden_box/interpolant_trajectory_pct": 100.0
                * _np_trajectory_forbidden_box_rate(interp_path_arr, bounds),
            }
        )

    return metrics


def _resolve_maizels_logging_pair_mode(cfg: config_dict.ConfigDict, maizels_cfg):
    pair_mode = getattr(
        maizels_cfg,
        "validation_pair_mode",
        getattr(
            maizels_cfg,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        ),
    )
    if pair_mode == "same_as_training":
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    return str(pair_mode)


def _maizels_validation_batch(cfg: config_dict.ConfigDict) -> Dict[str, jnp.ndarray]:
    maizels_cfg = cfg.logging.maizels
    dataset_location = getattr(cfg.problem, "dataset_location", None)
    n_val = max(1, int(getattr(maizels_cfg, "validation_bs", 1024)))
    seed = int(
        getattr(
            maizels_cfg,
            "validation_seed",
            int(getattr(getattr(cfg, "training", None), "seed", 0)) + 2701,
        )
    )
    pair_mode = _resolve_maizels_logging_pair_mode(cfg, maizels_cfg)
    cache_key = (
        str(dataset_location),
        str(getattr(cfg.problem, "source_time", "D3")),
        str(getattr(cfg.problem, "target_time", "D8")),
        pair_mode,
        n_val,
        seed,
        int(getattr(cfg.training, "seed", 0)),
        int(getattr(cfg.problem, "maizels_holdout_seed", 701)),
        float(getattr(cfg.problem, "maizels_holdout_fraction", 0.0)),
        int(getattr(cfg.problem, "maizels_holdout_n", 0)),
    )
    if cache_key in _MAIZELS_VALIDATION_CACHE:
        return _MAIZELS_VALIDATION_CACHE[cache_key]

    pairs, _ = maizels.make_heldout_pair_pool(
        cfg,
        n_val,
        dataset_location=dataset_location,
        pair_mode=pair_mode,
        seed=seed,
    )
    x0 = jnp.asarray(pairs["x0"], dtype=jnp.float32)
    x1 = jnp.asarray(pairs["x1"], dtype=jnp.float32)
    label = jnp.asarray(pairs["label"])
    bs = x0.shape[0]
    diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, bs)

    keys = jax.random.split(jax.random.PRNGKey(seed + 31), 5)
    if offdiag_bs == 0:
        sbatch, tbatch = loss_args._sample_diagonal(
            keys[0], bs, cfg.training.tmin, cfg.training.tmax
        )
    else:
        s_diag, t_diag = loss_args._sample_diagonal(
            keys[0], diag_bs, cfg.training.tmin, cfg.training.tmax
        )
        s_offdiag, t_offdiag = loss_args._sample_triangle(
            keys[1],
            keys[2],
            offdiag_bs,
            cfg.training.tmin,
            cfg.training.tmax,
        )
        sbatch, tbatch = loss_args._concat_diag_offdiag(
            s_diag,
            t_diag,
            s_offdiag,
            t_offdiag,
        )

    if cfg.training.psd_type == "midpoint":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = None
    elif cfg.training.psd_type == "uniform":
        hbatch = jax.random.uniform(keys[3], shape=(bs,), minval=0.0, maxval=1.0)
        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch
    elif cfg.training.psd_type is None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown psd_type: {cfg.training.psd_type}")

    batch = {
        "x0": x0,
        "x1": x1,
        "label": label,
        "s": sbatch,
        "t": tbatch,
        "u": ubatch,
        "h": hbatch,
        "dropout_keys": jax.random.split(keys[4], bs),
    }
    _MAIZELS_VALIDATION_CACHE[cache_key] = batch
    return batch


def compute_maizels_validation_metrics(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    step: jnp.ndarray,
) -> Dict[str, float]:
    """Evaluate the current objective on held-out Maizels D3/D8 endpoint pairs."""
    if getattr(cfg.problem, "target", None) != "maizels_pca50":
        return {}
    maizels_cfg = getattr(cfg.logging, "maizels", None)
    if maizels_cfg is None or not bool(getattr(maizels_cfg, "validation_enabled", False)):
        return {}

    batch = _maizels_validation_batch(cfg)
    params = dist_utils.safe_unreplicate(cfg, train_state.params)
    teacher_params = dist_utils.safe_unreplicate(
        cfg, loss_args.select_teacher_params(cfg, train_state)
    )
    constraint_scale = loss_args.compute_constraint_anneal_scale(cfg, step)
    stage2_scale = loss_args.compute_two_stage_scale(cfg, step)
    bs = batch["x0"].shape[0]
    constraint_scale_batch = jnp.full((bs,), constraint_scale, dtype=jnp.float32)
    stage2_scale_batch = jnp.full((bs,), stage2_scale, dtype=jnp.float32)

    val_loss = statics.loss(
        params,
        teacher_params,
        batch["x0"],
        batch["x1"],
        batch["label"],
        batch["s"],
        batch["t"],
        batch["u"],
        batch["h"],
        batch["dropout_keys"],
        constraint_scale_batch,
        stage2_scale_batch,
    )
    return {"validation_loss": val_loss}


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

    # Log held-out Maizels validation loss if configured.
    try:
        metrics.update(
            compute_maizels_validation_metrics(cfg, statics, train_state, step)
        )
    except Exception as e:
        print(f"Warning: Maizels validation metric computation failed: {e}")

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
    # Get parameters for visualization
    params_for_visual = get_params_for_sampling(cfg, train_state, param_type="visual")
    diagonal_only = is_diagonal_only_training(cfg)

    ## common plot parameters
    plt.close("all")
    sns.set_palette("deep")
    fw, fh = 4, 4
    fontsize = 12.5

    ## set up plot array
    steps = [1, 2, 5, 10, 25]
    titles = ["base and target"] + [rf"${step}$-step" for step in steps]

    ## extract target samples
    is_maizels = getattr(cfg.problem, "target", None) == "maizels_pca50"
    if is_maizels:
        maizels_cfg = getattr(cfg.logging, "maizels", None)
        plot_pair_mode = getattr(
            maizels_cfg,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        )
        if plot_pair_mode == "same_as_training":
            plot_pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
        plot_seed = int(
            getattr(
                maizels_cfg,
                "plot_seed",
                int(getattr(getattr(cfg, "training", None), "seed", 0)) + 997,
            )
        )
        plot_batch, _ = maizels.make_heldout_pair_pool(
            cfg,
            cfg.logging.plot_bs,
            dataset_location=getattr(cfg.problem, "dataset_location", None),
            pair_mode=str(plot_pair_mode),
            seed=plot_seed + 101,
        )
    else:
        plot_batch = next(statics.ds)
    paired_plot_x0s, plot_x1s, plot_labels = extract_lowd_batch_components(plot_batch)
    plot_x1s = jnp.asarray(plot_x1s)[: cfg.logging.plot_bs]
    if paired_plot_x0s is not None:
        paired_plot_x0s = jnp.asarray(paired_plot_x0s)[: cfg.logging.plot_bs]
    if plot_labels is not None:
        plot_labels = jnp.asarray(plot_labels)[: cfg.logging.plot_bs]

    ## draw multi-step samples from the model
    if is_maizels and paired_plot_x0s is not None:
        x0s = paired_plot_x0s
    else:
        x0s = statics.sample_rho0(cfg.logging.plot_bs, prng_key)
        prng_key = jax.random.split(prng_key)[0]
    xhats = np.zeros((len(steps), cfg.logging.plot_bs, cfg.problem.d))
    sample_labels = (
        plot_labels
        if is_maizels and plot_labels is not None
        else -jnp.ones(cfg.logging.plot_bs)
    )
    if diagonal_only:
        titles = ["base and target"] + [rf"${step}$-step Euler" for step in steps]
        for kk, step in enumerate(steps):
            xhats[kk] = _euler_paths(
                train_state.apply_fn,
                params_for_visual,
                x0s,
                sample_labels,
                step,
            )[:, -1, :]
    else:
        # Use flow map batch sampler for single-device visualization.
        batch_sample = flow_map.batch_sample
        for kk, step in enumerate(steps):
            xhats[kk] = batch_sample(
                train_state.apply_fn,
                params_for_visual,
                x0s,
                step,
                sample_labels,
            )

    # Track full direct, interpolant, and multi-step trajectories for a subset.
    line_bs = min(cfg.logging.plot_bs, getattr(cfg.logging, "line_plot_bs", 1000))
    n_line_times = max(2, int(getattr(cfg.logging, "line_plot_n_times", 20)))
    line_times = np.linspace(0.0, 1.0, n_line_times)
    one_step_line_paths = None
    ground_truth_line_paths = None
    multi_step_line_paths = []
    if paired_plot_x0s is not None:
        gt_line_bs = min(line_bs, paired_plot_x0s.shape[0])
        x0_gt_line = paired_plot_x0s[:gt_line_bs]
        x1_gt_line = plot_x1s[:gt_line_bs]
        labels_gt_line = None if plot_labels is None else plot_labels[:gt_line_bs]
        ground_truth_line_paths = _interpolant_paths(
            statics.interp,
            x0_gt_line,
            x1_gt_line,
            labels_gt_line,
            line_times,
        )

    if not diagonal_only:
        x0_line = x0s[:line_bs]
        x1_line = plot_x1s[:line_bs]
        labels_line = sample_labels[:line_bs]
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
    labels_multi_line = sample_labels[:multi_line_bs]
    if not diagonal_only:
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
    )

    for ax in axs.ravel():
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
            panel_xlim, panel_ylim = lowd_limits_for(cfg, x0s, plot_x1s)
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
            panel_xlim, panel_ylim = lowd_limits_for(cfg, plot_x1s, xhats[jj - 1])
        _draw_lowd_regions(ax, cfg)
        ax.set_xlim(panel_xlim)
        ax.set_ylim(panel_ylim)

    wandb.log({"samples": wandb.Image(fig)})

    if ground_truth_line_paths is not None:
        gt_fig, gt_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        gt_xlim, gt_ylim = lowd_limits_for(
            cfg,
            x0_gt_line,
            x1_gt_line,
            ground_truth_line_paths,
        )
        _draw_trajectory_paths(
            gt_ax,
            ground_truth_line_paths,
            np.asarray(x0_gt_line),
            np.asarray(x1_gt_line),
            title=f"Ground truth interpolant I_t(x0, x1), {n_line_times} times",
            xlim=gt_xlim,
            ylim=gt_ylim,
            fontsize=fontsize,
            cfg=cfg,
        )
        gt_ax.legend(loc="upper right", fontsize=10, markerscale=6, frameon=True)
        ground_truth_gate_metrics = _rollout_matched_gate_metrics(
            cfg, [("interpolant", ground_truth_line_paths)], "ground_truth"
        )
        ground_truth_dive_metrics = _rollout_dive_gate_metrics(
            cfg, [("interpolant", ground_truth_line_paths)], "ground_truth"
        )
        wandb.log(
            {
                "trajectory_ground_truth_lines": wandb.Image(gt_fig),
                **ground_truth_gate_metrics,
                **ground_truth_dive_metrics,
            }
        )

    if not diagonal_only:
        # Visualize direct trajectory slices X_{0,t}(x0) for fixed times.
        traj_bs = min(cfg.logging.plot_bs, getattr(cfg.logging, "traj_plot_bs", 2000))
        x0_traj = x0s[:traj_bs]
        x1_traj = plot_x1s[:traj_bs]
        labels = sample_labels[:traj_bs]
        time_points = [0.0, 0.25, 0.5, 0.75, 1.0]

        traj_fig, traj_axs = plt.subplots(
            nrows=1,
            ncols=len(time_points),
            figsize=(fw * len(time_points), fh),
            sharex=False,
            sharey=False,
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

            ax.scatter(
                x1_traj[:, 0], x1_traj[:, 1], s=0.3, alpha=0.5, marker="o", c="C0"
            )
            ax.scatter(xt[:, 0], xt[:, 1], s=0.3, alpha=0.5, marker="o", c="black")
            ax.set_title(rf"$t={tt:.2f}$", fontsize=fontsize)
            panel_xlim, panel_ylim = lowd_limits_for(cfg, x1_traj, xt)
            ax.set_xlim(panel_xlim)
            ax.set_ylim(panel_ylim)
            ax.set_aspect("equal")
            ax.grid(which="both", axis="both", color="0.90", alpha=0.2)
            ax.tick_params(axis="both", labelsize=fontsize)
            _draw_lowd_regions(ax, cfg)

        wandb.log({"trajectory_times": wandb.Image(traj_fig)})

        line_fig, line_ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        line_xlim, line_ylim = lowd_limits_for(
            cfg,
            x0_line,
            x1_line,
            one_step_line_paths,
        )
        _draw_trajectory_paths(
            line_ax,
            one_step_line_paths,
            np.asarray(x0_line),
            np.asarray(x1_line),
            title=f"1-step trajectory X_{{0,t}}(x), {n_line_times} times",
            xlim=line_xlim,
            ylim=line_ylim,
            fontsize=fontsize,
            cfg=cfg,
        )
        line_ax.legend(loc="upper right", fontsize=10, markerscale=6, frameon=True)
        one_step_gate_metrics = _rollout_matched_gate_metrics(
            cfg, [("direct", one_step_line_paths)], "direct"
        )
        one_step_dive_metrics = _rollout_dive_gate_metrics(
            cfg, [("direct", one_step_line_paths)], "direct"
        )
        wandb.log(
            {
                "trajectory_1step_lines": wandb.Image(line_fig),
                **one_step_gate_metrics,
                **one_step_dive_metrics,
            }
        )

        multi_fig, multi_axs = plt.subplots(
            nrows=1,
            ncols=len(multi_step_line_paths),
            figsize=(fw * len(multi_step_line_paths), fh),
            sharex=False,
            sharey=False,
            constrained_layout=True,
            squeeze=False,
        )
        for ax, (step, paths) in zip(multi_axs.ravel(), multi_step_line_paths):
            panel_xlim, panel_ylim = lowd_limits_for(
                cfg,
                x0_multi_line,
                x1_multi_line,
                paths,
            )
            _draw_trajectory_paths(
                ax,
                paths,
                np.asarray(x0_multi_line),
                np.asarray(x1_multi_line),
                title=f"{step}-step rollout",
                xlim=panel_xlim,
                ylim=panel_ylim,
                fontsize=fontsize,
                cfg=cfg,
            )
        multi_axs.ravel()[0].legend(
            loc="upper right", fontsize=10, markerscale=6, frameon=True
        )

        multistep_box_metrics = _rollout_forbidden_box_metrics(
            cfg, multi_step_line_paths, "multistep"
        )
        multistep_gate_metrics = _rollout_matched_gate_metrics(
            cfg, multi_step_line_paths, "multistep"
        )
        multistep_dive_metrics = _rollout_dive_gate_metrics(
            cfg, multi_step_line_paths, "multistep"
        )
        wandb.log(
            {
                "trajectory_multistep_lines": wandb.Image(multi_fig),
                **multistep_box_metrics,
                **multistep_gate_metrics,
                **multistep_dive_metrics,
            }
        )

    euler_fig, euler_axs = plt.subplots(
        nrows=1,
        ncols=len(euler_line_paths),
        figsize=(fw * len(euler_line_paths), fh),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (step, paths) in zip(euler_axs.ravel(), euler_line_paths):
        panel_xlim, panel_ylim = lowd_limits_for(
            cfg,
            x0_multi_line,
            x1_multi_line,
            paths,
        )
        _draw_trajectory_paths(
            ax,
            paths,
            np.asarray(x0_multi_line),
            np.asarray(x1_multi_line),
            title=f"{step}-step Euler",
            xlim=panel_xlim,
            ylim=panel_ylim,
            fontsize=fontsize,
            cfg=cfg,
        )
    euler_axs.ravel()[0].legend(
        loc="upper right", fontsize=10, markerscale=6, frameon=True
    )
    euler_box_metrics = _rollout_forbidden_box_metrics(
        cfg, euler_line_paths, "euler"
    )
    euler_gate_metrics = _rollout_matched_gate_metrics(
        cfg, euler_line_paths, "euler"
    )
    euler_dive_metrics = _rollout_dive_gate_metrics(
        cfg, euler_line_paths, "euler"
    )
    wandb.log(
        {
            "trajectory_euler_lines": wandb.Image(euler_fig),
            **euler_box_metrics,
            **euler_gate_metrics,
            **euler_dive_metrics,
        }
    )

    if getattr(cfg.problem, "target", None) == "maizels_pca50":
        try:
            _log_maizels_trajectory_diagnostics(
                cfg,
                statics,
                train_state,
                params_for_visual,
                paired_plot_x0s,
                plot_x1s,
                plot_labels,
                fontsize=fontsize,
            )
        except Exception as e:
            print(f"Warning: Maizels trajectory diagnostics failed: {e}")
        try:
            _log_maizels_distribution_eval(cfg, train_state, params_for_visual)
        except Exception as e:
            print(f"Warning: Maizels distribution eval failed: {e}")

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
                panel_xlim, panel_ylim = lowd_limits_for(cfg, x0batch)
            elif jj == 1:
                ax.scatter(x1batch[:, 0], x1batch[:, 1], s=0.1, alpha=0.5, marker="o")
                panel_xlim, panel_ylim = lowd_limits_for(cfg, x1batch)
            elif jj == 2:
                ax.scatter(xtbatch[:, 0], xtbatch[:, 1], s=0.1, alpha=0.5, marker="o")
                panel_xlim, panel_ylim = lowd_limits_for(cfg, xtbatch)
            elif jj == 3:
                ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")
            if jj < 3:
                _draw_lowd_regions(ax, cfg)
                ax.set_xlim(panel_xlim)
                ax.set_ylim(panel_ylim)
        else:
            ax.scatter(sbatch, tbatch, s=0.1, alpha=0.5, marker="o")

    wandb.log({"loss_fn_args": wandb.Image(fig)})

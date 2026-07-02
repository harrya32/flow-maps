"""Evaluate box-avoiding Bezier checkpoints against analytical marginals."""

import argparse
import csv
import functools
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import common.flow_map as flow_map
import common.interpolant as interpolant
import common.state_utils as state_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np


METADATA_COLUMNS = [
    "row_type",
    "mode",
    "training_seed",
    "seed_count",
    "run_name",
    "checkpoint",
    "cfg_path",
    "slurm_id",
    "ema_fac",
    "n_eval",
    "eval_seed",
]


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def time_tag(tau: float) -> str:
    return f"{tau:.2f}".replace(".", "_")


def box_bounds_from_cfg(cfg) -> Tuple[float, float, float, float]:
    box_cfg = getattr(cfg.logging, "forbidden_box", None)
    if box_cfg is not None and getattr(box_cfg, "enabled", True):
        xlim = getattr(box_cfg, "xlim", None)
        ylim = getattr(box_cfg, "ylim", None)
        if xlim is not None and ylim is not None:
            return tuple(float(v) for v in (xlim[0], xlim[1], ylim[0], ylim[1]))

    xlim = getattr(cfg.problem, "infeasible_box_xlim", [-1.5, 1.5])
    ylim = getattr(cfg.problem, "infeasible_box_ylim", [-1.0, 1.0])
    return tuple(float(v) for v in (xlim[0], xlim[1], ylim[0], ylim[1]))


def points_in_box(points: np.ndarray, bounds: Tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = bounds
    return (
        (points[..., 0] >= xmin)
        & (points[..., 0] <= xmax)
        & (points[..., 1] >= ymin)
        & (points[..., 1] <= ymax)
    )


def segments_intersect_box(
    starts: np.ndarray, ends: np.ndarray, bounds: Tuple[float, float, float, float]
) -> np.ndarray:
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


def path_violation_metrics(
    paths: np.ndarray, bounds: Tuple[float, float, float, float]
) -> Dict[str, float]:
    paths = np.asarray(paths, dtype=np.float32)
    inside = points_in_box(paths, bounds)
    node_rate = float(np.mean(np.any(inside, axis=1)))
    point_pct = 100.0 * float(np.mean(inside))

    if paths.shape[1] < 2:
        trajectory_rate = node_rate
    else:
        starts = paths[:, :-1, :].reshape((-1, paths.shape[-1]))
        ends = paths[:, 1:, :].reshape((-1, paths.shape[-1]))
        intersects = segments_intersect_box(starts, ends, bounds)
        intersects = intersects.reshape((paths.shape[0], paths.shape[1] - 1))
        trajectory_rate = float(np.mean(np.any(intersects, axis=1)))

    return {
        "point_pct": point_pct,
        "node_trajectory_rate": node_rate,
        "trajectory_rate": trajectory_rate,
    }


def sliced_wasserstein_2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_projections: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo sliced W2 distance between two empirical 2D distributions."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = min(x.shape[0], y.shape[0])
    if n <= 0:
        return float("nan")
    if x.shape[0] != n:
        x = x[rng.choice(x.shape[0], size=n, replace=False)]
    if y.shape[0] != n:
        y = y[rng.choice(y.shape[0], size=n, replace=False)]

    directions = rng.normal(size=(n_projections, x.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    x_proj = np.sort(x @ directions.T, axis=0)
    y_proj = np.sort(y @ directions.T, axis=0)
    return float(np.sqrt(np.mean((x_proj - y_proj) ** 2)))


def sample_box_avoiding_bezier_pairs(
    n_samples: int,
    key: jnp.ndarray,
    *,
    std: float,
    height: float,
    reject_box: bool,
    box_xlim,
    box_ylim,
    reject_times,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """Local copy of the synthetic Bezier sampler, avoiding TF dataset imports."""
    source_mean = jnp.asarray([-3.0, 0.0], dtype=jnp.float32)
    target_mean = jnp.asarray([3.0, 0.0], dtype=jnp.float32)

    def draw_candidates(draw_key, n_draw):
        sign_key, x0_key, x1_key = jax.random.split(draw_key, num=3)
        signs = 2.0 * jax.random.bernoulli(sign_key, p=0.5, shape=(n_draw,)) - 1.0
        x0s = source_mean + std * jax.random.normal(x0_key, shape=(n_draw, 2))
        x1s = target_mean + std * jax.random.normal(x1_key, shape=(n_draw, 2))
        return x0s, x1s, signs

    if not reject_box:
        x0s, x1s, signs = draw_candidates(key, n_samples)
        return {
            "x0": np.asarray(x0s, dtype=np.float32),
            "x1": np.asarray(x1s, dtype=np.float32),
            "label": np.asarray(signs, dtype=np.float32),
        }

    if reject_times is None:
        reject_times = np.linspace(0.0, 1.0, 81, dtype=np.float32)
    reject_times = jnp.asarray(reject_times, dtype=jnp.float32)
    box_xlim = jnp.asarray(box_xlim, dtype=jnp.float32)
    box_ylim = jnp.asarray(box_ylim, dtype=jnp.float32)

    def keep_mask(x0s, x1s, signs):
        controls = 0.5 * (x0s + x1s)
        controls = controls.at[:, 1].add(float(height) * signs)
        t = reject_times[:, None, None]
        paths = (
            ((1.0 - t) ** 2) * x0s[None, :, :]
            + 2.0 * t * (1.0 - t) * controls[None, :, :]
            + (t**2) * x1s[None, :, :]
        )
        inside = (
            (paths[..., 0] >= box_xlim[0])
            & (paths[..., 0] <= box_xlim[1])
            & (paths[..., 1] >= box_ylim[0])
            & (paths[..., 1] <= box_ylim[1])
        )
        return ~jnp.any(inside, axis=0)

    x0_chunks = []
    x1_chunks = []
    sign_chunks = []
    total = 0
    draw_key = key
    chunk_size = max(1, int(chunk_size))
    while total < n_samples:
        remaining = n_samples - total
        n_draw = min(max(2 * remaining, 4096), chunk_size)
        draw_key, subkey = jax.random.split(draw_key)
        x0cand, x1cand, signcand = draw_candidates(subkey, n_draw)
        keep = np.asarray(keep_mask(x0cand, x1cand, signcand))
        if not np.any(keep):
            continue

        x0_keep = np.asarray(x0cand, dtype=np.float32)[keep]
        x1_keep = np.asarray(x1cand, dtype=np.float32)[keep]
        sign_keep = np.asarray(signcand, dtype=np.float32)[keep]
        take = min(remaining, x0_keep.shape[0])
        x0_chunks.append(x0_keep[:take])
        x1_chunks.append(x1_keep[:take])
        sign_chunks.append(sign_keep[:take])
        total += take

    return {
        "x0": np.concatenate(x0_chunks, axis=0).astype(np.float32),
        "x1": np.concatenate(x1_chunks, axis=0).astype(np.float32),
        "label": np.concatenate(sign_chunks, axis=0).astype(np.float32),
    }


def sample_eval_pairs(cfg, n_eval: int, seed: int) -> Dict[str, np.ndarray]:
    key = jax.random.PRNGKey(seed)
    return sample_box_avoiding_bezier_pairs(
        n_eval,
        key,
        std=float(getattr(cfg.problem, "box_avoiding_std", 0.25)),
        height=float(getattr(cfg.problem, "bezier_height", 4.0)),
        reject_box=bool(getattr(cfg.problem, "reject_infeasible", False)),
        box_xlim=getattr(cfg.problem, "infeasible_box_xlim", [-1.5, 1.5]),
        box_ylim=getattr(cfg.problem, "infeasible_box_ylim", [-1.0, 1.0]),
        reject_times=getattr(cfg.problem, "reject_times", None),
        chunk_size=int(getattr(cfg.problem, "rejection_chunk_size", 65_536)),
    )


def rescale_cache_path(cfg, cache_dir: Optional[str]) -> Optional[Path]:
    if cache_dir in (None, ""):
        return None

    std = float(getattr(cfg.problem, "box_avoiding_std", 0.25))
    height = float(getattr(cfg.problem, "bezier_height", 4.0))
    reject = int(bool(getattr(cfg.problem, "reject_infeasible", False)))
    stem = (
        f"box_bezier_rescale_seed{cfg.training.seed}_n{int(cfg.problem.n)}"
        f"_std{std:g}_height{height:g}_reject{reject}"
    )
    return Path(cache_dir) / f"{stem.replace('.', 'p')}.txt"


def set_adaptive_box_rescale(cfg, cache_dir: Optional[str] = None) -> None:
    if getattr(cfg.problem, "gaussian_scale", None) != "adaptive":
        cfg.network.rescale = 1.0
        return

    cache_path = rescale_cache_path(cfg, cache_dir)
    if cache_path is not None and cache_path.exists():
        try:
            cfg.network.rescale = float(cache_path.read_text().strip())
            return
        except ValueError:
            pass

    data_key, _ = jax.random.split(jax.random.PRNGKey(cfg.training.seed))
    paired = sample_box_avoiding_bezier_pairs(
        int(cfg.problem.n),
        data_key,
        std=float(getattr(cfg.problem, "box_avoiding_std", 0.25)),
        height=float(getattr(cfg.problem, "bezier_height", 4.0)),
        reject_box=bool(getattr(cfg.problem, "reject_infeasible", False)),
        box_xlim=getattr(cfg.problem, "infeasible_box_xlim", [-1.5, 1.5]),
        box_ylim=getattr(cfg.problem, "infeasible_box_ylim", [-1.0, 1.0]),
        reject_times=getattr(cfg.problem, "reject_times", None),
        chunk_size=int(getattr(cfg.problem, "rejection_chunk_size", 65_536)),
    )
    cfg.network.rescale = float(
        np.std(np.concatenate([paired["x0"], paired["x1"]], axis=0))
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(f"{cfg.network.rescale:.12g}\n")


def analytical_paths(
    interp: interpolant.Interpolant,
    times: Sequence[float],
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    labels: jnp.ndarray,
) -> np.ndarray:
    paths = []
    for tau in times:
        tau_batch = jnp.full((x0.shape[0],), float(tau), dtype=x0.dtype)
        paths.append(np.asarray(interp.batch_calc_It(tau_batch, x0, x1, labels)))
    return np.stack(paths, axis=1).astype(np.float32)


def select_params(train_state, ema_fac: Optional[float]):
    if ema_fac is None:
        return train_state.params
    if ema_fac in train_state.ema_params:
        return train_state.ema_params[ema_fac]
    if len(train_state.ema_params) == 0:
        return train_state.params
    closest = min(train_state.ema_params.keys(), key=lambda fac: abs(fac - ema_fac))
    print(f"EMA factor {ema_fac} not found; using closest available factor {closest}.")
    return train_state.ema_params[closest]


def load_eval_objects(cfg, checkpoint: str, prng_key, rescale_cache_dir: Optional[str]):
    cfg.training.ndevices = jax.device_count()
    set_adaptive_box_rescale(cfg, cache_dir=rescale_cache_dir)
    interp = interpolant.setup_interpolant(cfg)
    ex_input = jnp.zeros((cfg.problem.d,), dtype=jnp.float32)
    net, params, prng_key = flow_map.initialize_flow_map(cfg.network, ex_input, prng_key)
    tx, _ = state_utils.setup_optimizer(cfg)
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={fac: params for fac in cfg.training.ema_facs},
        tx=tx,
    )
    with open(checkpoint, "rb") as f:
        train_state = flax.serialization.from_bytes(train_state, f.read())
    return cfg, interp, net, train_state, prng_key


def make_model_path_fns(net, params):
    @jax.jit
    def direct_batch(x0: jnp.ndarray, labels: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda x, lbl: net.apply(
                params,
                0.0,
                tau,
                x,
                label=lbl,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0, labels)

    @functools.partial(jax.jit, static_argnums=(2,))
    def euler_paths(x0: jnp.ndarray, labels: jnp.ndarray, n_steps: int) -> jnp.ndarray:
        ts = jnp.linspace(0.0, 1.0, n_steps + 1)

        def step(x, idx):
            t0 = ts[idx]
            dt = ts[idx + 1] - ts[idx]
            b = jax.vmap(
                lambda xi, lbl: net.apply(
                    params,
                    t0,
                    xi,
                    label=lbl,
                    train=False,
                    calc_weight=False,
                    method="calc_b",
                )
            )(x, labels)
            x_next = x + dt * b
            return x_next, x_next

        _, states = jax.lax.scan(step, x0, jnp.arange(n_steps))
        return jnp.swapaxes(jnp.concatenate([x0[None, ...], states], axis=0), 0, 1)

    @functools.partial(jax.jit, static_argnums=(2,))
    def flowmap_rollout_paths(
        x0: jnp.ndarray, labels: jnp.ndarray, n_steps: int
    ) -> jnp.ndarray:
        ts = jnp.linspace(0.0, 1.0, n_steps + 1)

        def step(x, idx):
            x_next = jax.vmap(
                lambda xi, lbl: net.apply(
                    params,
                    ts[idx],
                    ts[idx + 1],
                    xi,
                    label=lbl,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(x, labels)
            return x_next, x_next

        _, states = jax.lax.scan(step, x0, jnp.arange(n_steps))
        return jnp.swapaxes(jnp.concatenate([x0[None, ...], states], axis=0), 0, 1)

    def direct_paths(x0: jnp.ndarray, labels: jnp.ndarray, times: Sequence[float]) -> np.ndarray:
        paths = []
        for tau in times:
            tau_arr = jnp.asarray(float(tau), dtype=x0.dtype)
            paths.append(np.asarray(direct_batch(x0, labels, tau_arr), dtype=np.float32))
        return np.stack(paths, axis=1)

    return direct_paths, euler_paths, flowmap_rollout_paths


def eval_checkpoint(args) -> Tuple[Dict[str, str], Dict[str, float]]:
    if args.training_seed is not None:
        os.environ["BOX_BEZIER_SEED"] = str(args.training_seed)

    cfg_module = importlib.import_module(args.cfg_path)
    cfg = cfg_module.get_config(args.slurm_id, "", "")
    prng_key = jax.random.PRNGKey(args.eval_seed)
    cfg, interp, net, train_state, prng_key = load_eval_objects(
        cfg, args.checkpoint, prng_key, args.rescale_cache_dir
    )

    eval_params = select_params(train_state, args.ema_fac)
    mode = args.mode_name or getattr(cfg.logging, "comparison_mode", cfg.logging.wandb_name)
    run_name = getattr(cfg.logging, "wandb_name", f"{mode}-{args.training_seed}")
    bounds = box_bounds_from_cfg(cfg)
    wasserstein_rng = np.random.default_rng(args.eval_seed + 1009)

    pairs = sample_eval_pairs(cfg, args.n_eval, args.eval_seed + 17)
    x0 = jnp.asarray(pairs["x0"], dtype=jnp.float32)
    x1 = jnp.asarray(pairs["x1"], dtype=jnp.float32)
    labels = jnp.asarray(pairs["label"], dtype=jnp.float32)

    marginal_times = parse_float_list(args.marginal_times)
    path_times = np.linspace(0.0, 1.0, args.path_points, dtype=np.float32)
    flowmap_steps = parse_int_list(args.flowmap_steps)

    direct_paths, euler_paths_fn, flowmap_rollout_paths_fn = make_model_path_fns(
        net, eval_params
    )

    analytic_path = analytical_paths(interp, path_times, x0, x1, labels)
    direct_path = direct_paths(x0, labels, path_times)
    euler_path = np.asarray(euler_paths_fn(x0, labels, args.euler_steps), dtype=np.float32)
    flowmap_paths = {
        n_steps: np.asarray(flowmap_rollout_paths_fn(x0, labels, n_steps), dtype=np.float32)
        for n_steps in flowmap_steps
    }

    metrics: Dict[str, float] = {}
    analytic_endpoint = np.asarray(x1, dtype=np.float32)
    direct_endpoint = direct_path[:, -1, :]
    euler_endpoint = euler_path[:, -1, :]

    metrics["endpoint_sliced_w2_direct"] = sliced_wasserstein_2(
        direct_endpoint,
        analytic_endpoint,
        n_projections=args.wasserstein_projections,
        rng=wasserstein_rng,
    )
    metrics[f"endpoint_sliced_w2_euler_{args.euler_steps}"] = sliced_wasserstein_2(
        euler_endpoint,
        analytic_endpoint,
        n_projections=args.wasserstein_projections,
        rng=wasserstein_rng,
    )
    for n_steps, path in flowmap_paths.items():
        metrics[f"endpoint_sliced_w2_flowmap_{n_steps}"] = sliced_wasserstein_2(
            path[:, -1, :],
            analytic_endpoint,
            n_projections=args.wasserstein_projections,
            rng=wasserstein_rng,
        )

    for tau in marginal_times:
        tag = time_tag(tau)
        analytic_tau = analytical_paths(interp, [tau], x0, x1, labels)[:, 0, :]
        direct_tau = direct_paths(x0, labels, [tau])[:, 0, :]
        euler_idx = int(round(tau * args.euler_steps))
        euler_idx = min(max(euler_idx, 0), euler_path.shape[1] - 1)
        euler_tau = euler_path[:, euler_idx, :]
        metrics[f"marginal_sliced_w2_direct_t{tag}"] = sliced_wasserstein_2(
            direct_tau,
            analytic_tau,
            n_projections=args.wasserstein_projections,
            rng=wasserstein_rng,
        )
        metrics[f"marginal_sliced_w2_euler_{args.euler_steps}_t{tag}"] = (
            sliced_wasserstein_2(
                euler_tau,
                analytic_tau,
                n_projections=args.wasserstein_projections,
                rng=wasserstein_rng,
            )
        )

    for prefix, path in [
        ("analytic", analytic_path),
        ("direct", direct_path),
        (f"euler_{args.euler_steps}", euler_path),
    ]:
        violation = path_violation_metrics(path, bounds)
        metrics[f"{prefix}_point_violation_pct"] = violation["point_pct"]
        metrics[f"{prefix}_node_trajectory_violation_rate"] = violation[
            "node_trajectory_rate"
        ]
        metrics[f"{prefix}_trajectory_violation_rate"] = violation["trajectory_rate"]

    for n_steps, path in flowmap_paths.items():
        violation = path_violation_metrics(path, bounds)
        prefix = f"flowmap_{n_steps}"
        metrics[f"{prefix}_point_violation_pct"] = violation["point_pct"]
        metrics[f"{prefix}_node_trajectory_violation_rate"] = violation[
            "node_trajectory_rate"
        ]
        metrics[f"{prefix}_trajectory_violation_rate"] = violation["trajectory_rate"]

    metadata = {
        "row_type": "per_seed",
        "mode": mode,
        "training_seed": "" if args.training_seed is None else str(args.training_seed),
        "seed_count": "",
        "run_name": run_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "cfg_path": args.cfg_path,
        "slurm_id": str(args.slurm_id),
        "ema_fac": "" if args.ema_fac is None else str(args.ema_fac),
        "n_eval": str(args.n_eval),
        "eval_seed": str(args.eval_seed),
    }
    return metadata, metrics


def read_per_seed_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("row_type") == "per_seed"]


def finite_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def write_metrics_csv(
    csv_path: Path,
    per_seed_rows: List[Dict[str, str]],
    preferred_metric_columns: Iterable[str],
) -> None:
    metric_columns = []
    for column in preferred_metric_columns:
        if column not in metric_columns:
            metric_columns.append(column)
    for row in per_seed_rows:
        for column in row.keys():
            if column not in METADATA_COLUMNS and column not in metric_columns:
                metric_columns.append(column)

    aggregate_rows: List[Dict[str, str]] = []
    modes = sorted({row.get("mode", "") for row in per_seed_rows})
    for mode in modes:
        mode_rows = [row for row in per_seed_rows if row.get("mode", "") == mode]
        for row_type in ["mean", "std"]:
            aggregate = {column: "" for column in METADATA_COLUMNS + metric_columns}
            aggregate["row_type"] = row_type
            aggregate["mode"] = mode
            aggregate["seed_count"] = str(len(mode_rows))
            for column in metric_columns:
                values = [
                    parsed
                    for parsed in (finite_float(row.get(column, "")) for row in mode_rows)
                    if parsed is not None
                ]
                if len(values) == 0:
                    aggregate[column] = ""
                elif row_type == "mean":
                    aggregate[column] = f"{float(np.mean(values)):.10g}"
                elif len(values) == 1:
                    aggregate[column] = "0"
                else:
                    aggregate[column] = f"{float(np.std(values, ddof=1)):.10g}"
            aggregate_rows.append(aggregate)

    fieldnames = METADATA_COLUMNS + metric_columns
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            per_seed_rows,
            key=lambda item: (item.get("mode", ""), item.get("training_seed", "")),
        ):
            writer.writerow({column: row.get(column, "") for column in fieldnames})
        for row in aggregate_rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def upsert_and_aggregate(
    csv_path: Path,
    metadata: Dict[str, str],
    metrics: Dict[str, float],
) -> None:
    metric_row = {
        **metadata,
        **{key: f"{value:.10g}" for key, value in metrics.items()},
    }
    per_seed_rows = read_per_seed_rows(csv_path)
    per_seed_rows = [
        row
        for row in per_seed_rows
        if not (
            row.get("mode") == metric_row.get("mode")
            and row.get("training_seed") == metric_row.get("training_seed")
        )
    ]
    per_seed_rows.append(metric_row)
    write_metrics_csv(csv_path, per_seed_rows, metrics.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate box-avoiding Bezier checkpoints and update a CSV."
    )
    parser.add_argument("--cfg_path", default="configs.box_avoiding_bezier_comparison")
    parser.add_argument("--slurm_id", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--mode_name", default="")
    parser.add_argument("--training_seed", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=12345)
    parser.add_argument("--ema_fac", type=float, default=0.9999)
    parser.add_argument("--n_eval", type=int, default=4096)
    parser.add_argument("--wasserstein_projections", type=int, default=256)
    parser.add_argument("--marginal_times", default="0.25,0.5,0.75")
    parser.add_argument("--path_points", type=int, default=401)
    parser.add_argument("--euler_steps", type=int, default=200)
    parser.add_argument("--flowmap_steps", default="1,2,5,10,25")
    parser.add_argument("--rescale_cache_dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, metrics = eval_checkpoint(args)
    out_csv = Path(args.out_csv)
    upsert_and_aggregate(out_csv, metadata, metrics)

    print(f"Updated metrics CSV: {out_csv}")
    print(f"mode={metadata['mode']} seed={metadata['training_seed']}")
    print(
        "endpoint_sliced_w2_direct="
        f"{metrics['endpoint_sliced_w2_direct']:.6f}, "
        f"direct_trajectory_violation_rate="
        f"{metrics['direct_trajectory_violation_rate']:.6f}"
    )


if __name__ == "__main__":
    main()

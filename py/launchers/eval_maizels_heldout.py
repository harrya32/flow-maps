"""Evaluate lineage-aware cell trajectories on held-out intermediate days."""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import common.flow_map as flow_map
import common.cite_multi as cite_multi
import common.maizels as maizels
import common.state_utils as state_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np


def choose_heldout_timepoints(
    timepoints: np.ndarray,
    time_values: np.ndarray,
    source_time: str,
    target_time: str,
    max_times: int,
) -> List[str]:
    source_value = maizels.parse_timepoint(source_time)
    target_value = maizels.parse_timepoint(target_time)
    unique = sorted(
        {
            str(tp)
            for tp, value in zip(timepoints, time_values)
            if source_value < float(value) < target_value
        },
        key=maizels.parse_timepoint,
    )
    if max_times <= 0 or len(unique) <= max_times:
        return unique
    idx = np.linspace(0, len(unique) - 1, num=max_times, dtype=int)
    return [unique[ii] for ii in np.unique(idx)]


def random_subset(
    x: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    replace_if_needed: bool,
) -> np.ndarray:
    if x.shape[0] == 0:
        raise ValueError("Cannot sample from an empty array.")
    replace = replace_if_needed and (n > x.shape[0])
    idx = rng.choice(x.shape[0], size=n, replace=replace)
    return x[idx]


def covariance_fro_error(x_pred: np.ndarray, x_true: np.ndarray) -> float:
    cov_pred = np.cov(x_pred, rowvar=False)
    cov_true = np.cov(x_true, rowvar=False)
    return float(np.linalg.norm(cov_pred - cov_true, ord="fro"))


def plot_heldout_comparison(results: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_xy = np.concatenate(
        [
            np.concatenate([row["actual"][:, :2], row["pred"][:, :2]], axis=0)
            for row in results
        ],
        axis=0,
    )
    x_min, y_min = np.percentile(all_xy, 1, axis=0)
    x_max, y_max = np.percentile(all_xy, 99, axis=0)

    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(11, 4 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    for rr, row in enumerate(results):
        for ax, key, color, title in [
            (axes[rr, 0], "actual", "#1f77b4", f"Actual {row['timepoint']}"),
            (axes[rr, 1], "pred", "#111111", f"Predicted {row['timepoint']}"),
        ]:
            vals = row[key]
            ax.scatter(vals[:, 0], vals[:, 1], s=4, alpha=0.45, c=color, linewidths=0)
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.15)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfg_path",
        required=True,
        help="Config module, e.g. configs.maizels_pca50 or configs.cite_multi_pca100",
    )
    parser.add_argument("--slurm_id", type=int, default=0)
    parser.add_argument("--dataset_location", default="")
    parser.add_argument("--dataset_name", choices=("cite", "multi"), default=None)
    parser.add_argument("--heldout_day", choices=("3", "4"), default=None)
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema_fac", type=float, default=0.9999)
    parser.add_argument("--heldout_max_times", type=int, default=0)
    parser.add_argument("--points_per_time", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    cfg_module = importlib.import_module(args.cfg_path)
    get_config = cfg_module.get_config
    supported = inspect.signature(get_config).parameters
    optional = {
        "dataset_name": args.dataset_name,
        "heldout_day": args.heldout_day,
        "classifier_path": args.classifier_path,
    }
    kwargs = {
        name: value
        for name, value in optional.items()
        if value is not None and name in supported
    }
    cfg = get_config(args.slurm_id, args.dataset_location, args.out_dir, **kwargs)
    cfg.training.ndevices = jax.device_count()

    backend = (
        cite_multi
        if getattr(cfg.problem, "target", None) == "cite_multi_pca100"
        else maizels
    )
    data = backend.all_timepoint_data(cfg.problem.dataset_location)
    source_time = getattr(cfg.problem, "source_time", "D3")
    target_time = getattr(cfg.problem, "target_time", "D8")
    source_value = maizels.parse_timepoint(source_time)
    target_value = maizels.parse_timepoint(target_time)

    source_mask = data["timepoints"] == source_time
    target_mask = data["timepoints"] == target_time
    x0_all = data["x"][source_mask].astype(np.float32)
    x1_all = data["x"][target_mask].astype(np.float32)
    if x0_all.shape[0] == 0:
        raise RuntimeError(f"No source cells found for {source_time}.")
    if x1_all.shape[0] == 0:
        raise RuntimeError(f"No target cells found for {target_time}.")
    if getattr(cfg.problem, "gaussian_scale", None) == "adaptive":
        cfg.network.rescale = float(np.std(np.concatenate([x0_all, x1_all], axis=0)))

    configured_timepoints = getattr(
        getattr(cfg.logging, "maizels", None),
        "distribution_eval_timepoints",
        None,
    )
    if configured_timepoints is not None:
        heldout_timepoints = [str(value) for value in configured_timepoints]
    else:
        heldout_timepoints = choose_heldout_timepoints(
            data["timepoints"],
            data["time_values"],
            source_time,
            target_time,
            args.heldout_max_times,
        )
    if not heldout_timepoints:
        raise RuntimeError("No held-out timepoints found.")

    prng_key = jax.random.PRNGKey(args.seed)
    ex_input = jnp.asarray(x0_all[0], dtype=jnp.float32)
    net, params, _ = flow_map.initialize_flow_map(cfg.network, ex_input, prng_key)

    tx, _ = state_utils.setup_optimizer(cfg)
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={fac: params for fac in cfg.training.ema_facs},
        tx=tx,
    )
    with open(args.checkpoint, "rb") as f:
        train_state = flax.serialization.from_bytes(train_state, f.read())

    eval_params = train_state.ema_params.get(args.ema_fac, train_state.params)

    @jax.jit
    def pushforward_batch(x0_batch: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda x: net.apply(
                eval_params,
                0.0,
                tau,
                x,
                label=None,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            )
        )(x0_batch)

    rng = np.random.default_rng(args.seed + 123)
    results = []
    metrics = []
    for timepoint in heldout_timepoints:
        time_value = maizels.parse_timepoint(timepoint)
        actual_all = data["x"][data["timepoints"] == timepoint].astype(np.float32)
        n_compare = min(args.points_per_time, actual_all.shape[0])
        actual = random_subset(actual_all, n_compare, rng, replace_if_needed=False)
        x0_for_gen = random_subset(x0_all, n_compare, rng, replace_if_needed=True)
        if hasattr(backend, "normalized_time"):
            tau = float(backend.normalized_time(timepoint))
        else:
            tau = float(
                np.clip(
                    (time_value - source_value) / (target_value - source_value),
                    0.0,
                    1.0,
                )
            )
        pred = np.asarray(
            pushforward_batch(
                jnp.asarray(x0_for_gen, dtype=jnp.float32),
                jnp.asarray(tau, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )
        mean_mse = float(np.mean((pred.mean(axis=0) - actual.mean(axis=0)) ** 2))
        cov_err = covariance_fro_error(pred, actual)
        row = {
            "timepoint": timepoint,
            "day": time_value,
            "tau": tau,
            "n": n_compare,
            "mean_mse": mean_mse,
            "cov_fro_error": cov_err,
        }
        metrics.append(row)
        results.append({**row, "actual": actual, "pred": pred})
        print(
            f"{timepoint:>4s} tau={tau:.3f} n={n_compare:>4d} "
            f"mean_mse={mean_mse:.6g} cov_fro_error={cov_err:.6g}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = (
        str(cfg.problem.dataset_name)
        if getattr(cfg.problem, "target", None) == "cite_multi_pca100"
        else "maizels"
    )
    csv_path = out_dir / f"{output_prefix}_heldout_metrics.csv"
    with csv_path.open("w") as f:
        f.write("timepoint,day,tau,n,mean_mse,cov_fro_error\n")
        for row in metrics:
            f.write(
                f"{row['timepoint']},{row['day']},{row['tau']},{row['n']},"
                f"{row['mean_mse']},{row['cov_fro_error']}\n"
            )
    np.savez(
        out_dir / f"{output_prefix}_heldout_metrics.npz",
        timepoint=np.asarray([row["timepoint"] for row in metrics], dtype=object),
        day=np.asarray([row["day"] for row in metrics], dtype=np.float32),
        tau=np.asarray([row["tau"] for row in metrics], dtype=np.float32),
        n=np.asarray([row["n"] for row in metrics], dtype=np.int32),
        mean_mse=np.asarray([row["mean_mse"] for row in metrics], dtype=np.float32),
        cov_fro_error=np.asarray([row["cov_fro_error"] for row in metrics], dtype=np.float32),
    )
    plot_heldout_comparison(
        results,
        out_dir / f"{output_prefix}_heldout_vs_estimated.png",
    )
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

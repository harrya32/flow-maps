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
import common.wasserstein as wasserstein
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
            (
                axes[rr, 1],
                "pred",
                "#111111",
                f"Predicted {row['source_timepoint']}→{row['timepoint']}",
            ),
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
    parser.add_argument(
        "--maizels_ot_coupling",
        choices=("global_ot", "minibatch_ot"),
        default=None,
    )
    parser.add_argument(
        "--maizels_schedule",
        choices=("d3_d8", "d3_d3p8_d8"),
        default=None,
    )
    parser.add_argument(
        "--maizels_time_mode",
        choices=("real_time", "equal_time"),
        default=None,
    )
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
        "maizels_ot_coupling": args.maizels_ot_coupling,
        "maizels_schedule": args.maizels_schedule,
        "maizels_time_mode": args.maizels_time_mode,
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
        retained = [
            str(value)
            for value in getattr(
                cfg.problem,
                "retained_timepoints",
                [source_time, target_time],
            )
        ]
        scale_populations = []
        for left, right in zip(retained[:-1], retained[1:]):
            scale_populations.extend(
                [
                    data["x"][data["timepoints"] == left],
                    data["x"][data["timepoints"] == right],
                ]
            )
        cfg.network.rescale = float(np.std(np.concatenate(scale_populations, axis=0)))

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
    if (
        args.heldout_max_times > 0
        and len(heldout_timepoints) > args.heldout_max_times
    ):
        selected = np.linspace(
            0,
            len(heldout_timepoints) - 1,
            num=args.heldout_max_times,
            dtype=int,
        )
        heldout_timepoints = [heldout_timepoints[ii] for ii in np.unique(selected)]
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
    def pushforward_batch(
        x0_batch: jnp.ndarray,
        start_tau: jnp.ndarray,
        end_tau: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda x: net.apply(
                eval_params,
                start_tau,
                end_tau,
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
    interval_local = bool(
        getattr(
            getattr(cfg.logging, "maizels", None),
            "distribution_eval_interval_local",
            False,
        )
    )
    for timepoint in heldout_timepoints:
        time_value = maizels.parse_timepoint(timepoint)
        actual_all = data["x"][data["timepoints"] == timepoint].astype(np.float32)
        n_compare = min(args.points_per_time, actual_all.shape[0])
        actual = random_subset(actual_all, n_compare, rng, replace_if_needed=False)
        if interval_local and backend is maizels:
            interval_source, _ = maizels.retained_interval_for_timepoint(
                cfg, timepoint
            )
            source_population = data["x"][
                data["timepoints"] == interval_source
            ].astype(np.float32)
            start_tau = maizels.normalized_time(interval_source, cfg)
            tau = maizels.normalized_time(timepoint, cfg)
        else:
            interval_source = source_time
            source_population = x0_all
            start_tau = 0.0
            if backend is maizels:
                tau = maizels.normalized_time(timepoint, cfg)
            elif hasattr(backend, "normalized_time"):
                tau = float(backend.normalized_time(timepoint))
            else:
                tau = float(
                    np.clip(
                        (time_value - source_value) / (target_value - source_value),
                        0.0,
                        1.0,
                    )
                )
        x0_for_gen = random_subset(
            source_population, n_compare, rng, replace_if_needed=True
        )
        pred = np.asarray(
            pushforward_batch(
                jnp.asarray(x0_for_gen, dtype=jnp.float32),
                jnp.asarray(start_tau, dtype=jnp.float32),
                jnp.asarray(tau, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )
        emd = wasserstein.exact_emd(pred, actual)
        mean_mse = float(np.mean((pred.mean(axis=0) - actual.mean(axis=0)) ** 2))
        cov_err = covariance_fro_error(pred, actual)
        row = {
            "timepoint": timepoint,
            "source_timepoint": interval_source,
            "day": time_value,
            "start_tau": start_tau,
            "tau": tau,
            "n": n_compare,
            "emd": emd,
            "mean_mse": mean_mse,
            "cov_fro_error": cov_err,
        }
        metrics.append(row)
        results.append({**row, "actual": actual, "pred": pred})
        print(
            f"{interval_source:>4s}->{timepoint:<4s} "
            f"tau={start_tau:.3f}->{tau:.3f} n={n_compare:>4d} "
            f"emd={emd:.6g} mean_mse={mean_mse:.6g} "
            f"cov_fro_error={cov_err:.6g}"
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
        f.write(
            "timepoint,source_timepoint,day,start_tau,tau,n,emd,"
            "mean_mse,cov_fro_error\n"
        )
        for row in metrics:
            f.write(
                f"{row['timepoint']},{row['source_timepoint']},{row['day']},"
                f"{row['start_tau']},{row['tau']},{row['n']},"
                f"{row['emd']},{row['mean_mse']},{row['cov_fro_error']}\n"
            )
    np.savez(
        out_dir / f"{output_prefix}_heldout_metrics.npz",
        timepoint=np.asarray([row["timepoint"] for row in metrics], dtype=object),
        source_timepoint=np.asarray(
            [row["source_timepoint"] for row in metrics], dtype=object
        ),
        day=np.asarray([row["day"] for row in metrics], dtype=np.float32),
        start_tau=np.asarray(
            [row["start_tau"] for row in metrics], dtype=np.float32
        ),
        tau=np.asarray([row["tau"] for row in metrics], dtype=np.float32),
        n=np.asarray([row["n"] for row in metrics], dtype=np.int32),
        emd=np.asarray([row["emd"] for row in metrics], dtype=np.float32),
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

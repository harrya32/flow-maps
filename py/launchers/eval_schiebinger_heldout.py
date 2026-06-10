import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import click
import common.datasets as datasets
import common.flow_map as flow_map
import common.state_utils as state_utils
import flax
import jax
import jax.numpy as jnp
import numpy as np


def choose_heldout_times(unique_times: np.ndarray, max_times: int) -> np.ndarray:
    middle = unique_times[1:-1]
    if middle.size == 0:
        return np.array([], dtype=np.float32)
    if max_times <= 0:
        return middle
    count = min(max_times, middle.size)
    idx = np.linspace(0, middle.size - 1, num=count, dtype=int)
    return np.unique(middle[idx]).astype(np.float32)


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


def plot_heldout_comparison(results: List[Dict], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting") from exc

    if len(results) == 0:
        raise ValueError("No held-out results to plot.")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_xy = np.concatenate(
        [np.concatenate([row["actual"][:, :2], row["pred"][:, :2]], axis=0) for row in results],
        axis=0,
    )
    x_min, y_min = np.percentile(all_xy, 1, axis=0)
    x_max, y_max = np.percentile(all_xy, 99, axis=0)

    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 4 * n_rows), squeeze=False)

    for rr, row in enumerate(results):
        day = row["day"]
        tau = row["tau"]
        actual = row["actual"]
        pred = row["pred"]

        ax_actual = axes[rr, 0]
        ax_pred = axes[rr, 1]

        ax_actual.scatter(actual[:, 0], actual[:, 1], s=4, alpha=0.5, c="#1f77b4", linewidths=0)
        ax_pred.scatter(pred[:, 0], pred[:, 1], s=4, alpha=0.5, c="#ff7f0e", linewidths=0)

        ax_actual.set_title(f"Actual samples at day={day:g}")
        ax_pred.set_title(f"Flow-map estimate at day={day:g} (tau={tau:.3f})")

        for ax in [ax_actual, ax_pred]:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


@click.command()
@click.option("--cfg_path", required=True, help="Config module path (e.g. configs.schiebinger_lsd)")
@click.option("--slurm_id", default=0, show_default=True, type=int)
@click.option(
    "--dataset_location",
    required=True,
    help="Dataset directory (or full .h5ad path) passed to config.get_config",
)
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True, file_okay=True),
    help="Path to saved training checkpoint (.pkl)",
)
@click.option(
    "--ema_fac",
    default=0.9999,
    show_default=True,
    type=float,
    help="EMA factor to use for evaluation if present in checkpoint.",
)
@click.option(
    "--heldout_max_times",
    default=0,
    show_default=True,
    type=int,
    help="Max held-out intermediate times to evaluate. <=0 means use all.",
)
@click.option("--points_per_time", default=2000, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--out_dir", required=True, type=click.Path(file_okay=False))
def main(
    cfg_path: str,
    slurm_id: int,
    dataset_location: str,
    checkpoint: str,
    ema_fac: float,
    heldout_max_times: int,
    points_per_time: int,
    seed: int,
    out_dir: str,
):
    cfg_module = importlib.import_module(cfg_path)
    cfg = cfg_module.get_config(slurm_id, dataset_location, out_dir)
    cfg.training.ndevices = jax.device_count()

    split_data = datasets.load_schiebinger_splits(cfg, subsample_endpoints=False)
    embedding = split_data["embedding"]
    times = split_data["times"]
    unique_times = split_data["unique_times"]
    t_start = float(split_data["t_start"])
    t_end = float(split_data["t_end"])
    x0_all = split_data["x0_all"]

    heldout_times = choose_heldout_times(unique_times, max_times=heldout_max_times)
    if heldout_times.size == 0:
        raise RuntimeError("No held-out Schiebinger time points found.")

    prng_key = jax.random.PRNGKey(seed)
    ex_input = jnp.asarray(x0_all[0], dtype=jnp.float32)
    net, params, _ = flow_map.initialize_flow_map(cfg.network, ex_input, prng_key)

    tx, _ = state_utils.setup_optimizer(cfg)
    train_state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={fac: params for fac in cfg.training.ema_facs},
        tx=tx,
    )
    with open(checkpoint, "rb") as f:
        train_state = flax.serialization.from_bytes(train_state, f.read())

    if ema_fac in train_state.ema_params:
        eval_params = train_state.ema_params[ema_fac]
        print(f"Using EMA parameters at factor={ema_fac}.")
    else:
        eval_params = train_state.params
        print(f"EMA factor {ema_fac} not found. Using instantaneous parameters.")

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

    rng = np.random.default_rng(seed + 123)
    results = []
    metrics = []

    for day in heldout_times:
        actual_all = embedding[times == day]
        if actual_all.shape[0] == 0:
            continue

        n_compare = min(points_per_time, actual_all.shape[0])
        actual = random_subset(
            actual_all,
            n_compare,
            rng,
            replace_if_needed=False,
        ).astype(np.float32)
        x0_for_gen = random_subset(
            x0_all,
            n_compare,
            rng,
            replace_if_needed=True,
        ).astype(np.float32)

        tau = float(np.clip((float(day) - t_start) / (t_end - t_start), 0.0, 1.0))
        pred = np.asarray(
            pushforward_batch(
                jnp.asarray(x0_for_gen, dtype=jnp.float32),
                jnp.asarray(tau, dtype=jnp.float32),
            ),
            dtype=np.float32,
        )

        mean_mse = float(np.mean((pred.mean(axis=0) - actual.mean(axis=0)) ** 2))
        cov_err = covariance_fro_error(pred, actual)

        results.append(
            {
                "day": float(day),
                "tau": tau,
                "actual": actual,
                "pred": pred,
            }
        )
        metrics.append(
            {
                "day": float(day),
                "tau": tau,
                "n": int(n_compare),
                "mean_mse": mean_mse,
                "cov_fro_error": cov_err,
            }
        )
        print(
            f"day={day:>6.2f} tau={tau:.3f} n={n_compare:>4d} "
            f"mean_mse={mean_mse:.6f} cov_fro={cov_err:.6f}"
        )

    if len(results) == 0:
        raise RuntimeError("No held-out rows evaluated.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_path / "schiebinger_heldout_metrics.csv"
    with metrics_csv.open("w", newline="") as f:
        f.write("day,tau,n,mean_mse,cov_fro_error\n")
        for row in metrics:
            f.write(
                f"{row['day']},{row['tau']},{row['n']},"
                f"{row['mean_mse']},{row['cov_fro_error']}\n"
            )

    npz_path = out_path / "schiebinger_heldout_metrics.npz"
    np.savez(
        npz_path,
        day=np.asarray([row["day"] for row in metrics], dtype=np.float32),
        tau=np.asarray([row["tau"] for row in metrics], dtype=np.float32),
        n=np.asarray([row["n"] for row in metrics], dtype=np.int32),
        mean_mse=np.asarray([row["mean_mse"] for row in metrics], dtype=np.float32),
        cov_fro_error=np.asarray([row["cov_fro_error"] for row in metrics], dtype=np.float32),
    )

    if embedding.shape[1] >= 2:
        plot_path = out_path / "schiebinger_heldout_vs_estimated.png"
        plot_heldout_comparison(results, str(plot_path))
        print(f"Saved held-out comparison plot: {plot_path}")

    print(f"Saved held-out metrics CSV: {metrics_csv}")
    print(f"Saved held-out metrics NPZ: {npz_path}")


if __name__ == "__main__":
    main()

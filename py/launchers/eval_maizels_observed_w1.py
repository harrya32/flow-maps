"""Evaluate deterministic Maizels checkpoints at every marginal and test time.

The ``interval`` protocol mirrors the held-out distribution evaluation:
each target starts from the real population at its preceding observed time.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import flax
import jax
import jax.numpy as jnp
import numpy as np

import common.flow_map as flow_map
import common.maizels as maizels
import common.maizels_checkpoint_eval as checkpoint_eval
import common.state_utils as state_utils
from common.deterministic_checkpoint import pushforward as _pushforward


def _checkpoint_spec(value: str) -> Tuple[str, int, Path]:
    """Parse METHOD:SEED:PATH while allowing colons inside PATH."""
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "checkpoint must have the form METHOD:SEED:PATH"
        )
    method, seed_text, path_text = parts
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint seed must be an integer") from exc
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return method, seed, path


def _adaptive_rescale(cfg) -> None:
    if getattr(cfg.problem, "gaussian_scale", None) != "adaptive":
        return
    # Reproduce the scalar used during training exactly. Dynamic minibatch OT
    # draws equal-sized batches per adjacent interval, so each interval endpoint
    # has equal weight and an interior retained population is counted twice.
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    means = []
    second_moments = []
    for left, right in zip(
        maizels.retained_timepoints(cfg)[:-1],
        maizels.retained_timepoints(cfg)[1:],
    ):
        for timepoint in (left, right):
            population = np.asarray(pools[timepoint]["train_x"], dtype=np.float64)
            means.append(float(np.mean(population)))
            second_moments.append(float(np.mean(population * population)))
    mean = float(np.mean(means))
    variance = max(float(np.mean(second_moments)) - mean * mean, 0.0)
    cfg.network.rescale = math.sqrt(variance)


def _restore(checkpoint: Path, cfg, example: np.ndarray, ema_fac: float | None):
    key = jax.random.PRNGKey(0)
    net, params, _ = flow_map.initialize_flow_map(
        cfg.network, jnp.asarray(example, dtype=jnp.float32), key
    )
    tx, _ = state_utils.setup_optimizer(cfg)
    state = state_utils.EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params={fac: params for fac in cfg.training.ema_facs},
        tx=tx,
    )
    state = flax.serialization.from_bytes(state, checkpoint.read_bytes())
    if ema_fac is None:
        return net, state.params
    return net, state.ema_params.get(ema_fac, state.params)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=_checkpoint_spec,
        metavar="METHOD:SEED:PATH",
    )
    parser.add_argument("--cfg_path", default="configs.maizels_pca50")
    parser.add_argument("--slurm_id", type=int, default=7)
    parser.add_argument(
        "--dataset_location", default="celltype_classification_pca50_dataset.csv.gz"
    )
    parser.add_argument("--maizels_schedule", default="d3_d3p8_d8")
    parser.add_argument("--maizels_time_mode", default="real_time")
    parser.add_argument(
        "--ema_fac",
        type=float,
        default=None,
        help="Optional EMA decay. By default use instantaneous parameters, matching training evaluation.",
    )
    parser.add_argument(
        "--n_steps_per_interval", type=int, default=50,
        help="Number of equal steps from each observed source to each target marginal.",
    )
    parser.add_argument(
        "--euler", action="store_true",
        help="Evaluate only Euler sampling of the diagonal vector field instead of composed maps.",
    )
    parser.add_argument(
        "--hparam_val_times", default=None,
        help="Comma-separated validation days excluded from test_times; defaults to the training config.",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_dir", default="outputs/maizels_observed_w1")
    args = parser.parse_args()
    if args.n_steps_per_interval <= 0 or args.batch_size <= 0:
        parser.error("n_steps_per_interval and batch_size must be positive")

    cfg_module = importlib.import_module(args.cfg_path)
    supported = inspect.signature(cfg_module.get_config).parameters
    optional = {
        "maizels_schedule": args.maizels_schedule,
        "maizels_time_mode": args.maizels_time_mode,
    }
    kwargs = {key: value for key, value in optional.items() if key in supported}
    cfg = cfg_module.get_config(
        args.slurm_id,
        args.dataset_location,
        args.out_dir,
        **kwargs,
    )
    cfg.training.ndevices = jax.device_count()
    data = maizels.all_timepoint_data(cfg.problem.dataset_location)
    marginals = checkpoint_eval.build_marginals(cfg, data, args.hparam_val_times)
    _adaptive_rescale(cfg)
    sampler = "euler" if args.euler else "flowmap"

    rows: List[Dict] = []
    for method, seed, checkpoint in args.checkpoint:
        print(f"Restoring {method} seed={seed}: {checkpoint}", flush=True)
        net, params = _restore(
            checkpoint, cfg, marginals.populations[marginals.retained[0]][0], args.ema_fac
        )

        def pushforward(source, *, start_time, end_time, target_index):
            return _pushforward(
                net, params, source,
                start_time=start_time,
                end_time=end_time,
                n_steps=args.n_steps_per_interval,
                batch_size=args.batch_size,
                euler=args.euler,
            )

        def progress(day, interval_w1):
            print(
                f"  {day}: interval W1={interval_w1:.6f}",
                flush=True,
            )

        rows.extend(checkpoint_eval.evaluate_marginals(
            marginals, pushforward,
            context={
                "method": method, "seed": seed, "checkpoint": str(checkpoint),
                "sampler": sampler, "n_steps_per_interval": args.n_steps_per_interval,
            },
            progress=progress,
        ))

    summaries = checkpoint_eval.summary_rows(
        rows, ("method", "sampler", "n_steps_per_interval", "protocol", "timepoint"),
        count_field="n_checkpoints",
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval.write_csv(out_dir / "per_checkpoint.csv", rows)
    checkpoint_eval.write_csv(out_dir / "summary.csv", summaries)
    (out_dir / "metrics.json").write_text(
        json.dumps(checkpoint_eval.metric_records(rows), indent=2) + "\n"
    )
    metadata = {
        **marginals.metadata(),
        "dataset_location": str(Path(args.dataset_location).expanduser().resolve()),
        "time_mode": args.maizels_time_mode,
        "parameters": "instantaneous" if args.ema_fac is None else f"EMA {args.ema_fac}",
        "sampler": sampler,
        "n_steps_per_interval": args.n_steps_per_interval,
        "protocols": {
            "interval": "restart from the real population at each observed left endpoint",
        },
        "checkpoints": [
            {"method": method, "seed": seed, "path": str(path)}
            for method, seed, path in args.checkpoint
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nSummary (mean +/- s.e.):")
    for row in summaries:
        print(
            f"  {row['method']:>24s} {row['protocol']:>8s} "
            f"{row['timepoint']:>10s}: {row['w1_mean']:.6f} +/- {row['w1_se']:.6f}"
        )
        if row["timepoint"] == "test_times":
            print(f"    final_eval/{sampler}_mean_emd_test_times={row['w1_mean']:.6f}")
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

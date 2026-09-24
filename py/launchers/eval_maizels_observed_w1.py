"""Evaluate deterministic Maizels flow-map checkpoints at observed marginals.

The primary ``interval`` protocol mirrors the held-out distribution evaluation:
each learned interval starts from the real population at its left endpoint.  The
additional ``rollout`` protocol starts at D3 once and carries the model samples
through every retained interval, without resetting them at D3.8.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import flax
import jax
import jax.numpy as jnp
import numpy as np

import common.flow_map as flow_map
import common.maizels as maizels
import common.state_utils as state_utils
import common.wasserstein as wasserstein


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


def _composed_pushforward(
    net,
    params,
    source: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    n_steps: int,
    batch_size: int,
) -> np.ndarray:
    if end_time == start_time:
        return np.asarray(source, dtype=np.float32).copy()
    times = jnp.linspace(start_time, end_time, n_steps + 1, dtype=jnp.float32)

    @jax.jit
    def run_batch(x):
        def step(current, interval):
            left, right = interval
            result = jax.vmap(
                lambda point: net.apply(
                    params,
                    left,
                    right,
                    point,
                    label=None,
                    train=False,
                    calc_weight=False,
                    return_X_and_phi=False,
                )
            )(current)
            return result, None

        result, _ = jax.lax.scan(step, x, (times[:-1], times[1:]))
        return result

    source = np.asarray(source, dtype=np.float32)
    pieces = []
    for start in range(0, source.shape[0], batch_size):
        result = run_batch(jnp.asarray(source[start : start + batch_size]))
        pieces.append(np.asarray(jax.device_get(result), dtype=np.float32))
    return np.concatenate(pieces, axis=0)


def _summary_rows(rows: Iterable[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["protocol"], row["timepoint"])].append(
            float(row["w1"])
        )
    summaries = []
    for (method, protocol, timepoint), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        summaries.append(
            {
                "method": method,
                "protocol": protocol,
                "timepoint": timepoint,
                "n_checkpoints": int(array.size),
                "w1_mean": float(np.mean(array)),
                "w1_sd": sd,
                "w1_se": sd / math.sqrt(array.size),
            }
        )
    return summaries


def _write_csv(path: Path, rows: List[Dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--n_steps_per_interval", type=int, default=50)
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
    retained = maizels.retained_timepoints(cfg)
    data = maizels.all_timepoint_data(cfg.problem.dataset_location)
    populations = {
        timepoint: np.asarray(
            data["x"][data["timepoints"] == timepoint], dtype=np.float32
        )
        for timepoint in retained
    }
    if any(population.shape[0] == 0 for population in populations.values()):
        raise RuntimeError(f"Empty retained population; sizes={ {k: len(v) for k, v in populations.items()} }")
    _adaptive_rescale(cfg)

    rows: List[Dict] = []
    for method, seed, checkpoint in args.checkpoint:
        print(f"Restoring {method} seed={seed}: {checkpoint}", flush=True)
        net, params = _restore(
            checkpoint, cfg, populations[retained[0]][0], args.ema_fac
        )

        # At the initial observed marginal the model distribution is exactly the
        # empirical source distribution, hence W1 is zero by construction.
        for protocol in ("interval", "rollout"):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "protocol": protocol,
                    "timepoint": retained[0],
                    "source_timepoint": retained[0],
                    "source_n": populations[retained[0]].shape[0],
                    "target_n": populations[retained[0]].shape[0],
                    "w1": 0.0,
                    "checkpoint": str(checkpoint),
                }
            )

        rollout = populations[retained[0]]
        for source_timepoint, target_timepoint in zip(retained[:-1], retained[1:]):
            start_time = maizels.normalized_time(source_timepoint, cfg)
            end_time = maizels.normalized_time(target_timepoint, cfg)
            target = populations[target_timepoint]

            interval_prediction = _composed_pushforward(
                net,
                params,
                populations[source_timepoint],
                start_time=start_time,
                end_time=end_time,
                n_steps=args.n_steps_per_interval,
                batch_size=args.batch_size,
            )
            interval_w1 = wasserstein.exact_emd(interval_prediction, target)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "protocol": "interval",
                    "timepoint": target_timepoint,
                    "source_timepoint": source_timepoint,
                    "source_n": interval_prediction.shape[0],
                    "target_n": target.shape[0],
                    "w1": interval_w1,
                    "checkpoint": str(checkpoint),
                }
            )

            if source_timepoint == retained[0]:
                # Both protocols are identical in the first interval.
                rollout = interval_prediction
                rollout_w1 = interval_w1
            else:
                rollout = _composed_pushforward(
                    net,
                    params,
                    rollout,
                    start_time=start_time,
                    end_time=end_time,
                    n_steps=args.n_steps_per_interval,
                    batch_size=args.batch_size,
                )
                rollout_w1 = wasserstein.exact_emd(rollout, target)
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "protocol": "rollout",
                    "timepoint": target_timepoint,
                    "source_timepoint": retained[0],
                    "source_n": rollout.shape[0],
                    "target_n": target.shape[0],
                    "w1": rollout_w1,
                    "checkpoint": str(checkpoint),
                }
            )
            print(
                f"  {target_timepoint}: interval W1={interval_w1:.6f}; "
                f"D3 rollout W1={rollout_w1:.6f}",
                flush=True,
            )

    summaries = _summary_rows(rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_checkpoint.csv", rows)
    _write_csv(out_dir / "summary.csv", summaries)
    metadata = {
        "dataset_location": str(Path(args.dataset_location).expanduser().resolve()),
        "retained_timepoints": list(retained),
        "time_mode": args.maizels_time_mode,
        "parameters": "instantaneous" if args.ema_fac is None else f"EMA {args.ema_fac}",
        "sampler": "composed deterministic flow map",
        "n_steps_per_interval": args.n_steps_per_interval,
        "population_sizes": {key: int(value.shape[0]) for key, value in populations.items()},
        "protocols": {
            "interval": "restart from the real population at each observed left endpoint",
            "rollout": "start from D3 once and carry predictions across observed intervals",
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
            f"{row['timepoint']:>4s}: {row['w1_mean']:.6f} +/- {row['w1_se']:.6f}"
        )
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()

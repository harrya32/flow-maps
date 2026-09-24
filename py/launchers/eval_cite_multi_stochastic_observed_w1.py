"""Evaluate CITE/Multi SSFM checkpoints at their observed training marginals.

The interval protocol restarts from the real population at each retained
left endpoint, matching the held-out-day distributional evaluation.  The
rollout protocol starts from day 2 once and carries samples through all
retained intervals.  Both protocols use composed stochastic flow maps.
"""

from __future__ import annotations

import argparse
import csv
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

import flax.serialization
import jax
import ml_collections
import numpy as np

from common import cite_multi
from common import maizels_stochastic_eval
from common import stochastic_flow_map
from common import wasserstein


def _checkpoint_spec(value: str) -> Tuple[str, int, Path]:
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


def _load_checkpoint(path: Path):
    payload = flax.serialization.msgpack_restore(path.read_bytes())
    config_path = path.with_name("config.json")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Expected the checkpoint's companion config at {config_path}."
        )
    cfg = ml_collections.ConfigDict(json.loads(config_path.read_text()))
    pc_scale = np.asarray(payload["pc_scale"], dtype=np.float32)
    model = stochastic_flow_map.StrongStochasticFlowMapMLP(
        data_dim=int(cfg.problem.d),
        n_coefficients=int(cfg.ssfm.n_coefficients),
        hidden_dim=int(cfg.ssfm.hidden_dim),
        n_hidden=int(cfg.ssfm.n_hidden),
        rescale=float(np.sqrt(np.mean(np.square(pc_scale)))),
        uncertainty_hidden_dim=int(cfg.ssfm.uncertainty_hidden_dim),
    )
    return model, payload["ema_params"], cfg, payload


def _draw_key(seed: int, draw: int, interval: int, protocol: int):
    key = jax.random.PRNGKey(int(seed))
    key = jax.random.fold_in(key, int(draw))
    key = jax.random.fold_in(key, int(interval))
    return jax.random.fold_in(key, int(protocol))


def _summary_rows(rows: Iterable[Dict], group_fields: Tuple[str, ...]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(float(row["w1"]))
    result = []
    for group, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        result.append(
            {
                **dict(zip(group_fields, group)),
                "n": int(array.size),
                "w1_mean": float(np.mean(array)),
                "w1_sd": sd,
                "w1_se": sd / math.sqrt(array.size),
            }
        )
    return result


def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
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
    parser.add_argument(
        "--dataset_location",
        default="",
        help="Optional override; otherwise use the path embedded in each checkpoint.",
    )
    parser.add_argument("--noise_draws", type=int, default=3)
    parser.add_argument(
        "--n_steps_per_interval",
        type=int,
        default=0,
        help="Zero uses evaluation.flowmap_n_steps from each checkpoint.",
    )
    parser.add_argument("--evaluation_seed", type=int, default=2701)
    parser.add_argument(
        "--out_dir", default="outputs/cite_multi_stochastic_observed_w1"
    )
    args = parser.parse_args()
    if args.noise_draws <= 0:
        parser.error("noise_draws must be positive")
    if args.n_steps_per_interval < 0:
        parser.error("n_steps_per_interval must be non-negative")

    rows: List[Dict] = []
    checkpoint_metadata = []
    expected_identity = None
    population_sizes = None
    dataset_locations = set()
    time_mode = None
    for method, seed, checkpoint in args.checkpoint:
        print(f"Restoring {method} seed={seed}: {checkpoint}", flush=True)
        model, params, cfg, payload = _load_checkpoint(checkpoint)
        dataset_name = cite_multi.canonical_dataset_name(cfg.problem.dataset_name)
        heldout = str(cfg.problem.heldout_timepoint)
        retained = cite_multi.retained_timepoints(cfg)
        identity = (dataset_name, heldout, tuple(retained))
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(
                "One invocation must contain one dataset/held-out task; got "
                f"{expected_identity} and {identity}."
            )
        current_time_mode = cite_multi.time_mode_from_config(cfg)
        if time_mode is None:
            time_mode = current_time_mode
        elif current_time_mode != time_mode:
            raise ValueError(
                f"Checkpoints use different time modes: {time_mode} and "
                f"{current_time_mode}."
            )

        dataset_location = args.dataset_location or str(cfg.problem.dataset_location)
        dataset_locations.add(str(Path(dataset_location).expanduser().resolve()))
        data = cite_multi.all_timepoint_data(dataset_location, dataset_name)
        populations = {
            timepoint: np.asarray(
                data["x"][data["timepoints"] == timepoint], dtype=np.float32
            )
            for timepoint in retained
        }
        current_sizes = {
            timepoint: int(population.shape[0])
            for timepoint, population in populations.items()
        }
        if population_sizes is None:
            population_sizes = current_sizes
        elif current_sizes != population_sizes:
            raise ValueError(
                f"Checkpoints resolve to datasets with different sizes: "
                f"{population_sizes} and {current_sizes}."
            )
        n_steps = int(
            args.n_steps_per_interval
            or getattr(cfg.evaluation, "flowmap_n_steps", 50)
        )

        for draw in range(args.noise_draws):
            rollout = populations[retained[0]]
            for protocol in ("interval", "rollout"):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "heldout_day": heldout,
                        "method": method,
                        "seed": seed,
                        "noise_draw": draw,
                        "protocol": protocol,
                        "timepoint": retained[0],
                        "source_timepoint": retained[0],
                        "source_n": rollout.shape[0],
                        "target_n": rollout.shape[0],
                        "w1": 0.0,
                        "checkpoint": str(checkpoint),
                    }
                )

            for interval_index, (source_timepoint, target_timepoint) in enumerate(
                zip(retained[:-1], retained[1:])
            ):
                start_time = cite_multi.normalized_time(source_timepoint, cfg)
                end_time = cite_multi.normalized_time(target_timepoint, cfg)
                target = populations[target_timepoint]
                interval_prediction = (
                    maizels_stochastic_eval.sample_composed_pushforward(
                        model,
                        params,
                        populations[source_timepoint],
                        start_time,
                        end_time,
                        _draw_key(
                            args.evaluation_seed,
                            draw,
                            interval_index,
                            protocol=0,
                        ),
                        n_coefficients=int(cfg.ssfm.n_coefficients),
                        n_steps=n_steps,
                    )
                )
                interval_w1 = wasserstein.exact_emd(interval_prediction, target)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "heldout_day": heldout,
                        "method": method,
                        "seed": seed,
                        "noise_draw": draw,
                        "protocol": "interval",
                        "timepoint": target_timepoint,
                        "source_timepoint": source_timepoint,
                        "source_n": interval_prediction.shape[0],
                        "target_n": target.shape[0],
                        "w1": interval_w1,
                        "checkpoint": str(checkpoint),
                    }
                )

                if interval_index == 0:
                    rollout = interval_prediction
                    rollout_w1 = interval_w1
                else:
                    rollout = maizels_stochastic_eval.sample_composed_pushforward(
                        model,
                        params,
                        rollout,
                        start_time,
                        end_time,
                        _draw_key(
                            args.evaluation_seed,
                            draw,
                            interval_index,
                            protocol=1,
                        ),
                        n_coefficients=int(cfg.ssfm.n_coefficients),
                        n_steps=n_steps,
                    )
                    rollout_w1 = wasserstein.exact_emd(rollout, target)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "heldout_day": heldout,
                        "method": method,
                        "seed": seed,
                        "noise_draw": draw,
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
                    f"  draw={draw} day {target_timepoint}: "
                    f"interval W1={interval_w1:.6f}; "
                    f"day-{retained[0]} rollout W1={rollout_w1:.6f}",
                    flush=True,
                )

        checkpoint_metadata.append(
            {
                "method": method,
                "seed": seed,
                "path": str(checkpoint),
                "best_step": int(np.asarray(payload["best_step"])),
                "best_validation_loss": float(
                    np.asarray(payload["best_validation_loss"])
                ),
                "n_steps_per_interval": n_steps,
                "learning_rate": float(cfg.optimization.learning_rate),
                "constraint_weight": float(cfg.constraints.weight),
                "diffusion_scale": float(cfg.ssfm.diffusion_scale),
            }
        )

    grouping = (
        "dataset",
        "heldout_day",
        "method",
        "seed",
        "protocol",
        "timepoint",
    )
    per_checkpoint = _summary_rows(rows, grouping)
    # Paper-style uncertainty is across independently trained checkpoints:
    # first average Brownian draws within a checkpoint, then compute s.e. over seeds.
    seed_mean_rows = [
        {
            "dataset": row["dataset"],
            "heldout_day": row["heldout_day"],
            "method": row["method"],
            "protocol": row["protocol"],
            "timepoint": row["timepoint"],
            "w1": row["w1_mean"],
        }
        for row in per_checkpoint
    ]
    summary = _summary_rows(
        seed_mean_rows,
        ("dataset", "heldout_day", "method", "protocol", "timepoint"),
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "per_draw.csv", rows)
    _write_csv(out_dir / "per_checkpoint.csv", per_checkpoint)
    _write_csv(out_dir / "summary.csv", summary)
    metadata = {
        "dataset": expected_identity[0] if expected_identity else None,
        "heldout_day": expected_identity[1] if expected_identity else None,
        "retained_timepoints": list(expected_identity[2] if expected_identity else ()),
        "time_mode": time_mode,
        "parameters": "EMA from each best checkpoint",
        "sampler": "composed stochastic flow map",
        "noise_draws": args.noise_draws,
        "evaluation_seed": args.evaluation_seed,
        "dataset_locations": sorted(dataset_locations),
        "population_sizes": population_sizes,
        "protocols": {
            "interval": "restart from the real population at each retained left endpoint",
            "rollout": "start from day 2 once and carry predictions across retained intervals",
        },
        "checkpoints": checkpoint_metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nSummary (mean +/- s.e. over trained seeds):")
    for row in summary:
        print(
            f"  {row['method']:>25s} {row['protocol']:>8s} "
            f"day {row['timepoint']:>1s}: "
            f"{row['w1_mean']:.6f} +/- {row['w1_se']:.6f}"
        )
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()

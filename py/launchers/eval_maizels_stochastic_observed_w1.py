"""Evaluate Maizels SSFM checkpoints at every marginal and test time."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import flax.serialization
import jax
import ml_collections
import numpy as np

from common import maizels
from common import maizels_checkpoint_eval as checkpoint_eval
from common import maizels_stochastic_eval
from common import stochastic_flow_map


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


def _draw_key(seed: int, draw: int, target_index: int, protocol: int):
    key = jax.random.PRNGKey(int(seed))
    key = jax.random.fold_in(key, int(draw))
    key = jax.random.fold_in(key, int(target_index))
    return jax.random.fold_in(key, int(protocol))


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
        help="Steps from each observed source to each target; zero uses the selected sampler's checkpoint default.",
    )
    parser.add_argument(
        "--euler", action="store_true",
        help="Evaluate only Euler--Maruyama sampling instead of composed stochastic maps.",
    )
    parser.add_argument(
        "--hparam_val_times", default=None,
        help="Comma-separated validation days excluded from test_times; defaults to the saved config.",
    )
    parser.add_argument("--evaluation_seed", type=int, default=2701)
    parser.add_argument(
        "--out_dir", default="outputs/maizels_stochastic_observed_w1"
    )
    args = parser.parse_args()
    if args.noise_draws <= 0:
        parser.error("noise_draws must be positive")
    if args.n_steps_per_interval < 0:
        parser.error("n_steps_per_interval must be non-negative")

    rows: List[Dict] = []
    checkpoint_metadata = []
    expected_marginals = None
    time_mode = None
    dataset_locations = set()
    sampler = "euler_maruyama" if args.euler else "flowmap"
    sample = (
        maizels_stochastic_eval.sample_euler_maruyama_pushforward if args.euler
        else maizels_stochastic_eval.sample_composed_pushforward
    )
    for method, seed, checkpoint in args.checkpoint:
        print(f"Restoring {method} seed={seed}: {checkpoint}", flush=True)
        model, params, cfg, payload = _load_checkpoint(checkpoint)
        if not args.euler and maizels_stochastic_eval.uses_euler_maruyama_evaluation(cfg):
            raise ValueError(
                "This checkpoint trained only local updates; select --euler for Euler--Maruyama evaluation."
            )
        dataset_location = args.dataset_location or str(cfg.problem.dataset_location)
        dataset_locations.add(str(Path(dataset_location).expanduser().resolve()))
        data = maizels.all_timepoint_data(dataset_location)
        marginals = checkpoint_eval.build_marginals(cfg, data, args.hparam_val_times)
        current_marginals = marginals.metadata()
        if expected_marginals is None:
            expected_marginals = current_marginals
            time_mode = str(cfg.problem.maizels_time_mode)
        elif current_marginals != expected_marginals:
            raise ValueError(
                "Checkpoints use different marginal populations, model clocks, or validation days."
            )
        default_steps_field = "euler_maruyama_n_steps" if args.euler else "flowmap_n_steps"
        n_steps = int(
            args.n_steps_per_interval
            or getattr(cfg.evaluation, default_steps_field, 50)
        )
        if n_steps <= 0:
            raise ValueError(f"Checkpoint {default_steps_field} must be positive.")

        for draw in range(args.noise_draws):
            def pushforward(source, *, start_time, end_time, target_index, protocol):
                return sample(
                    model, params, source, start_time, end_time,
                    _draw_key(args.evaluation_seed, draw, target_index, int(protocol == "rollout")),
                    n_coefficients=int(cfg.ssfm.n_coefficients),
                    n_steps=n_steps,
                )

            def progress(day, interval_w1, rollout_w1):
                print(
                    f"  draw={draw} {day}: "
                    f"interval W1={interval_w1:.6f}; "
                    f"rollout W1={rollout_w1:.6f}",
                    flush=True,
                )

            rows.extend(checkpoint_eval.evaluate_marginals(
                marginals, pushforward,
                context={
                    "method": method, "seed": seed, "checkpoint": str(checkpoint),
                    "sampler": sampler, "n_steps_per_interval": n_steps,
                    "noise_draw": draw,
                },
                progress=progress,
            ))

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

    per_checkpoint = checkpoint_eval.summary_rows(
        rows,
        ("method", "seed", "checkpoint", "sampler", "n_steps_per_interval", "protocol", "timepoint"),
    )
    # The paper-style uncertainty is across independently trained checkpoints:
    # first average Brownian draws within a checkpoint, then compute s.e. over seeds.
    seed_mean_rows = [
        {
            **row,
            "w1": row["w1_mean"],
        }
        for row in per_checkpoint
    ]
    summary = checkpoint_eval.summary_rows(
        seed_mean_rows, ("method", "sampler", "n_steps_per_interval", "protocol", "timepoint")
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_eval.write_csv(out_dir / "per_draw.csv", rows)
    checkpoint_eval.write_csv(out_dir / "per_checkpoint.csv", per_checkpoint)
    checkpoint_eval.write_csv(out_dir / "summary.csv", summary)
    (out_dir / "metrics.json").write_text(
        json.dumps(checkpoint_eval.metric_records(per_checkpoint, "w1_mean"), indent=2) + "\n"
    )
    metadata = {
        **expected_marginals,
        "time_mode": time_mode,
        "parameters": "EMA from each best checkpoint",
        "sampler": sampler,
        "noise_draws": args.noise_draws,
        "evaluation_seed": args.evaluation_seed,
        "dataset_locations": sorted(dataset_locations),
        "protocols": {
            "interval": "restart from the real population at each observed left endpoint",
            "rollout": "start from D3 once and carry predictions across observed intervals",
        },
        "checkpoints": checkpoint_metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nSummary (mean +/- s.e. over trained seeds):")
    for row in summary:
        print(
            f"  {row['method']:>25s} {row['protocol']:>8s} "
            f"{row['timepoint']:>10s} ({row['n_steps_per_interval']} steps): "
            f"{row['w1_mean']:.6f} +/- {row['w1_se']:.6f}"
        )
        if row["timepoint"] == "test_times":
            prefix = "final_eval" if row["protocol"] == "interval" else "rollout_eval"
            print(f"    {prefix}/{sampler}_mean_emd_test_times={row['w1_mean']:.6f}")
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()

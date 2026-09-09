#!/usr/bin/env python3
"""Run a resumable multi-seed hyperparameter grid for Maizels SSFMs."""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from scripts import sweep_maizels_hparams as sweep_utils  # noqa: E402


DEFAULT_LEARNING_RATES = (3e-4, 1e-3, 3e-3)
DEFAULT_DIFFUSION_SCALES = (0.1, 0.2, 0.4)
DEFAULT_CONSTRAINT_WEIGHTS = (1.0, 10.0, 100.0)
DEFAULT_ENTROPY_WEIGHTS = (0.0, 0.01, 0.1)
OBJECTIVE_METRIC = "final_eval/ssfm_mean_emd_hparam_val_times"


def build_grid(
    spec: Dict[str, object],
    learning_rates: Iterable[float],
    diffusion_scales: Iterable[float],
    constraint_weights: Iterable[float],
    entropy_weights: Iterable[float],
) -> List[Dict[str, float | None]]:
    """Build only dimensions that affect the selected stochastic variant."""
    dimensions = [
        ("learning_rate", tuple(learning_rates)),
        ("diffusion_scale", tuple(diffusion_scales)),
    ]
    if bool(spec["constraint_weight"]):
        dimensions.append(("constraint_weight", tuple(constraint_weights)))
    if bool(spec["entropy_weight"]):
        dimensions.append(("entropy_weight", tuple(entropy_weights)))
    names = [name for name, _ in dimensions]
    result = []
    for values in itertools.product(*(values for _, values in dimensions)):
        selected = dict(zip(names, values))
        result.append(
            {
                "learning_rate": selected["learning_rate"],
                "diffusion_scale": selected["diffusion_scale"],
                "constraint_weight": selected.get("constraint_weight"),
                "entropy_weight": selected.get("entropy_weight"),
            }
        )
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep direct Maizels SSFM hyperparameters and select them by "
            "held-out validation-time EMD."
        )
    )
    parser.add_argument("--slurm-id", "--slurm_id", required=True, type=int)
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.maizels_stochastic"
    )
    parser.add_argument("--dataset-location", "--dataset_location", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/maizels_stochastic_hparam_sweep"
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument(
        "--learning-rates",
        default=",".join(f"{value:g}" for value in DEFAULT_LEARNING_RATES),
    )
    parser.add_argument(
        "--diffusion-scales",
        default=",".join(f"{value:g}" for value in DEFAULT_DIFFUSION_SCALES),
    )
    parser.add_argument(
        "--constraint-weights",
        default=",".join(f"{value:g}" for value in DEFAULT_CONSTRAINT_WEIGHTS),
    )
    parser.add_argument(
        "--entropy-weights",
        default=",".join(f"{value:g}" for value in DEFAULT_ENTROPY_WEIGHTS),
    )
    parser.add_argument(
        "--maizels-schedule",
        choices=("d3_d8", "d3_d3p8_d8"),
        default="d3_d3p8_d8",
    )
    parser.add_argument(
        "--maizels-time-mode",
        choices=("real_time", "equal_time"),
        default="real_time",
    )
    parser.add_argument("--hparam-val-times", default="D3.4,D6")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _setting_id(args, values: Dict[str, float | None]) -> str:
    base = sweep_utils.setting_id_for(
        cfg_path=args.cfg_path,
        slurm_id=args.slurm_id,
        schedule=args.maizels_schedule,
        time_mode=args.maizels_time_mode,
        hparam_val_times=args.hparam_val_times,
        values=values,
    )
    diffusion = sweep_utils._float_slug(values["diffusion_scale"])
    return base.replace("_", f"_diff{diffusion}_", 1)


def _optional_command_arg(command: List[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def main(argv=None) -> int:
    args = parse_args(argv)
    module = importlib.import_module(args.cfg_path)
    if not hasattr(module, "get_hparam_sweep_spec"):
        raise AttributeError(
            f"{args.cfg_path} must expose get_hparam_sweep_spec(slurm_id)."
        )
    spec = dict(module.get_hparam_sweep_spec(args.slurm_id))
    if not bool(spec.get("diffusion_scale", False)):
        raise ValueError(f"{args.cfg_path} does not declare diffusion_scale sweepable.")

    learning_rates = sweep_utils.parse_float_grid(args.learning_rates, "learning rates")
    diffusion_scales = sweep_utils.parse_float_grid(
        args.diffusion_scales, "diffusion scales"
    )
    constraint_weights = sweep_utils.parse_float_grid(
        args.constraint_weights, "constraint weights"
    )
    entropy_weights = sweep_utils.parse_float_grid(
        args.entropy_weights, "entropy weights"
    )
    seeds = sweep_utils.parse_seed_grid(args.seeds)
    if any(value <= 0.0 for value in learning_rates):
        raise ValueError("All learning rates must be positive.")
    if any(value < 0.0 for value in diffusion_scales):
        raise ValueError("Diffusion scales must be non-negative.")
    if any(value < 0.0 for value in constraint_weights + entropy_weights):
        raise ValueError("Constraint and entropy weights must be non-negative.")

    grid = build_grid(
        spec,
        learning_rates,
        diffusion_scales,
        constraint_weights,
        entropy_weights,
    )
    variant_name = str(spec["variant_name"])
    output_root = Path(args.output_dir).expanduser().resolve()
    csv_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else output_root / f"slurm_{args.slurm_id}_results.csv"
    )
    summary_path = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else output_root / f"slurm_{args.slurm_id}_summary.csv"
    )
    rows: List[Dict[str, object]] = list(sweep_utils.read_rows(csv_path))
    completed = {
        str(row["run_id"]) for row in rows if str(row.get("status", "")) == "complete"
    }

    settings = []
    planned_runs = []
    for values in grid:
        setting_id = _setting_id(args, values)
        setting = {
            "setting_id": setting_id,
            "slurm_id": args.slurm_id,
            "variant_name": variant_name,
            "maizels_schedule": args.maizels_schedule,
            "maizels_time_mode": args.maizels_time_mode,
            "hparam_val_times": args.hparam_val_times,
            "learning_rate": values["learning_rate"],
            "diffusion_scale": values["diffusion_scale"],
            "constraint_weight": (
                ""
                if values["constraint_weight"] is None
                else values["constraint_weight"]
            ),
            "entropy_weight": (
                "" if values["entropy_weight"] is None else values["entropy_weight"]
            ),
        }
        settings.append(setting)
        for seed in seeds:
            planned_runs.append((setting_id, values, seed))

    total_runs = len(planned_runs)
    print(
        f"Variant {variant_name!r}: {len(grid)} settings x {len(seeds)} seeds "
        f"= {total_runs} runs; objective={OBJECTIVE_METRIC}."
    )
    for index, (setting_id, values, seed) in enumerate(planned_runs, start=1):
        run_id = sweep_utils.run_id_for(setting_id, seed)
        if run_id in completed and not args.rerun_completed:
            print(f"[{index}/{total_runs}] Skipping completed {run_id}.")
            continue

        run_root = output_root / run_id
        metrics_path = run_root / "final_metrics.json"
        command = [
            str(Path(args.python).expanduser()),
            str(REPO_ROOT / "py" / "launchers" / "maizels_stochastic.py"),
            "--cfg_path",
            args.cfg_path,
            "--slurm_id",
            str(args.slurm_id),
            "--dataset_location",
            str(Path(args.dataset_location).expanduser()),
            "--output_folder",
            str(run_root / "checkpoints"),
            "--maizels_schedule",
            args.maizels_schedule,
            "--maizels_time_mode",
            args.maizels_time_mode,
            "--hparam_val_times",
            args.hparam_val_times,
            "--learning_rate",
            f"{values['learning_rate']:g}",
            "--diffusion_scale",
            f"{values['diffusion_scale']:g}",
            "--seed",
            str(seed),
            "--final_metrics_path",
            str(metrics_path),
        ]
        if values["constraint_weight"] is not None:
            command.extend(["--constraint_weight", f"{values['constraint_weight']:g}"])
        if values["entropy_weight"] is not None:
            command.extend(["--entropy_weight", f"{values['entropy_weight']:g}"])
        _optional_command_arg(command, "--total_steps", args.total_steps)
        _optional_command_arg(command, "--batch_size", args.batch_size)
        _optional_command_arg(command, "--n_pairs", args.n_pairs)
        _optional_command_arg(
            command, "--early_stopping_patience", args.early_stopping_patience
        )
        _optional_command_arg(command, "--wandb_mode", args.wandb_mode)

        print(f"[{index}/{total_runs}] {run_id}")
        print(shlex.join(command))
        if args.dry_run:
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        started = time.monotonic()
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
        duration = time.monotonic() - started

        metrics = {}
        error = ""
        status = "failed"
        if result.returncode == 0 and metrics_path.is_file():
            with metrics_path.open() as handle:
                metrics = json.load(handle)
            objective = metrics.get(OBJECTIVE_METRIC)
            if isinstance(objective, (int, float)) and math.isfinite(float(objective)):
                status = "complete"
            else:
                error = f"Missing or non-finite objective {OBJECTIVE_METRIC}."
        elif result.returncode == 0:
            error = "Training completed without writing final_metrics.json."
        else:
            error = f"Training exited with status {result.returncode}."

        row: Dict[str, object] = {
            "run_id": run_id,
            "setting_id": setting_id,
            "status": status,
            "slurm_id": args.slurm_id,
            "seed": seed,
            "variant_name": variant_name,
            "maizels_schedule": args.maizels_schedule,
            "maizels_time_mode": args.maizels_time_mode,
            "hparam_val_times": args.hparam_val_times,
            "learning_rate": values["learning_rate"],
            "diffusion_scale": values["diffusion_scale"],
            "constraint_weight": (
                ""
                if values["constraint_weight"] is None
                else values["constraint_weight"]
            ),
            "entropy_weight": (
                "" if values["entropy_weight"] is None else values["entropy_weight"]
            ),
            "objective_metric": OBJECTIVE_METRIC,
            "objective_value": metrics.get(OBJECTIVE_METRIC, ""),
            "duration_seconds": round(duration, 3),
            "return_code": result.returncode,
            "run_output_folder": str(run_root),
            "metrics_json": str(metrics_path),
            "error": error,
            **sweep_utils._metric_row(metrics),
        }
        sweep_utils.upsert_row(rows, row)
        sweep_utils.write_rows(csv_path, rows)
        sweep_utils.write_summary_rows(
            summary_path,
            sweep_utils.summarize_settings(rows, settings, seeds, OBJECTIVE_METRIC),
        )
        print(f"Recorded {status} result in {csv_path}.")
        if status != "complete" and args.fail_fast:
            return result.returncode or 1

    summaries = sweep_utils.summarize_settings(rows, settings, seeds, OBJECTIVE_METRIC)
    if not args.dry_run:
        sweep_utils.write_summary_rows(summary_path, summaries)
    completed_settings = [
        row
        for row in summaries
        if row["status"] == "complete"
        and sweep_utils._is_finite_number(row["objective_mean"])
    ]
    if completed_settings:
        best = min(completed_settings, key=lambda row: float(row["objective_mean"]))
        print(
            f"Best mean {OBJECTIVE_METRIC}={float(best['objective_mean']):.8g}: "
            f"{best['setting_id']}"
        )
    elif not args.dry_run:
        print("No stochastic grid setting completed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run selected LARRY SPRING2D SSFM variants over multiple seeds.

Runs are sequential so that they do not compete for one accelerator.  Each
launcher writes a final-metrics JSON file; this wrapper maintains resumable
per-run and across-seed CSV files for every selected stochastic setting.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import larry  # noqa: E402
from scripts import run_maizels_stochastic_multiseed as runner_utils  # noqa: E402
from scripts import sweep_maizels_hparams as sweep_utils  # noqa: E402


DEFAULT_OBJECTIVE_METRIC = "final_eval/d2_to_d4_flowmap_emd"
parse_slurm_ids = runner_utils.parse_slurm_ids


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected LARRY SPRING2D SSFM settings over multiple seeds "
            "and collect their population, lineage, and clonal metrics."
        )
    )
    parser.add_argument(
        "--slurm-ids",
        "--slurm_ids",
        default="all",
        help="Comma-separated stochastic setting IDs, or 'all' (default: all).",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated non-negative training seeds.",
    )
    parser.add_argument(
        "--cfg-path",
        "--cfg_path",
        default="configs.larry_spring2d_stochastic",
    )
    parser.add_argument(
        "--dataset-location",
        "--dataset_location",
        default=str(larry.DEFAULT_DATA_DIR),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/larry_spring2d_stochastic_multiseed",
    )
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--classifier-path", "--classifier_path", default=None)
    parser.add_argument(
        "--full-data-classifier-path",
        "--full_data_classifier_path",
        default=None,
    )
    parser.add_argument("--learning-rate", "--learning_rate", type=float)
    parser.add_argument("--diffusion-scale", "--diffusion_scale", type=float)
    parser.add_argument("--gamma-scale", "--gamma_scale", type=float)
    parser.add_argument("--constraint-weight", "--constraint_weight", type=float)
    parser.add_argument("--entropy-weight", "--entropy_weight", type=float)
    parser.add_argument("--total-steps", "--total_steps", type=int)
    parser.add_argument("--batch-size", "--batch_size", type=int)
    parser.add_argument("--n-pairs", "--n_pairs", type=int)
    parser.add_argument("--ot-minibatch-size", "--ot_minibatch_size", type=int)
    parser.add_argument("--validation-frequency", "--validation_frequency", type=int)
    parser.add_argument(
        "--early-stopping-patience", "--early_stopping_patience", type=int
    )
    parser.add_argument("--eval-noise-draws", "--eval_noise_draws", type=int)
    parser.add_argument("--eval-max-points", "--eval_max_points", type=int)
    parser.add_argument(
        "--lineage-eval-max-points", "--lineage_eval_max_points", type=int
    )
    parser.add_argument("--eval-flowmap-steps", "--eval_flowmap_steps", type=int)
    parser.add_argument(
        "--clone-samples-per-source",
        "--clone_samples_per_source",
        type=int,
    )
    parser.add_argument("--clone-noise-draws", "--clone_noise_draws", type=int)
    parser.add_argument(
        "--clone-target-times",
        "--clone_target_times",
        default=None,
        help="Comma-separated targets such as D4,D6 (default from config).",
    )
    parser.add_argument("--clone-eval-batch-size", "--clone_eval_batch_size", type=int)
    parser.add_argument("--visual-frequency", "--visual_frequency", type=int)
    parser.add_argument(
        "--objective-metric",
        default=DEFAULT_OBJECTIVE_METRIC,
        help=(
            "Metric highlighted in summary.csv. All numerical final_eval "
            "metrics are summarized regardless."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _optional_arg(command: List[str], name: str, value) -> None:
    if value is not None:
        command.extend([name, str(value)])


def build_command(
    args: argparse.Namespace,
    *,
    slurm_id: int,
    seed: int,
    constrained: bool,
    minibatch_ot: bool,
    output_folder: Path,
    metrics_path: Path,
) -> List[str]:
    """Build one LARRY launcher command with relevant overrides only."""
    command = [
        str(Path(args.python).expanduser()),
        str(REPO_ROOT / "py" / "launchers" / "larry_stochastic.py"),
        "--cfg_path",
        str(args.cfg_path),
        "--slurm_id",
        str(int(slurm_id)),
        "--dataset_location",
        str(Path(args.dataset_location).expanduser()),
        "--output_folder",
        str(output_folder),
        "--seed",
        str(int(seed)),
        "--final_metrics_path",
        str(metrics_path),
    ]
    _optional_arg(command, "--classifier_path", args.classifier_path)
    _optional_arg(
        command, "--full_data_classifier_path", args.full_data_classifier_path
    )
    _optional_arg(command, "--learning_rate", args.learning_rate)
    _optional_arg(command, "--diffusion_scale", args.diffusion_scale)
    _optional_arg(command, "--gamma_scale", args.gamma_scale)
    if constrained:
        _optional_arg(command, "--constraint_weight", args.constraint_weight)
        _optional_arg(command, "--entropy_weight", args.entropy_weight)
    if minibatch_ot:
        _optional_arg(command, "--ot_minibatch_size", args.ot_minibatch_size)
    _optional_arg(command, "--total_steps", args.total_steps)
    _optional_arg(command, "--batch_size", args.batch_size)
    _optional_arg(command, "--n_pairs", args.n_pairs)
    _optional_arg(command, "--validation_frequency", args.validation_frequency)
    _optional_arg(command, "--early_stopping_patience", args.early_stopping_patience)
    _optional_arg(command, "--eval_noise_draws", args.eval_noise_draws)
    _optional_arg(command, "--eval_max_points", args.eval_max_points)
    _optional_arg(command, "--lineage_eval_max_points", args.lineage_eval_max_points)
    _optional_arg(command, "--eval_flowmap_steps", args.eval_flowmap_steps)
    _optional_arg(command, "--clone_samples_per_source", args.clone_samples_per_source)
    _optional_arg(command, "--clone_noise_draws", args.clone_noise_draws)
    _optional_arg(command, "--clone_target_times", args.clone_target_times)
    _optional_arg(command, "--clone_eval_batch_size", args.clone_eval_batch_size)
    _optional_arg(command, "--visual_frequency", args.visual_frequency)
    _optional_arg(command, "--wandb_mode", args.wandb_mode)
    return command


def _setting_id(
    args: argparse.Namespace,
    *,
    slurm_id: int,
    variant_name: str,
    constrained: bool,
    minibatch_ot: bool,
) -> str:
    ignored = {
        "slurm_ids",
        "seeds",
        "output_dir",
        "results_csv",
        "summary_csv",
        "objective_metric",
        "python",
        "rerun_completed",
        "fail_fast",
        "dry_run",
    }
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    if not constrained:
        payload.pop("constraint_weight", None)
        payload.pop("entropy_weight", None)
    if not minibatch_ot:
        payload.pop("ot_minibatch_size", None)
    payload["slurm_id"] = int(slurm_id)
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    safe_variant = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(variant_name)
    )
    return f"sid{int(slurm_id)}_{safe_variant}_{digest}"


def _load_metrics(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return metrics


def _validate_numeric_args(args: argparse.Namespace) -> None:
    positive = (
        "learning_rate",
        "gamma_scale",
        "total_steps",
        "batch_size",
        "n_pairs",
        "ot_minibatch_size",
        "validation_frequency",
        "eval_noise_draws",
        "eval_flowmap_steps",
        "clone_samples_per_source",
        "clone_noise_draws",
        "clone_eval_batch_size",
    )
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive.")
    nonnegative = (
        "diffusion_scale",
        "constraint_weight",
        "entropy_weight",
        "early_stopping_patience",
        "eval_max_points",
        "lineage_eval_max_points",
        "visual_frequency",
    )
    for name in nonnegative:
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative.")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    module = importlib.import_module(args.cfg_path)
    if not hasattr(module, "VARIANTS") or not hasattr(module, "get_hparam_sweep_spec"):
        raise AttributeError(
            f"{args.cfg_path} must expose VARIANTS and get_hparam_sweep_spec()."
        )
    slurm_ids = parse_slurm_ids(args.slurm_ids, len(module.VARIANTS))
    seeds = sweep_utils.parse_seed_grid(args.seeds)
    _validate_numeric_args(args)

    output_root = Path(args.output_dir).expanduser().resolve()
    results_path = (
        Path(args.results_csv).expanduser().resolve()
        if args.results_csv
        else output_root / "results.csv"
    )
    summary_path = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else output_root / "summary.csv"
    )
    rows: List[Dict[str, object]] = list(sweep_utils.read_rows(results_path))
    settings = []
    planned_runs = []

    for slurm_id in slurm_ids:
        spec = dict(module.get_hparam_sweep_spec(slurm_id))
        variant_name = str(spec["variant_name"])
        constrained = bool(spec.get("constraint_weight", False))
        minibatch_ot = bool(spec.get("minibatch_ot", False))
        setting_id = _setting_id(
            args,
            slurm_id=slurm_id,
            variant_name=variant_name,
            constrained=constrained,
            minibatch_ot=minibatch_ot,
        )
        setting = {
            "setting_id": setting_id,
            "slurm_id": int(slurm_id),
            "variant_name": variant_name,
            "learning_rate": (
                args.learning_rate if args.learning_rate is not None else "default"
            ),
            "diffusion_scale": (
                args.diffusion_scale if args.diffusion_scale is not None else "default"
            ),
            "gamma_scale": (
                args.gamma_scale if args.gamma_scale is not None else "default"
            ),
            "constraint_weight": (
                args.constraint_weight
                if constrained and args.constraint_weight is not None
                else "default" if constrained else ""
            ),
            "entropy_weight": (
                args.entropy_weight
                if constrained and args.entropy_weight is not None
                else "default" if constrained else ""
            ),
            "ot_minibatch_size": (
                args.ot_minibatch_size
                if minibatch_ot and args.ot_minibatch_size is not None
                else "default" if minibatch_ot else ""
            ),
            "clone_samples_per_source": (
                args.clone_samples_per_source
                if args.clone_samples_per_source is not None
                else "default"
            ),
            "clone_noise_draws": (
                args.clone_noise_draws
                if args.clone_noise_draws is not None
                else "default"
            ),
            "clone_target_times": args.clone_target_times or "default",
        }
        settings.append(setting)
        for seed in seeds:
            planned_runs.append((setting, constrained, minibatch_ot, int(seed)))

    if (args.constraint_weight is not None or args.entropy_weight is not None) and any(
        not constrained for _, constrained, _, _ in planned_runs
    ):
        print(
            "Constraint/entropy overrides will be applied only to constrained "
            "settings."
        )
    if args.ot_minibatch_size is not None and any(
        not minibatch_ot for _, _, minibatch_ot, _ in planned_runs
    ):
        print("The OT minibatch override will be applied only to OT settings.")

    total_runs = len(planned_runs)
    print(
        f"Planned {len(slurm_ids)} setting(s) x {len(seeds)} seed(s) "
        f"= {total_runs} sequential runs."
    )
    for index, (setting, constrained, minibatch_ot, seed) in enumerate(
        planned_runs, start=1
    ):
        setting_id = str(setting["setting_id"])
        run_id = sweep_utils.run_id_for(setting_id, seed)
        run_root = output_root / setting_id / f"seed_{seed}"
        metrics_path = run_root / "final_metrics.json"
        checkpoint_root = run_root / "checkpoints"
        completed_row = next(
            (
                row
                for row in rows
                if str(row.get("run_id", "")) == run_id
                and str(row.get("status", "")) == "complete"
            ),
            None,
        )
        if (
            not args.rerun_completed
            and completed_row is not None
            and metrics_path.is_file()
        ):
            print(f"[{index}/{total_runs}] Skipping completed {run_id}.")
            continue

        command = build_command(
            args,
            slurm_id=int(setting["slurm_id"]),
            seed=seed,
            constrained=constrained,
            minibatch_ot=minibatch_ot,
            output_folder=checkpoint_root,
            metrics_path=metrics_path,
        )
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

        metrics: Dict[str, object] = {}
        error = ""
        status = "failed"
        if result.returncode == 0 and metrics_path.is_file():
            try:
                metrics = _load_metrics(metrics_path)
                status = "complete"
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                error = f"Could not read final metrics: {exc}"
        elif result.returncode == 0:
            error = "Training completed without writing final_metrics.json."
        else:
            error = f"Training exited with status {result.returncode}."

        row: Dict[str, object] = {
            "run_id": run_id,
            **setting,
            "status": status,
            "seed": seed,
            "objective_metric": args.objective_metric,
            "objective_value": metrics.get(args.objective_metric, ""),
            "duration_seconds": round(duration, 3),
            "return_code": result.returncode,
            "run_output_folder": str(run_root),
            "metrics_json": str(metrics_path),
            "error": error,
            **sweep_utils._metric_row(metrics),
        }
        sweep_utils.upsert_row(rows, row)
        sweep_utils.write_rows(results_path, rows)
        sweep_utils.write_summary_rows(
            summary_path,
            sweep_utils.summarize_settings(
                rows, settings, seeds, args.objective_metric
            ),
        )
        print(f"Recorded {status} result in {results_path}.")
        if status != "complete" and args.fail_fast:
            return result.returncode or 1

    if not args.dry_run:
        summaries = sweep_utils.summarize_settings(
            rows, settings, seeds, args.objective_metric
        )
        sweep_utils.write_summary_rows(summary_path, summaries)
        completed = sum(row["status"] == "complete" for row in summaries)
        print(
            f"Completed summaries for {completed}/{len(settings)} settings: "
            f"{summary_path}"
        )
        if completed != len(settings):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run selected CITE/Multi SSFM variants over datasets, holdouts, and seeds.

Runs are sequential so they do not compete for one accelerator.  Each launcher
writes a metrics JSON file; this wrapper maintains resumable per-run and
across-seed CSV files.
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

from scripts import run_maizels_stochastic_multiseed as runner_utils  # noqa: E402
from scripts import sweep_maizels_hparams as sweep_utils  # noqa: E402


DEFAULT_OBJECTIVE_METRIC = "final_eval/evaluation_mean_emd"
parse_slurm_ids = runner_utils.parse_slurm_ids


def _parse_selection(
    text: str,
    *,
    name: str,
    allowed: tuple[str, ...],
) -> tuple[str, ...]:
    raw = str(text).strip().lower()
    if raw == "all":
        return allowed
    values = tuple(item.strip().lower() for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"{name} must contain only {list(allowed)}, got {invalid}."
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values.")
    return values


def parse_datasets(text: str) -> tuple[str, ...]:
    return _parse_selection(
        text,
        name="datasets",
        allowed=("cite", "multi"),
    )


def parse_heldout_days(text: str) -> tuple[str, ...]:
    return _parse_selection(
        text,
        name="held-out days",
        allowed=("3", "4"),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected direct CITE/Multi SSFM variants over multiple seeds "
            "and collect their final metrics."
        )
    )
    parser.add_argument(
        "--slurm-ids",
        "--slurm_ids",
        default="all",
        help="Comma-separated stochastic IDs, or 'all' (default: all).",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated non-negative training seeds.",
    )
    parser.add_argument(
        "--datasets",
        default="cite,multi",
        help="Comma-separated selection from cite,multi, or 'all'.",
    )
    parser.add_argument(
        "--heldout-days",
        "--heldout_days",
        default="3,4",
        help="Comma-separated selection from 3,4, or 'all'.",
    )
    parser.add_argument(
        "--cite-multi-time-mode",
        "--cite_multi_time_mode",
        choices=("equal_time", "real_time"),
        default="equal_time",
        help="Clock used for every run (default: equal_time).",
    )
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.cite_multi_stochastic"
    )
    parser.add_argument("--dataset-location", "--dataset_location", required=True)
    parser.add_argument(
        "--output-dir", default="outputs/cite_multi_stochastic_multiseed"
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
    parser.add_argument(
        "--validation-frequency", "--validation_frequency", type=int
    )
    parser.add_argument(
        "--early-stopping-patience", "--early_stopping_patience", type=int
    )
    parser.add_argument("--eval-noise-draws", "--eval_noise_draws", type=int)
    parser.add_argument("--eval-max-points", "--eval_max_points", type=int)
    parser.add_argument(
        "--lineage-eval-max-points", "--lineage_eval_max_points", type=int
    )
    parser.add_argument(
        "--eval-flowmap-steps", "--eval_flowmap_steps", type=int
    )
    parser.add_argument("--visual-frequency", "--visual_frequency", type=int)
    parser.add_argument(
        "--objective-metric",
        default=DEFAULT_OBJECTIVE_METRIC,
        help="Metric summarized across seeds.",
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
    dataset_name: str,
    heldout_day: str,
    slurm_id: int,
    seed: int,
    constrained: bool,
    output_folder: Path,
    metrics_path: Path,
) -> List[str]:
    """Build one launcher command, omitting irrelevant constraint options."""
    command = [
        str(Path(args.python).expanduser()),
        str(REPO_ROOT / "py" / "launchers" / "cite_multi_stochastic.py"),
        "--cfg_path",
        str(args.cfg_path),
        "--slurm_id",
        str(int(slurm_id)),
        "--dataset_name",
        str(dataset_name),
        "--heldout_day",
        str(heldout_day),
        "--cite_multi_time_mode",
        str(args.cite_multi_time_mode),
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
    _optional_arg(command, "--total_steps", args.total_steps)
    _optional_arg(command, "--batch_size", args.batch_size)
    _optional_arg(command, "--n_pairs", args.n_pairs)
    _optional_arg(command, "--validation_frequency", args.validation_frequency)
    _optional_arg(command, "--early_stopping_patience", args.early_stopping_patience)
    _optional_arg(command, "--eval_noise_draws", args.eval_noise_draws)
    _optional_arg(command, "--eval_max_points", args.eval_max_points)
    _optional_arg(
        command, "--lineage_eval_max_points", args.lineage_eval_max_points
    )
    _optional_arg(command, "--eval_flowmap_steps", args.eval_flowmap_steps)
    _optional_arg(command, "--visual_frequency", args.visual_frequency)
    _optional_arg(command, "--wandb_mode", args.wandb_mode)
    return command


def _setting_id(
    args: argparse.Namespace,
    *,
    dataset_name: str,
    heldout_day: str,
    slurm_id: int,
    variant_name: str,
    constrained: bool,
) -> str:
    ignored = {
        "slurm_ids",
        "seeds",
        "datasets",
        "heldout_days",
        "output_dir",
        "results_csv",
        "summary_csv",
        "python",
        "rerun_completed",
        "fail_fast",
        "dry_run",
    }
    payload = {key: value for key, value in vars(args).items() if key not in ignored}
    if not constrained:
        payload.pop("constraint_weight", None)
        payload.pop("entropy_weight", None)
    payload.update(
        {
            "dataset_name": dataset_name,
            "heldout_day": heldout_day,
            "slurm_id": int(slurm_id),
        }
    )
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    safe_variant = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(variant_name)
    )
    return (
        f"{dataset_name}_{args.cite_multi_time_mode}_day{heldout_day}_"
        f"sid{int(slurm_id)}_"
        f"{safe_variant}_{digest}"
    )


def _load_metrics(path: Path) -> Dict[str, object]:
    with path.open() as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return metrics


def _validate_numeric_args(args: argparse.Namespace) -> None:
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    for name in ("diffusion_scale", "constraint_weight", "entropy_weight"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if args.gamma_scale is not None and args.gamma_scale <= 0.0:
        raise ValueError("gamma_scale must be positive.")


def main(argv=None) -> int:
    args = parse_args(argv)
    module = importlib.import_module(args.cfg_path)
    if not hasattr(module, "VARIANTS") or not hasattr(
        module, "get_hparam_sweep_spec"
    ):
        raise AttributeError(
            f"{args.cfg_path} must expose VARIANTS and get_hparam_sweep_spec()."
        )
    slurm_ids = parse_slurm_ids(args.slurm_ids, len(module.VARIANTS))
    seeds = sweep_utils.parse_seed_grid(args.seeds)
    datasets = parse_datasets(args.datasets)
    heldout_days = parse_heldout_days(args.heldout_days)
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

    for dataset_name in datasets:
        for heldout_day in heldout_days:
            for slurm_id in slurm_ids:
                spec = dict(module.get_hparam_sweep_spec(slurm_id))
                variant_name = str(spec["variant_name"])
                constrained = bool(spec.get("constraint_weight", False))
                setting_id = _setting_id(
                    args,
                    dataset_name=dataset_name,
                    heldout_day=heldout_day,
                    slurm_id=slurm_id,
                    variant_name=variant_name,
                    constrained=constrained,
                )
                setting = {
                    "setting_id": setting_id,
                    "dataset_name": dataset_name,
                    "heldout_day": heldout_day,
                    "cite_multi_time_mode": args.cite_multi_time_mode,
                    "slurm_id": int(slurm_id),
                    "variant_name": variant_name,
                    "learning_rate": (
                        args.learning_rate
                        if args.learning_rate is not None
                        else "default"
                    ),
                    "diffusion_scale": (
                        args.diffusion_scale
                        if args.diffusion_scale is not None
                        else "default"
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
                }
                settings.append(setting)
                for seed in seeds:
                    planned_runs.append((setting, constrained, int(seed)))

    if (args.constraint_weight is not None or args.entropy_weight is not None) and any(
        not constrained for _, constrained, _ in planned_runs
    ):
        print(
            "Constraint/entropy overrides will be applied only to constrained "
            "variants."
        )

    total_runs = len(planned_runs)
    print(
        f"Planned {len(datasets)} dataset(s) x {len(heldout_days)} holdout(s) x "
        f"{len(slurm_ids)} variant(s) x {len(seeds)} seed(s) = "
        f"{total_runs} sequential runs ({args.cite_multi_time_mode})."
    )
    for index, (setting, constrained, seed) in enumerate(planned_runs, start=1):
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
            dataset_name=str(setting["dataset_name"]),
            heldout_day=str(setting["heldout_day"]),
            slurm_id=int(setting["slurm_id"]),
            seed=seed,
            constrained=constrained,
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

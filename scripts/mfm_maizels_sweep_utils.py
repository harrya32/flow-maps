#!/usr/bin/env python3
"""Shared resumable grid-search machinery for Maizels MFM baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = REPO_ROOT / "metric-flow-matching"

BASE_COLUMNS = [
    "run_id",
    "setting_id",
    "status",
    "method",
    "seed",
    "base_config",
    "base_config_sha256",
    "maizels_schedule",
    "maizels_time_mode",
    "hparam_val_times",
    "objective_metric",
    "objective_value",
    "duration_seconds",
    "return_code",
    "run_output_folder",
    "derived_config",
    "metrics_json",
    "error",
]

SUMMARY_BASE_COLUMNS = [
    "setting_id",
    "status",
    "objective_rank",
    "method",
    "base_config",
    "base_config_sha256",
    "maizels_schedule",
    "maizels_time_mode",
    "hparam_val_times",
    "seeds",
    "n_seeds_planned",
    "n_seeds_completed",
    "objective_metric",
    "objective_mean",
    "objective_std",
]


def parse_float_grid(text: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated list of numbers."
        ) from exc
    return _validate_grid(values, name)


def parse_int_grid(text: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated list of integers."
        ) from exc
    return _validate_grid(values, name)


def parse_seed_grid(text: str) -> tuple[int, ...]:
    values = parse_int_grid(text, "seeds")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("seeds must be non-negative.")
    return values


def parse_timepoints(text: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(text).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("hparam validation times cannot be empty.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "hparam validation times contain duplicate values."
        )
    return values


def _validate_grid(values: Sequence, name: str):
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values.")
    return tuple(values)


def build_grid(parameter_grids: Mapping[str, Iterable]) -> List[Dict[str, object]]:
    """Return the Cartesian product in the insertion order of ``parameter_grids``."""
    names = list(parameter_grids)
    values = [tuple(parameter_grids[name]) for name in names]
    for name, candidates in zip(names, values):
        _validate_grid(candidates, name)
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_config: str,
    default_output_dir: str,
    default_objective_metric: str,
) -> None:
    parser.add_argument("--config-path", default=default_config)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--classifier-path", default="")
    parser.add_argument("--output-dir", default=default_output_dir)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument(
        "--seeds",
        default="0,1",
        help="Comma-separated seeds run for every hyperparameter setting.",
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
    parser.add_argument("--objective-metric", default=default_objective_metric)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=None)
    parser.add_argument("--validation-pairs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--val-check-interval", type=int, default=None)
    parser.add_argument("--eval-every-n-steps", type=int, default=None)
    parser.add_argument("--eval-points-per-time", type=int, default=None)
    parser.add_argument("--eval-euler-steps", type=int, default=None)
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--flow-accelerator", default=None)
    parser.add_argument("--geopath-accelerator", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def common_runtime_overrides(args: argparse.Namespace) -> Dict[str, object]:
    mapping = {
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "maizels_n_pairs": args.n_pairs,
        "maizels_validation_pairs": args.validation_pairs,
        "patience": args.patience,
        "val_check_interval": args.val_check_interval,
        "maizels_eval_every_n_steps": args.eval_every_n_steps,
        "maizels_eval_points_per_time": args.eval_points_per_time,
        "maizels_eval_euler_steps": args.eval_euler_steps,
        "accelerator": args.accelerator,
        "flow_accelerator": args.flow_accelerator,
        "geopath_accelerator": args.geopath_accelerator,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def validate_common_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "batch_size",
        "n_pairs",
        "validation_pairs",
        "val_check_interval",
        "eval_every_n_steps",
        "eval_points_per_time",
        "eval_euler_steps",
    )
    for name in positive:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if args.patience is not None and args.patience < 0:
        raise ValueError("patience must be non-negative.")


def resolve_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, MFM_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find MFM config {value!r}; checked "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_base_config(path: Path, method: str) -> Dict[str, object]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    if str(config.get("data_type", "")) != "maizels":
        raise ValueError(f"{path} is not a Maizels configuration.")
    sf2m = bool(config.get("sf2m", False))
    mfm = bool(config.get("mfm", True))
    if method == "sf2m" and (not sf2m or mfm):
        raise ValueError("The SF2M sweep requires a config with sf2m=true, mfm=false.")
    if method == "mfm" and (sf2m or not mfm):
        raise ValueError("The MFM sweep requires a config with mfm=true, sf2m=false.")
    return config


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def setting_id_for(
    *,
    method: str,
    base_config_path: Path,
    base_config_digest: str,
    dataset_path: Path,
    classifier_path: Optional[Path],
    schedule: str,
    time_mode: str,
    hparam_val_times: Sequence[str],
    objective_metric: str,
    hparams: Mapping[str, object],
    runtime_overrides: Mapping[str, object],
    slug_keys: Sequence[str],
) -> str:
    payload = {
        "method": method,
        "base_config": str(base_config_path),
        "base_config_sha256": base_config_digest,
        "dataset_path": str(dataset_path),
        "classifier_path": str(classifier_path or ""),
        "schedule": schedule,
        "time_mode": time_mode,
        "hparam_val_times": list(hparam_val_times),
        "objective_metric": objective_metric,
        "hparams": dict(hparams),
        "runtime_overrides": dict(runtime_overrides),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    readable = [method]
    for key in slug_keys:
        if key in hparams:
            readable.append(f"{key}_{_slug(hparams[key])}")
    return "_".join(readable + [digest])


def run_id_for(setting_id: str, seed: int) -> str:
    digest = hashlib.sha1(f"{setting_id}:{seed}".encode("utf-8")).hexdigest()[:6]
    return f"{setting_id}_seed{seed}_{digest}"


def _atomic_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)
    os.replace(temporary, path)


def derived_config(
    base_config: Mapping[str, object],
    *,
    method: str,
    setting_id: str,
    seed: int,
    hparams: Mapping[str, object],
    runtime_overrides: Mapping[str, object],
    schedule: str,
    time_mode: str,
    hparam_val_times: Sequence[str],
    shared_cache_root: Path,
) -> Dict[str, object]:
    config = dict(base_config)
    # ``merge_config`` applies YAML after argparse, so remove values that must
    # remain specific to this subprocess and are supplied on its command line.
    for command_line_key in (
        "config_path",
        "working_dir",
        "maizels_dataset_path",
        "maizels_classifier_path",
        "final_metrics_path",
    ):
        config.pop(command_line_key, None)
    config.update(hparams)
    config.update(runtime_overrides)
    config.update(
        {
            "seeds": [int(seed)],
            "t_exclude": None,
            "maizels_schedule": schedule,
            "maizels_time_mode": time_mode,
            "maizels_hparam_val_times": list(hparam_val_times),
            "wandb_name": f"maizels_{setting_id}_seed{seed}",
        }
    )
    if (
        method == "sf2m"
        and str(config.get("sf2m_ot_cost", "euclidean")) == "geodesic"
        and not str(config.get("sf2m_geodesic_cache_dir", "")).strip()
    ):
        config["sf2m_geodesic_cache_dir"] = str(
            shared_cache_root / ".sf2m_geodesic_cache"
        )
    return config


def build_command(
    args: argparse.Namespace,
    *,
    derived_config_path: Path,
    run_root: Path,
    metrics_path: Path,
) -> List[str]:
    command = [
        str(Path(args.python).expanduser()),
        "-m",
        "mfm.train.main",
        "--config_path",
        str(derived_config_path),
        "--working_dir",
        str(run_root),
        "--maizels_dataset_path",
        str(Path(args.dataset_path).expanduser().resolve()),
        "--final_metrics_path",
        str(metrics_path),
    ]
    if str(args.classifier_path).strip():
        command.extend(
            [
                "--maizels_classifier_path",
                str(Path(args.classifier_path).expanduser().resolve()),
            ]
        )
    return command


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, object]], base_columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row if key not in base_columns})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(base_columns) + extras,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv(path, rows, BASE_COLUMNS)


def write_summary_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv(path, rows, SUMMARY_BASE_COLUMNS)


def upsert_row(rows: List[Dict[str, object]], row: Dict[str, object]) -> None:
    for index, existing in enumerate(rows):
        if existing.get("run_id") == row["run_id"]:
            rows[index] = row
            return
    rows.append(row)


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_row(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith("final_eval/")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }


def summarize_settings(
    rows: List[Dict[str, object]],
    settings: List[Dict[str, object]],
    seeds: Sequence[int],
    objective_metric: str,
) -> List[Dict[str, object]]:
    expected_seeds = set(int(seed) for seed in seeds)
    summaries = []
    for setting in settings:
        matching = [
            row
            for row in rows
            if row.get("setting_id") == setting["setting_id"]
            and row.get("status") == "complete"
            and int(row["seed"]) in expected_seeds
        ]
        completed_seeds = {int(row["seed"]) for row in matching}
        metric_names = sorted(
            {
                key
                for row in matching
                for key, value in row.items()
                if key.startswith("final_eval/") and _is_finite_number(value)
            }
        )
        summary = {
            **setting,
            "status": (
                "complete"
                if completed_seeds == expected_seeds
                else "partial" if completed_seeds else "failed"
            ),
            "objective_rank": "",
            "seeds": ",".join(str(seed) for seed in seeds),
            "n_seeds_planned": len(expected_seeds),
            "n_seeds_completed": len(completed_seeds),
            "objective_metric": objective_metric,
            "objective_mean": "",
            "objective_std": "",
        }
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in matching
                if _is_finite_number(row.get(metric_name))
            ]
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric_name}_mean_across_seeds"] = mean
            summary[f"{metric_name}_std_across_seeds"] = std
            if metric_name == objective_metric:
                summary["objective_mean"] = mean
                summary["objective_std"] = std
        summaries.append(summary)

    ranked = sorted(
        (
            summary
            for summary in summaries
            if summary["status"] == "complete"
            and _is_finite_number(summary["objective_mean"])
        ),
        key=lambda row: float(row["objective_mean"]),
    )
    for rank, summary in enumerate(ranked, start=1):
        summary["objective_rank"] = rank
    return summaries


def run_sweep(
    args: argparse.Namespace,
    *,
    method: str,
    parameter_grids: Mapping[str, Iterable],
    slug_keys: Sequence[str],
    extra_runtime_overrides: Optional[Mapping[str, object]] = None,
) -> int:
    """Run one method-specific grid and maintain resumable CSV summaries."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    validate_common_args(args)
    seeds = parse_seed_grid(args.seeds)
    hparam_val_times = parse_timepoints(args.hparam_val_times)
    base_config_path = resolve_config_path(args.config_path)
    base_config = load_base_config(base_config_path, method)
    base_digest = config_sha256(base_config_path)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Maizels dataset not found: {dataset_path}")
    classifier_path = None
    if str(args.classifier_path).strip():
        classifier_path = Path(args.classifier_path).expanduser().resolve()
        if not classifier_path.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Maizels classifier not found: {classifier_path}")

    runtime_overrides = common_runtime_overrides(args)
    runtime_overrides.update(
        {
            key: value
            for key, value in dict(extra_runtime_overrides or {}).items()
            if value is not None
        }
    )
    grid = build_grid(parameter_grids)
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
    rows: List[Dict[str, object]] = list(read_rows(results_path))

    settings = []
    planned_runs = []
    for hparams in grid:
        setting_id = setting_id_for(
            method=method,
            base_config_path=base_config_path,
            base_config_digest=base_digest,
            dataset_path=dataset_path,
            classifier_path=classifier_path,
            schedule=args.maizels_schedule,
            time_mode=args.maizels_time_mode,
            hparam_val_times=hparam_val_times,
            objective_metric=args.objective_metric,
            hparams=hparams,
            runtime_overrides=runtime_overrides,
            slug_keys=slug_keys,
        )
        setting = {
            "setting_id": setting_id,
            "method": method,
            "base_config": str(base_config_path),
            "base_config_sha256": base_digest,
            "maizels_schedule": args.maizels_schedule,
            "maizels_time_mode": args.maizels_time_mode,
            "hparam_val_times": ",".join(hparam_val_times),
            **hparams,
            **runtime_overrides,
        }
        settings.append(setting)
        for seed in seeds:
            planned_runs.append((setting, hparams, int(seed)))

    print(
        f"{method.upper()}: {len(grid)} settings x {len(seeds)} seeds = "
        f"{len(planned_runs)} runs; objective={args.objective_metric}."
    )
    for index, (setting, hparams, seed) in enumerate(planned_runs, start=1):
        setting_id = str(setting["setting_id"])
        run_id = run_id_for(setting_id, seed)
        run_root = output_root / setting_id / f"seed_{seed}"
        config_path = run_root / "sweep_config.yaml"
        metrics_path = run_root / "final_metrics.json"
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
            print(f"[{index}/{len(planned_runs)}] Skipping completed {run_id}.")
            continue

        config = derived_config(
            base_config,
            method=method,
            setting_id=setting_id,
            seed=seed,
            hparams=hparams,
            runtime_overrides=runtime_overrides,
            schedule=args.maizels_schedule,
            time_mode=args.maizels_time_mode,
            hparam_val_times=hparam_val_times,
            shared_cache_root=output_root,
        )
        command = build_command(
            args,
            derived_config_path=config_path,
            run_root=run_root,
            metrics_path=metrics_path,
        )
        print(f"[{index}/{len(planned_runs)}] {run_id}")
        print(shlex.join(command))
        if args.dry_run:
            print("  config overrides: " + json.dumps({**hparams, **runtime_overrides}))
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        _atomic_yaml(config_path, config)
        if metrics_path.exists():
            metrics_path.unlink()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(MFM_ROOT), environment.get("PYTHONPATH", "")],
            )
        )
        if args.wandb_mode is not None:
            environment["WANDB_MODE"] = args.wandb_mode

        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=MFM_ROOT,
            env=environment,
            check=False,
        )
        duration = time.monotonic() - started

        metrics: Dict[str, object] = {}
        error = ""
        status = "failed"
        if result.returncode == 0 and metrics_path.is_file():
            try:
                with metrics_path.open() as handle:
                    metrics = json.load(handle)
                objective = metrics.get(args.objective_metric)
                if _is_finite_number(objective):
                    status = "complete"
                else:
                    error = (
                        "Missing or non-finite objective metric "
                        f"{args.objective_metric}."
                    )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                error = f"Could not read final metrics: {exc}"
        elif result.returncode == 0:
            error = "Training completed without writing final_metrics.json."
        else:
            error = f"Training exited with status {result.returncode}."

        row = {
            "run_id": run_id,
            **setting,
            "status": status,
            "seed": seed,
            "objective_metric": args.objective_metric,
            "objective_value": metrics.get(args.objective_metric, ""),
            "duration_seconds": round(duration, 3),
            "return_code": result.returncode,
            "run_output_folder": str(run_root),
            "derived_config": str(config_path),
            "metrics_json": str(metrics_path),
            "error": error,
            **_metric_row(metrics),
        }
        upsert_row(rows, row)
        write_rows(results_path, rows)
        write_summary_rows(
            summary_path,
            summarize_settings(rows, settings, seeds, args.objective_metric),
        )
        print(f"Recorded {status} result in {results_path}.")
        if status != "complete" and args.fail_fast:
            return result.returncode or 1

    summaries = summarize_settings(rows, settings, seeds, args.objective_metric)
    if not args.dry_run:
        write_summary_rows(summary_path, summaries)
    completed = [
        row
        for row in summaries
        if row["status"] == "complete" and _is_finite_number(row["objective_mean"])
    ]
    if completed:
        best = min(completed, key=lambda row: float(row["objective_mean"]))
        print(
            f"Best mean {args.objective_metric}="
            f"{float(best['objective_mean']):.8g}: {best['setting_id']}"
        )
    elif not args.dry_run:
        print("No complete hyperparameter setting is available.")
        return 1
    return 0

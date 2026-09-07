#!/usr/bin/env python3
"""Run a resumable Maizels hyperparameter grid and collect final metrics.

The selected config declares which dimensions affect each Slurm variant.
Learning rate is always swept; constraint and entropy weights are included only
when the variant enables the corresponding lineage-constraint terms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
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
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

DEFAULT_LEARNING_RATES = (3e-4, 1e-3, 3e-3)
DEFAULT_CONSTRAINT_WEIGHTS = (100.0, 350.0, 1000.0)
DEFAULT_ENTROPY_WEIGHTS = (0.0, 0.01, 0.1)

BASE_COLUMNS = [
    "run_id",
    "status",
    "slurm_id",
    "variant_name",
    "maizels_schedule",
    "maizels_time_mode",
    "hparam_val_times",
    "learning_rate",
    "constraint_weight",
    "entropy_weight",
    "objective_metric",
    "objective_value",
    "duration_seconds",
    "return_code",
    "run_output_folder",
    "metrics_json",
    "error",
]


def parse_float_grid(text: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers.") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values.")
    return values


def build_grid(
    spec: Dict[str, object],
    learning_rates: Iterable[float],
    constraint_weights: Iterable[float],
    entropy_weights: Iterable[float],
) -> List[Dict[str, float | None]]:
    """Build only the dimensions consumed by the selected variant."""
    dimensions = [("learning_rate", tuple(learning_rates))]
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
                "constraint_weight": selected.get("constraint_weight"),
                "entropy_weight": selected.get("entropy_weight"),
            }
        )
    return result


def _float_slug(value: float | None) -> str:
    if value is None:
        return "na"
    return f"{float(value):g}".replace("-", "m").replace(".", "p").replace("+", "")


def run_id_for(
    *,
    cfg_path: str,
    slurm_id: int,
    schedule: str,
    time_mode: str,
    hparam_val_times: str,
    values: Dict[str, float | None],
) -> str:
    payload = {
        "cfg_path": cfg_path,
        "slurm_id": int(slurm_id),
        "schedule": schedule,
        "time_mode": time_mode,
        "hparam_val_times": hparam_val_times,
        **values,
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    parts = [f"sid{slurm_id}", f"lr{_float_slug(values['learning_rate'])}"]
    if values["constraint_weight"] is not None:
        parts.append(f"cw{_float_slug(values['constraint_weight'])}")
    if values["entropy_weight"] is not None:
        parts.append(f"ew{_float_slug(values['entropy_weight'])}")
    return "_".join(parts + [digest])


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted(
        {key for row in rows for key in row if key not in BASE_COLUMNS}
    )
    fieldnames = BASE_COLUMNS + extra_columns
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def upsert_row(rows: List[Dict[str, object]], row: Dict[str, object]) -> None:
    for index, existing in enumerate(rows):
        if existing.get("run_id") == row["run_id"]:
            rows[index] = row
            return
    rows.append(row)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Maizels hyperparameter grid and save final held-out EMD "
            "metrics to a resumable CSV."
        )
    )
    parser.add_argument("--slurm-id", "--slurm_id", required=True, type=int)
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.maizels_pca50"
    )
    parser.add_argument("--dataset-location", "--dataset_location", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/maizels_hparam_sweep",
        help="Root for per-run checkpoints, metric JSON files, and the CSV.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--learning-rates",
        default=",".join(f"{value:g}" for value in DEFAULT_LEARNING_RATES),
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
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument(
        "--objective-sampler",
        choices=("auto", "direct", "flowmap", "euler"),
        default="auto",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for each training subprocess.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sweep_spec(cfg_path: str, slurm_id: int) -> Dict[str, object]:
    module = importlib.import_module(cfg_path)
    if not hasattr(module, "get_hparam_sweep_spec"):
        raise AttributeError(
            f"{cfg_path} must expose get_hparam_sweep_spec(slurm_id)."
        )
    supported = inspect.signature(module.get_config).parameters
    missing = [
        name
        for name in ("learning_rate", "constraint_weight", "entropy_weight")
        if name not in supported
    ]
    spec = dict(module.get_hparam_sweep_spec(slurm_id))
    relevant_missing = [
        name
        for name in missing
        if name == "learning_rate" or bool(spec.get(name, False))
    ]
    if relevant_missing:
        raise TypeError(
            f"{cfg_path}.get_config does not accept relevant sweep arguments: "
            f"{relevant_missing}."
        )
    return spec


def _objective_sampler(requested: str, variant_name: str) -> str:
    if requested != "auto":
        return requested
    return "euler" if "flow_matching" in variant_name else "direct"


def _metric_row(metrics: Dict[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = _sweep_spec(args.cfg_path, args.slurm_id)
    learning_rates = parse_float_grid(args.learning_rates, "learning rates")
    constraint_weights = parse_float_grid(
        args.constraint_weights, "constraint weights"
    )
    entropy_weights = parse_float_grid(args.entropy_weights, "entropy weights")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("All learning rates must be positive.")
    if any(value < 0 for value in constraint_weights + entropy_weights):
        raise ValueError("Constraint and entropy weights must be non-negative.")

    grid = build_grid(spec, learning_rates, constraint_weights, entropy_weights)
    variant_name = str(spec["variant_name"])
    sampler = _objective_sampler(args.objective_sampler, variant_name)
    objective_metric = f"final_eval/{sampler}_mean_emd_hparam_val_times"
    output_root = Path(args.output_dir).expanduser().resolve()
    csv_path = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else output_root / f"slurm_{args.slurm_id}_results.csv"
    )
    rows: List[Dict[str, object]] = list(read_rows(csv_path))
    completed = {
        str(row["run_id"])
        for row in rows
        if str(row.get("status", "")) == "complete"
    }

    ignored = []
    if not bool(spec["constraint_weight"]):
        ignored.append("constraint weight")
    if not bool(spec["entropy_weight"]):
        ignored.append("entropy weight")
    print(
        f"Variant {variant_name!r}: {len(grid)} grid runs; objective="
        f"{objective_metric}."
    )
    if ignored:
        print("Not sweeping irrelevant dimensions: " + ", ".join(ignored) + ".")

    current_run_ids = set()
    for index, values in enumerate(grid, start=1):
        run_id = run_id_for(
            cfg_path=args.cfg_path,
            slurm_id=args.slurm_id,
            schedule=args.maizels_schedule,
            time_mode=args.maizels_time_mode,
            hparam_val_times=args.hparam_val_times,
            values=values,
        )
        current_run_ids.add(run_id)
        if run_id in completed and not args.rerun_completed:
            print(f"[{index}/{len(grid)}] Skipping completed {run_id}.")
            continue

        run_root = output_root / run_id
        checkpoint_dir = run_root / "checkpoints"
        metrics_path = run_root / "final_metrics.json"
        command = [
            str(Path(args.python).expanduser()),
            str(REPO_ROOT / "py" / "launchers" / "learn.py"),
            "--cfg_path",
            args.cfg_path,
            "--slurm_id",
            str(args.slurm_id),
            "--dataset_location",
            str(Path(args.dataset_location).expanduser()),
            "--output_folder",
            str(checkpoint_dir),
            "--maizels_schedule",
            args.maizels_schedule,
            "--maizels_time_mode",
            args.maizels_time_mode,
            "--hparam_val_times",
            args.hparam_val_times,
            "--learning_rate",
            f"{values['learning_rate']:g}",
            "--final_metrics_path",
            str(metrics_path),
        ]
        if values["constraint_weight"] is not None:
            command.extend(
                ["--constraint_weight", f"{values['constraint_weight']:g}"]
            )
        if values["entropy_weight"] is not None:
            command.extend(["--entropy_weight", f"{values['entropy_weight']:g}"])
        if args.early_stopping_patience is not None:
            command.extend(
                ["--early_stopping_patience", str(args.early_stopping_patience)]
            )

        print(f"[{index}/{len(grid)}] {run_id}")
        print(shlex.join(command))
        if args.dry_run:
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        if args.wandb_mode is not None:
            environment["WANDB_MODE"] = args.wandb_mode
        started = time.monotonic()
        result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
        duration = time.monotonic() - started

        metrics = {}
        error = ""
        status = "failed"
        if result.returncode == 0 and metrics_path.is_file():
            with metrics_path.open() as handle:
                metrics = json.load(handle)
            objective_value = metrics.get(objective_metric)
            if (
                isinstance(objective_value, (int, float))
                and math.isfinite(float(objective_value))
            ):
                status = "complete"
            else:
                error = f"Missing or non-finite objective metric {objective_metric}."
        elif result.returncode == 0:
            error = "Training completed without writing final_metrics.json."
        else:
            error = f"Training exited with status {result.returncode}."

        row: Dict[str, object] = {
            "run_id": run_id,
            "status": status,
            "slurm_id": args.slurm_id,
            "variant_name": variant_name,
            "maizels_schedule": args.maizels_schedule,
            "maizels_time_mode": args.maizels_time_mode,
            "hparam_val_times": args.hparam_val_times,
            "learning_rate": values["learning_rate"],
            "constraint_weight": (
                "" if values["constraint_weight"] is None else values["constraint_weight"]
            ),
            "entropy_weight": (
                "" if values["entropy_weight"] is None else values["entropy_weight"]
            ),
            "objective_metric": objective_metric,
            "objective_value": metrics.get(objective_metric, ""),
            "duration_seconds": round(duration, 3),
            "return_code": result.returncode,
            "run_output_folder": str(run_root),
            "metrics_json": str(metrics_path),
            "error": error,
            **_metric_row(metrics),
        }
        upsert_row(rows, row)
        write_rows(csv_path, rows)
        print(f"Recorded {status} result in {csv_path}.")
        if status != "complete" and args.fail_fast:
            return result.returncode or 1

    completed_rows = [
        row
        for row in rows
        if row.get("run_id") in current_run_ids
        and row.get("status") == "complete"
        and row.get("objective_metric") == objective_metric
        and str(row.get("objective_value", "")) != ""
    ]
    if completed_rows:
        best = min(completed_rows, key=lambda row: float(row["objective_value"]))
        print(
            f"Best {objective_metric}={float(best['objective_value']):.8g}: "
            f"{best['run_id']}"
        )
    elif not args.dry_run:
        print("No grid run completed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

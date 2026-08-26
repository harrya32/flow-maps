#!/usr/bin/env python3
"""Run and summarize the Maizels constrained-flow-map hyperparameter sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


DEFAULT_WEIGHTS = (400.0, 700.0, 1000.0, 1500.0)
DEFAULT_ENTROPY_WEIGHTS = (0.0, 0.03, 0.1)
DEFAULT_SEEDS = (1, 2)
RUN_PREFIX = "maizels_pca50_bio_prior_ot_flowmap_loss_points_nll"

METRICS = {
    "model_direct_invalid_trajectory_pct": (
        "maizels/model_direct_invalid_trajectory_pct"
    ),
    "model_flowmap_invalid_trajectory_pct": (
        "maizels/model_flowmap_invalid_trajectory_pct"
    ),
    "model_euler_invalid_trajectory_pct": (
        "maizels/model_euler_invalid_trajectory_pct"
    ),
    "direct_emd_mean": "distribution_eval/direct_emd_mean",
    "flowmap_emd_mean": "distribution_eval/flowmap_emd_mean",
    "euler_emd_mean": "distribution_eval/euler_emd_mean",
}
EMD_COLUMNS = (
    "direct_emd_mean",
    "flowmap_emd_mean",
    "euler_emd_mean",
)
PRIMARY_INVALID = "model_direct_invalid_trajectory_pct"
PRIMARY_EMD = "direct_emd_mean"


def value_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


@dataclass(frozen=True)
class RunSpec:
    weight: float
    entropy_weight: float
    seed: int

    @property
    def name(self) -> str:
        return (
            f"{RUN_PREFIX}_w{value_label(self.weight)}"
            f"_ent{value_label(self.entropy_weight)}_seed{self.seed}"
        )


@dataclass
class ActiveRun:
    spec: RunSpec
    gpu: str
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path


def parse_gpus(raw: str) -> list[str]:
    gpus = [item for item in raw.replace(",", " ").split() if item]
    if not gpus:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    return gpus


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-location",
        default=os.getenv(
            "DATASET_LOCATION",
            "/mnt/pdata/hmka3/flow-maps/celltype_classification_pca50_dataset.csv.gz",
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.getenv(
            "OUTPUT_ROOT",
            "/mnt/pdata/hmka3/flow-maps/outputs/maizels_flowmap_loss_points_hparam_sweep",
        ),
    )
    parser.add_argument(
        "--ot-cache-dir",
        default=os.getenv("MAIZELS_OT_CACHE_DIR", ""),
        help=(
            "Directory containing the existing Maizels exact-OT cache. "
            "Defaults to OUTPUT_ROOT/maizels_ot_cache."
        ),
    )
    parser.add_argument(
        "--python-bin", default=os.getenv("PYTHON_BIN", sys.executable)
    )
    parser.add_argument(
        "--cfg-path",
        default="configs.maizels_flowmap_loss_points_hparam_sweep",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=parse_gpus(os.getenv("GPUS", "0,1")),
        help="Comma- or space-separated GPU ids (default: 0,1).",
    )
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--entropy-weights",
        type=float,
        nargs="+",
        default=DEFAULT_ENTROPY_WEIGHTS,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY", ""))
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "self-distill-flow-maps"),
    )
    parser.add_argument(
        "--max-emd",
        "--max-sliced-w2",
        dest="max_emd",
        type=float,
        default=float("inf"),
        help=(
            "Per-seed ceiling applied to all three exact-EMD metrics when "
            "marking a run stable (default: no ceiling). The old "
            "--max-sliced-w2 spelling remains as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--summary-wait-seconds",
        type=int,
        default=180,
        help="Maximum time to wait for final W&B summaries to become visible.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training and regenerate CSV/JSON results from W&B.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    # Adjacent seeds make the initial pair of workers run the two replicates of
    # each setting together; the shared queue still fills whichever GPU frees first.
    return [
        RunSpec(weight, entropy_weight, seed)
        for weight in args.weights
        for entropy_weight in args.entropy_weights
        for seed in args.seeds
    ]


def validate_training_python(args: argparse.Namespace) -> None:
    probe = (
        "import jax, ml_collections, tensorflow, wandb; "
        "assert callable(getattr(wandb, 'init', None)), "
        "'wandb.init is unavailable (wrong environment or W&B is not installed)'; "
        "print('python=' + __import__('sys').executable); "
        "print('wandb=' + str(getattr(wandb, '__file__', None))); "
        "print('wandb_version=' + str(getattr(wandb, '__version__', None)))"
    )
    result = subprocess.run(
        [args.python_bin, "-c", probe],
        cwd=args.repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise SystemExit(
            "Training-environment preflight failed. Activate the flow-maps "
            "Conda environment or pass --python-bin /path/to/env/bin/python.\n\n"
            + result.stdout
        )
    print("Training-environment preflight passed:\n" + result.stdout.rstrip())


def start_run(
    spec: RunSpec,
    gpu: str,
    args: argparse.Namespace,
    log_dir: Path,
) -> ActiveRun:
    log_path = log_dir / f"{spec.name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    command = [
        args.python_bin,
        "py/launchers/learn.py",
        "--cfg_path",
        args.cfg_path,
        "--dataset_location",
        args.dataset_location,
        "--output_folder",
        args.output_root,
        "--slurm_id",
        "0",
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "MAIZELS_SEED": str(spec.seed),
            "MAIZELS_CONSTRAINT_WEIGHT": f"{spec.weight:g}",
            "MAIZELS_ENTROPY_WEIGHT": f"{spec.entropy_weight:g}",
            "MAIZELS_OT_CACHE_DIR": args.ot_cache_dir,
            "WANDB_ENTITY": args.wandb_entity,
            "WANDB_PROJECT": args.wandb_project,
        }
    )
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    print(f"START gpu={gpu} {spec.name} (log: {log_path})", flush=True)
    process = subprocess.Popen(
        command,
        cwd=args.repo_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveRun(spec, gpu, process, log_handle, log_path)


def run_sweep(specs: Iterable[RunSpec], args: argparse.Namespace) -> list[ActiveRun]:
    pending = deque(specs)
    available_gpus = deque(args.gpus)
    active: dict[str, ActiveRun] = {}
    failures: list[ActiveRun] = []
    log_dir = Path(args.output_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Avoid two cache-miss processes writing the same exact-OT plan. When a
        # cache already exists this gate is skipped entirely.
        ot_cache_dir = Path(args.ot_cache_dir)
        if pending and not any(ot_cache_dir.glob("*.npz")):
            warm_gpu = available_gpus.popleft()
            warm_run = start_run(pending.popleft(), warm_gpu, args, log_dir)
            active[warm_gpu] = warm_run
            print("Waiting for the shared exact-OT cache before starting GPU 1...")
            while not any(ot_cache_dir.glob("*.npz")):
                returncode = warm_run.process.poll()
                if returncode is not None:
                    warm_run.log_handle.close()
                    if returncode != 0:
                        failures.append(warm_run)
                        return failures
                    raise RuntimeError(
                        "The warm-up run finished without creating an OT cache."
                    )
                time.sleep(1.0)
            print("Shared exact-OT cache is ready; enabling both GPUs.", flush=True)

        while pending or active:
            while pending and available_gpus:
                gpu = available_gpus.popleft()
                active[gpu] = start_run(pending.popleft(), gpu, args, log_dir)

            completed_gpus = []
            for gpu, run in active.items():
                returncode = run.process.poll()
                if returncode is None:
                    continue
                run.log_handle.close()
                status = "DONE" if returncode == 0 else f"FAILED({returncode})"
                print(f"{status} gpu={gpu} {run.spec.name}", flush=True)
                if returncode != 0:
                    failures.append(run)
                completed_gpus.append(gpu)

            for gpu in completed_gpus:
                del active[gpu]
                available_gpus.append(gpu)

            if active and not completed_gpus:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Interrupted; terminating active training processes.", file=sys.stderr)
        for run in active.values():
            run.process.terminate()
        for run in active.values():
            try:
                run.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                run.process.kill()
            run.log_handle.close()
        raise

    return failures


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def public_wandb_api():
    import wandb

    api_class = getattr(wandb, "Api", None)
    if api_class is None:
        try:
            from wandb.apis.public import Api as api_class
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "The real W&B package is unavailable. Activate the flow-maps "
                "Conda environment before running this launcher."
            ) from exc
    return api_class(timeout=60)


def fetch_runs(
    specs: list[RunSpec], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    expected_names = {spec.name for spec in specs}
    runs = public_wandb_api().runs(
        f"{args.wandb_entity}/{args.wandb_project}",
        filters={"display_name": {"$in": sorted(expected_names)}},
    )
    latest: dict[str, tuple[str, dict[str, Any]]] = {}
    for run in runs:
        if run.name not in expected_names:
            continue
        timestamp = str(getattr(run, "updated_at", ""))
        record = {
            "wandb_id": run.id,
            "wandb_url": run.url,
            "state": run.state,
            **{
                column: finite_float(run.summary.get(wandb_key))
                for column, wandb_key in METRICS.items()
            },
        }
        if run.name not in latest or timestamp > latest[run.name][0]:
            latest[run.name] = (timestamp, record)
    return {name: value[1] for name, value in latest.items()}


def record_complete(record: dict[str, Any]) -> bool:
    return record.get("state") == "finished" and all(
        record.get(column) is not None for column in METRICS
    )


def results_ready(specs: list[RunSpec], runs: dict[str, dict[str, Any]]) -> bool:
    return all(spec.name in runs and record_complete(runs[spec.name]) for spec in specs)


def fetch_with_wait(
    specs: list[RunSpec], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + max(0, args.summary_wait_seconds)
    while True:
        runs = fetch_runs(specs, args)
        if results_ready(specs, runs) or time.monotonic() >= deadline:
            return runs
        ready_count = sum(
            1
            for spec in specs
            if spec.name in runs and record_complete(runs[spec.name])
        )
        print(
            f"Waiting for W&B summaries ({ready_count}/{len(specs)} ready)...",
            flush=True,
        )
        time.sleep(10)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_is_stable(row: dict[str, Any], max_emd: float) -> bool:
    return (
        row["state"] == "finished"
        and all(row[column] is not None for column in METRICS)
        and all(row[column] <= max_emd for column in EMD_COLUMNS)
    )


def summarize(specs: list[RunSpec], args: argparse.Namespace) -> bool:
    runs = fetch_with_wait(specs, args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for spec in specs:
        record = runs.get(spec.name, {})
        row = {
            "weight": spec.weight,
            "entropy_weight": spec.entropy_weight,
            "seed": spec.seed,
            "run_name": spec.name,
            "state": record.get("state", "missing"),
            **{column: record.get(column) for column in METRICS},
            "wandb_id": record.get("wandb_id", ""),
            "wandb_url": record.get("wandb_url", ""),
        }
        row["stable"] = run_is_stable(row, args.max_emd)
        run_rows.append(row)

    run_csv = output_root / "flowmap_loss_points_hparam_runs.csv"
    write_csv(run_csv, run_rows)

    aggregate_rows = []
    for weight in args.weights:
        for entropy_weight in args.entropy_weights:
            setting_rows = [
                row
                for row in run_rows
                if row["weight"] == weight
                and row["entropy_weight"] == entropy_weight
            ]
            complete_rows = [row for row in setting_rows if record_complete(row)]
            stable_rows = [row for row in setting_rows if row["stable"]]
            aggregate = {
                "weight": weight,
                "entropy_weight": entropy_weight,
                "completed_seeds": len(complete_rows),
                "stable_seeds": len(stable_rows),
                "expected_seeds": len(args.seeds),
                "stable": len(stable_rows) == len(args.seeds),
            }
            for column in METRICS:
                values = [row[column] for row in complete_rows]
                aggregate[f"{column}_mean"] = mean(values) if values else None
                aggregate[f"{column}_std"] = pstdev(values) if values else None
            aggregate["rank"] = None
            aggregate_rows.append(aggregate)

    eligible = [row for row in aggregate_rows if row["stable"]]
    eligible.sort(
        key=lambda row: (
            row[f"{PRIMARY_INVALID}_mean"],
            row["model_flowmap_invalid_trajectory_pct_mean"],
            row[f"{PRIMARY_EMD}_mean"],
            row["flowmap_emd_mean_mean"],
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["rank"] = rank

    aggregate_rows.sort(
        key=lambda row: (
            row["rank"] is None,
            row["rank"] if row["rank"] is not None else math.inf,
            row["weight"],
            row["entropy_weight"],
        )
    )
    aggregate_csv = output_root / "flowmap_loss_points_hparam_summary.csv"
    write_csv(aggregate_csv, aggregate_rows)

    best = eligible[0] if eligible else None
    best_json = output_root / "flowmap_loss_points_hparam_best.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "stability_rule": (
                    "both seeds finished with all six finite metrics and each "
                    f"exact EMD <= {args.max_emd:g}"
                ),
                "selection_rule": (
                    "lowest mean direct invalid-trajectory percentage; tie-break "
                    "by composed-flow-map invalid percentage, direct EMD, "
                    "then composed-flow-map EMD"
                ),
                "best": best,
                "run_results_csv": str(run_csv),
                "aggregate_results_csv": str(aggregate_csv),
            },
            handle,
            indent=2,
        )

    print(f"Saved per-run metrics: {run_csv}")
    print(f"Saved aggregate ranking: {aggregate_csv}")
    print(f"Saved best setting: {best_json}")
    if best is None:
        print("No setting passed the two-seed stability rule.", file=sys.stderr)
    else:
        print(
            "BEST "
            f"weight={best['weight']:g} entropy={best['entropy_weight']:g} "
            f"direct_invalid={best[f'{PRIMARY_INVALID}_mean']:.6g} "
            f"direct_emd={best[f'{PRIMARY_EMD}_mean']:.6g}"
        )

    return all(row["completed_seeds"] == len(args.seeds) for row in aggregate_rows)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    if args.ot_cache_dir:
        args.ot_cache_dir = str(Path(args.ot_cache_dir).expanduser().resolve())
    else:
        args.ot_cache_dir = str(Path(args.output_root) / "maizels_ot_cache")
    if not args.wandb_entity:
        raise SystemExit(
            "Set WANDB_ENTITY or pass --wandb-entity so runs can be summarized."
        )
    if os.getenv("WANDB_MODE", "online").lower() == "offline":
        raise SystemExit(
            "Online W&B mode is required for automatic final-metric comparison."
        )

    specs = build_specs(args)
    print(
        f"Sweep: {len(args.weights) * len(args.entropy_weights)} settings, "
        f"{len(args.seeds)} seeds, {len(specs)} runs across GPUs {args.gpus}."
    )

    failures: list[ActiveRun] = []
    if not args.summary_only:
        validate_training_python(args)
        failures = run_sweep(specs, args)
        if failures:
            print("Failed training runs:", file=sys.stderr)
            for run in failures:
                print(f"  {run.spec.name}: {run.log_path}", file=sys.stderr)
            args.summary_wait_seconds = 0

    all_complete = summarize(specs, args)
    return 0 if not failures and all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())

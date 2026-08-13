#!/usr/bin/env python3
"""Run and summarize the Maizels velocity-endpoint hyperparameter sweep.

The launcher keeps one training process on each GPU, dynamically assigning the
next pending run whenever a GPU becomes free. After training, it reads the final
W&B summaries, saves per-run and two-seed aggregate CSV files, and selects the
stable setting with the lowest mean Euler invalid-trajectory percentage (using
Euler sliced-W2 as the tie-breaker).
"""

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


DEFAULT_WEIGHTS = (200.0, 350.0, 550.0, 750.0)
DEFAULT_ENTROPY_WEIGHTS = (0.0, 0.01, 0.1)
DEFAULT_SEEDS = (1, 2)
RUN_PREFIX = "maizels_pca50_bio_prior_ot_velocity_endpoint"
INVALID_METRIC = "maizels/model_euler_invalid_trajectory_pct"
W2_METRIC = "distribution_eval/euler_sliced_w2_mean"


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
            "/mnt/pdata/hmka3/flow-maps/outputs/maizels_velocity_endpoint_hparam_sweep",
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
        default="configs.maizels_velocity_endpoint_hparam_sweep",
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
    parser.add_argument(
        "--wandb-entity", default=os.getenv("WANDB_ENTITY", "")
    )
    parser.add_argument(
        "--wandb-project",
        default=os.getenv("WANDB_PROJECT", "self-distill-flow-maps"),
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
        help="Skip training and regenerate result files from W&B.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    # Keeping seeds adjacent causes the two GPUs to start both replicates of a
    # setting together, while the dynamic queue still avoids idle GPU time.
    return [
        RunSpec(weight, entropy_weight, seed)
        for weight in args.weights
        for entropy_weight in args.entropy_weights
        for seed in args.seeds
    ]


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
        str(args.output_root),
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
        # Every run uses the same exact OT plan. Its cache writer is atomic but
        # intentionally single-writer, so let the first process create it before
        # admitting a second process. GPU 1 joins immediately after the plan is
        # visible; the warm-up run itself continues training on GPU 0.
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
                        print(
                            "The OT cache warm-up run failed; not starting the "
                            "remaining jobs.",
                            file=sys.stderr,
                        )
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


def fetch_runs(
    specs: list[RunSpec], args: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    import wandb

    expected_names = {spec.name for spec in specs}
    api = wandb.Api(timeout=60)
    runs = api.runs(
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
            "model_euler_invalid_trajectory_pct": finite_float(
                run.summary.get(INVALID_METRIC)
            ),
            "euler_sliced_w2_mean": finite_float(run.summary.get(W2_METRIC)),
        }
        if run.name not in latest or timestamp > latest[run.name][0]:
            latest[run.name] = (timestamp, record)
    return {name: value[1] for name, value in latest.items()}


def results_ready(specs: list[RunSpec], runs: dict[str, dict[str, Any]]) -> bool:
    return all(
        spec.name in runs
        and runs[spec.name]["state"] == "finished"
        and runs[spec.name]["model_euler_invalid_trajectory_pct"] is not None
        and runs[spec.name]["euler_sliced_w2_mean"] is not None
        for spec in specs
    )


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
            if spec.name in runs
            and runs[spec.name]["model_euler_invalid_trajectory_pct"] is not None
            and runs[spec.name]["euler_sliced_w2_mean"] is not None
        )
        print(
            f"Waiting for W&B summaries ({ready_count}/{len(specs)} ready)...",
            flush=True,
        )
        time.sleep(10)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(specs: list[RunSpec], args: argparse.Namespace) -> bool:
    runs = fetch_with_wait(specs, args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for spec in specs:
        record = runs.get(spec.name, {})
        run_rows.append(
            {
                "weight": spec.weight,
                "entropy_weight": spec.entropy_weight,
                "seed": spec.seed,
                "run_name": spec.name,
                "state": record.get("state", "missing"),
                "model_euler_invalid_trajectory_pct": record.get(
                    "model_euler_invalid_trajectory_pct"
                ),
                "euler_sliced_w2_mean": record.get("euler_sliced_w2_mean"),
                "wandb_id": record.get("wandb_id", ""),
                "wandb_url": record.get("wandb_url", ""),
            }
        )

    run_csv = output_root / "velocity_endpoint_hparam_runs.csv"
    write_csv(run_csv, run_rows, list(run_rows[0]))

    aggregate_rows = []
    for weight in args.weights:
        for entropy_weight in args.entropy_weights:
            setting_rows = [
                row
                for row in run_rows
                if row["weight"] == weight
                and row["entropy_weight"] == entropy_weight
            ]
            valid_rows = [
                row
                for row in setting_rows
                if row["state"] == "finished"
                and row["model_euler_invalid_trajectory_pct"] is not None
                and row["euler_sliced_w2_mean"] is not None
            ]
            invalid_values = [
                row["model_euler_invalid_trajectory_pct"] for row in valid_rows
            ]
            w2_values = [row["euler_sliced_w2_mean"] for row in valid_rows]
            complete = len(valid_rows) == len(args.seeds)
            aggregate_rows.append(
                {
                    "weight": weight,
                    "entropy_weight": entropy_weight,
                    "completed_seeds": len(valid_rows),
                    "expected_seeds": len(args.seeds),
                    "stable": complete,
                    "model_euler_invalid_trajectory_pct_mean": (
                        mean(invalid_values) if invalid_values else None
                    ),
                    "model_euler_invalid_trajectory_pct_std": (
                        pstdev(invalid_values) if invalid_values else None
                    ),
                    "euler_sliced_w2_mean_mean": mean(w2_values) if w2_values else None,
                    "euler_sliced_w2_mean_std": pstdev(w2_values) if w2_values else None,
                    "rank": None,
                }
            )

    eligible = [row for row in aggregate_rows if row["stable"]]
    eligible.sort(
        key=lambda row: (
            row["model_euler_invalid_trajectory_pct_mean"],
            row["euler_sliced_w2_mean_mean"],
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
    aggregate_csv = output_root / "velocity_endpoint_hparam_summary.csv"
    write_csv(aggregate_csv, aggregate_rows, list(aggregate_rows[0]))

    best = eligible[0] if eligible else None
    best_json = output_root / "velocity_endpoint_hparam_best.json"
    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selection_rule": (
                    "lowest stable two-seed mean model_euler_invalid_trajectory_pct; "
                    "tie-break by lowest mean euler_sliced_w2_mean"
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
    if best is not None:
        print(
            "BEST "
            f"weight={best['weight']:g} entropy={best['entropy_weight']:g} "
            f"invalid_pct={best['model_euler_invalid_trajectory_pct_mean']:.6g} "
            f"euler_sliced_w2={best['euler_sliced_w2_mean_mean']:.6g}"
        )
    else:
        print("No setting has both seeds and both final metrics.", file=sys.stderr)

    return all(row["stable"] for row in aggregate_rows)


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = str(Path(args.output_root).resolve())
    if args.ot_cache_dir:
        args.ot_cache_dir = str(Path(args.ot_cache_dir).expanduser().resolve())
    else:
        args.ot_cache_dir = str(Path(args.output_root) / "maizels_ot_cache")
    if not args.wandb_entity:
        raise SystemExit(
            "Set WANDB_ENTITY or pass --wandb-entity so runs can be logged and summarized."
        )
    if os.getenv("WANDB_MODE", "online").lower() == "offline":
        raise SystemExit(
            "This launcher requires online W&B mode for automatic final-metric comparison."
        )

    specs = build_specs(args)
    print(
        f"Sweep: {len(args.weights) * len(args.entropy_weights)} settings, "
        f"{len(args.seeds)} seeds, {len(specs)} runs across GPUs {args.gpus}."
    )

    failures: list[ActiveRun] = []
    if not args.summary_only:
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

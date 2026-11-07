#!/usr/bin/env python3
"""Run and summarize the cross-dataset SSFM endpoint-EMD comparison.

The comparison is intentionally fixed to three full stochastic flow maps:

* ID 2: endpoint-filtered SSFM with the differentiable lineage constraint;
* ID 5: plain minibatch-OT SSFM; and
* ID 3: endpoint-filtered minibatch-OT SSFM.

All endpoint-preservation metrics use composed stochastic flow-map sampling.
For the final observed marginal, the evaluator compares (i) restarting from
the nearest retained observation and (ii) carrying samples from the base
distribution through the complete retained schedule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep this ordering in the commands and final tables.
VARIANTS = (
    (2, "constrained_ssfm"),
    (5, "ot_ssfm"),
    (3, "filtered_ot_ssfm"),
)
VARIANT_LABELS = dict(VARIANTS)
SLURM_IDS = ",".join(str(slurm_id) for slurm_id, _ in VARIANTS)

RESULT_FIELDS = (
    "dataset",
    "heldout_day",
    "time_mode",
    "method",
    "variant_name",
    "slurm_id",
    "seed",
    "source_protocol",
    "source_day",
    "target_day",
    "sampling_method",
    "composition_steps",
    "learning_rate",
    "constraint_weight",
    "n_noise_draws",
    "emd",
    "emd_std_over_noise",
    "best_step",
    "metrics_json",
)

SUMMARY_FIELDS = (
    "dataset",
    "heldout_day",
    "time_mode",
    "method",
    "variant_name",
    "slurm_id",
    "source_protocol",
    "source_day",
    "target_day",
    "sampling_method",
    "composition_steps",
    "learning_rate",
    "constraint_weight",
    "n_noise_draws",
    "seeds",
    "n_seeds",
    "emd_mean_across_seeds",
    "emd_std_across_seeds",
)


def _parse_csv_ints(value: str, *, name: str, allowed=None) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated list of integers."
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} contains duplicate values.")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must be non-negative.")
    if allowed is not None:
        invalid = [item for item in values if item not in allowed]
        if invalid:
            raise argparse.ArgumentTypeError(
                f"{name} must contain only {sorted(allowed)}, got {invalid}."
            )
    return values


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare constrained, plain-OT, and filtered-OT stochastic flow "
            "maps on final-marginal preservation across Maizels, CITE, and Multi."
        )
    )
    parser.add_argument(
        "--maizels-dataset-location",
        "--maizels_dataset_location",
        default=os.getenv(
            "MAIZELS_DATASET_PATH",
            str(REPO_ROOT / "celltype_classification_pca50_dataset.csv.gz"),
        ),
    )
    parser.add_argument(
        "--cite-multi-dataset-location",
        "--cite_multi_dataset_location",
        default=os.getenv(
            "CITE_MULTI_DATA_DIR",
            str(Path.home() / "Desktop" / "flow-maps-data"),
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stochastic_end_distribution_comparison",
    )
    parser.add_argument(
        "--seeds",
        default="0,1",
        help="Exactly two seeds by default (0,1).",
    )
    parser.add_argument(
        "--heldout-days",
        "--heldout_days",
        default="3,4",
        help="CITE/Multi omitted-day experiments to run (default: 3,4).",
    )
    parser.add_argument(
        "--learning-rate", "--learning_rate", type=float, default=1e-3
    )
    parser.add_argument(
        "--constraint-weight", "--constraint_weight", type=float, default=10.0
    )
    parser.add_argument(
        "--maizels-schedule",
        "--maizels_schedule",
        choices=("d3_d3p8_d8",),
        default="d3_d3p8_d8",
    )
    parser.add_argument(
        "--maizels-time-mode",
        "--maizels_time_mode",
        choices=("real_time", "equal_time"),
        default="real_time",
    )
    parser.add_argument(
        "--cite-multi-time-mode",
        "--cite_multi_time_mode",
        choices=("equal_time", "real_time"),
        default="equal_time",
    )
    parser.add_argument("--maizels-classifier-path", default=None)
    parser.add_argument("--total-steps", "--total_steps", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--n-pairs", "--n_pairs", type=int, default=None)
    parser.add_argument(
        "--validation-frequency", "--validation_frequency", type=int, default=None
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval-noise-draws", "--eval_noise_draws", type=int, default=None
    )
    parser.add_argument(
        "--eval-max-points", "--eval_max_points", type=int, default=None
    )
    parser.add_argument(
        "--eval-flowmap-steps", "--eval_flowmap_steps", type=int, default=None
    )
    parser.add_argument(
        "--visual-frequency", "--visual_frequency", type=int, default=None
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
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild the combined CSVs from already completed runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.seed_values = _parse_csv_ints(args.seeds, name="seeds")
    if len(args.seed_values) != 2:
        parser.error("--seeds must contain exactly two seeds for this comparison.")
    args.heldout_day_values = _parse_csv_ints(
        args.heldout_days,
        name="heldout days",
        allowed={3, 4},
    )
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive.")
    if args.constraint_weight < 0.0:
        parser.error("--constraint-weight must be non-negative.")
    for name in (
        "total_steps",
        "batch_size",
        "n_pairs",
        "validation_frequency",
        "eval_noise_draws",
        "eval_flowmap_steps",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be non-negative.")
    if args.eval_max_points is not None and args.eval_max_points < 0:
        parser.error("--eval-max-points must be non-negative.")
    if args.visual_frequency is not None and args.visual_frequency < 0:
        parser.error("--visual-frequency must be non-negative.")
    return args


def _append_optional(command: List[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _common_runner_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    for flag, value in (
        ("--total-steps", args.total_steps),
        ("--batch-size", args.batch_size),
        ("--n-pairs", args.n_pairs),
        ("--validation-frequency", args.validation_frequency),
        ("--early-stopping-patience", args.early_stopping_patience),
        ("--eval-noise-draws", args.eval_noise_draws),
        ("--eval-max-points", args.eval_max_points),
        ("--eval-flowmap-steps", args.eval_flowmap_steps),
        ("--visual-frequency", args.visual_frequency),
        ("--wandb-mode", args.wandb_mode),
    ):
        _append_optional(flags, flag, value)
    if args.rerun_completed:
        flags.append("--rerun-completed")
    if args.fail_fast:
        flags.append("--fail-fast")
    if args.dry_run:
        flags.append("--dry-run")
    return flags


def _commands(args: argparse.Namespace, output_root: Path):
    python = str(Path(args.python).expanduser())
    seeds = ",".join(str(seed) for seed in args.seed_values)
    common = _common_runner_flags(args)

    maizels = [
        python,
        str(REPO_ROOT / "scripts" / "run_maizels_stochastic_multiseed.py"),
        "--slurm-ids",
        SLURM_IDS,
        "--seeds",
        seeds,
        "--dataset-location",
        str(Path(args.maizels_dataset_location).expanduser()),
        "--output-dir",
        str(output_root / "maizels"),
        "--maizels-schedule",
        args.maizels_schedule,
        "--maizels-time-mode",
        args.maizels_time_mode,
        "--learning-rate",
        f"{args.learning_rate:g}",
        "--constraint-weight",
        f"{args.constraint_weight:g}",
        "--python",
        python,
    ]
    _append_optional(
        maizels, "--classifier-path", args.maizels_classifier_path
    )
    maizels.extend(common)

    cite_multi = [
        python,
        str(REPO_ROOT / "scripts" / "run_cite_multi_stochastic_multiseed.py"),
        "--slurm-ids",
        SLURM_IDS,
        "--seeds",
        seeds,
        "--datasets",
        "cite,multi",
        "--heldout-days",
        ",".join(str(day) for day in args.heldout_day_values),
        "--cite-multi-time-mode",
        args.cite_multi_time_mode,
        "--dataset-location",
        str(Path(args.cite_multi_dataset_location).expanduser()),
        "--output-dir",
        str(output_root / "cite_multi"),
        "--learning-rate",
        f"{args.learning_rate:g}",
        "--constraint-weight",
        f"{args.constraint_weight:g}",
        "--python",
        python,
    ]
    cite_multi.extend(common)
    return (("Maizels", maizels), ("CITE/Multi", cite_multi))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_matches(value, expected: float) -> bool:
    return _finite(value) and math.isclose(
        float(value), float(expected), rel_tol=1e-12, abs_tol=1e-15
    )


def _load_metrics(path_value: str) -> Dict[str, object]:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    with path.open() as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return values


def _dataset_protocols(dataset: str, heldout_day: str):
    if dataset == "maizels":
        return {
            "nearest_observed": {
                "source_day": "D3.8",
                "target_day": "D8",
                "metric_prefix": "final_eval/observed_D3p8_to_D8_flowmap",
            },
            "base_rollout": {
                "source_day": "D3",
                "target_day": "D8",
                "metric_prefix": "final_eval/D3_to_D8_rollout_flowmap",
            },
        }
    nearest = "D4" if str(heldout_day) == "3" else "D3"
    nearest_tag = nearest.lower().replace("d", "day", 1)
    return {
        "nearest_observed": {
            "source_day": nearest,
            "target_day": "D7",
            "metric_prefix": f"final_eval/observed_{nearest_tag}_to_day7_flowmap",
        },
        "base_rollout": {
            "source_day": "D2",
            "target_day": "D7",
            "metric_prefix": "final_eval/day2_to_day7_rollout_flowmap",
        },
    }


def _metric_value(
    metrics: Mapping[str, object],
    *,
    metric_prefix: str,
    suffix: str,
):
    value = metrics.get(f"{metric_prefix}_{suffix}")
    if _finite(value):
        return float(value)
    return ""


def _selected_result_rows(args: argparse.Namespace, output_root: Path):
    selected_seeds = set(args.seed_values)
    selected_holdouts = {str(value) for value in args.heldout_day_values}
    sources = (
        ("maizels", output_root / "maizels" / "results.csv"),
        ("cite_multi", output_root / "cite_multi" / "results.csv"),
    )
    for source, path in sources:
        for row in _read_csv(path):
            if row.get("status") != "complete":
                continue
            try:
                slurm_id = int(row.get("slurm_id", ""))
                seed = int(row.get("seed", ""))
            except ValueError:
                continue
            if slurm_id not in VARIANT_LABELS or seed not in selected_seeds:
                continue
            if not _float_matches(row.get("learning_rate"), args.learning_rate):
                continue
            if slurm_id == 2 and not _float_matches(
                row.get("constraint_weight"), args.constraint_weight
            ):
                continue
            if source == "maizels":
                if row.get("maizels_schedule") != args.maizels_schedule:
                    continue
                if row.get("maizels_time_mode") != args.maizels_time_mode:
                    continue
            else:
                if row.get("heldout_day") not in selected_holdouts:
                    continue
                if row.get("cite_multi_time_mode") != args.cite_multi_time_mode:
                    continue
            yield source, row, slurm_id, seed


def _collect_results(args: argparse.Namespace, output_root: Path):
    collected = {}
    warnings = []
    for source, row, slurm_id, seed in _selected_result_rows(args, output_root):
        dataset = "maizels" if source == "maizels" else str(row["dataset_name"])
        heldout_day = "" if dataset == "maizels" else str(row["heldout_day"])
        try:
            metrics = _load_metrics(str(row["metrics_json"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {row.get('metrics_json')}: {exc}")
            continue

        for protocol, metadata in _dataset_protocols(dataset, heldout_day).items():
            metric_prefix = str(metadata["metric_prefix"])
            values = {
                "emd": _metric_value(
                    metrics,
                    metric_prefix=metric_prefix,
                    suffix="emd",
                ),
                "emd_std_over_noise": _metric_value(
                    metrics,
                    metric_prefix=metric_prefix,
                    suffix="emd_std_over_noise",
                ),
            }
            missing = [name for name, value in values.items() if value == ""]
            if missing:
                warnings.append(
                    f"{dataset}, holdout={heldout_day or 'n/a'}, "
                    f"ID={slurm_id}, seed={seed}, protocol={protocol}: "
                    f"missing composed-flow-map {missing}. Re-run with "
                    "--rerun-completed if the stored result predates the "
                    "observed-marginal evaluation."
                )
            result = {
                "dataset": dataset,
                "heldout_day": heldout_day,
                "time_mode": (
                    args.maizels_time_mode
                    if dataset == "maizels"
                    else args.cite_multi_time_mode
                ),
                "method": VARIANT_LABELS[slurm_id],
                "variant_name": row.get("variant_name", ""),
                "slurm_id": slurm_id,
                "seed": seed,
                "source_protocol": protocol,
                "source_day": metadata["source_day"],
                "target_day": metadata["target_day"],
                "sampling_method": "composed_flow_map",
                "composition_steps": (
                    args.eval_flowmap_steps
                    if args.eval_flowmap_steps is not None
                    else 50 if dataset == "maizels" else 100
                ),
                "learning_rate": args.learning_rate,
                "constraint_weight": (
                    args.constraint_weight if slurm_id == 2 else ""
                ),
                "n_noise_draws": (
                    args.eval_noise_draws
                    if args.eval_noise_draws is not None
                    else 3
                ),
                **values,
                "best_step": metrics.get("training/best_step", ""),
                "metrics_json": row["metrics_json"],
            }
            key = (dataset, heldout_day, slurm_id, seed, protocol)
            collected[key] = result
    return list(collected.values()), warnings


def _sort_key(row: Mapping[str, object]):
    dataset_order = {"maizels": 0, "cite": 1, "multi": 2}
    method_order = {name: index for index, (_, name) in enumerate(VARIANTS)}
    protocol_order = {"nearest_observed": 0, "base_rollout": 1}
    return (
        dataset_order.get(str(row["dataset"]), 99),
        str(row["heldout_day"]),
        method_order.get(str(row["method"]), 99),
        int(row.get("seed", 0)),
        protocol_order.get(str(row["source_protocol"]), 99),
    )


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _mean_and_std(values: Sequence[float]):
    if not values:
        return "", ""
    return (
        statistics.fmean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def _summarize(rows: Sequence[Mapping[str, object]]):
    group_fields = SUMMARY_FIELDS[:14]
    grouped: Dict[tuple, List[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in grouped.items():
        metadata = dict(zip(group_fields, key))
        emd = [float(row["emd"]) for row in group if _finite(row["emd"])]
        emd_mean, emd_std = _mean_and_std(emd)
        summaries.append(
            {
                **metadata,
                "seeds": ",".join(
                    str(row["seed"])
                    for row in sorted(group, key=lambda item: int(item["seed"]))
                ),
                "n_seeds": len({int(row["seed"]) for row in group}),
                "emd_mean_across_seeds": emd_mean,
                "emd_std_across_seeds": emd_std,
            }
        )
    return summaries


def _print_summary(rows: Sequence[Mapping[str, object]]) -> None:
    print("\nFinal composed-flow-map endpoint comparison:")
    for row in rows:
        holdout = f"/holdout-D{row['heldout_day']}" if row["heldout_day"] else ""
        emd = row["emd_mean_across_seeds"]
        emd_text = f"{float(emd):.6g}" if _finite(emd) else "missing"
        print(
            f"  {row['dataset']}{holdout} | {row['method']} | "
            f"{row['source_protocol']}: EMD={emd_text}"
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_dir).expanduser().resolve()
    failures = []

    if not args.aggregate_only:
        print(
            "Comparison: IDs 2,5,3; learning rate="
            f"{args.learning_rate:g}; constraint weight={args.constraint_weight:g}; "
            f"seeds={','.join(str(seed) for seed in args.seed_values)}."
        )
        print(
            "Endpoint evaluation uses composed stochastic flow-map sampling "
            "for both nearest-observed and base-rollout protocols."
        )
        for label, command in _commands(args, output_root):
            print(f"\n=== {label} ===")
            result = subprocess.run(command, cwd=REPO_ROOT, check=False)
            if result.returncode != 0:
                failures.append((label, result.returncode))
                if args.fail_fast:
                    break

    if args.dry_run:
        return 1 if failures else 0

    results, warnings = _collect_results(args, output_root)
    results = sorted(results, key=_sort_key)
    summaries = sorted(_summarize(results), key=_sort_key)
    results_path = output_root / "end_distribution_results.csv"
    summary_path = output_root / "end_distribution_summary.csv"
    _write_csv(results_path, RESULT_FIELDS, results)
    _write_csv(summary_path, SUMMARY_FIELDS, summaries)

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if summaries:
        _print_summary(summaries)
    print(f"\nPer-seed results: {results_path}")
    print(f"Across-seed summary: {summary_path}")

    expected_settings = 1 + 2 * len(args.heldout_day_values)
    expected_rows = expected_settings * len(VARIANTS) * len(args.seed_values) * 2
    incomplete = len(results) != expected_rows or bool(warnings)
    if len(results) != expected_rows:
        print(
            f"WARNING: collected {len(results)}/{expected_rows} expected "
            "dataset/method/seed/protocol rows.",
            file=sys.stderr,
        )
    if failures:
        detail = ", ".join(f"{label}={code}" for label, code in failures)
        print(f"WARNING: child runner failures: {detail}", file=sys.stderr)
    return 1 if failures or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

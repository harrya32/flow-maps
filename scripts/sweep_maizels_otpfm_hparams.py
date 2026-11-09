#!/usr/bin/env python3
"""Resumable OTP-FM grid selected by best-checkpoint EMD at D3.4 and D6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mfm_maizels_sweep_utils as sweep


DEFAULT_LEARNING_RATES = (3e-4, 1e-3, 3e-3)
DEFAULT_STRENGTHS = (100.0, 300.0, 1000.0)
DEFAULT_GAUSSIAN_WIDTHS = (0.05, 0.1, 0.2)
HPARAM_VAL_TIMES = ("D3.4", "D6")
HPARAM_KEYS = ("lr", "strength", "lambda_type", "lambda_width")


def _grid_text(values):
    return ",".join(f"{value:g}" for value in values)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-location", "--dataset-path", required=True)
    parser.add_argument(
        "--config-path", type=Path, help="Optional base JSON overrides."
    )
    parser.add_argument("--output-dir", default="outputs/maizels_otpfm_hparam_sweep")
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument(
        "--seeds", default="0,1", help="Comma-separated training seeds."
    )
    parser.add_argument("--learning-rates", default=_grid_text(DEFAULT_LEARNING_RATES))
    parser.add_argument("--strengths", default=_grid_text(DEFAULT_STRENGTHS))
    parser.add_argument("--kernels", default="delta,gaussian")
    parser.add_argument(
        "--gaussian-widths", "--widths", default=_grid_text(DEFAULT_GAUSSIAN_WIDTHS)
    )
    parser.add_argument(
        "--potential", type=str.lower, choices=("w2inf", "w2"), default=None
    )
    parser.add_argument(
        "--ot-coupling",
        "--ot_coupling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ot-method", choices=("emd", "sinkhorn"), default=None)
    parser.add_argument("--ot-reg", type=float, default=None)
    parser.add_argument(
        "--maizels-schedule", choices=("d3_d3p8_d8", "d3_d8"), default="d3_d3p8_d8"
    )
    parser.add_argument(
        "--maizels-time-mode", choices=("real_time", "equal_time"), default="real_time"
    )
    parser.add_argument(
        "--objective-sampler", choices=("flowmap", "euler", "direct"), default="flowmap"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=0,
        help="0 evaluates only the validation-selected checkpoint (default).",
    )
    parser.add_argument("--eval-max-points", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--classifier-path", default="")
    parser.add_argument("--full-data-classifier-path", default="")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default=None
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use one training/validation batch and capped final evaluation.",
    )
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def parameter_grid(args):
    kernels = tuple(
        item.strip().lower() for item in args.kernels.split(",") if item.strip()
    )
    if (
        not kernels
        or len(set(kernels)) != len(kernels)
        or set(kernels) - {"delta", "gaussian"}
    ):
        raise ValueError(
            "kernels must be a non-empty, unique list of delta and/or gaussian."
        )
    base = {
        "lr": sweep.parse_float_grid(args.learning_rates, "learning rates"),
        "strength": sweep.parse_float_grid(args.strengths, "strengths"),
    }
    widths = (
        sweep.parse_float_grid(args.gaussian_widths, "Gaussian widths")
        if "gaussian" in kernels
        else ()
    )
    for name, values in (*base.items(), ("Gaussian widths", widths)):
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{name} must be finite and positive.")
    grid = []
    for values in sweep.build_grid(base):
        for kernel in kernels:
            # Delta's ignored width is canonicalized, never crossed with the
            # Gaussian width grid, so it cannot generate duplicate trials.
            for width in widths if kernel == "gaussian" else (0.0,):
                grid.append({**values, "lambda_type": kernel, "lambda_width": width})
    return grid


def resolved_base_config(args):
    # Reuse the training runner's preset selection rather than reproducing
    # the 80/200-epoch, coupling, and solver rules in the sweep.
    from otpfm_baseline.train import experiment_config, parse_args as train_args

    argv = ["--dataset_name", "maizels"]
    for field, option in (
        ("config_path", "--config_path"),
        ("potential", "--potential"),
        ("ot_method", "--ot_method"),
        ("ot_reg", "--ot_reg"),
        ("epochs", "--epochs"),
        ("batch_size", "--batch_size"),
    ):
        value = getattr(args, field)
        if value is not None:
            argv.extend((option, str(value)))
    if args.ot_coupling is not None:
        argv.append("--ot_coupling" if args.ot_coupling else "--no-ot_coupling")
    if args.smoke_test:
        argv.append("--smoke_test")
    config = experiment_config(train_args(argv))
    # These aliases take precedence over the scalar values in the upstream
    # factory; remove them so every requested grid dimension actually applies.
    for key in ("strengths", "widths", "width"):
        config.pop(key, None)
    return config


def derived_config(base_config, hparams, setting_id, run_id, seed, objective):
    return {
        **base_config,
        **hparams,
        "seed": int(seed),
        "sweep_setting_id": setting_id,
        "sweep_run_id": run_id,
        "sweep_objective_metric": objective,
        "sweep_hparam_val_times": list(HPARAM_VAL_TIMES),
    }


def build_command(args, config_path, output_root, run_id, seed):
    command = [
        str(Path(args.python).expanduser()),
        "-m",
        "otpfm_baseline.train",
        "--dataset_name",
        "maizels",
        "--dataset_location",
        str(Path(args.dataset_location).expanduser().resolve()),
        "--config_path",
        str(config_path),
        "--working_dir",
        str(output_root),
        "--run_name",
        run_id,
        "--seeds",
        str(seed),
        "--maizels_schedule",
        args.maizels_schedule,
        "--maizels_time_mode",
        args.maizels_time_mode,
        "--device",
        args.device,
        "--eval_every_epochs",
        str(args.eval_every_epochs),
        "--eval_max_points",
        str(args.eval_max_points),
        "--eval_batch_size",
        str(args.eval_batch_size),
    ]
    for field, option in (
        ("classifier_path", "--classifier_path"),
        ("full_data_classifier_path", "--full_data_classifier_path"),
        ("wandb_project", "--wandb_project"),
        ("wandb_entity", "--wandb_entity"),
        ("wandb_mode", "--wandb_mode"),
    ):
        value = getattr(args, field)
        if value is not None and (value or field == "wandb_entity"):
            command.extend((option, str(value)))
    if args.smoke_test:
        command.append("--smoke_test")
    return command


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_metrics(path, objective):
    metrics = json.loads(path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError("Final metrics must be a JSON object.")
    value = metrics.get(objective)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"Missing or non-finite objective metric {objective}.")
    return metrics


def run_sweep(args):
    seeds = sweep.parse_seed_grid(args.seeds)
    grid = parameter_grid(args)
    if args.maizels_schedule != "d3_d3p8_d8":
        raise ValueError(
            "Strength/kernel sweeps require retained D3.8; d3_d8 has no intermediate potential."
        )
    if (
        args.eval_every_epochs < 0
        or args.eval_max_points < 0
        or args.eval_batch_size <= 0
    ):
        raise ValueError(
            "Evaluation intervals/caps must be non-negative and batch size positive."
        )
    base = resolved_base_config(args)
    from otpfm_baseline import UPSTREAM_ROOT
    from otpfm_baseline.data import _maizels_path

    dataset_path = _maizels_path(args.dataset_location)
    if not dataset_path.is_file() and not args.dry_run:
        raise FileNotFoundError(f"Maizels dataset not found: {dataset_path}")
    base_path = (
        (
            args.config_path
            or UPSTREAM_ROOT
            / "configs"
            / "citeseq"
            / f"{base['upstream_config_preset']}.json"
        )
        .expanduser()
        .resolve()
    )
    # Hash all effective preset values, including overrides and variant, rather
    # than only the optional JSON file, so changed training defaults invalidate resume IDs.
    base_digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    objective = f"final_eval/{args.objective_sampler}_mean_emd_hparam_val_times"
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
    runtime = {
        "eval_every_epochs": args.eval_every_epochs,
        "eval_max_points": args.eval_max_points,
        "eval_batch_size": args.eval_batch_size,
        "device": args.device,
        "full_data_classifier_path": (
            str(Path(args.full_data_classifier_path).expanduser().resolve())
            if args.full_data_classifier_path
            else ""
        ),
        "smoke_test": args.smoke_test,
    }
    settings, configs = [], {}
    for hparams in grid:
        setting_id = sweep.setting_id_for(
            method="otpfm",
            base_config_path=base_path,
            base_config_digest=base_digest,
            dataset_path=dataset_path,
            classifier_path=(
                Path(args.classifier_path).expanduser().resolve()
                if args.classifier_path
                else None
            ),
            schedule=args.maizels_schedule,
            time_mode=args.maizels_time_mode,
            hparam_val_times=HPARAM_VAL_TIMES,
            objective_metric=objective,
            hparams=hparams,
            runtime_overrides=runtime,
            slug_keys=HPARAM_KEYS,
        )
        configs[setting_id] = hparams
        settings.append(
            {
                "setting_id": setting_id,
                "method": "otpfm",
                "base_config": str(base_path),
                "base_config_sha256": base_digest,
                "maizels_schedule": args.maizels_schedule,
                "maizels_time_mode": args.maizels_time_mode,
                "hparam_val_times": ",".join(HPARAM_VAL_TIMES),
                "potential": base["potential"],
                "ot_coupling": base["ot_coupling"],
                "epochs": base["epochs"],
                **hparams,
                **runtime,
            }
        )

    counts = {
        kernel: sum(row["lambda_type"] == kernel for row in grid)
        for kernel in ("delta", "gaussian")
    }
    print(
        f"OTP-FM: potential={base['potential']}, coupling={'OT' if base['ot_coupling'] else 'independent'}, "
        f"schedule={args.maizels_schedule}, clock={args.maizels_time_mode}."
    )
    print(
        f"Learning rates: {sorted({row['lr'] for row in grid})}; strengths: {sorted({row['strength'] for row in grid})}."
    )
    print(
        f"Kernel settings: {counts['delta']} delta (width ignored), {counts['gaussian']} Gaussian "
        f"(widths {sorted({row['lambda_width'] for row in grid if row['lambda_type'] == 'gaussian'})})."
    )
    print(
        f"{len(grid)} settings x {len(seeds)} seeds {list(seeds)} = {len(grid) * len(seeds)} runs; "
        f"{base['epochs']} epochs/run; eval_every_epochs={args.eval_every_epochs}."
    )
    print(
        f"Checkpoint selection: retained-day val_loss; sweep objective: {objective} (D3.4/D6 mean EMD)."
    )
    if not args.dry_run:
        _atomic_json(
            output_root / "sweep_plan.json",
            {
                "settings": settings,
                "seeds": list(seeds),
                "n_settings": len(grid),
                "n_runs": len(grid) * len(seeds),
                "objective_metric": objective,
                "effective_base_config": base,
            },
        )
    rows = list(sweep.read_rows(results_path))
    failures, index = 0, 0
    for setting in settings:
        setting_id = setting["setting_id"]
        for seed in seeds:
            index += 1
            run_id = sweep.run_id_for(setting_id, seed)
            run_root = output_root / run_id
            config_path, metrics_path = (
                run_root / "sweep_config.json",
                run_root / "final_metrics.json",
            )
            completed = any(
                row.get("run_id") == run_id and row.get("status") == "complete"
                for row in rows
            )
            if completed and not args.rerun_completed:
                try:
                    _read_metrics(metrics_path, objective)
                    print(
                        f"[{index}/{len(grid) * len(seeds)}] Skipping completed {run_id}."
                    )
                    continue
                except (OSError, ValueError):
                    pass
            config = derived_config(
                base, configs[setting_id], setting_id, run_id, seed, objective
            )
            command = build_command(args, config_path, output_root, run_id, seed)
            print(f"[{index}/{len(grid) * len(seeds)}] {run_id}")
            print(shlex.join(command))
            if args.dry_run:
                continue
            _atomic_json(config_path, config)
            if metrics_path.exists():
                metrics_path.unlink()
            environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
            started = time.monotonic()
            result = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, check=False
            )
            metrics, error, status = {}, "", "failed"
            if result.returncode == 0:
                try:
                    metrics = _read_metrics(metrics_path, objective)
                    status = "complete"
                except (OSError, ValueError) as exc:
                    error = f"Could not read final objective: {exc}"
            else:
                error = f"Training exited with status {result.returncode}."
            row = {
                "run_id": run_id,
                **setting,
                "status": status,
                "seed": seed,
                "objective_metric": objective,
                "objective_value": metrics.get(objective, ""),
                "duration_seconds": round(time.monotonic() - started, 3),
                "return_code": result.returncode,
                "run_output_folder": str(run_root),
                "derived_config": str(config_path),
                "metrics_json": str(metrics_path),
                "error": error,
                **sweep._metric_row(metrics),
            }
            sweep.upsert_row(rows, row)
            sweep.write_rows(results_path, rows)
            sweep.write_summary_rows(
                summary_path, sweep.summarize_settings(rows, settings, seeds, objective)
            )
            print(f"Recorded {status} result in {results_path}.")
            if status != "complete":
                failures += 1
                if args.fail_fast:
                    return result.returncode or 1

    if args.dry_run:
        return 0
    summaries = sweep.summarize_settings(rows, settings, seeds, objective)
    sweep.write_summary_rows(summary_path, summaries)
    complete = [
        row
        for row in summaries
        if row["status"] == "complete"
        and sweep._is_finite_number(row["objective_mean"])
    ]
    if not complete:
        print("No complete hyperparameter setting is available.")
        return 1
    best = min(complete, key=lambda row: float(row["objective_mean"]))
    selected = configs[best["setting_id"]]
    _atomic_json(output_root / "best_config.json", {**base, **selected})
    _atomic_json(
        output_root / "best_setting.json",
        {
            **best,
            "selected_hparams": selected,
            "config_path": str(output_root / "best_config.json"),
        },
    )
    print(f"Best mean D3.4/D6 EMD={best['objective_mean']:.8g}: {best['setting_id']}")
    print(f"Selected config: {output_root / 'best_config.json'}")
    return 1 if failures else 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    return run_sweep(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

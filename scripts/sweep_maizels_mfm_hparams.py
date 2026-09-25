#!/usr/bin/env python3
"""Sweep metric-flow-matching hyperparameters on held-out Maizels days."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mfm_maizels_sweep_utils as sweep


# The CITE/Multi MFM configs inherit the parser's flow_lr default of 1e-3.
DEFAULT_FLOW_LEARNING_RATES = (1e-3,)
DEFAULT_FLOW_WEIGHT_DECAYS = (1e-5,)
DEFAULT_GEOPATH_LEARNING_RATES = (1e-4,)
DEFAULT_GEOPATH_WEIGHT_DECAYS = (1e-5,)
DEFAULT_METRIC_LEARNING_RATES = (1e-2,)
DEFAULT_RHOS = (-1.5, -2.75, -5.0)
DEFAULT_KAPPAS = (1.0, 1.5, 2.0)
DEFAULT_ALPHA_METRICS = (1.0,)
DEFAULT_N_CENTERS = (150,)
DEFAULT_OBJECTIVE = "final_eval/euler_mean_emd_hparam_val_times"


def _grid_text(values) -> str:
    return ",".join(f"{value:g}" for value in values)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable MFM grid on Maizels and select by mean exact "
            "Euler EMD at the held-out validation days."
        )
    )
    sweep.add_common_arguments(
        parser,
        default_config=(
            "metric-flow-matching/configs/single_cell/50dims/"
            "i-mfm_maizels_3marginal.yaml"
        ),
        default_output_dir="outputs/maizels_mfm_hparam_sweep",
        default_objective_metric=DEFAULT_OBJECTIVE,
    )
    parser.add_argument(
        "--flow-learning-rates", default=_grid_text(DEFAULT_FLOW_LEARNING_RATES)
    )
    parser.add_argument(
        "--flow-weight-decays", default=_grid_text(DEFAULT_FLOW_WEIGHT_DECAYS)
    )
    parser.add_argument(
        "--geopath-learning-rates",
        default=_grid_text(DEFAULT_GEOPATH_LEARNING_RATES),
    )
    parser.add_argument(
        "--geopath-weight-decays",
        default=_grid_text(DEFAULT_GEOPATH_WEIGHT_DECAYS),
    )
    parser.add_argument(
        "--metric-learning-rates",
        default=_grid_text(DEFAULT_METRIC_LEARNING_RATES),
    )
    parser.add_argument(
        "--rhos",
        default=_grid_text(DEFAULT_RHOS),
        help=(
            "RBF metric epsilon controls. Negative values use MFM's adaptive "
            "epsilon rule; pass this option as --rhos=-1.5,-2.75,-5."
        ),
    )
    parser.add_argument("--kappas", default=_grid_text(DEFAULT_KAPPAS))
    parser.add_argument("--alpha-metrics", default=_grid_text(DEFAULT_ALPHA_METRICS))
    parser.add_argument("--n-centers", default=_grid_text(DEFAULT_N_CENTERS))
    parser.add_argument("--metric-epochs", type=int, default=None)
    parser.add_argument("--metric-train-batches", type=int, default=None)
    parser.add_argument("--metric-val-batches", type=int, default=None)
    parser.add_argument("--metric-patience", type=int, default=None)
    parser.add_argument("--patience-geopath", type=int, default=None)
    parser.add_argument("--reference-loss-batches", type=int, default=None)
    return parser.parse_args(argv)


def parameter_grid(args: argparse.Namespace) -> Dict[str, tuple]:
    grid = {
        "flow_lr": sweep.parse_float_grid(
            args.flow_learning_rates, "flow learning rates"
        ),
        "flow_weight_decay": sweep.parse_float_grid(
            args.flow_weight_decays, "flow weight decays"
        ),
        "geopath_lr": sweep.parse_float_grid(
            args.geopath_learning_rates, "geopath learning rates"
        ),
        "geopath_weight_decay": sweep.parse_float_grid(
            args.geopath_weight_decays, "geopath weight decays"
        ),
        "metric_lr": sweep.parse_float_grid(
            args.metric_learning_rates, "metric learning rates"
        ),
        "rho": sweep.parse_float_grid(args.rhos, "rho values"),
        "kappa": sweep.parse_float_grid(args.kappas, "kappa values"),
        "alpha_metric": sweep.parse_float_grid(args.alpha_metrics, "metric exponents"),
        "n_centers": sweep.parse_int_grid(args.n_centers, "RBF center counts"),
    }
    positive = ("flow_lr", "geopath_lr", "metric_lr", "kappa", "alpha_metric")
    for name in positive:
        if any(value <= 0.0 for value in grid[name]):
            raise ValueError(f"{name} values must be positive.")
    for name in ("flow_weight_decay", "geopath_weight_decay"):
        if any(value < 0.0 for value in grid[name]):
            raise ValueError(f"{name} values must be non-negative.")
    if any(value == 0.0 for value in grid["rho"]):
        raise ValueError("rho values must be non-zero.")
    if any(value <= 1 for value in grid["n_centers"]):
        raise ValueError("RBF center counts must be greater than one.")
    return grid


def extra_runtime_overrides(args: argparse.Namespace) -> Dict[str, object]:
    values = {
        "metric_epochs": args.metric_epochs,
        "metric_train_batches": args.metric_train_batches,
        "metric_val_batches": args.metric_val_batches,
        "metric_patience": args.metric_patience,
        "patience_geopath": args.patience_geopath,
        "reference_loss_batches": args.reference_loss_batches,
    }
    if values["metric_epochs"] is not None and values["metric_epochs"] <= 0:
        raise ValueError("metric_epochs must be positive.")
    for name in ("metric_train_batches", "metric_val_batches"):
        value = values[name]
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative.")
    for name in ("metric_patience", "patience_geopath"):
        value = values[name]
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative.")
    if (
        values["reference_loss_batches"] is not None
        and values["reference_loss_batches"] < 0
    ):
        raise ValueError("reference_loss_batches must be non-negative.")
    return {key: value for key, value in values.items() if value is not None}


def main(argv=None) -> int:
    args = parse_args(argv)
    return sweep.run_sweep(
        args,
        method="mfm",
        parameter_grids=parameter_grid(args),
        slug_keys=("flow_lr", "rho", "kappa"),
        extra_runtime_overrides=extra_runtime_overrides(args),
    )


if __name__ == "__main__":
    raise SystemExit(main())

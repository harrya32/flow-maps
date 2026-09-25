#!/usr/bin/env python3
"""Sweep SF2M hyperparameters on held-out Maizels validation days."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import mfm_maizels_sweep_utils as sweep


# Match the CITE/Multi flow-learning-rate default.
DEFAULT_FLOW_LEARNING_RATES = (1e-3,)
DEFAULT_FLOW_WEIGHT_DECAYS = (1e-5,)
DEFAULT_SIGMAS = (0.2,)
DEFAULT_SCORE_WEIGHTS = (0.3, 1.0, 3.0)
DEFAULT_TIME_EPSILONS = (1e-3,)
DEFAULT_OBJECTIVE = "final_eval/euler_maruyama_mean_emd_hparam_val_times"


def _grid_text(values) -> str:
    return ",".join(f"{value:g}" for value in values)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable SF2M grid on Maizels and select by mean exact "
            "Euler--Maruyama EMD at the held-out validation days."
        )
    )
    sweep.add_common_arguments(
        parser,
        default_config=(
            "metric-flow-matching/configs/single_cell/50dims/"
            "sf2m_maizels_3marginal.yaml"
        ),
        default_output_dir="outputs/maizels_sf2m_hparam_sweep",
        default_objective_metric=DEFAULT_OBJECTIVE,
    )
    parser.add_argument(
        "--flow-learning-rates", default=_grid_text(DEFAULT_FLOW_LEARNING_RATES)
    )
    parser.add_argument(
        "--flow-weight-decays", default=_grid_text(DEFAULT_FLOW_WEIGHT_DECAYS)
    )
    parser.add_argument("--sf2m-sigmas", default=_grid_text(DEFAULT_SIGMAS))
    parser.add_argument(
        "--sf2m-score-weights", default=_grid_text(DEFAULT_SCORE_WEIGHTS)
    )
    parser.add_argument(
        "--sf2m-time-epsilons", default=_grid_text(DEFAULT_TIME_EPSILONS)
    )
    return parser.parse_args(argv)


def parameter_grid(args: argparse.Namespace) -> Dict[str, tuple[float, ...]]:
    grid = {
        "flow_lr": sweep.parse_float_grid(
            args.flow_learning_rates, "flow learning rates"
        ),
        "flow_weight_decay": sweep.parse_float_grid(
            args.flow_weight_decays, "flow weight decays"
        ),
        "sf2m_sigma": sweep.parse_float_grid(args.sf2m_sigmas, "SF2M sigmas"),
        "sf2m_score_weight": sweep.parse_float_grid(
            args.sf2m_score_weights, "SF2M score weights"
        ),
        "sf2m_time_eps": sweep.parse_float_grid(
            args.sf2m_time_epsilons, "SF2M endpoint margins"
        ),
    }
    if any(value <= 0.0 for value in grid["flow_lr"]):
        raise ValueError("Flow learning rates must be positive.")
    if any(value < 0.0 for value in grid["flow_weight_decay"]):
        raise ValueError("Flow weight decays must be non-negative.")
    if any(value <= 0.0 for value in grid["sf2m_sigma"]):
        raise ValueError("SF2M sigmas must be positive.")
    if any(value < 0.0 for value in grid["sf2m_score_weight"]):
        raise ValueError("SF2M score weights must be non-negative.")
    if any(not 0.0 < value < 0.5 for value in grid["sf2m_time_eps"]):
        raise ValueError("SF2M endpoint margins must lie strictly between 0 and 0.5.")
    return grid


def main(argv=None) -> int:
    args = parse_args(argv)
    return sweep.run_sweep(
        args,
        method="sf2m",
        parameter_grids=parameter_grid(args),
        slug_keys=("flow_lr", "sf2m_sigma", "sf2m_score_weight"),
    )


if __name__ == "__main__":
    raise SystemExit(main())

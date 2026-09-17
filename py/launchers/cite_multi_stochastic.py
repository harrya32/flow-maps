#!/usr/bin/env python3
"""Train direct strong stochastic flow maps on CITE-seq or Multiome PCA100."""

from __future__ import annotations

# isort: off
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PY_DIR = SCRIPT_DIR.parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# isort: on

import argparse
import importlib
import inspect

from common import cite_multi
from common import cite_multi_stochastic_eval
from launchers import maizels_stochastic as ssfm_launcher


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direct SSFM on CITE-seq or Multiome PCA100."
    )
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.cite_multi_stochastic"
    )
    parser.add_argument("--slurm-id", "--slurm_id", type=int, required=True)
    parser.add_argument(
        "--dataset-name",
        "--dataset_name",
        choices=("cite", "multi"),
        default="cite",
    )
    parser.add_argument(
        "--heldout-day", "--heldout_day", choices=("3", "4"), default="4"
    )
    parser.add_argument("--dataset-location", "--dataset_location", default="")
    parser.add_argument("--output-folder", "--output_folder", default="")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", "--total_steps", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--n-pairs", "--n_pairs", type=int, default=None)
    parser.add_argument(
        "--validation-frequency",
        "--validation_frequency",
        "--eval-frequency",
        "--eval_frequency",
        dest="validation_frequency",
        type=int,
        default=None,
        help="Steps between fixed held-out objective checks (default: 100).",
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
        "--lineage-eval-max-points",
        "--lineage_eval_max_points",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval-flowmap-steps",
        "--eval_flowmap_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--visual-frequency", "--visual_frequency", type=int, default=None
    )
    parser.add_argument("--final-metrics-path", "--final_metrics_path", default=None)
    parser.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace):
    module = importlib.import_module(args.cfg_path)
    arguments = {
        "dataset_location": args.dataset_location,
        "output_folder": args.output_folder,
        "dataset_name": args.dataset_name,
        "heldout_day": args.heldout_day,
        "classifier_path": args.classifier_path,
        "full_data_classifier_path": args.full_data_classifier_path,
        "learning_rate": args.learning_rate,
        "constraint_weight": args.constraint_weight,
        "entropy_weight": args.entropy_weight,
        "diffusion_scale": args.diffusion_scale,
        "gamma_scale": args.gamma_scale,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "n_pairs": args.n_pairs,
    }
    supported = inspect.signature(module.get_config).parameters
    kwargs = {
        key: value
        for key, value in arguments.items()
        if value is not None and key in supported
    }
    cfg = module.get_config(args.slurm_id, **kwargs)
    if args.validation_frequency is not None:
        if args.validation_frequency <= 0:
            raise ValueError("validation_frequency must be positive.")
        cfg.optimization.early_stopping.check_freq = int(args.validation_frequency)
    if args.eval_noise_draws is not None:
        if args.eval_noise_draws <= 0:
            raise ValueError("eval_noise_draws must be positive.")
        cfg.evaluation.n_noise_draws = int(args.eval_noise_draws)
    if args.eval_max_points is not None:
        if args.eval_max_points < 0:
            raise ValueError("eval_max_points must be non-negative (zero means all).")
        cfg.evaluation.max_source_points = int(args.eval_max_points)
        cfg.evaluation.max_target_points = int(args.eval_max_points)
    if args.lineage_eval_max_points is not None:
        if args.lineage_eval_max_points < 0:
            raise ValueError(
                "lineage_eval_max_points must be non-negative (zero means all)."
            )
        cfg.evaluation.lineage_max_source_points = int(
            args.lineage_eval_max_points
        )
    if args.eval_flowmap_steps is not None:
        if args.eval_flowmap_steps <= 0:
            raise ValueError("eval_flowmap_steps must be positive.")
        cfg.evaluation.flowmap_n_steps = int(args.eval_flowmap_steps)
    if args.visual_frequency is not None:
        if args.visual_frequency < 0:
            raise ValueError("visual_frequency must be non-negative.")
        cfg.logging.visual_freq = int(args.visual_frequency)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    ssfm_launcher.train(
        cfg,
        final_metrics_path=args.final_metrics_path,
        wandb_mode=args.wandb_mode,
        data_backend=cite_multi,
        evaluation_backend=cite_multi_stochastic_eval,
        dataset_label=f"{cfg.problem.dataset_name.upper()} PCA100",
        default_output_folder="outputs/cite_multi_stochastic",
        trajectory_tag="heldout_day2",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

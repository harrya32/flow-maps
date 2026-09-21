#!/usr/bin/env python3
"""Train strong stochastic flow maps on a configured LARRY representation."""

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

from common import larry
from common import larry_stochastic_eval
from launchers import maizels_stochastic as ssfm_launcher


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a strong stochastic flow map on LARRY."
    )
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.larry_spring2d_stochastic"
    )
    parser.add_argument("--slurm-id", "--slurm_id", type=int, required=True)
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
        "--ot-minibatch-size", "--ot_minibatch_size", type=int, default=None
    )
    parser.add_argument(
        "--validation-frequency",
        "--validation_frequency",
        "--eval-frequency",
        "--eval_frequency",
        dest="validation_frequency",
        type=int,
        default=None,
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
        "--eval-flowmap-steps", "--eval_flowmap_steps", type=int, default=None
    )
    parser.add_argument(
        "--clone-samples-per-source",
        "--clone_samples_per_source",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--clone-noise-draws",
        "--clone_noise_draws",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--clone-target-times",
        "--clone_target_times",
        default=None,
        help="Comma-separated clone-evaluation targets (default: D4,D6).",
    )
    parser.add_argument(
        "--clone-eval-batch-size",
        "--clone_eval_batch_size",
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
        "ot_minibatch_size": args.ot_minibatch_size,
        "clone_samples_per_source": args.clone_samples_per_source,
        "clone_noise_draws": args.clone_noise_draws,
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
        cfg.evaluation.lineage_max_source_points = int(args.lineage_eval_max_points)
    if args.eval_flowmap_steps is not None:
        if args.eval_flowmap_steps <= 0:
            raise ValueError("eval_flowmap_steps must be positive.")
        cfg.evaluation.flowmap_n_steps = int(args.eval_flowmap_steps)
        cfg.evaluation.lineage_n_steps = int(args.eval_flowmap_steps)
        cfg.evaluation.clone_flowmap_n_steps = int(args.eval_flowmap_steps)
    if args.clone_target_times is not None:
        target_times = [
            larry.format_timepoint(value)
            for value in args.clone_target_times.split(",")
            if value.strip()
        ]
        if not target_times:
            raise ValueError("clone_target_times must contain at least one day.")
        cfg.evaluation.clone_target_times = target_times
    if args.clone_eval_batch_size is not None:
        if args.clone_eval_batch_size <= 0:
            raise ValueError("clone_eval_batch_size must be positive.")
        cfg.evaluation.clone_batch_size = int(args.clone_eval_batch_size)
    if args.visual_frequency is not None:
        if args.visual_frequency < 0:
            raise ValueError("visual_frequency must be non-negative.")
        cfg.logging.visual_freq = int(args.visual_frequency)
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    representation = larry.canonical_representation(
        getattr(cfg.problem, "larry_representation", larry.PCA50_REPRESENTATION)
    )
    representation_label = (
        "SPRING2D" if representation == larry.SPRING2D_REPRESENTATION else "PCA50"
    )
    ssfm_launcher.train(
        cfg,
        final_metrics_path=args.final_metrics_path,
        wandb_mode=args.wandb_mode,
        data_backend=larry,
        evaluation_backend=larry_stochastic_eval,
        dataset_label=f"LARRY {representation_label}",
        default_output_folder=f"outputs/larry_{representation}_stochastic",
        trajectory_tag="heldout_d2",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

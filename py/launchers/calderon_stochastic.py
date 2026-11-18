#!/usr/bin/env python3
"""Prepare, train, or evaluate the three Calderon minibatch-OT SSFM variants."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import importlib
import inspect
import json
import math

import flax.serialization
import numpy as np
from ml_collections import ConfigDict

from common import calderon, calderon_logging, calderon_stochastic_data, calderon_stochastic_eval
from launchers import maizels_stochastic as shared


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", "--cfg_path", "--config", default="configs.calderon_stochastic")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--variant", choices=("ot", "filtered", "constrained"))
    variants.add_argument("--slurm-id", "--slurm_id", type=int, choices=range(3))
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--dataset-location", "--dataset_location", default="")
    parser.add_argument("--output-folder", "--output_folder", default="")
    parser.add_argument("--classifier-path", "--classifier-manifest", dest="classifier_path")
    parser.add_argument("--evaluation-classifier-path", "--evaluation-classifier-manifest", dest="evaluation_classifier_path")
    parser.add_argument("--classifier-export-path", "--classifier-export", dest="classifier_export_path")
    parser.add_argument("--evaluation-classifier-export-path", "--evaluation-classifier-export", dest="evaluation_classifier_export_path")
    for name in ("learning-rate", "diffusion-scale", "gamma-scale", "constraint-weight", "entropy-weight"):
        parser.add_argument(f"--{name}", f"--{name.replace('-', '_')}", type=float)
    for name in ("seed", "total-steps", "batch-size", "n-pairs", "early-stopping-patience",
                 "validation-frequency", "validation-bs", "eval-noise-draws", "eval-flowmap-steps",
                 "lineage-eval-max-points", "visual-frequency"):
        flags = (f"--{name}", f"--{name.replace('-', '_')}") if "-" in name else (f"--{name}",)
        parser.add_argument(*flags, type=int)
    parser.add_argument("--eval-max-points", "--eval_max_points", "--marginal-cap", dest="eval_max_points", type=int,
                        help="Source and reference W1 cap (default 4096); zero uses all cells.")
    parser.add_argument("--checkpoint", type=Path, help="Saved best.msgpack for --stage evaluate.")
    parser.add_argument("--evaluation-output-dir", type=Path, help="Separate directory for checkpoint re-evaluation.")
    parser.add_argument("--final-metrics-path", "--final_metrics_path")
    parser.add_argument("--wandb-mode", "--wandb_mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--smoke", action="store_true", help="Two small optimizer steps and capped final evaluation.")
    return parser.parse_args(argv)


def _runtime_overrides(cfg, args):
    # Apply the current reporting protocol to older saved runs as well.
    cfg.evaluation.include_direct = False
    cfg.evaluation.include_mmd = False
    for argument, section, field, minimum in (
        ("validation_frequency", cfg.optimization.early_stopping, "check_freq", 1),
        ("validation_bs", cfg.logging.maizels, "validation_bs", 2),
        ("eval_noise_draws", cfg.evaluation, "n_noise_draws", 1),
        ("eval_max_points", cfg.evaluation, "max_source_points", 0),
        ("eval_flowmap_steps", cfg.evaluation, "flowmap_n_steps", 1),
        ("lineage_eval_max_points", cfg.evaluation, "lineage_max_source_points", 0),
        ("visual_frequency", cfg.logging, "visual_freq", 0),
    ):
        value = getattr(args, argument)
        if value is not None:
            if value < minimum:
                raise ValueError(f"{argument} must be at least {minimum}.")
            section[field] = value
    if args.eval_max_points is not None:
        cfg.evaluation.max_target_points = args.eval_max_points
    cfg.logging.maizels.distribution_eval_points_per_time = cfg.evaluation.max_target_points
    cfg.logging.maizels.distribution_eval_source_max_points = cfg.evaluation.max_source_points


def _build_config(args):
    module = importlib.import_module(args.cfg_path)
    supported = inspect.signature(module.get_config).parameters
    kwargs = {key: value for key, value in vars(args).items()
              if key in supported and value is not None and key != "slurm_id"}
    index = (("ot", "filtered", "constrained").index(args.variant) if args.variant is not None
             else 0 if args.slurm_id is None else args.slurm_id)
    cfg = module.get_config(index, **kwargs)
    _runtime_overrides(cfg, args)
    if args.smoke:
        cfg.optimization.bs = 16
        cfg.optimization.total_steps = 2
        cfg.optimization.decay_steps = 2
        cfg.optimization.warmup_steps = 1
        cfg.optimization.early_stopping.check_freq = 1
        cfg.logging.maizels.validation_bs = 16
        cfg.ssfm.hidden_dim = 32
        cfg.ssfm.uncertainty_hidden_dim = 16
        cfg.constraints.constraint_batch_size = 4
        cfg.evaluation.n_noise_draws = 1
        cfg.evaluation.max_source_points = cfg.evaluation.max_target_points = 16
        cfg.evaluation.lineage_max_source_points = 16
        cfg.evaluation.flowmap_n_steps = cfg.evaluation.lineage_n_steps = 4
        cfg.evaluation.save_plot = False
        cfg.logging.visual_freq = 0
        cfg.logging.output_name += "_smoke"
        cfg.logging.wandb_name += "_smoke"
        cfg.logging.maizels.distribution_eval_points_per_time = 16
        cfg.logging.maizels.distribution_eval_source_max_points = 16
    for field in ("learning_rate",):
        if not math.isfinite(cfg.optimization[field]) or cfg.optimization[field] <= 0:
            raise ValueError(f"{field} must be finite and positive.")
    for section, field in ((cfg.ssfm, "gamma_scale"), (cfg.ssfm, "diffusion_scale"),
                           (cfg.constraints, "weight"), (cfg.constraints, "loss_point_entropy_weight")):
        if not math.isfinite(section[field]) or section[field] < 0:
            raise ValueError(f"{field} must be finite and nonnegative.")
    return cfg


def _prepare_data(cfg):
    data = calderon.load_endpoint_experiment(cfg)
    for label, export, manifest in (
        ("training", cfg.classifier.jax_export_path, cfg.classifier.manifest_path),
        ("evaluation", cfg.classifier.evaluation_jax_export_path, cfg.classifier.evaluation_manifest_path),
    ):
        with np.load(export, allow_pickle=False) as stored:
            metadata = json.loads(stored["metadata"].item())
        if metadata["manifest_sha256"] != calderon.sha256_file(manifest):
            raise ValueError(f"The {label} classifier export differs from its selected manifest.")
        data.provenance[f"{label}_classifier_export_sha256"] = calderon.sha256_file(export)
    calderon_logging.register(cfg, data)
    return data


def _evaluate_checkpoint(args):
    if args.checkpoint is None:
        raise ValueError("--stage evaluate requires --checkpoint from a saved stochastic run.")
    training_fields = ("variant", "slurm_id", "dataset_location", "output_folder", "classifier_path",
        "evaluation_classifier_path", "classifier_export_path", "evaluation_classifier_export_path",
        "seed", "total_steps", "batch_size", "n_pairs", "learning_rate", "constraint_weight",
        "entropy_weight", "gamma_scale", "diffusion_scale", "early_stopping_patience",
        "validation_frequency", "validation_bs")
    if args.smoke or any(getattr(args, name) not in (None, "") for name in training_fields):
        raise ValueError("Checkpoint evaluation reloads saved training/data settings; only evaluation overrides apply.")
    checkpoint = args.checkpoint.expanduser().resolve()
    payload = flax.serialization.msgpack_restore(checkpoint.read_bytes())
    saved_config = json.loads((checkpoint.parent / "config.json").read_text())
    if flax.serialization.to_state_dict(saved_config) != payload["config"]:
        raise ValueError("The saved config differs from the config embedded in the stochastic checkpoint.")
    cfg = ConfigDict(saved_config)
    _runtime_overrides(cfg, args)
    data = _prepare_data(cfg)
    prepared = json.loads((checkpoint.parent / "data_summary.json").read_text())["provenance"]
    for field, value in prepared.items():
        if field.endswith("sha256") and data.provenance.get(field) != value:
            raise ValueError(f"Saved stochastic experiment input changed: {field}.")
    result_dir = (args.evaluation_output_dir.expanduser().resolve() if args.evaluation_output_dir else
                  checkpoint.parent / "evaluations" / f"w1_{cfg.evaluation.max_source_points}")
    if result_dir == checkpoint.parent:
        raise ValueError("Use a separate evaluation output directory to preserve the original results.")
    result_dir.mkdir(parents=True, exist_ok=True)
    model, _, _, _ = shared._create_model_and_state(cfg, np.asarray(payload["pc_scale"]))
    metrics = shared._evaluate_best_parameters(model, payload["state"]["params"], payload["ema_params"],
                                               cfg, result_dir, calderon_stochastic_eval)
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=calderon.sha256_file(checkpoint),
                parameters="EMA" if not cfg.evaluation.get("evaluate_instantaneous_and_ema", False) else "instantaneous and EMA",
                best_step=int(np.asarray(payload["best_step"])),
                best_validation_loss=float(np.asarray(payload["best_validation_loss"])), provenance=data.provenance)
    path = Path(args.final_metrics_path).expanduser().resolve() if args.final_metrics_path else result_dir / "final_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (result_dir / "evaluation_info.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    (result_dir / "evaluation_config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    print(f"Re-evaluated stochastic checkpoint: {path}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "evaluate":
        return _evaluate_checkpoint(args)
    if args.checkpoint is not None or args.evaluation_output_dir is not None:
        raise ValueError("--checkpoint and --evaluation-output-dir apply to --stage evaluate.")
    cfg = _build_config(args)
    run_dir = Path(cfg.logging.output_folder) / cfg.logging.output_name
    if any((run_dir / name).exists() for name in ("best.msgpack", "latest.msgpack", "final_metrics.json")):
        raise FileExistsError("This run already has training artifacts; choose another output folder or seed.")
    data = _prepare_data(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "data_summary.json").write_text(json.dumps(dict(marginals=data.summary(), provenance=data.provenance), indent=2) + "\n")
    data.split_table().to_csv(run_dir / "cell_splits.csv.gz", index=False)
    print(json.dumps(data.summary(), indent=2), flush=True)
    print(f"Stochastic artifacts: {run_dir}", flush=True)
    if args.stage == "prepare":
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
        return 0
    shared.train(cfg, final_metrics_path=args.final_metrics_path, wandb_mode=args.wandb_mode,
                 data_backend=calderon_stochastic_data, evaluation_backend=calderon_stochastic_eval,
                 dataset_label="Calderon PCA100", default_output_folder="outputs/calderon_stochastic",
                 trajectory_tag="heldout_4_6h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

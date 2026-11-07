"""Run upstream OTP-FM on the CITE/Multi PCA100 benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from common import cite_multi, maizels
from experiments.common.trainer import Trainer
from experiments.train import (
    create_model,
    create_potentials,
    load_config,
    load_json_config,
    merge_configs,
)

from . import REPO_ROOT, UPSTREAM_ROOT
from .data import prepare_data
from .evaluation import classifier_paths, evaluate
from .model import OTPFMSampler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", choices=("cite", "multi"), default="cite")
    parser.add_argument(
        "--heldout_day", nargs="+", choices=("3", "4"), default=["3", "4"]
    )
    parser.add_argument(
        "--cite_multi_time_mode", choices=cite_multi.TIME_MODES, default="equal_time"
    )
    parser.add_argument(
        "--dataset_location", default="", help="Benchmark data directory or H5AD path."
    )
    parser.add_argument("--working_dir", default=str(REPO_ROOT / "outputs" / "otpfm"))
    parser.add_argument(
        "--config_path",
        type=Path,
        help="JSON overrides layered on the selected upstream CITE preset.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, help="Defaults to the upstream seed (42)."
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--potential",
        type=str.lower,
        choices=("w2inf", "w2"),
        help="w2inf uses paired targets; w2 selects upstream W2Potential.",
    )
    parser.add_argument(
        "--ot_coupling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Precompute upstream pairwise OT chains instead of independent tuples.",
    )
    parser.add_argument(
        "--ot_method",
        choices=("emd", "sinkhorn"),
        help="Startup OT solver (default: emd).",
    )
    parser.add_argument(
        "--ot_reg", type=float, help="Startup Sinkhorn regularization (default: 0.01)."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument(
        "--eval_every_epochs", type=int, help="Defaults to upstream traj_skips (20)."
    )
    parser.add_argument(
        "--eval_max_points",
        type=int,
        default=0,
        help="Population cap; 0 evaluates every cell.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument(
        "--classifier_path",
        default="",
        help="Observed-days classifier override (evaluation only).",
    )
    parser.add_argument(
        "--full_data_classifier_path", default="", help="All-days classifier override."
    )
    parser.add_argument("--wandb_project", default="self-distill-flow-maps")
    parser.add_argument(
        "--wandb_entity",
        default=os.getenv("WANDB_ENTITY", "harryamad-university-of-cambridge"),
    )
    parser.add_argument(
        "--wandb_mode", choices=("online", "offline", "disabled"), default=None
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="One epoch, one train/val batch, evaluation capped at 64 cells.",
    )
    args = parser.parse_args(argv)
    if args.eval_max_points < 0 or args.eval_batch_size <= 0:
        parser.error(
            "Evaluation needs a non-negative population cap and a positive batch size."
        )
    if args.eval_every_epochs is not None and args.eval_every_epochs <= 0:
        parser.error("--eval_every_epochs must be positive.")
    return args


def experiment_config(args):
    overrides = load_json_config(args.config_path) if args.config_path else {}
    for key in (
        "epochs",
        "lr",
        "batch_size",
        "potential",
        "ot_coupling",
        "ot_method",
        "ot_reg",
    ):
        if getattr(args, key) is not None:
            overrides[key] = getattr(args, key)
    potential = overrides.get("potential", "w2inf").lower()
    ot_coupling = overrides.get("ot_coupling", False)
    if potential not in ("w2inf", "w2"):
        raise ValueError("Supported potentials are w2inf and w2 (W2Potential).")
    if not isinstance(ot_coupling, bool):
        raise ValueError("ot_coupling must be a JSON boolean or a CLI flag.")
    # The CITE W2 preset is W2Inf with precomputed OT. The actual W2Potential
    # class has no CITE preset and inherits the shared CITE training defaults.
    preset = ("W2" if ot_coupling else "W2Inf") if potential == "w2inf" else None
    config = merge_configs(load_config("citeseq", preset), overrides)
    config.update(
        potential=potential,
        ot_coupling=ot_coupling,
        upstream_config_preset=preset or "defaults",
    )
    config.setdefault("ot_reg", 0.01)
    config["ot_reg"] = float(config["ot_reg"])
    if config["ot_method"] not in ("emd", "sinkhorn") or not (
        np.isfinite(config["ot_reg"]) and config["ot_reg"] > 0
    ):
        raise ValueError(
            "OT needs emd or sinkhorn and a positive Sinkhorn regularization."
        )
    if (
        int(config["epochs"]) <= 0
        or int(config["batch_size"]) <= 0
        or float(config["lr"]) <= 0
    ):
        raise ValueError("Epochs, batch size, and learning rate must be positive.")
    # The only model-space change: use the benchmark representation and clock.
    config["pca_dim"] = 100
    # W2Inf has a linear force with a fixed assignment and permits a cached
    # direct solve. W2Potential changes its OT map as predicted positions move.
    config["solver_type"] = config.get("solver_type") or (
        "direct" if config["potential"] == "w2inf" else "anderson_safe"
    )
    if config["potential"] == "w2" and config["solver_type"] == "direct":
        raise ValueError(
            "W2Potential requires an iterative solver, e.g. anderson_safe."
        )
    if args.smoke_test:
        config["epochs"] = 1
    return config


def resolve_device(requested):
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    return torch.device(requested)


class BenchmarkTrainer(Trainer):
    """Reuse upstream training, adding cell-held-out validation and W&B hooks."""

    def __init__(self, *, data, config, args, classifiers, **kwargs):
        self.data = data
        self.config = config
        self.args = args
        self.classifiers = classifiers
        self.best_validation_loss = float("inf")
        self.best_step = 0
        self.final_metrics = {}
        super().__init__(**kwargs)
        self.sampler = OTPFMSampler(
            self.model, data.scaler, batch_size=args.eval_batch_size, ema=self.ema_eval
        )

    def _setup_file_logging(self):
        before = set(logging.getLogger().handlers)
        super()._setup_file_logging()
        self.file_handlers = set(logging.getLogger().handlers) - before

    def on_epoch_start(self, epoch, batch=None):
        if self.config["reshuffle_each_epoch"]:
            self.data.train_dataset.reshuffle()

    @torch.no_grad()
    def validate(self):
        # Compare a fixed full-OTP objective across curriculum stages. MSE is a
        # validation diagnostic; adaptive weighting remains the training loss.
        self.model.eval()
        training_lossfn = self.model.lossfn
        total, count, ret_batch = 0.0, 0, None
        try:
            self.model.lossfn = "mse"
            for batch in self.val_loader:
                ret_batch = self._process_batch(batch).to(self.device)
                loss = self.model.forward_with_loss(ret_batch, otp_alpha=1.0)
                total += float(loss) * len(ret_batch)
                count += len(ret_batch)
        finally:
            self.model.lossfn = training_lossfn
        value = total / count
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite OTP-FM validation loss.")
        if self.global_step > 0 and value < self.best_validation_loss:
            self.best_validation_loss, self.best_step = value, self.global_step
            self.save_checkpoint("best.pt")
        metrics = {
            "OTPFM/val_loss": value,
            "OTPFM/otp_alpha": self.curriculum(self.global_step),
            "OTPFM/lr": self.optimizers["flow"].param_groups[0]["lr"],
        }
        if self.losses["train_loss"]:
            metrics["OTPFM/train_loss"] = self.losses["train_loss"][-1]
        wandb.log(metrics, step=self.global_step)
        return value, ret_batch

    def _evaluate(self, *, final=False):
        metrics, figure = evaluate(
            self.data,
            self.sampler,
            self.classifiers,
            seed=int(self.config["seed"]),
            max_points=(
                min(self.args.eval_max_points or 64, 64)
                if self.args.smoke_test
                else self.args.eval_max_points
            ),
            lineage_mode=os.getenv("MAIZELS_LINEAGE_TRANSITION_MODE", "descendant"),
        )
        if not final:
            metrics = {
                key: value
                for key, value in metrics.items()
                if not key.startswith("final_eval/")
            }
        else:
            metrics.update(
                {
                    "final_eval/best_step": self.best_step,
                    "final_eval/best_validation_loss": self.best_validation_loss,
                }
            )
            self.final_metrics = metrics
            (self.save_dir / "final_metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n"
            )
            wandb.run.summary.update(metrics)
            wandb.run.summary["final_eval/best_checkpoint_path"] = str(
                self.models_dir / "best.pt"
            )
        if figure is not None:
            metrics = {
                **metrics,
                "plots/cite_multi_classifier_validity_paths": wandb.Image(figure),
            }
        wandb.log(metrics, step=self.global_step)
        if figure is not None:
            plt.close(figure)
        self.logger.info(
            "Held-out day %s: flowmap EMD %.6g; invalid trajectories %.4g%%",
            self.data.heldout_day,
            metrics["distribution_eval/flowmap_emd_mean"],
            metrics[
                "maizels/full_data_classifier/model_flowmap_invalid_trajectory_pct"
            ],
        )

    def on_epoch_end(self, epoch, batch=None):
        self._evaluate()

    def on_train_end(self):
        # Upstream loading also restores counters/history. Keep the completed
        # run's history and W&B step when restoring an earlier best model.
        completed_step, completed_losses = self.global_step, self.losses
        self.load_checkpoint(self.models_dir / "best.pt")
        self.global_step, self.losses = completed_step, completed_losses
        self._evaluate(final=True)
        self.save_losses_csv()


def run_experiment(args, config, heldout_day, seed):
    config = {**config, "seed": int(seed)}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    data = prepare_data(
        args.dataset_name,
        heldout_day,
        args.cite_multi_time_mode,
        args.dataset_location,
        config,
        smoke_test=args.smoke_test,
    )
    config["tks"] = [data.time_map[day] for day in data.retained[1:-1]]
    classifiers = classifier_paths(
        args.dataset_name,
        heldout_day,
        classifier_path=args.classifier_path,
        full_data_classifier_path=args.full_data_classifier_path,
    )
    potentials = create_potentials(config)
    model = create_model(config, 100, potentials, str(device))
    method = f"otpfm_{config['potential']}"
    if config["ot_coupling"]:
        method += f"_ot_{config['ot_method']}"
        if config["ot_method"] == "sinkhorn":
            method += f"_reg{config['ot_reg']}"
    name = f"{args.dataset_name}_pca100_holdout_day{heldout_day}_{args.cite_multi_time_mode}_{method}_seed{seed}"
    if args.smoke_test:
        name += "_smoke"
    output = Path(args.working_dir).expanduser().resolve() / name
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        **config,
        "dataset_name": args.dataset_name,
        "heldout_day": heldout_day,
        "cite_multi_time_mode": args.cite_multi_time_mode,
        "dataset_path": str(data.path),
        "retained_days": list(data.retained),
        "time_map": data.time_map,
        "device": str(device),
        "comparison_mode": method,
        "coupling_type": "precomputed_ot" if config["ot_coupling"] else "independent",
        "potential_class": type(next(iter(potentials.values()))).__name__,
        "upstream_repo": str(UPSTREAM_ROOT),
        "classifiers": {key: str(path) for key, path in classifiers.items()},
        "normalization_mean": (
            None if data.scaler is None else data.scaler.mean_.tolist()
        ),
        "normalization_scale": None if data.scaler is None else data.scaler.scale_,
        "train_cells_per_day": {
            day: len(data.timepoint_splits[day]["train_x"]) for day in data.retained
        },
        "validation_cells_per_day": {
            day: len(data.timepoint_splits[day]["holdout_x"]) for day in data.retained
        },
        "tuple_train_cells_per_day": {
            day: len(data.train_dataset.data_by_time[cite_multi.TIMEPOINTS.index(day)])
            for day in data.retained
        },
        "tuple_validation_cells_per_day": {
            day: len(data.val_dataset.data_by_time[cite_multi.TIMEPOINTS.index(day)])
            for day in data.retained
        },
        "validation_metric": f"full_strength_{config['consistency_loss']}_mse_on_retained_day_validation_cells",
        "evaluation_weights": "ema" if float(config["ema_decay"]) > 0 else "model",
        "distribution_steps": 50,
        "trajectory_check_times": 50,
        "mfm_steps": 100,
        "eval_max_points": (
            min(args.eval_max_points or 64, 64)
            if args.smoke_test
            else args.eval_max_points
        ),
        "smoke_test": args.smoke_test,
    }
    (output / "experiment.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    # W&B 0.17's Python backend crashes on null ServerSettings introspection
    # from the current cloud API. Its bundled core backend handles this schema.
    wandb.require("core")
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=name,
        config=metadata,
        dir=str(output),
        mode=args.wandb_mode,
    ):
        trainer = BenchmarkTrainer(
            data=data,
            config=config,
            args=args,
            classifiers=classifiers,
            model=model,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
            save_dir=output,
            lr=float(config["lr"]),
            epochs=int(config["epochs"]),
            optimizer=config["optimizer"],
            grad_clip=float(config["grad_clip"]),
            lr_schedule=config["lr_schedule"],
            otp_alpha_type=config["otp_alpha_type"],
            otp_alpha_slope=float(config["otp_alpha_slope"]),
            otp_alpha_mean_scale=float(config["otp_alpha_mean_scale"]),
            sampling_steps=int(config["sampling_steps"]),
            ema_eval=float(config["ema_decay"]) > 0,
            traj_skips=args.eval_every_epochs or int(config["traj_skips"]),
            potentials=potentials,
            device=str(device),
        )
        try:
            trainer.train()
        finally:
            gc.enable()
            for handler in trainer.file_handlers:
                logging.getLogger().removeHandler(handler)
                handler.close()
        return trainer.final_metrics


def main(argv=None):
    args = parse_args(argv)
    config = experiment_config(args)
    for heldout_day in args.heldout_day:
        for seed in args.seeds if args.seeds is not None else [int(config["seed"])]:
            run_experiment(args, config, heldout_day, seed)


if __name__ == "__main__":
    main()

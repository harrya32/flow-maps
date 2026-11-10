"""Run Cell-MNN on CITE/Multi PCA100 and Maizels PCA50."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from . import REPO_ROOT, UPSTREAM_ROOT
from .data import prepare_data
from .evaluation import classifier_paths, evaluate
from .model import BenchmarkCellMNN, CellMNNSampler


HPARAMS = (
    "epochs",
    "lr",
    "batch_size",
    "patience",
    "time_limit",
    "check_val_every_n_epoch",
    "weight_decay",
    "lambda_kinetic",
    "gamma",
    "width",
    "depth",
    "init_scale",
    "mmd_sigma",
    "num_const_dims",
)

# Maizels-selected settings and our constrained flow map's hidden layout.
BENCHMARK_DEFAULTS = {
    "lr": 3e-4,
    "batch_size": 1028,
    "lambda_kinetic": 0.1,
    "gamma": 0.1,
    "width": 1024,
    "depth": 3,
    "mmd_sigma": 0.3,
}


def upstream_defaults():
    """Read their argparse defaults without importing/executing their runner."""
    defaults = {}
    for node in ast.walk(ast.parse((UPSTREAM_ROOT / "train_mnn.py").read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            name = ast.literal_eval(node.args[0]).removeprefix("--")
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[name] = ast.literal_eval(keyword.value)
    return {key: defaults[key] for key in (*HPARAMS, "seed")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name", choices=("cite", "multi", "maizels"), default="cite"
    )
    parser.add_argument(
        "--heldout_day", nargs="+", choices=("3", "4"), default=["3", "4"]
    )
    parser.add_argument(
        "--cite_multi_time_mode",
        choices=("equal_time", "real_time"),
        default="equal_time",
    )
    parser.add_argument(
        "--maizels_schedule", choices=("d3_d3p8_d8", "d3_d8"), default="d3_d3p8_d8"
    )
    parser.add_argument(
        "--maizels_time_mode", choices=("equal_time", "real_time"), default="real_time"
    )
    parser.add_argument(
        "--dataset_location", default="", help="H5AD/CSV file or dataset directory."
    )
    parser.add_argument("--working_dir", default=str(REPO_ROOT / "outputs" / "cellmnn"))
    parser.add_argument(
        "--run_name", help="Output folder and W&B name override, useful for sweeps."
    )
    parser.add_argument(
        "--preset",
        choices=("auto", "cite100", "cite50", "script"),
        default="auto",
        help="auto uses the tuned 100D CITE/Multi preset, or 50D for Maizels.",
    )
    parser.add_argument(
        "--config_path", type=Path, help="JSON hyperparameter overrides."
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    for key in HPARAMS:
        parser.add_argument(
            f"--{key}",
            type=(
                int
                if key
                in (
                    "epochs",
                    "batch_size",
                    "patience",
                    "check_val_every_n_epoch",
                    "width",
                    "depth",
                    "num_const_dims",
                )
                else float
            ),
        )
    parser.add_argument(
        "--train_batches",
        type=int,
        help="Batches/epoch; default follows upstream dataset length.",
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        help="Fixed validation batches; default follows validation dataset length.",
    )
    parser.add_argument("--val_split", type=float)
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=20,
        help="Periodic evaluation; 0 evaluates only the final best checkpoint.",
    )
    parser.add_argument(
        "--eval_max_points", type=int, default=0, help="0 uses complete populations."
    )
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument(
        "--classifier_path",
        default="",
        help="Observed-days classifier, evaluation only.",
    )
    parser.add_argument(
        "--full_data_classifier_path",
        default="",
        help="All-days classifier, evaluation only.",
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
        help="One epoch and train/val batch; evaluate at most 64 cells.",
    )
    args = parser.parse_args(argv)
    if args.run_name and (
        args.run_name in (".", "..") or "/" in args.run_name or "\\" in args.run_name
    ):
        parser.error("--run_name must be a single output-folder name.")
    if (
        args.eval_every_epochs < 0
        or args.eval_max_points < 0
        or args.eval_batch_size <= 0
    ):
        parser.error(
            "Evaluation intervals/caps must be non-negative; batch size must be positive."
        )
    return args


def experiment_config(args):
    overrides = json.loads(args.config_path.read_text()) if args.config_path else {}
    json_preset = overrides.pop("preset", "auto")
    preset = args.preset if args.preset != "auto" else json_preset
    if preset == "auto":
        preset = "cite50" if args.dataset_name == "maizels" else "cite100"
    if preset not in ("cite100", "cite50", "script"):
        raise ValueError("Preset must be cite100, cite50, script, or auto.")
    config = {
        **upstream_defaults(),
        "preset": preset,
        "normalize": True,
        "val_split": 0.1,
        "train_batches": None,
        "val_batches": None,
        "kinetic_dt": 0.1,
    }
    if preset != "script":
        config.update(BENCHMARK_DEFAULTS)
    unknown = set(overrides) - set(config)
    if unknown:
        raise ValueError(
            f"Unknown Cell-MNN hyperparameter overrides: {sorted(unknown)}"
        )
    config.update(overrides)
    for key in (*HPARAMS, "train_batches", "val_batches", "val_split", "normalize"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    config["pca_dim"] = 50 if args.dataset_name == "maizels" else 100
    for key in (
        "epochs",
        "batch_size",
        "patience",
        "check_val_every_n_epoch",
        "width",
        "depth",
        "num_const_dims",
        "seed",
    ):
        if not isinstance(config[key], int) or isinstance(config[key], bool):
            raise ValueError(f"{key} must be an integer.")
    for key in ("weight_decay", "lambda_kinetic", "init_scale"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and non-negative.")
    if (
        not 0 <= config["seed"] < 2**32 - 2701
        or args.seeds is not None
        and any(not 0 <= seed < 2**32 - 2701 for seed in args.seeds)
    ):
        raise ValueError("Seeds must be non-negative 32-bit integers.")
    for key in (
        "epochs",
        "batch_size",
        "check_val_every_n_epoch",
        "width",
        "depth",
        "lr",
        "mmd_sigma",
        "kinetic_dt",
        "time_limit",
    ):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    for key in ("train_batches", "val_batches"):
        if config[key] is not None and int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive when specified.")
    if not 0 < config["val_split"] < 1 or not 0 <= config["gamma"] <= 1:
        raise ValueError("val_split must be in (0,1), and gamma in [0,1].")
    if not 0 <= config["num_const_dims"] < config["pca_dim"] or config["patience"] < 0:
        raise ValueError("Invalid constant dimension count or early-stopping patience.")
    if not isinstance(config["normalize"], bool):
        raise ValueError("normalize must be a boolean.")
    if args.smoke_test:
        config["epochs"] = 1
    return config


def resolve_device(requested):
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(requested)


@contextlib.contextmanager
def validation_rng(seed):
    """Repeat upstream validation batches without advancing training RNGs."""
    state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(int(seed))
        torch.random.default_generator.manual_seed(int(seed))
        try:
            yield
        finally:
            np.random.set_state(state)


class BenchmarkTrainer:
    def __init__(self, model, data, config, args, classifiers, output):
        self.model, self.data, self.config, self.args = model, data, config, args
        self.classifiers, self.output = classifiers, Path(output)
        self.optimizer = model.configure_optimizers()
        self.device = next(model.parameters()).device
        self.global_step, self.best_step, self.best_epoch = 0, 0, 0
        self.best_loss, self.stale_checks = float("inf"), 0
        self.history = []

    def _epoch_loss(self, loader, *, training):
        self.model.train(training)
        totals, count = {}, 0
        with torch.enable_grad() if training else torch.no_grad():
            for batch in loader:
                batch = tuple(value.to(self.device) for value in batch)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                terms = self.model.loss_terms(batch)
                if not all(torch.isfinite(value).item() for value in terms.values()):
                    raise FloatingPointError(
                        "Non-finite Cell-MNN training/validation loss."
                    )
                if training:
                    terms["loss"].backward()
                    if not all(
                        torch.isfinite(parameter.grad).all().item()
                        for parameter in self.model.parameters()
                        if parameter.grad is not None
                    ):
                        raise FloatingPointError("Non-finite Cell-MNN gradient.")
                    self.optimizer.step()
                    self.global_step += 1
                batch_n = len(batch[0])
                count += batch_n
                for key, value in terms.items():
                    totals[key] = totals.get(key, 0.0) + batch_n * value.detach().item()
                if self.device.type == "mps":
                    # Evict unused buffers from the caller after Metal completes,
                    # rather than letting allocation trigger eviction on its queue.
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        if count == 0:
            raise ValueError("Cell-MNN loader produced no batches.")
        return {key: value / count for key, value in totals.items()}

    def validation_loss(self):
        with validation_rng(int(self.config["seed"]) + 2701):
            return self._epoch_loss(self.data.val_loader, training=False)

    def evaluate(self, *, final=False):
        cap = (
            min(self.args.eval_max_points or 64, 64)
            if self.args.smoke_test
            else self.args.eval_max_points
        )
        sampler = CellMNNSampler(
            self.model,
            self.data.scaler,
            time_origin=self.data.time_origin,
            time_scale=self.data.time_scale,
            batch_size=self.args.eval_batch_size,
        )
        metrics, figures = evaluate(
            self.data,
            sampler,
            self.classifiers,
            seed=self.config["seed"],
            max_points=cap,
            include_observed=final,
        )
        if final:
            metrics.update(
                {
                    "final_eval/best_step": self.best_step,
                    "final_eval/best_epoch": self.best_epoch,
                    "final_eval/best_validation_loss": self.best_loss,
                    "final_eval/best_checkpoint_path": str(self.output / "best.pt"),
                }
            )
            (self.output / "final_metrics.json").write_text(
                json.dumps(metrics, indent=2, allow_nan=False) + "\n"
            )
        for name, figure in figures.items():
            figure.savefig(
                self.output / f"{'final_' if final else ''}{name}.png", dpi=150
            )
        wandb.log(
            {
                **metrics,
                "train/global_step": self.global_step,
                **{
                    f"plots/{name}": wandb.Image(figure)
                    for name, figure in figures.items()
                },
            }
        )
        for figure in figures.values():
            plt.close(figure)
        if final and wandb.run is not None:
            wandb.run.summary.update(metrics)
        if final:
            print(
                f"Final Cell-MNN {self.data.dataset_name}: best_val_loss={self.best_loss:.7g}, "
                f"test_EMD={metrics['cellmnn/test_EMD']:.7g}, "
                f"direct_invalid_trajectory_pct={metrics['final_eval/direct_invalid_trajectory_pct']:.7g}; "
                f"outputs={self.output}",
                flush=True,
            )
        return metrics

    def train(self):
        started = time.monotonic()
        for epoch in range(1, int(self.config["epochs"]) + 1):
            terms = self._epoch_loss(self.data.train_loader, training=True)
            timed_out = (
                time.monotonic() - started >= float(self.config["time_limit"]) * 60
            )
            last = epoch == int(self.config["epochs"]) or timed_out
            record = {
                "epoch": epoch,
                "global_step": self.global_step,
                **{f"train_{key}": value for key, value in terms.items()},
            }
            should_validate = (
                epoch % int(self.config["check_val_every_n_epoch"]) == 0 or last
            )
            if should_validate:
                validation = self.validation_loss()
                record.update(
                    {f"val_{key}": value for key, value in validation.items()}
                )
                if validation["loss"] < self.best_loss:
                    self.best_loss, self.best_step, self.best_epoch = (
                        validation["loss"],
                        self.global_step,
                        epoch,
                    )
                    self.stale_checks = 0
                    torch.save(
                        {
                            "state_dict": self.model.state_dict(),
                            "config": self.config,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "val_loss": self.best_loss,
                        },
                        self.output / "best.pt",
                    )
                else:
                    self.stale_checks += 1
                print(
                    f"epoch={epoch} step={self.global_step} train_loss={terms['loss']:.7g} val_loss={validation['loss']:.7g}",
                    flush=True,
                )
            self.history.append(record)
            wandb.log({**record, "train/global_step": self.global_step})
            if (
                self.args.eval_every_epochs > 0
                and epoch % self.args.eval_every_epochs == 0
            ):
                self.evaluate()
            if (
                last
                or should_validate
                and self.stale_checks >= max(1, int(self.config["patience"]))
            ):
                break
        columns = list(dict.fromkeys(key for row in self.history for key in row))
        with (self.output / "losses.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history)
        checkpoint = torch.load(
            self.output / "best.pt", map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        return self.evaluate(final=True)


def run_experiment(args, config, heldout_day, seed):
    config = {**config, "seed": int(seed)}
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(args.device)
    data = prepare_data(args, config, heldout_day, seed)
    classifiers = classifier_paths(args, heldout_day)
    model = BenchmarkCellMNN(
        latent_dim=config["pca_dim"],
        retained_times=[data.model_time_map[day] for day in data.retained],
        kinetic_dt=config["kinetic_dt"],
        **{
            key: config[key]
            for key in (
                "lr",
                "weight_decay",
                "lambda_kinetic",
                "gamma",
                "width",
                "depth",
                "init_scale",
                "mmd_sigma",
                "num_const_dims",
            )
        },
    ).to(device)
    fold = (
        args.maizels_schedule
        if args.dataset_name == "maizels"
        else f"holdout_day{heldout_day}"
    )
    method = (
        "cellmnn"
        if config["num_const_dims"] == 0
        else f"cellmnn_{config['num_const_dims']}ev0"
    )
    name = f"{args.dataset_name}_pca{config['pca_dim']}_{fold}_{data.time_mode}_{method}_seed{seed}"
    if args.smoke_test:
        name += "_smoke"
    name = args.run_name or name
    output = Path(args.working_dir).expanduser().resolve() / name
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        **config,
        "run_name": name,
        "dataset_name": args.dataset_name,
        "dataset_path": str(data.path),
        "heldout_day": heldout_day,
        "retained_days": list(data.retained),
        "time_mode": data.time_mode,
        "time_map": data.time_map,
        "model_time_map": data.model_time_map,
        "device": str(device),
        "population_loss_device": "cpu" if device.type == "mps" else str(device),
        "determinant_device": "cpu" if device.type == "mps" else str(device),
        "matrix_inverse_device": "cpu" if device.type == "mps" else str(device),
        "mps_cache_policy": (
            "release_after_each_batch" if device.type == "mps" else None
        ),
        "upstream_repo": str(UPSTREAM_ROOT),
        "comparison_mode": method,
        "primary_sampler": "direct",
        "evaluation_samplers": ["direct"],
        "checkpoint_monitor": "val_loss",
        "validation_metric": "discounted_laplacian_mmd_plus_kinetic_plus_invertibility_on_retained_validation_cells",
        "normalization": "global_training_pc1_std" if data.scaler else "none",
        "normalization_scale": data.scaler.scale_ if data.scaler else 1.0,
        "cell_split_seed": 701,
        "validation_batch_seed": int(seed) + 2701,
        "train_cells_per_day": {
            day: len(data.timepoint_splits[day]["train_x"]) for day in data.retained
        },
        "validation_cells_per_day": {
            day: len(data.timepoint_splits[day]["holdout_x"]) for day in data.retained
        },
        "train_batches_per_epoch": len(data.train_dataset),
        "validation_batches": len(data.val_dataset),
        "classifiers": {key: str(path) for key, path in classifiers.items()},
        "distribution_steps": 1,
        "trajectory_check_times": 50,
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
    print(
        f"Cell-MNN {args.dataset_name}: preset={config['preset']}, retained_days={list(data.retained)}, "
        f"lr={config['lr']}, batch_size={config['batch_size']}, device={device}",
        flush=True,
    )
    wandb.require("core")
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=name,
        config=metadata,
        dir=str(output),
        mode=args.wandb_mode,
    ):
        return BenchmarkTrainer(model, data, config, args, classifiers, output).train()


def main(argv=None):
    args = parse_args(argv)
    config = experiment_config(args)
    folds = [None] if args.dataset_name == "maizels" else args.heldout_day
    for fold in folds:
        for seed in args.seeds if args.seeds is not None else [int(config["seed"])]:
            run_experiment(args, config, fold, seed)


if __name__ == "__main__":
    main()

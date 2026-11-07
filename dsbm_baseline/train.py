"""Train the authors' vector DSBM implementation on flow-maps benchmarks."""

from __future__ import annotations

import argparse
import copy
import math
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml


FLOW_MAPS_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = FLOW_MAPS_ROOT / "metric-flow-matching"
PY_ROOT = FLOW_MAPS_ROOT / "py"
for source_root in (MFM_ROOT, PY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from common import cite_multi, maizels, wasserstein  # noqa: E402
from dsbm_baseline.evaluation import run_final_evaluation  # noqa: E402
from dsbm_baseline.model import (  # noqa: E402
    PiecewiseDSBM,
    import_original_dsbm,
    make_original_dsbm,
    sample_local_forward,
)
from mfm.dataloaders.maizels_data import MaizelsEndpointDataModule  # noqa: E402
from mfm.dataloaders.trajectory_data import TemporalDataModule  # noqa: E402
from mfm.flow_matchers.cite_multi_eval import (  # noqa: E402
    resolve_all_days_classifier_path,
    validate_cite_multi_evaluation_args,
)
from mfm.train import parsers as mfm_parsers  # noqa: E402
from mfm.train.train_utils import dataset_name2datapath  # noqa: E402
from mfm.utils import set_seed  # noqa: E402


def _add_dsbm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--working_dir",
        type=str,
        default="./",
        help="Directory for DSBM checkpoints and W&B run files.",
    )
    parser.add_argument(
        "--dsbm_dir",
        type=str,
        default=os.environ.get(
            "DSBM_DIR",
            str(Path.home() / "Desktop" / "dsbm-pytorch"),
        ),
        help="Path to the unchanged upstream dsbm-pytorch checkout.",
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--t_exclude", nargs="+", type=int, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--final_metrics_path", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--split_ratios", nargs=2, type=float, default=[0.9, 0.1])
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_name", type=str, default="")

    parser.add_argument(
        "--mfm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--optimal_transport_method",
        type=str,
        default="None",
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--dsbm_first_coupling",
        choices=["ind", "ref"],
        default="ind",
        help="Upstream 'ind' is DSBM-IMF; 'ref' is DSBM-IPF.",
    )
    parser.add_argument("--dsbm_sigma", type=float, default=0.2)
    parser.add_argument("--dsbm_num_steps", type=int, default=20)
    parser.add_argument("--dsbm_time_eps", type=float, default=1e-3)
    parser.add_argument("--dsbm_learning_rate", type=float, default=1e-4)
    parser.add_argument("--dsbm_hidden_dims", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--dsbm_activation", type=str, default="silu")
    parser.add_argument(
        "--dsbm_outer_iterations",
        type=int,
        default=40,
        help="Number of alternating upstream backward/forward fitting passes.",
    )
    parser.add_argument(
        "--dsbm_inner_iterations",
        type=int,
        default=250,
        help="Optimizer steps in each upstream fitting pass, per interval.",
    )
    parser.add_argument(
        "--dsbm_train_points_per_marginal",
        type=int,
        default=10_000,
        help="Training population cap per marginal; 0 uses every training cell.",
    )
    parser.add_argument(
        "--dsbm_normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply one shared translation and scalar scale to all retained data.",
    )
    parser.add_argument(
        "--dsbm_scale_sigma_by_interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale local diffusion by sqrt(global interval duration).",
    )
    parser.add_argument("--dsbm_validation_points", type=int, default=1024)
    parser.add_argument("--dsbm_validation_every_forward", type=int, default=1)
    parser.add_argument("--dsbm_eval_points_per_time", type=int, default=0)
    parser.add_argument("--dsbm_eval_euler_steps", type=int, default=100)
    parser.add_argument("--dsbm_eval_batch_size", type=int, default=1024)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one original DSBM bridge per retained interval and evaluate "
            "the composed stochastic process on flow-maps benchmarks."
        )
    )
    _add_dsbm_arguments(parser)
    mfm_parsers.datasets_parser(parser)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config_path", type=str, default="")
    preliminary, _ = config_parser.parse_known_args(argv)
    config = {}
    if preliminary.config_path:
        with Path(preliminary.config_path).expanduser().open("r") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise TypeError("DSBM config must contain a YAML mapping.")

    parser = build_parser()
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"Unknown DSBM config fields: {unknown}")
    parser.set_defaults(**config)
    return parser.parse_args(argv)


def _resolve_device(accelerator: str) -> torch.device:
    requested = str(accelerator).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested in {"gpu", "cuda"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("accelerator=gpu was requested, but no GPU is available.")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("accelerator=mps was requested, but MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    device = torch.device(accelerator)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is unavailable.")
    return device


def _resolve_data_path(args: argparse.Namespace) -> str:
    if args.data_path:
        return str(Path(args.data_path).expanduser().resolve())
    if args.data_type == "maizels":
        configured = str(args.maizels_dataset_path or "")
        if configured:
            return str(Path(configured).expanduser().resolve())
        local_copy = FLOW_MAPS_ROOT / "celltype_classification_pca50_dataset.csv.gz"
        return str(local_copy if local_copy.is_file() else Path(maizels.DEFAULT_DATASET))
    return str(Path(dataset_name2datapath(args.data_name, args.working_dir)).resolve())


def _build_data(args: argparse.Namespace, t_exclude: int | None):
    dataset_args = copy.deepcopy(args)
    dataset_args.data_path = _resolve_data_path(dataset_args)
    if not Path(dataset_args.data_path).is_file():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_args.data_path}. Pass --data_path, "
            "or set CITE_MULTI_DATA_DIR for CITE/Multi."
        )

    if dataset_args.data_type == "maizels":
        if dataset_args.maizels_schedule != "d3_d3p8_d8":
            raise ValueError(
                "DSBM's Maizels baseline requires maizels_schedule=d3_d3p8_d8."
            )
        if t_exclude is not None:
            raise ValueError("The Maizels 3-marginal run does not use t_exclude.")
        if not str(dataset_args.maizels_classifier_path or ""):
            repository_classifier = (
                FLOW_MAPS_ROOT / "celltype_classifier_pca50_d3_d3p8_d8.pt"
            )
            if repository_classifier.is_file():
                dataset_args.maizels_classifier_path = str(repository_classifier)
        datamodule = MaizelsEndpointDataModule(args=dataset_args)
        retained = tuple(datamodule.retained_timepoints)
        times = tuple(float(value) for value in datamodule.times.tolist())
    elif dataset_args.data_type == "scrna" and dataset_args.data_name in {
        "cite",
        "multi",
    }:
        if t_exclude not in (1, 2):
            raise ValueError("CITE/Multi DSBM requires --t_exclude 1 or 2.")
        datamodule = TemporalDataModule(
            args=dataset_args,
            skipped_datapoint=int(t_exclude),
        )
        ordered = tuple(datamodule.timepoint_splits)
        retained_indices = [
            index for index in range(len(ordered)) if index != int(t_exclude)
        ]
        retained = tuple(ordered[index] for index in retained_indices)
        times = tuple(float(datamodule.times[index]) for index in retained_indices)
    else:
        raise ValueError("DSBM is configured only for CITE, Multi, and Maizels.")
    return dataset_args, datamodule, retained, times


def _shared_scalar_normalization(datamodule, retained, enabled: bool):
    arrays = [
        np.asarray(datamodule.timepoint_splits[timepoint]["train_x"], dtype=np.float32)
        for timepoint in retained
    ]
    dim = arrays[0].shape[1]
    if not enabled:
        return np.zeros(dim, dtype=np.float32), 1.0

    count = sum(array.shape[0] for array in arrays)
    center = sum(
        (array.sum(axis=0, dtype=np.float64) for array in arrays),
        start=np.zeros(dim, dtype=np.float64),
    ) / float(count)
    squared_error = sum(
        (
            np.square(array.astype(np.float64) - center).sum(dtype=np.float64)
            for array in arrays
        ),
        start=0.0,
    )
    scale = math.sqrt(float(squared_error) / float(count * dim))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Invalid DSBM normalization scale: {scale}.")
    return center.astype(np.float32), float(scale)


def _draw_rows(values, count: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == 0:
        raise ValueError("Cannot sample an empty DSBM marginal.")
    replace = count > values.shape[0]
    indices = rng.choice(values.shape[0], size=int(count), replace=replace)
    return values[indices]


def _training_pairs(
    source,
    target,
    *,
    max_points: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source_n = source.shape[0] if max_points <= 0 else min(source.shape[0], max_points)
    target_n = target.shape[0] if max_points <= 0 else min(target.shape[0], max_points)
    paired_n = max(int(batch_size), source_n, target_n)
    return np.stack(
        [_draw_rows(source, paired_n, rng), _draw_rows(target, paired_n, rng)],
        axis=1,
    ), paired_n


def _validation_populations(
    source,
    target,
    *,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    count = min(source.shape[0], target.shape[0])
    if max_points > 0:
        count = min(count, int(max_points))
    return _draw_rows(source, count, rng), _draw_rows(target, count, rng)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _safe_tag(value: str) -> str:
    return str(value).replace(".", "p").replace("/", "_")


def _train_segment(
    upstream,
    args,
    *,
    source_timepoint: str,
    target_timepoint: str,
    start_time: float,
    end_time: float,
    source_train: np.ndarray,
    target_train: np.ndarray,
    source_validation: np.ndarray,
    target_validation: np.ndarray,
    center: np.ndarray,
    scale: float,
    device: torch.device,
    segment_index: int,
    checkpoint_root: Path,
    optimizer_step_offset: int,
) -> tuple[torch.nn.Module, dict, int]:
    duration = float(end_time) - float(start_time)
    effective_sigma = float(args.dsbm_sigma)
    if bool(args.dsbm_scale_sigma_by_interval):
        effective_sigma *= math.sqrt(duration)
    model = make_original_dsbm(
        upstream,
        dim=int(args.dim),
        hidden_dims=args.dsbm_hidden_dims,
        activation=args.dsbm_activation,
        num_steps=int(args.dsbm_num_steps),
        sigma=effective_sigma,
        time_eps=float(args.dsbm_time_eps),
        first_coupling=args.dsbm_first_coupling,
        device=device,
        learning_rate=float(args.dsbm_learning_rate),
    )

    rng = np.random.default_rng(int(args.seed_current) + 10_000 * segment_index + 17)
    x_pairs, train_count = _training_pairs(
        (source_train - center) / scale,
        (target_train - center) / scale,
        max_points=int(args.dsbm_train_points_per_marginal),
        batch_size=int(args.batch_size),
        rng=rng,
    )
    x_pairs_tensor = torch.as_tensor(x_pairs, dtype=torch.float32, device=device)
    validation_source, validation_target = _validation_populations(
        source_validation,
        target_validation,
        max_points=int(args.dsbm_validation_points),
        rng=np.random.default_rng(
            int(args.seed_current) + 10_000 * segment_index + 701
        ),
    )
    normalized_validation_source = torch.as_tensor(
        (validation_source - center) / scale,
        dtype=torch.float32,
        device=device,
    )

    segment_tag = f"{_safe_tag(source_timepoint)}_to_{_safe_tag(target_timepoint)}"
    checkpoint_dir = checkpoint_root / f"segment_{segment_index:02d}_{segment_tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    best_emd = float("inf")
    best_outer_iteration = -1
    best_state = None
    previous_model = None
    optimizer_steps = 0
    forward_pass = 0

    print(
        f"Training DSBM segment {source_timepoint}->{target_timepoint}: "
        f"n={train_count}, duration={duration:.6g}, sigma={effective_sigma:.6g}, "
        f"device={device}"
    )
    for outer_iteration in range(1, int(args.dsbm_outer_iterations) + 1):
        direction = "b" if outer_iteration % 2 == 1 else "f"
        first_iteration = outer_iteration == 1
        model.train()
        model, loss_curve = upstream.train_dsbm(
            model,
            x_pairs_tensor,
            int(args.batch_size),
            int(args.dsbm_inner_iterations),
            prev_model=previous_model,
            fb=direction,
            first_it=first_iteration,
        )
        optimizer_steps += int(args.dsbm_inner_iterations)
        log_loss = float(loss_curve[-1]) if loss_curve else float("nan")
        wandb.log(
            {
                "training/optimizer_steps": optimizer_step_offset + optimizer_steps,
                f"training/{segment_tag}/outer_iteration": outer_iteration,
                f"training/{segment_tag}/{direction}_log_mse": log_loss,
            }
        )

        if direction == "f":
            forward_pass += 1
            should_validate = (
                forward_pass % int(args.dsbm_validation_every_forward) == 0
                or outer_iteration == int(args.dsbm_outer_iterations)
            )
            if should_validate:
                was_training = model.training
                model.eval()
                prediction_normalized = sample_local_forward(
                    model,
                    normalized_validation_source,
                    n_steps=int(args.dsbm_eval_euler_steps),
                    seed=(
                        int(args.seed_current)
                        + 10_000 * segment_index
                        + 1701
                    ),
                )
                if was_training:
                    model.train()
                prediction = (
                    prediction_normalized.detach().cpu().numpy() * scale + center
                ).astype(np.float32)
                validation_emd = wasserstein.exact_emd(
                    prediction,
                    validation_target,
                )
                wandb.log(
                    {
                        f"validation/{segment_tag}_emd": validation_emd,
                        f"validation/{segment_tag}_forward_pass": forward_pass,
                    }
                )
                print(
                    f"  pass {outer_iteration}/{args.dsbm_outer_iterations} "
                    f"({direction}): held-out endpoint EMD={validation_emd:.8g}"
                )
                if validation_emd < best_emd:
                    best_emd = float(validation_emd)
                    best_outer_iteration = int(outer_iteration)
                    best_state = _cpu_state_dict(model)
                    torch.save(
                        {
                            "state_dict": best_state,
                            "source_timepoint": source_timepoint,
                            "target_timepoint": target_timepoint,
                            "start_time": float(start_time),
                            "end_time": float(end_time),
                            "effective_sigma": effective_sigma,
                            "normalization_center": center,
                            "normalization_scale": float(scale),
                            "best_validation_emd": best_emd,
                            "best_outer_iteration": best_outer_iteration,
                        },
                        checkpoint_path,
                    )

        previous_model = copy.deepcopy(model).eval()
        for parameter in previous_model.parameters():
            parameter.requires_grad_(False)

    if best_state is None:
        raise RuntimeError(
            f"DSBM segment {segment_tag} completed without a forward validation."
        )
    model.load_state_dict(best_state, strict=True)
    model.eval()
    del previous_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "source_timepoint": source_timepoint,
        "target_timepoint": target_timepoint,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration": duration,
        "effective_sigma": effective_sigma,
        "training_population": int(train_count),
        "validation_population": int(validation_source.shape[0]),
        "best_validation_emd": best_emd,
        "best_outer_iteration": best_outer_iteration,
        "checkpoint_path": str(checkpoint_path),
        "optimizer_steps": optimizer_steps,
    }
    if wandb.run is not None and hasattr(wandb.run, "summary"):
        wandb.run.summary[f"validation/{segment_tag}_best_emd"] = best_emd
        wandb.run.summary[f"validation/{segment_tag}_best_outer_iteration"] = (
            best_outer_iteration
        )
    return model, metadata, optimizer_steps


def run_one(args: argparse.Namespace, *, seed: int, t_exclude: int | None) -> None:
    run_args = copy.deepcopy(args)
    run_args.seed_current = int(seed)
    run_args.t_exclude_current = t_exclude
    run_args.working_dir = str(Path(args.working_dir).expanduser().resolve())
    Path(run_args.working_dir).mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))

    dataset_args, datamodule, retained, times = _build_data(run_args, t_exclude)
    run_args.data_path = dataset_args.data_path
    if run_args.data_type == "scrna":
        validate_cite_multi_evaluation_args(run_args)
        run_args.cite_multi_eval_classifier_path = str(
            resolve_all_days_classifier_path(run_args)
        )
    else:
        run_args.maizels_classifier_path = str(datamodule.classifier_path)

    heldout_suffix = ""
    if t_exclude is not None:
        heldout_day = tuple(datamodule.timepoint_splits)[int(t_exclude)]
        heldout_suffix = f"_holdout_day{heldout_day}"
    method_name = (
        "dsbm_imf" if run_args.dsbm_first_coupling == "ind" else "dsbm_ipf"
    )
    default_name = f"{run_args.data_name}_{method_name}{heldout_suffix}_seed{seed}"
    run_name = run_args.wandb_name or run_args.run_name or default_name
    run_args.run_name = run_name
    device = _resolve_device(run_args.accelerator)

    upstream = import_original_dsbm(run_args.dsbm_dir)
    run = wandb.init(
        project=run_args.wandb_project or "self-distill-flow-maps",
        entity=run_args.wandb_entity or None,
        name=run_name,
        group=run_args.group_name,
        config=vars(run_args),
        dir=run_args.working_dir,
    )
    try:
        center, scale = _shared_scalar_normalization(
            datamodule,
            retained,
            bool(run_args.dsbm_normalize),
        )
        interval_config = [
            {
                "source_timepoint": source,
                "target_timepoint": target,
                "start_time": float(start),
                "end_time": float(end),
                "effective_sigma": float(run_args.dsbm_sigma)
                * (
                    math.sqrt(float(end) - float(start))
                    if bool(run_args.dsbm_scale_sigma_by_interval)
                    else 1.0
                ),
            }
            for source, target, start, end in zip(
                retained[:-1], retained[1:], times[:-1], times[1:]
            )
        ]
        wandb.config.update(
            {
                "method": method_name,
                "evaluation_sampler": PiecewiseDSBM.evaluation_sampler,
                "dsbm_source_checkout": str(
                    Path(run_args.dsbm_dir).expanduser().resolve()
                ),
                "dsbm_source_file": "DSBM-Gaussian.py",
                "protocol": "piecewise_adjacent_retained_marginals",
                "retained_timepoints": list(retained),
                "retained_times": list(times),
                "dsbm_intervals": interval_config,
                "training_uses_cell_type_labels": False,
                "classifier_is_evaluation_only": True,
                "normalization_center_l2": float(np.linalg.norm(center)),
                "normalization_scalar": float(scale),
            },
            allow_val_change=True,
        )

        checkpoint_root = (
            Path(run_args.working_dir) / "checkpoints" / "dsbm" / str(run.id)
        )
        bridges = []
        segment_metadata = []
        total_optimizer_steps = 0
        for segment_index, (source, target, start, end) in enumerate(
            zip(retained[:-1], retained[1:], times[:-1], times[1:])
        ):
            source_pool = datamodule.timepoint_splits[source]
            target_pool = datamodule.timepoint_splits[target]
            bridge, metadata, optimizer_steps = _train_segment(
                upstream,
                run_args,
                source_timepoint=source,
                target_timepoint=target,
                start_time=float(start),
                end_time=float(end),
                source_train=source_pool["train_x"],
                target_train=target_pool["train_x"],
                source_validation=source_pool["holdout_x"],
                target_validation=target_pool["holdout_x"],
                center=center,
                scale=scale,
                device=device,
                segment_index=segment_index,
                checkpoint_root=checkpoint_root,
                optimizer_step_offset=total_optimizer_steps,
            )
            bridges.append(bridge)
            segment_metadata.append(metadata)
            total_optimizer_steps += optimizer_steps

        model = PiecewiseDSBM(
            bridges,
            timepoints=retained,
            times=times,
            center=torch.from_numpy(center),
            scale=scale,
        ).to(device)
        model.eval()
        combined_checkpoint = checkpoint_root / "piecewise_best.pt"
        combined_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": _cpu_state_dict(model),
                "method": method_name,
                "source_checkout": str(
                    Path(run_args.dsbm_dir).expanduser().resolve()
                ),
                "timepoints": list(retained),
                "times": list(times),
                "normalization_center": center,
                "normalization_scale": float(scale),
                "segments": segment_metadata,
                "config": vars(run_args),
            },
            combined_checkpoint,
        )
        best_validation_emd = float(
            np.mean([item["best_validation_emd"] for item in segment_metadata])
        )
        wandb.log(
            {
                "validation/dsbm_endpoint_emd_mean": best_validation_emd,
                "training/total_optimizer_steps": total_optimizer_steps,
                "final_eval/best_validation_emd": best_validation_emd,
            }
        )
        if wandb.run is not None and hasattr(wandb.run, "summary"):
            wandb.run.summary["final_eval/best_checkpoint_path"] = str(
                combined_checkpoint
            )
            wandb.run.summary["validation/dsbm_endpoint_emd_mean"] = (
                best_validation_emd
            )
            wandb.run.summary["final_eval/best_validation_emd"] = (
                best_validation_emd
            )

        run_final_evaluation(
            run_args,
            datamodule,
            model,
            global_step=total_optimizer_steps,
            best_validation_emd=best_validation_emd,
            checkpoint_path=str(combined_checkpoint),
        )
    finally:
        wandb.finish()


def _validate_args(args: argparse.Namespace) -> None:
    if args.dsbm_first_coupling not in {"ind", "ref"}:
        raise ValueError("dsbm_first_coupling must be 'ind' or 'ref'.")
    if int(args.dsbm_outer_iterations) < 2 or int(args.dsbm_outer_iterations) % 2:
        raise ValueError("dsbm_outer_iterations must be a positive even integer >= 2.")
    for name in (
        "batch_size",
        "dsbm_num_steps",
        "dsbm_inner_iterations",
        "dsbm_validation_points",
        "dsbm_validation_every_forward",
        "dsbm_eval_euler_steps",
        "dsbm_eval_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(args.dsbm_train_points_per_marginal) < 0:
        raise ValueError("dsbm_train_points_per_marginal must be non-negative.")
    if int(args.dsbm_eval_points_per_time) < 0:
        raise ValueError("dsbm_eval_points_per_time must be non-negative.")
    if float(args.dsbm_sigma) < 0.0:
        raise ValueError("dsbm_sigma must be non-negative.")
    if float(args.dsbm_learning_rate) <= 0.0:
        raise ValueError("dsbm_learning_rate must be positive.")
    if not 0.0 < float(args.dsbm_time_eps) < 0.5:
        raise ValueError("dsbm_time_eps must lie strictly between 0 and 0.5.")
    if any(int(width) <= 0 for width in args.dsbm_hidden_dims):
        raise ValueError("Every dsbm_hidden_dims value must be positive.")
    if args.data_type == "maizels" and (int(args.dim) != 50 or bool(args.whiten)):
        raise ValueError("Maizels evaluation requires unwhitened PCA50 coordinates.")


def main(argv=None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    args.working_dir = str(Path(args.working_dir).expanduser().resolve())
    args.group_name = secrets.token_urlsafe(12)
    exclusions = list(args.t_exclude or [None])
    for seed in args.seeds:
        for t_exclude in exclusions:
            run_one(args, seed=int(seed), t_exclude=t_exclude)


if __name__ == "__main__":
    main()

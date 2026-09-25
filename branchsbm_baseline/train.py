"""Train the original BranchSBM model on CITE, Multi, or Maizels data."""

from __future__ import annotations

import argparse
import copy
import os
import secrets
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchcfm.optimal_transport import OTPlanSampler


FLOW_MAPS_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = FLOW_MAPS_ROOT / "metric-flow-matching"
PY_ROOT = FLOW_MAPS_ROOT / "py"
for source_root in (MFM_ROOT, PY_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from common import cite_multi, maizels  # noqa: E402
from mfm.dataloaders.maizels_data import MaizelsEndpointDataModule  # noqa: E402
from mfm.dataloaders.trajectory_data import TemporalDataModule  # noqa: E402
from mfm.flow_matchers.cite_multi_eval import (  # noqa: E402
    CiteMultiEvaluationCallback,
    resolve_all_days_classifier_path,
    validate_cite_multi_evaluation_args,
)
from mfm.flow_matchers.maizels_eval import MaizelsEvaluationCallback  # noqa: E402
from mfm.train import parsers as mfm_parsers  # noqa: E402
from mfm.train.train_utils import dataset_name2datapath  # noqa: E402
from mfm.utils import set_seed  # noqa: E402

from branchsbm_baseline.data import BranchEndpointDataModule  # noqa: E402
from branchsbm_baseline.evaluation import (  # noqa: E402
    BranchSBMEvaluationMixin,
    CiteMultiDistributionEvaluationCallback,
    endpoint_maizels_evaluation_view,
)


def _add_branchsbm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument(
        "--working_dir",
        type=str,
        default="./",
        help="Directory for checkpoints, logs, and other run outputs.",
    )
    parser.add_argument(
        "--branchsbm_dir",
        type=str,
        default=os.environ.get(
            "BRANCHSBM_DIR",
            str(Path.home() / "Desktop" / "BranchSBM"),
        ),
        help="Path to the original BranchSBM checkout.",
    )
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--t_exclude", nargs="+", type=int, default=None)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--branches", type=int, default=0)
    parser.add_argument("--metric_clusters", type=int, default=2)
    parser.add_argument(
        "--branchsbm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--manifold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--branchsbm_metric_max_points_per_cluster",
        type=int,
        default=0,
        help="Optional deterministic cap per RBF metric cluster; 0 uses all cells.",
    )
    parser.add_argument(
        "--branchsbm_eval_points_per_time",
        type=int,
        default=0,
        help="Optional CITE/Multi EMD/MMD population cap; 0 uses full populations.",
    )
    parser.add_argument("--branchsbm_eval_euler_steps", type=int, default=100)
    parser.add_argument("--branchsbm_eval_batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dims_growth", nargs="+", type=int, default=[64] * 3)
    parser.add_argument("--activation_growth", type=str, default="tanh")
    parser.add_argument("--growth_optimizer", type=str, default="adamw")
    parser.add_argument("--growth_lr", type=float, default=1e-3)
    parser.add_argument("--growth_weight_decay", type=float, default=1e-5)
    parser.add_argument("--lambda_energy", type=float, default=1.0)
    parser.add_argument("--lambda_mass", type=float, default=100.0)
    parser.add_argument("--lambda_match", type=float, default=1000.0)
    parser.add_argument("--lambda_recons", type=float, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unmodified BranchSBM components on flow-maps benchmarks."
    )
    _add_branchsbm_arguments(parser)
    mfm_parsers.datasets_parser(parser)
    mfm_parsers.metric_parser(parser)
    mfm_parsers.general_training_parser(parser)
    mfm_parsers.geopath_network_parser(parser)
    mfm_parsers.flow_network_parser(parser)
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
            raise TypeError("BranchSBM config must contain a YAML mapping.")

    parser = build_parser()
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"Unknown BranchSBM config fields: {unknown}")
    parser.set_defaults(**config)
    return parser.parse_args(argv)


def _import_original_branchsbm(branchsbm_dir: str):
    root = Path(branchsbm_dir).expanduser().resolve()
    required = (
        root / "src" / "branchsbm.py",
        root / "src" / "branch_interpolant_train.py",
        root / "src" / "branch_flow_net_train.py",
        root / "src" / "branch_growth_net_train.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Could not find the original BranchSBM checkout files: "
            + ", ".join(missing)
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.branch_flow_net_train import FlowNetTrainCell
    from src.branch_growth_net_train import GrowthNetTrainCell
    from src.branch_interpolant_train import BranchInterpolantTrain
    from src.branchsbm import BranchSBM
    from src.geo_metrics.metric_factory import DataManifoldMetric
    from src.networks.flow_mlp import VelocityNet
    from src.networks.growth_mlp import GrowthNet
    from src.networks.interpolant_mlp import GeoPathMLP

    class BranchSBMJointTrain(BranchSBMEvaluationMixin, GrowthNetTrainCell):
        """Original joint trainer plus the repository's evaluation interface."""

    return {
        "BranchSBM": BranchSBM,
        "BranchInterpolantTrain": BranchInterpolantTrain,
        "FlowNetTrainCell": FlowNetTrainCell,
        "GrowthNetTrainCell": GrowthNetTrainCell,
        "BranchSBMJointTrain": BranchSBMJointTrain,
        "DataManifoldMetric": DataManifoldMetric,
        "VelocityNet": VelocityNet,
        "GrowthNet": GrowthNet,
        "GeoPathMLP": GeoPathMLP,
    }


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
    if dataset_args.data_type == "maizels":
        if dataset_args.maizels_schedule != "d3_d3p8_d8":
            raise ValueError(
                "This baseline configuration expects the 3-marginal Maizels "
                "protocol so D3.8 is available to the learned metric."
            )
        base = MaizelsEndpointDataModule(args=dataset_args)
        retained = tuple(base.retained_timepoints)
        class_order = tuple(maizels.CLASS_NAMES)
    elif dataset_args.data_type == "scrna" and dataset_args.data_name in {
        "cite",
        "multi",
    }:
        if t_exclude not in (1, 2):
            raise ValueError("CITE/Multi BranchSBM requires t_exclude 1 or 2.")
        base = TemporalDataModule(
            args=dataset_args,
            skipped_datapoint=t_exclude,
        )
        ordered = tuple(base.timepoint_splits)
        retained = tuple(
            timepoint
            for index, timepoint in enumerate(ordered)
            if index != int(t_exclude)
        )
        class_order = tuple(cite_multi.CLASS_NAMES)
    else:
        raise ValueError("BranchSBM is configured only for CITE, Multi, and Maizels.")

    adapter = BranchEndpointDataModule(
        base,
        retained_timepoints=retained,
        class_order=class_order,
        seed=int(args.seed_current),
        batch_size=int(args.batch_size),
        metric_max_points_per_cluster=int(
            args.branchsbm_metric_max_points_per_cluster
        ),
    )
    return dataset_args, adapter


def _phase_callbacks(
    args: argparse.Namespace,
    *,
    phase: str,
    monitor: str,
    run_id: str,
    evaluation_callbacks=(),
):
    checkpoint_dir = (
        Path(args.working_dir)
        / "checkpoints"
        / "branchsbm"
        / str(run_id)
        / f"{phase}_model"
    )
    patience = int(args.patience_geopath if phase == "geopath" else args.patience)
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best",
        monitor=monitor,
        mode="min",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode="min",
    )
    return [checkpoint, early_stopping, *evaluation_callbacks]


def _trainer_kwargs(args: argparse.Namespace, phase: str):
    if int(args.max_steps) > 0:
        limits = {
            "max_epochs": -1,
            "max_steps": int(args.max_steps),
            "val_check_interval": int(args.val_check_interval),
            "check_val_every_n_epoch": None,
        }
    else:
        limits = {
            "max_epochs": int(args.epochs),
            "check_val_every_n_epoch": (
                1 if phase == "geopath" else int(args.check_val_every_n_epoch)
            ),
        }
    return {
        **limits,
        "accelerator": args.accelerator,
        "default_root_dir": args.working_dir,
        "num_sanity_val_steps": 0,
    }


def _restore_best(model: pl.LightningModule, checkpoint: ModelCheckpoint) -> str:
    path = str(checkpoint.best_model_path or "")
    if not path:
        raise RuntimeError("Training completed without a best checkpoint.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    return path


def _fit_stage(
    args,
    *,
    phase,
    monitor,
    model,
    datamodule,
    wandb_logger,
    run_id,
    evaluation_callbacks=(),
):
    callbacks = _phase_callbacks(
        args,
        phase=phase,
        monitor=monitor,
        run_id=run_id,
        evaluation_callbacks=evaluation_callbacks,
    )
    trainer = pl.Trainer(
        **_trainer_kwargs(args, phase),
        callbacks=callbacks,
        logger=wandb_logger,
    )
    trainer.fit(model, datamodule=datamodule)
    return trainer, callbacks[0]


def _evaluation_callbacks(args, datamodule):
    if args.data_type == "maizels":
        endpoint_view = endpoint_maizels_evaluation_view(datamodule)
        return [MaizelsEvaluationCallback(args=args, datamodule=endpoint_view)]
    return [
        CiteMultiDistributionEvaluationCallback(args=args, datamodule=datamodule),
        CiteMultiEvaluationCallback(args=args, datamodule=datamodule),
    ]


def run_one(args: argparse.Namespace, *, seed: int, t_exclude: int | None) -> None:
    run_args = copy.deepcopy(args)
    run_args.seed_current = int(seed)
    run_args.t_exclude_current = t_exclude
    gamma_index = 0 if t_exclude is None else list(args.t_exclude).index(t_exclude)
    run_args.gamma_current = float(args.gammas[min(gamma_index, len(args.gammas) - 1)])
    run_args.working_dir = str(Path(args.working_dir).expanduser().resolve())
    Path(run_args.working_dir).mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    dataset_args, datamodule = _build_data(run_args, t_exclude)
    run_args.data_path = dataset_args.data_path
    branch_args = copy.deepcopy(run_args)
    # The original RBF module has separate unpacking branches for the author's
    # ``lightning`` and ``pytorch_lightning`` CombinedLoader variants.  This
    # adapter uses the latter (as does metric-flow-matching), whose native batch
    # is the generic dictionary form selected by this internal tag.
    branch_args.data_type = "flowmaps"
    branch_args.data_path = dataset_args.data_path
    branch_args.branches = datamodule.num_branches
    branch_args.metric_clusters = 2
    branch_args.run_name = run_args.run_name

    if run_args.data_type == "scrna":
        validate_cite_multi_evaluation_args(run_args)
        run_args.cite_multi_eval_classifier_path = str(
            resolve_all_days_classifier_path(run_args)
        )
        branch_args.cite_multi_eval_classifier_path = (
            run_args.cite_multi_eval_classifier_path
        )

    heldout_suffix = (
        ""
        if t_exclude is None
        else f"_holdout_day{tuple(datamodule.timepoint_splits)[t_exclude]}"
    )
    default_name = (
        f"{run_args.data_name}_branchsbm{heldout_suffix}_seed{seed}"
    )
    run_name = run_args.wandb_name or run_args.run_name or default_name
    run_args.run_name = run_name
    branch_args.run_name = run_name

    components = _import_original_branchsbm(run_args.branchsbm_dir)
    run = wandb.init(
        project=run_args.wandb_project or "self-distill-flow-maps",
        entity=run_args.wandb_entity or None,
        name=run_name,
        group=run_args.group_name,
        config=vars(run_args),
        dir=run_args.working_dir,
    )
    wandb.config.update(
        {
            "method": "branchsbm",
            "evaluation_sampler": "branchsbm",
            "branchsbm_source_checkout": str(
                Path(run_args.branchsbm_dir).expanduser().resolve()
            ),
            "protocol": (
                "D3_to_D8_endpoint_metric_sees_D3p8"
                if run_args.data_type == "maizels"
                else "D2_to_D7_endpoint_metric_sees_retained_middle_day"
            ),
            "branches": int(datamodule.num_branches),
            "source_timepoint": datamodule.source_timepoint,
            "terminal_timepoint": datamodule.terminal_timepoint,
            "retained_metric_timepoints": list(datamodule.retained_timepoints),
            "terminal_branch_metadata": datamodule.branch_metadata,
            "terminal_split_repairs": datamodule.terminal_split_repairs,
            "metric_group_sizes": list(datamodule.metric_group_sizes),
            "small_terminal_branches_filtered": False,
        },
        allow_val_change=True,
    )
    print("BranchSBM terminal branches:")
    for metadata in datamodule.branch_metadata:
        print(
            "  branch {branch_index}: {cell_type} "
            "(train={train_count}, val={holdout_count}, mass={target_mass:.6f})".format(
                **metadata
            )
        )
    if datamodule.terminal_split_repairs:
        print(
            "Adjusted the terminal validation split to retain every rare "
            f"cell type on both sides: {datamodule.terminal_split_repairs}"
        )

    wandb_logger = WandbLogger(experiment=run)
    try:
        flow_nets = nn.ModuleList()
        geopath_nets = nn.ModuleList()
        growth_nets = nn.ModuleList()
        for branch_index in range(datamodule.num_branches):
            flow_nets.append(
                components["VelocityNet"](
                    dim=branch_args.dim,
                    hidden_dims=branch_args.hidden_dims_flow,
                    activation=branch_args.activation_flow,
                    batch_norm=False,
                )
            )
            geopath_nets.append(
                components["GeoPathMLP"](
                    input_dim=branch_args.dim,
                    hidden_dims=branch_args.hidden_dims_geopath,
                    time_geopath=branch_args.time_geopath,
                    activation=branch_args.activation_geopath,
                    batch_norm=False,
                )
            )
            growth_nets.append(
                components["GrowthNet"](
                    dim=branch_args.dim,
                    hidden_dims=branch_args.hidden_dims_growth,
                    activation=branch_args.activation_growth,
                    batch_norm=False,
                    negative=branch_index == 0,
                )
            )

        ot_sampler = (
            None
            if str(branch_args.optimal_transport_method).lower() == "none"
            else OTPlanSampler(method=branch_args.optimal_transport_method)
        )
        flow_matcher = components["BranchSBM"](
            geopath_nets=geopath_nets,
            sigma=branch_args.sigma,
            alpha=int(branch_args.branchsbm),
        )
        data_metric = components["DataManifoldMetric"](
            args=branch_args,
            skipped_time_points=[],
            datamodule=datamodule,
        )

        geopath_model = components["BranchInterpolantTrain"](
            flow_matcher=flow_matcher,
            skipped_time_points=[],
            ot_sampler=ot_sampler,
            args=branch_args,
            data_manifold_metric=data_metric,
        )
        _, checkpoint = _fit_stage(
            branch_args,
            phase="geopath",
            monitor="BranchPathNet/val_loss_geopath",
            model=geopath_model,
            datamodule=datamodule,
            wandb_logger=wandb_logger,
            run_id=run.id,
        )
        _restore_best(geopath_model, checkpoint)
        flow_matcher.geopath_nets = geopath_model.geopath_nets

        flow_model = components["FlowNetTrainCell"](
            flow_matcher=flow_matcher,
            flow_nets=flow_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=[],
            args=branch_args,
        )
        # BranchSBM's flow matcher is not an nn.Module, so its geopath nets are
        # otherwise invisible to Lightning's device transfer at this stage.
        # Register the already-trained nets on the flow-stage module while
        # keeping them frozen, as intended by the staged BranchSBM procedure.
        for parameter in flow_matcher.geopath_nets.parameters():
            parameter.requires_grad_(False)
        flow_model.geopath_nets = flow_matcher.geopath_nets
        _, checkpoint = _fit_stage(
            branch_args,
            phase="flow",
            monitor="FlowNet/val_loss_cfm",
            model=flow_model,
            datamodule=datamodule,
            wandb_logger=wandb_logger,
            run_id=run.id,
        )
        _restore_best(flow_model, checkpoint)
        flow_nets = flow_model.flow_nets

        growth_model = components["GrowthNetTrainCell"](
            flow_nets=flow_nets,
            growth_nets=growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=[],
            args=branch_args,
            data_manifold_metric=data_metric,
            joint=False,
        )
        _, checkpoint = _fit_stage(
            branch_args,
            phase="growth",
            monitor="GrowthNet/val_loss",
            model=growth_model,
            datamodule=datamodule,
            wandb_logger=wandb_logger,
            run_id=run.id,
        )
        _restore_best(growth_model, checkpoint)
        flow_nets = growth_model.flow_nets
        growth_nets = growth_model.growth_nets

        # The preceding growth-only stage freezes the shared velocity objects.
        # Restore their trainability before invoking BranchSBM's intended joint
        # optimization stage; no loss or network implementation is changed.
        for parameter in flow_nets.parameters():
            parameter.requires_grad_(True)

        joint_model = components["BranchSBMJointTrain"](
            flow_nets=flow_nets,
            growth_nets=growth_nets,
            ot_sampler=ot_sampler,
            skipped_time_points=[],
            args=branch_args,
            data_manifold_metric=data_metric,
            joint=True,
            terminal_branch_weights=datamodule.branch_weights,
        )
        trainer, checkpoint = _fit_stage(
            branch_args,
            phase="joint",
            monitor="JointTrain/val_loss",
            model=joint_model,
            datamodule=datamodule,
            wandb_logger=wandb_logger,
            run_id=run.id,
            evaluation_callbacks=_evaluation_callbacks(run_args, datamodule),
        )
        trainer.test(joint_model, datamodule=datamodule, ckpt_path="best")
        if wandb.run is not None and hasattr(wandb.run, "summary"):
            wandb.run.summary["final_eval/best_checkpoint_path"] = (
                checkpoint.best_model_path
            )
    finally:
        wandb.finish()


def main(argv=None) -> None:
    args = parse_args(argv)
    args.working_dir = str(Path(args.working_dir).expanduser().resolve())
    args.group_name = secrets.token_urlsafe(12)
    if not bool(args.branchsbm):
        raise ValueError("The BranchSBM baseline requires branchsbm=true.")
    if not bool(args.manifold):
        raise ValueError(
            "These benchmark configs require BranchSBM's learned manifold "
            "metric so the retained middle marginal is used."
        )
    if int(args.metric_clusters) != 2:
        raise ValueError(
            "The adapter uses two metric clusters: source and all retained "
            "post-source observations."
        )
    if int(args.branchsbm_metric_max_points_per_cluster) < 0:
        raise ValueError("branchsbm_metric_max_points_per_cluster must be non-negative.")
    if int(args.branchsbm_eval_points_per_time) < 0:
        raise ValueError("branchsbm_eval_points_per_time must be non-negative.")
    if int(args.branchsbm_eval_batch_size) <= 0:
        raise ValueError("branchsbm_eval_batch_size must be positive.")
    if args.data_type == "maizels" and (
        int(args.dim) != 50 or bool(args.whiten)
    ):
        raise ValueError("Maizels evaluation requires unwhitened PCA50 coordinates.")

    exclusions = list(args.t_exclude or [None])
    for seed in args.seeds:
        for t_exclude in exclusions:
            run_one(args, seed=int(seed), t_exclude=t_exclude)


if __name__ == "__main__":
    main()

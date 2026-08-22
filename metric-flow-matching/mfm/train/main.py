import argparse
import copy
import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from torchcfm.optimal_transport import OTPlanSampler

from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from mfm.flow_matchers.flow_net_train import (
    FlowNetTrainTrajectory,
    FlowNetTrainLidar,
    FlowNetTrainImage,
)
from mfm.flow_matchers.geopath_net_train import GeoPathNetTrain
from mfm.dataloaders.trajectory_data import TemporalDataModule
from mfm.dataloaders.maizels_data import MaizelsEndpointDataModule
from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.geopath_networks.mlp import GeoPathMLP
from mfm.utils import set_seed
from mfm.train.parsers import parse_args
from mfm.flow_matchers.ema import EMA
from mfm.train.train_utils import (
    load_config,
    merge_config,
    generate_group_string,
    dataset_name2datapath,
    create_callbacks,
)


def trainer_limit_kwargs(args, *, check_val_every_n_epoch: int = 1):
    if args.max_steps > 0:
        return {
            "max_epochs": -1,
            "max_steps": args.max_steps,
            "val_check_interval": args.val_check_interval,
            "check_val_every_n_epoch": None,
        }
    return {
        "max_epochs": args.epochs,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }


def phase_accelerator(args, phase: str) -> str:
    """Resolve phase devices, avoiding unsupported geopath JVP backward on MPS."""
    override = str(getattr(args, f"{phase}_accelerator", "auto") or "auto")
    if override != "auto":
        return override

    requested = str(args.accelerator)
    if (
        phase == "geopath"
        and args.data_type == "maizels"
        and torch.backends.mps.is_available()
        and requested in ("auto", "gpu", "mps")
    ):
        return "cpu"
    return requested


def torch_device_for_accelerator(accelerator: str) -> torch.device:
    accelerator = str(accelerator)
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "mps":
        return torch.device("mps")
    if accelerator == "cuda":
        return torch.device("cuda")
    if accelerator in ("auto", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def main(args: argparse.Namespace, seed: int, t_exclude: int) -> None:
    set_seed(seed)
    if args.data_type == "lidar":
        assert args.dim == 3 and args.data_name == "lidar"
    elif args.data_type == "arch":
        assert args.dim == 2
    elif args.data_type == "sphere":
        assert args.dim == 3
    elif args.data_type == "image":
        assert not args.whiten
        assert args.data_name == "afhq"
    elif args.data_type == "maizels":
        assert args.dim == 50
        assert not args.whiten

    skipped_time_points = [t_exclude] if t_exclude else []
    geopath_accelerator = phase_accelerator(args, "geopath")
    flow_accelerator = phase_accelerator(args, "flow")
    if geopath_accelerator != args.accelerator:
        print(
            "Using CPU for metric/geopath training because PyTorch's "
            "time-JVP backward is unsupported on Apple MPS."
        )

    ### DATAMODULES
    if args.data_type in ["arch", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )
    elif args.data_type == "maizels":
        datamodule = MaizelsEndpointDataModule(args=args)
    elif args.data_type == "lidar":
        from mfm.dataloaders.lidar_data import LidarDataModule

        datamodule = LidarDataModule(args=args)
    elif args.data_type == "image":
        from mfm.dataloaders.image_data import ImageDataModule

        datamodule = ImageDataModule(args=args)
    else:
        raise ValueError("Data type not recognized")

    ### Interpolation and Vector Field Networks
    if args.data_type in ["arch", "scrna", "maizels", "lidar", "sphere"]:
        flow_net = VelocityNet(
            dim=args.dim,
            hidden_dims=args.hidden_dims_flow,
            activation=args.activation_flow,
            batch_norm=False,
        )
        geopath_net = GeoPathMLP(
            input_dim=args.dim,
            hidden_dims=args.hidden_dims_geopath,
            time_geopath=args.time_geopath,
            activation=args.activation_geopath,
            batch_norm=False,
        )
    elif args.data_type == "image":
        from mfm.networks.unet_base import UNetModelWrapper as UNetModel
        from mfm.networks.geopath_networks.unet import GeoPathUNet

        flow_net = UNetModel(
            geopath_model=False,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels,
            num_res_blocks=args.unet_num_res_blocks,
            channel_mult=args.unet_channel_mult,
            dropout=args.unet_dropout,
            resblock_updown=args.unet_resblock_updown,
            use_new_attention_order=args.unet_use_new_attention_order,
            attention_resolutions=args.unet_attention_resolutions,
            num_heads=args.unet_num_heads,
        )
        geopath_net = GeoPathUNet(
            geopath_model=True,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels_geopath,
            num_res_blocks=args.unet_num_res_blocks_geopath,
            channel_mult=args.unet_channel_mult_geopath,
            dropout=args.unet_dropout_geopath,
            use_checkpoint=False,
        )

    if args.ema_decay is not None:
        flow_net = EMA(model=flow_net, decay=args.ema_decay)
        geopath_net = EMA(model=geopath_net, decay=args.ema_decay)

    ot_sampler = (
        OTPlanSampler(method=args.optimal_transport_method)
        if args.optimal_transport_method != "None" and args.data_type != "maizels"
        else None
    )

    project = args.wandb_project or (
        "self-distill-flow-maps"
        if args.data_type == "maizels"
        else f"mfm-{args.data_type}-{args.data_name}"
    )
    run_name = args.wandb_name or (
        f"maizels_pca50_mfm_{args.maizels_pair_mode}_seed{seed}"
        if args.data_type == "maizels"
        else None
    )
    wandb.init(
        project=project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.group_name,
        config=vars(args),
        dir=args.working_dir,
    )
    if args.data_type == "maizels":
        wandb.config.update(
            {
                "method": "mfm",
                "protocol": "D3_to_D8_endpoint",
                "maizels_pair_mode_requested": datamodule.requested_pair_mode,
                "maizels_geopath_pair_mode": datamodule.geopath_pair_mode,
                "maizels_interpolant_check_times": int(
                    args.maizels_interpolant_check_times
                ),
                "geopath_accelerator_resolved": geopath_accelerator,
                "flow_accelerator_resolved": flow_accelerator,
                "maizels_pair_stats": datamodule.pair_stats,
                "maizels_validation_pair_stats": datamodule.validation_pair_stats,
                "maizels_eval_pair_stats": datamodule.eval_pair_stats,
            },
            allow_val_change=True,
        )

    ### Metric Flow Matching Module
    flow_matcher_base = MetricFlowMatcher(
        geopath_net=geopath_net,
        sigma=args.sigma,
        alpha=int(args.mfm),
    )

    ##### ALGO 1: Training of Geodesic Interpolants Beginning #####
    if args.mfm:
        data_manifold_metric = DataManifoldMetric(
            args=args,
            skipped_time_points=skipped_time_points,
            datamodule=datamodule,
            accelerator=geopath_accelerator,
        )
        geopath_callbacks = create_callbacks(
            args, phase="geopath", data_type=args.data_type, run_id=wandb.run.id
        )

        geopath_model = GeoPathNetTrain(
            flow_matcher=flow_matcher_base,
            skipped_time_points=skipped_time_points,
            ot_sampler=ot_sampler,
            data_manifold_metric=data_manifold_metric,
            args=args,
        )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            **trainer_limit_kwargs(args),
            callbacks=geopath_callbacks,
            accelerator=geopath_accelerator,
            logger=wandb_logger,
            num_sanity_val_steps=0,
            default_root_dir=args.working_dir,
            gradient_clip_val=(1.0 if args.data_type == "image" else None),
        )
        if args.load_geopath_model_ckpt:
            best_model_path = args.load_geopath_model_ckpt
        else:
            trainer.fit(
                geopath_model,
                datamodule=datamodule,
            )
            best_model_path = geopath_callbacks[0].best_model_path
        geopath_model = GeoPathNetTrain.load_from_checkpoint(
            best_model_path,
            flow_matcher=flow_matcher_base,
            skipped_time_points=skipped_time_points,
            ot_sampler=ot_sampler,
            data_manifold_metric=data_manifold_metric,
            args=args,
        )

        flow_matcher_base.geopath_net = geopath_model.geopath_net

        if args.data_type == "maizels":
            rebuilt = datamodule.apply_learned_geopath_prior(
                geopath_model.geopath_net,
                device=torch_device_for_accelerator(flow_accelerator),
            )
            if rebuilt:
                print(
                    "Rebuilt Maizels velocity-training pairs by checking "
                    f"{args.maizels_interpolant_check_times} points along each "
                    "frozen learned geodesic."
                )
            wandb.config.update(
                {
                    "maizels_pair_mode_active_for_flow": datamodule.active_pair_mode,
                    "maizels_geopath_pair_stats": datamodule.geopath_pair_stats,
                    "maizels_geopath_validation_pair_stats": (
                        datamodule.geopath_validation_pair_stats
                    ),
                    "maizels_geopath_eval_pair_stats": datamodule.geopath_eval_pair_stats,
                    "maizels_pair_stats": datamodule.pair_stats,
                    "maizels_validation_pair_stats": datamodule.validation_pair_stats,
                    "maizels_eval_pair_stats": datamodule.eval_pair_stats,
                },
                allow_val_change=True,
            )

    ##### ALGO 1: Training of Geodesic Interpolants END #####

    ##### ALGO 2: (Metric) Flow Matching Beginning #####
    if args.data_type in ["arch", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )
    flow_callbacks = create_callbacks(
        args,
        phase="flow",
        data_type=args.data_type,
        run_id=wandb.run.id,
        datamodule=datamodule,
    )

    if args.data_type in ["arch", "scrna", "maizels", "sphere"]:
        FlowNetTrain = FlowNetTrainTrajectory
    elif args.data_type == "lidar":
        FlowNetTrain = FlowNetTrainLidar
    elif args.data_type == "image":
        FlowNetTrain = FlowNetTrainImage
    else:
        raise ValueError("Data type not recognized")

    flow_train = FlowNetTrain(
        flow_matcher=flow_matcher_base,
        flow_net=flow_net,
        ot_sampler=ot_sampler,
        skipped_time_points=skipped_time_points,
        args=args,
    )

    wandb_logger = WandbLogger()

    trainer = Trainer(
        **trainer_limit_kwargs(
            args,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
        ),
        callbacks=flow_callbacks,
        accelerator=flow_accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )

    trainer.fit(
        flow_train, datamodule=datamodule, ckpt_path=args.resume_flow_model_ckpt
    )
    trainer.test(flow_train, datamodule=datamodule)
    wandb.finish()
    ##### ALGO 2: (Metric) Flow Matching END #####


if __name__ == "__main__":
    args = parse_args()
    updated_args = copy.deepcopy(args)
    if args.config_path:
        config = load_config(args.config_path)
        updated_args = merge_config(updated_args, config)

    updated_args.group_name = generate_group_string()
    if updated_args.data_type == "maizels":
        updated_args.data_path = updated_args.maizels_dataset_path
    else:
        updated_args.data_path = dataset_name2datapath(
            updated_args.data_name, updated_args.working_dir
        )
    for seed in updated_args.seeds:
        if updated_args.t_exclude:
            for i, t_exclude in enumerate(updated_args.t_exclude):
                updated_args.t_exclude_current = t_exclude
                updated_args.seed_current = seed
                updated_args.gamma_current = updated_args.gammas[i]
                main(updated_args, seed=seed, t_exclude=t_exclude)
        else:
            updated_args.seed_current = seed
            updated_args.gamma_current = updated_args.gammas[0]
            main(updated_args, seed=seed, t_exclude=None)

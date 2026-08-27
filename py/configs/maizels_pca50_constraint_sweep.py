"""Seed sweep for Maizels bio-prior flow-map lineage constraints.

The seed is controlled by MAIZELS_SEED so shell scripts can sweep seeds without
writing temporary config modules.
"""

import os

from configs.maizels_pca50 import get_config as _base_get_config


VARIANTS = [
    {
        "mode": "vanilla_flow_map",
        "base_slurm_id": 1,
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "vanilla_flow_matching",
        "base_slurm_id": 0,
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_flow_map",
        "base_slurm_id": 3,
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_flow_matching",
        "base_slurm_id": 2,
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "ot_flow_map",
        "base_slurm_id": 3,
        "pair_mode": "ot_plain",
        "constraints_enabled": False,
        "path_mode": "loss_points_nll",
        "weight": 0.0,
    },
    {
        "mode": "ot_flow_matching",
        "base_slurm_id": 2,
        "pair_mode": "ot_plain",
        "constraints_enabled": False,
        "path_mode": "loss_points_nll",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_ot_flow_map",
        "base_slurm_id": 3,
        "pair_mode": "ot_endpoint_interpolant",
        "constraints_enabled": False,
        "path_mode": "loss_points_nll",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_ot_flow_matching",
        "base_slurm_id": 2,
        "pair_mode": "ot_endpoint_interpolant",
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_constrained_flow_map_w1000_e01",
        "base_slurm_id": 4,
        "pair_mode": "endpoint_interpolant",
        "constraints_enabled": True,
        "path_mode": "loss_points_nll",
        "weight": 1000.0,
        "loss_point_entropy_weight": 0.1,
    },
    {
        "mode": "bio_prior_ot_constrained_flow_map_w1000_e01",
        "base_slurm_id": 4,
        "pair_mode": "ot_endpoint_interpolant",
        "constraints_enabled": True,
        "path_mode": "loss_points_nll",
        "weight": 1000.0,
        "loss_point_entropy_weight": 0.1,
    },
    {
        "mode": "bio_prior_constrained_flow_matching_w1000_e01",
        "base_slurm_id": 2,
        "pair_mode": "endpoint_interpolant",
        "constraints_enabled": True,
        "path_mode": "velocity_loss_points_nll",
        "weight": 1000.0,
        "loss_point_entropy_weight": 0.1,
        "diag_fraction": 1.0,
        "velocity_rollout_loss_scope": "endpoints",
    },
    {
        "mode": "bio_prior_ot_constrained_flow_matching_w350_e001",
        "base_slurm_id": 2,
        "pair_mode": "ot_endpoint_interpolant",
        "constraints_enabled": True,
        "path_mode": "velocity_loss_points_nll",
        "weight": 350.0,
        "loss_point_entropy_weight": 0.01,
        "diag_fraction": 1.0,
        "velocity_rollout_loss_scope": "endpoints",
    },
]


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
    classifier_path=None,
    maizels_schedule=None,
    maizels_time_mode=None,
):
    variant = VARIANTS[slurm_id % len(VARIANTS)]
    cfg = _base_get_config(
        variant["base_slurm_id"],
        dataset_location,
        output_folder,
        early_stopping_patience=early_stopping_patience,
        maizels_ot_coupling=maizels_ot_coupling,
        classifier_path=classifier_path,
        maizels_schedule=maizels_schedule,
        maizels_time_mode=maizels_time_mode,
    )
    seed = int(os.getenv("MAIZELS_SEED", str(cfg.training.seed)))

    cfg.training.seed = seed
    cfg.problem.maizels_pair_mode = variant.get(
        "pair_mode", cfg.problem.maizels_pair_mode
    )
    cfg.constraints.enabled = variant["constraints_enabled"]
    cfg.constraints.path_mode = variant["path_mode"]
    cfg.constraints.weight = variant["weight"]
    cfg.constraints.loss_point_entropy_weight = variant.get(
        "loss_point_entropy_weight", 0.0
    )
    cfg.constraints.velocity_rollout_loss_scope = variant.get(
        "velocity_rollout_loss_scope",
        getattr(cfg.constraints, "velocity_rollout_loss_scope", "endpoints"),
    )
    cfg.constraints.velocity_rollout_batch_size = variant.get(
        "velocity_rollout_batch_size",
        getattr(cfg.constraints, "velocity_rollout_batch_size", 0),
    )
    cfg.constraints.velocity_rollout_reference_diag_fraction = variant.get(
        "velocity_rollout_reference_diag_fraction",
        getattr(cfg.constraints, "velocity_rollout_reference_diag_fraction", 0.75),
    )
    cfg.constraints.velocity_rollout_max_step = variant.get(
        "velocity_rollout_max_step",
        getattr(cfg.constraints, "velocity_rollout_max_step", 0.05),
    )
    cfg.constraints.velocity_rollout_max_steps = variant.get(
        "velocity_rollout_max_steps",
        getattr(cfg.constraints, "velocity_rollout_max_steps", 0),
    )
    cfg.optimization.diag_fraction = variant.get("diag_fraction", 0.75)

    mode = variant["mode"]
    run_name = f"maizels_pca50_{mode}_seed{seed}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = mode

    return cfg

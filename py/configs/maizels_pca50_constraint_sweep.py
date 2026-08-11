"""Seed sweep for Maizels bio-prior flow-map lineage constraints.

The seed is controlled by MAIZELS_SEED so shell scripts can sweep seeds without
writing temporary config modules.
"""

import os

from configs.maizels_pca50 import get_config as _base_get_config


VARIANTS = [
    {
        "mode": "bio_prior_flow_map_unconstrained",
        "base_slurm_id": 3,
        "constraints_enabled": False,
        "path_mode": "loss_points",
        "weight": 0.0,
    },
    {
        "mode": "bio_prior_flow_map_constrained_loss_points_w1000",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "loss_points",
        "weight": 1000.0,
    },
    {
        "mode": "bio_prior_flow_map_constrained_direct_w100",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "direct",
        "weight": 100.0,
    },
    {
        "mode": "bio_prior_flow_map_constrained_direct_w1000_diag_06",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "direct",
        "weight": 1000.0,
        "diag_fraction": 0.6
    },
    {
        "mode": "bio_prior_flow_map_constrained_direct_w1000_diag_075",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "direct",
        "weight": 1000.0,
        "diag_fraction": 0.75
    },
    {
        "mode": "bio_prior_flow_map_constrained_direct_w1000_diag_085",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "direct",
        "weight": 1000.0,
        "diag_fraction": 0.85
    },

    {
        "mode": "bio_prior_flow_map_constrained_rollout_w1000",
        "base_slurm_id": 4,
        "constraints_enabled": True,
        "path_mode": "flowmap",
        "weight": 10.0,
        "diag_fraction": 0.8
    },
]


def get_config(slurm_id: int, dataset_location: str = "", output_folder: str = ""):
    variant = VARIANTS[slurm_id % len(VARIANTS)]
    cfg = _base_get_config(variant["base_slurm_id"], dataset_location, output_folder)
    seed = int(os.getenv("MAIZELS_SEED", str(cfg.training.seed)))

    cfg.training.seed = seed
    cfg.constraints.enabled = variant["constraints_enabled"]
    cfg.constraints.path_mode = variant["path_mode"]
    cfg.constraints.weight = variant["weight"]
    cfg.optimization.diag_fraction = variant.get("diag_fraction", 0.75)

    mode = variant["mode"]
    run_name = f"maizels_pca50_{mode}_seed{seed}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = mode

    return cfg

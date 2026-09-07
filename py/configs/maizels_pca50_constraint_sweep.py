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
        "mode": "bio_prior_ot_constrained_flow_map_w350_e001",
        "base_slurm_id": 4,
        "pair_mode": "ot_endpoint_interpolant",
        "constraints_enabled": True,
        "path_mode": "loss_points_nll",
        "weight": 350.0,
        "loss_point_entropy_weight": 0.01,
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


def get_hparam_sweep_spec(slurm_id: int) -> dict:
    """Describe the hyperparameters consumed by a constraint-sweep variant."""
    variant = VARIANTS[int(slurm_id) % len(VARIANTS)]
    constraints_enabled = bool(variant["constraints_enabled"])
    return {
        "variant_name": variant["mode"],
        "learning_rate": True,
        "constraint_weight": constraints_enabled,
        "entropy_weight": bool(
            constraints_enabled and "nll" in str(variant["path_mode"])
        ),
    }


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
    classifier_path=None,
    maizels_schedule=None,
    maizels_time_mode=None,
    hparam_val_times=None,
    learning_rate=None,
    constraint_weight=None,
    entropy_weight=None,
    seed=None,
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
        hparam_val_times=hparam_val_times,
        learning_rate=learning_rate,
        seed=seed,
    )
    seed = int(
        seed if seed is not None else os.getenv("MAIZELS_SEED", str(cfg.training.seed))
    )

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
    if constraint_weight is not None:
        if not cfg.constraints.enabled:
            raise ValueError(
                "constraint_weight is not relevant for this unconstrained Slurm ID."
            )
        if float(constraint_weight) < 0:
            raise ValueError("constraint_weight must be non-negative.")
        cfg.constraints.weight = float(constraint_weight)
    if entropy_weight is not None:
        entropy_relevant = bool(
            cfg.constraints.enabled and "nll" in str(cfg.constraints.path_mode)
        )
        if not entropy_relevant:
            raise ValueError(
                "entropy_weight is relevant only to an enabled NLL lineage constraint."
            )
        if float(entropy_weight) < 0:
            raise ValueError("entropy_weight must be non-negative.")
        cfg.constraints.loss_point_entropy_weight = float(entropy_weight)
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
    override_tags = []
    if learning_rate is not None:
        override_tags.append(f"lr{float(learning_rate):g}")
    if constraint_weight is not None:
        override_tags.append(f"cw{float(constraint_weight):g}")
    if entropy_weight is not None:
        override_tags.append(f"ew{float(entropy_weight):g}")
    if override_tags:
        suffix = "_".join(override_tags).replace(".", "p").replace("-", "m")
        run_name = f"{run_name}_{suffix}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = mode

    return cfg

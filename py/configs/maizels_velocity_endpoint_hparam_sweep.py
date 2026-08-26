"""Hyperparameter sweep config for Maizels OT velocity-endpoint constraints.

The launcher sets the weight, entropy penalty, and seed through environment
variables so every run receives a unique, descriptive W&B/checkpoint name.
"""

import os

from configs.maizels_pca50 import get_config as _base_get_config


RUN_PREFIX = "maizels_pca50_bio_prior_ot_velocity_endpoint"


def _value_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
):
    del slurm_id

    # Base variant 2 is bio-prior flow matching. The sweep explicitly replaces
    # its pairing with OT-filtered, path-valid endpoint interpolants.
    cfg = _base_get_config(
        2,
        dataset_location,
        output_folder,
        early_stopping_patience=early_stopping_patience,
        maizels_ot_coupling=maizels_ot_coupling,
    )
    seed = int(os.getenv("MAIZELS_SEED", str(cfg.training.seed)))
    weight = float(os.environ["MAIZELS_CONSTRAINT_WEIGHT"])
    entropy_weight = float(os.environ["MAIZELS_ENTROPY_WEIGHT"])

    cfg.training.seed = seed
    cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
    ot_cache_dir = os.getenv("MAIZELS_OT_CACHE_DIR", "")
    if ot_cache_dir:
        cfg.problem.ot_cache_dir = ot_cache_dir

    cfg.optimization.diag_fraction = 1.0
    cfg.constraints.enabled = True
    cfg.constraints.path_mode = "velocity_loss_points_nll"
    cfg.constraints.velocity_rollout_loss_scope = "endpoints"
    cfg.constraints.weight = weight
    cfg.constraints.loss_point_entropy_weight = entropy_weight

    # The requested comparison metrics are expensive visual-time diagnostics.
    # Evaluate them once at the final step while retaining scalar training logs.
    cfg.logging.visual_freq = cfg.optimization.total_steps
    cfg.logging.wandb_project = os.getenv(
        "WANDB_PROJECT", cfg.logging.wandb_project
    )
    cfg.logging.wandb_entity = os.getenv("WANDB_ENTITY", cfg.logging.wandb_entity)

    setting_name = (
        f"bio_prior_ot_velocity_endpoint_w{_value_label(weight)}"
        f"_ent{_value_label(entropy_weight)}"
    )
    run_name = f"{RUN_PREFIX}_w{_value_label(weight)}_ent{_value_label(entropy_weight)}_seed{seed}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = setting_name

    return cfg

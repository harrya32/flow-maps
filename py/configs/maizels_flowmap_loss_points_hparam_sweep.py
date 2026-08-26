"""Maizels OT flow-map sweep with finite-transition NLL constraints."""

import os

from configs.maizels_pca50 import get_config as _base_get_config


RUN_PREFIX = "maizels_pca50_bio_prior_ot_flowmap_loss_points_nll"


def _value_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
):
    del slurm_id

    # Variant 4 supplies the LSD flow-map setup. The values relevant to this
    # sweep are repeated explicitly to keep its experimental contract obvious.
    cfg = _base_get_config(
        4,
        dataset_location,
        output_folder,
        early_stopping_patience=early_stopping_patience,
    )
    seed = int(os.getenv("MAIZELS_SEED", str(cfg.training.seed)))
    weight = float(os.environ["MAIZELS_CONSTRAINT_WEIGHT"])
    entropy_weight = float(os.environ["MAIZELS_ENTROPY_WEIGHT"])

    cfg.training.seed = seed
    cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
    ot_cache_dir = os.getenv("MAIZELS_OT_CACHE_DIR", "")
    if ot_cache_dir:
        cfg.problem.ot_cache_dir = ot_cache_dir

    cfg.optimization.diag_fraction = 0.75
    cfg.constraints.enabled = True
    cfg.constraints.path_mode = "loss_points_nll"
    cfg.constraints.weight = weight
    cfg.constraints.loss_point_entropy_weight = entropy_weight

    # Direct, composed-flow-map, and Euler diagnostics are all produced by the
    # visual logger. Running them once at the final step keeps the sweep cheap.
    cfg.logging.visual_freq = cfg.optimization.total_steps
    cfg.logging.wandb_project = os.getenv(
        "WANDB_PROJECT", cfg.logging.wandb_project
    )
    cfg.logging.wandb_entity = os.getenv("WANDB_ENTITY", cfg.logging.wandb_entity)

    setting_name = (
        f"bio_prior_ot_flowmap_loss_points_nll_w{_value_label(weight)}"
        f"_ent{_value_label(entropy_weight)}"
    )
    run_name = (
        f"{RUN_PREFIX}_w{_value_label(weight)}"
        f"_ent{_value_label(entropy_weight)}_seed{seed}"
    )
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = setting_name

    return cfg

"""Four-mode comparison config for the box-avoiding Bezier experiment.

The seed is controlled by BOX_BEZIER_SEED so shell scripts can sweep seeds
without writing temporary config modules.
"""

import os

from configs.box_avoiding_bezier import get_config as _base_get_config


VARIANTS = [
    {
        "mode": "vanilla-flow-matching",
        "diag_fraction": 1.0,
        "constraints_enabled": False,
        "constraint_mode": "flow_matching",
    },
    {
        "mode": "vanilla-flow-map",
        "diag_fraction": 0.75,
        "constraints_enabled": False,
        "constraint_mode": "flow_map",
    },
    {
        "mode": "constrained-flow-matching",
        "diag_fraction": 1.0,
        "constraints_enabled": True,
        "constraint_mode": "flow_matching",
    },
    {
        "mode": "constrained-flow-map",
        "diag_fraction": 0.75,
        "constraints_enabled": True,
        "constraint_mode": "flow_map",
    },
]


def get_config(slurm_id: int, dataset_location: str = "", output_folder: str = ""):
    cfg = _base_get_config(slurm_id, dataset_location, output_folder)
    variant = VARIANTS[slurm_id % len(VARIANTS)]
    seed = int(os.getenv("BOX_BEZIER_SEED", str(cfg.training.seed)))

    cfg.training.seed = seed
    cfg.optimization.diag_fraction = variant["diag_fraction"]

    cfg.constraints.enabled = variant["constraints_enabled"]
    cfg.constraints.constraint_mode = variant["constraint_mode"]
    cfg.constraints.box_path_mode = "loss_points"
    cfg.constraints.constraint_reference_diag_fraction = 0.75

    run_name = f"{variant['mode']}-{seed}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = variant["mode"]

    return cfg

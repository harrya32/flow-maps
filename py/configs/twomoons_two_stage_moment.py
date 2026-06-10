"""
Two-stage training on two moons:
1) Learn baseline flow map without trajectory constraints.
2) Transition to off-diagonal self-distillation + endpoint MMD +
   mid-time moment constraint.
"""

import ml_collections

from . import twomoons as twomoons_cfg


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    cfg = twomoons_cfg.get_config(slurm_id, dataset_location, output_folder)

    # Much longer run and more off-diagonal coverage for stage-2.
    cfg.optimization.diag_fraction = 0.5
    cfg.optimization.total_samples = 500_000
    cfg.optimization.total_steps = int(
        cfg.optimization.total_samples // cfg.optimization.bs
    )
    cfg.optimization.learning_rate = 1e-4
    cfg.optimization.schedule_type = "constant"
    cfg.training.teacher_ema_factor = 0.999

    stage2_start = max(1, (3 * cfg.optimization.total_steps) // 5)
    stage2_ramp = max(1, (7 * cfg.optimization.total_steps) // 20)

    cfg.training.two_stage = ml_collections.ConfigDict()
    cfg.training.two_stage.enabled = True
    cfg.training.two_stage.start_step = stage2_start
    cfg.training.two_stage.ramp_steps = stage2_ramp
    cfg.training.two_stage.power = 1.5
    cfg.training.two_stage.diag_weight_stage1 = 1.0
    cfg.training.two_stage.diag_weight_stage2 = 0.2
    cfg.training.two_stage.offdiag_weight_stage1 = 1.0
    cfg.training.two_stage.offdiag_weight_stage2 = 1.0
    cfg.training.two_stage.endpoint_mmd_weight = 0.2
    cfg.training.two_stage.endpoint_mmd_bandwidths = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    cfg.training.two_stage.endpoint_mmd_x_clip = 10.0
    cfg.training.two_stage.endpoint_mmd_clip_mode = "tanh"
    cfg.training.two_stage.endpoint_mmd_normalize = True
    cfg.training.two_stage.endpoint_mmd_normalize_eps = 1e-3
    cfg.training.two_stage.endpoint_mmd_detach_self = True
    cfg.training.two_stage.endpoint_mmd_self_weight = 0.25
    cfg.training.two_stage.endpoint_mmd_cross_weight = 2.0

    # Use mid-time moment constraint, applied only during stage-2.
    cfg.constraints.enabled = True
    cfg.constraints.type = "mid_moment"
    cfg.constraints.weight = 0.01
    cfg.constraints.stage2_only = True
    cfg.constraints.x_clip = 10.0
    cfg.constraints.x_clip_mode = "tanh"

    cfg.constraints.anneal.enabled = True
    cfg.constraints.anneal.start_step = stage2_start
    cfg.constraints.anneal.end_step = cfg.optimization.total_steps
    cfg.constraints.anneal.power = 1.5

    cfg.constraints.time = 0.5
    cfg.constraints.lambda_mean = 1.0
    cfg.constraints.lambda_cov = 0.0
    cfg.constraints.target_mean = [0.0, 0.55]
    cfg.constraints.target_cov = [[0.25, 0.0], [0.0, 0.25]]

    cfg.logging.wandb_name = "twomoons_lsd_two_stage_mmd_midmoment"
    cfg.logging.output_name = cfg.logging.wandb_name
    cfg.logging.visual_freq = 250
    cfg.logging.visual_ema_factor = None

    return cfg

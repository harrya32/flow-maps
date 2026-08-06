"""Constrained local config for the sharp dive-gate toy process."""

from configs import dive_gate


def get_config(slurm_id: int, dataset_location: str = "", output_folder: str = ""):
    cfg = dive_gate.get_config(slurm_id, dataset_location, output_folder)

    cfg.constraints.enabled = False
    cfg.constraints.type = "dive_gate_path"
    cfg.constraints.path_mode = "flow_map"
    #cfg.optimization.clip = 0.25


    cfg.logging.wandb_name = cfg.logging.wandb_name.replace(
        "dive_gate_vanilla",
        "dive_gate_constrained",
    )
    cfg.logging.output_name = cfg.logging.wandb_name

    return cfg

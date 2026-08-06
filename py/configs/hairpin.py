"""Vanilla local config for fast-turn hairpin interpolants.

The endpoints are paired Gaussians with p0 near the origin and p1 directly
below it. The true interpolants move right, make a smooth downward 180-degree
turn, and return leftward to p1. A beta-shaped time warp makes particles move
slowly near the endpoint distributions and quickly through the far turn.
"""

import os

import ml_collections

experiments = [
    ("lsd", None, "convex"),
]

variants = [
    ("vanilla_flow_matching", 1.0),
    ("vanilla_flow_map", 0.75),
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    import jax

    del dataset_location  # Synthetic data generated on the fly.

    variant_name, diag_fraction = variants[slurm_id % len(variants)]
    experiment_id = (slurm_id // len(variants)) % len(experiments)
    loss_type, psd_type, stopgrad_type = experiments[experiment_id]

    config = ml_collections.ConfigDict()

    # Training config.
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = False
    config.training.class_dropout = 0.0
    config.training.stopgrad_type = stopgrad_type
    config.training.psd_type = psd_type
    config.training.loss_type = loss_type
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 0
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    # Problem config.
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 100_000
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "hairpin"
    config.problem.dataset_location = None
    config.problem.interp_type = "hairpin"
    config.problem.interp_uses_labels = False
    config.problem.base = "hairpin_source"
    config.problem.gaussian_scale = "adaptive"

    # Endpoint and hairpin geometry.
    config.problem.hairpin_source_mean = [0.0, 0.0]
    config.problem.hairpin_target_mean = [0.0, -1.0]
    config.problem.hairpin_source_std = 0.08
    config.problem.hairpin_target_std = 0.08
    config.problem.hairpin_out_length = 4.0
    config.problem.hairpin_drop = 1
    config.problem.hairpin_turn_start = 0.35
    config.problem.hairpin_turn_end = 0.65
    config.problem.hairpin_tangent_speed = 0.0
    config.problem.hairpin_endpoint_tangent_speed = 0.0
    config.problem.hairpin_speed_scale = 15.0

    # Optimization config.
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 2048
    config.optimization.diag_fraction = diag_fraction
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 5_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 3_000
    config.optimization.schedule_type = "sqrt"

    # Logging config. The generic low-d plots include learned and true trajectories.
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 512
    config.logging.traj_plot_bs = 256
    config.logging.line_plot_bs = 96
    config.logging.line_plot_n_times = 161
    config.logging.multi_step_line_plot_bs = 96
    config.logging.multi_step_line_steps = [10, 25, 100]
    config.logging.euler_line_steps = [10, 25, 100]
    config.logging.scalar_freq = 1
    config.logging.progress_freq = 1
    config.logging.visual_freq = 1_000
    config.logging.save_freq = 1_000
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    config.logging.wandb_name = f"hairpin_{variant_name}_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name
    config.logging.comparison_mode = variant_name

    # FID not relevant for low-dimensional synthetic trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # Constraint training is intentionally off for this vanilla experiment.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = False

    # Network config.
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 4
    config.network.n_neurons = 512
    config.network.output_dim = 2
    config.network.act = "gelu"
    config.network.use_residual = False
    config.network.use_weight = False
    config.network.use_bfloat16 = False
    config.network.rescale = 0.5

    # Required but not used for MLP.
    config.network.load_path = ""
    config.network.input_dims = (2,)
    config.network.load_ema_fac = None
    config.network.img_resolution = None
    config.network.img_channels = None
    config.network.label_dim = None
    config.network.logvar_channels = None
    config.network.reset_optimizer = True
    config.network.unet_kwargs = None

    return config

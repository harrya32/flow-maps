"""
Lagrangian self-distillation on a constrained four-Gaussian transport toy.

Source distribution: A=(-3, 3) and C=(-3, -3).
Target distribution: B=(3, 3) and D=(3, -3).
Interpolant endpoint coupling: A -> D and C -> B.
"""

import os

import ml_collections

experiments = [
    ("lsd", None, "convex"),
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (weird issue with importlib)
    import jax

    del dataset_location  # Synthetic dataset generated on the fly.

    loss_type, psd_type, stopgrad_type = experiments[slurm_id % len(experiments)]

    config = ml_collections.ConfigDict()

    # training config
    config.training = ml_collections.ConfigDict()
    config.training.shuffle = True
    config.training.conditional = False
    config.training.class_dropout = 0.0
    config.training.stopgrad_type = stopgrad_type
    config.training.psd_type = psd_type
    config.training.loss_type = loss_type
    config.training.tmin = 0.0
    config.training.tmax = 1.0
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    # Branch-conditional MMD on the direct endpoint map X_{0,1}(x0) -> x1.
    config.training.endpoint_matching = ml_collections.ConfigDict()
    config.training.endpoint_matching.enabled = True
    config.training.endpoint_matching.weight = 1.0
    config.training.endpoint_matching.branch_conditional = True
    config.training.endpoint_matching.branch_axis = 1
    config.training.endpoint_matching.branch_threshold = 0.0
    config.training.endpoint_matching.bandwidths = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    config.training.endpoint_matching.eps = 1e-6
    config.training.endpoint_matching.min_branch_mass = 1.0

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 500_000
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "four_gaussians"
    config.problem.dataset_location = None
    config.problem.interp_type = "linear"
    config.problem.base = "four_gaussians_source"
    config.problem.gaussian_scale = "adaptive"
    config.problem.four_gaussians_std = 0.35
    config.problem.coupling = "A_to_D_C_to_B"

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 512
    config.optimization.diag_fraction = 0.75
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 20_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 5_000
    config.optimization.schedule_type = "sqrt"

    # One-shot midpoint bias: push the direct X_{0,0.5}(x0) distribution upward.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = True
    config.constraints.type = "mid_moment"
    config.constraints.weight = 1.0
    config.constraints.stage2_only = False
    config.constraints.x_clip = 10.0
    config.constraints.x_clip_mode = "tanh"

    config.constraints.time = 0.5
    config.constraints.lambda_mean = 1.0
    config.constraints.lambda_cov = 0.0
    config.constraints.target_mean = [0.0, 2.0]
    config.constraints.target_cov = [[1.0, 0.0], [0.0, 1.0]]

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5000
    config.logging.traj_plot_bs = 1000
    config.logging.line_plot_bs = 1000
    config.logging.line_plot_n_times = 20
    config.logging.multi_step_line_plot_bs = 500
    config.logging.multi_step_line_steps = [1, 2, 5, 10, 25]
    config.logging.visual_freq = 250
    config.logging.save_freq = 500
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    endpoint_suffix = (
        "_endpoint_mmd"
        if getattr(config.training.endpoint_matching, "enabled", False)
        else ""
    )
    constraint_suffix = (
        "_midup"
        if getattr(config.constraints, "enabled", False)
        and getattr(config.constraints, "type", "") == "mid_moment"
        else ""
    )
    config.logging.wandb_name = (
        f"four_gaussians_{method_str}{endpoint_suffix}{constraint_suffix}"
    )
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # FID not relevant for low-dimensional synthetic trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 3
    config.network.n_neurons = 256
    config.network.output_dim = 2
    config.network.act = "gelu"
    config.network.use_residual = False
    config.network.use_weight = False
    config.network.use_bfloat16 = False
    config.network.rescale = 0.5

    # required but not used for MLP
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

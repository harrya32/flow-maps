"""
Nicholas M. Boffi
10/5/25

Lagrangian self-distillation on a two-moons transport problem
with left moon as source and right moon as target.
"""

import os

import ml_collections

# Keep a list for compatibility with the slurm_id API used by other configs.
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

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 500_000
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "twomoons"
    config.problem.dataset_location = None
    config.problem.interp_type = "linear"
    config.problem.base = "left_moon"
    config.problem.gaussian_scale = "adaptive"
    config.problem.moons_noise = 0.05
    config.problem.moons_gap = 0.5
    config.problem.base_pool_size = 20_000

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 100
    config.optimization.diag_fraction = 0.75
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_samples = 10_000
    config.optimization.total_steps = int(
        config.optimization.total_samples // config.optimization.bs
    )
    config.optimization.decay_steps = 100
    config.optimization.schedule_type = "sqrt"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5000
    config.logging.traj_plot_bs = 1000
    config.logging.line_plot_bs = 1000
    config.logging.line_plot_n_times = 20
    config.logging.visual_freq = 5
    config.logging.save_freq = 20
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    config.logging.wandb_name = f"twomoons_{method_str}"
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

    # Optional trajectory constraint (enabled by default for this demo).
    # We default to a KDE path constraint that encourages trajectories to remain
    # in high-density regions of observed data.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = True
    config.constraints.type = "kde_path"
    config.constraints.weight = 0.05
    config.constraints.x_clip = 5.0
    config.constraints.lambda_kde = 1.0
    config.constraints.kde_obs = "base_target"  # {"target", "base_target"}
    config.constraints.kde_bandwidth = 0.25
    config.constraints.kde_max_points = 200
    config.constraints.kde_penalty = "hinge"  # {"hinge", "nll"}
    config.constraints.kde_logp_floor = -2.0
    config.constraints.kde_penalty_clip = 25.0

    config.constraints.anneal = ml_collections.ConfigDict()
    config.constraints.anneal.enabled = True
    config.constraints.anneal.start_step = 0
    config.constraints.anneal.end_step = max(1, config.optimization.total_steps // 2)
    config.constraints.anneal.power = 1.0

    # Parameters used only when type == "mid_moment".
    config.constraints.time = 0.5
    config.constraints.lambda_mean = 1.0
    config.constraints.lambda_cov = 0.0
    config.constraints.target_mean = [0.0, 0.55]
    config.constraints.target_cov = [[0.25, 0.0], [0.0, 0.25]]

    # network config
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 2
    config.network.n_neurons = 128
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

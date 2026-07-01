"""
Lagrangian self-distillation on a triangular Gaussian temporal process.

Endpoint pairs are independent samples from p0 = N((0, 0), sigma^2 I) and
p1 = N((3, 0), sigma^2 I). The custom triangle interpolant moves the mean
along (0, 0) -> (1.5, 3) -> (3, 0); its per-sample diagonal target is
x1 - x0 plus (0, 6) before t=0.5 and (0, -6) after t=0.5.
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

    # problem config
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 500_000
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "triangle_gaussian"
    config.problem.dataset_location = None
    config.problem.interp_type = "triangle"
    config.problem.base = "triangle_gaussian_source"
    config.problem.gaussian_scale = "adaptive"
    config.problem.triangle_std = 0.18
    config.problem.triangle_height = 3.0

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
    config.logging.wandb_name = f"triangle_gaussian_{method_str}"
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

    # Optional constraints are disabled; the triangle is specified by
    # independent endpoint pairs plus the custom interpolant velocity.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = False

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

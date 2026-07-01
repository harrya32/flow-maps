"""
Lagrangian self-distillation on 2D box-avoiding Bezier interpolants.

p0 = N((-3, 0), 0.25^2 I), p1 = N((3, 0), 0.25^2 I). Each endpoint pair is
assigned an independent branch sign s in {-1, +1}; the sign places the
quadratic Bezier control point above or below the infeasible box. Since true
Gaussians have unbounded tails, a small tail of unfiltered paths can still
intersect the infeasible box.
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
    config.problem.target = "box_avoiding_bezier"
    config.problem.dataset_location = None
    config.problem.interp_type = "bezier_box"
    config.problem.interp_uses_labels = True
    config.problem.base = "box_avoiding_source"
    config.problem.gaussian_scale = "adaptive"
    config.problem.box_avoiding_std = 0.25
    config.problem.bezier_height = 4.0
    config.problem.reject_infeasible = True
    config.problem.infeasible_box_xlim = [-1.5, 1.5]
    config.problem.infeasible_box_ylim = [-1.0, 1.0]
    config.problem.reject_times = [
        0.00,
        0.025,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.225,
        0.25,
        0.275,
        0.30,
        0.325,
        0.35,
        0.375,
        0.40,
        0.425,
        0.45,
        0.475,
        0.50,
        0.525,
        0.55,
        0.575,
        0.60,
        0.625,
        0.65,
        0.675,
        0.70,
        0.725,
        0.75,
        0.775,
        0.80,
        0.825,
        0.85,
        0.875,
        0.90,
        0.925,
        0.95,
        0.975,
        1.00,
    ]
    config.problem.rejection_chunk_size = 65_536

    # optimization config
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 512
    config.optimization.diag_fraction = 1.0
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 20_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 20_000
    config.optimization.schedule_type = "sqrt"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 5000
    config.logging.traj_plot_bs = 1000
    config.logging.line_plot_bs = 1000
    config.logging.line_plot_n_times = 25
    config.logging.multi_step_line_plot_bs = 500
    config.logging.multi_step_line_steps = [1, 2, 5, 10, 25]
    config.logging.euler_line_steps = [5, 10, 25, 100]
    config.logging.visual_freq = 1000
    config.logging.save_freq = 500
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    config.logging.wandb_name = f"box_avoiding_bezier_{method_str}_boxpath"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    config.logging.forbidden_box = ml_collections.ConfigDict()
    config.logging.forbidden_box.enabled = True
    config.logging.forbidden_box.freq = config.logging.visual_freq
    config.logging.forbidden_box.xlim = [-1.5, 1.5]
    config.logging.forbidden_box.ylim = [-1.0, 1.0]
    config.logging.forbidden_box.time = 0.5
    config.logging.forbidden_box.times = [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
    ]

    # FID not relevant for low-dimensional synthetic trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # Optimize the ordinary loss and configured box path penalty jointly.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = True
    config.constraints.type = "box_path"
    config.constraints.box_path_mode = "loss_points"
    config.constraints.constraint_mode = "flow_matching"  # {"flow_map", "flow_matching"}
    config.constraints.euler_steps = 25
    config.constraints.constraint_reference_diag_fraction = 0.75
    config.constraints.weight = 5.0
    config.constraints.lambda_box = 1.0
    config.constraints.stage2_only = False
    config.constraints.margin = 0
    config.constraints.xlim = [-1.75, 1.75]
    config.constraints.ylim = [-1.0, 1.0]
    config.constraints.x_clip = 0.0
    config.constraints.box_penalty_clip = 0.0
    config.constraints.log_freq = config.logging.visual_freq

    config.constraints.anneal = ml_collections.ConfigDict()
    config.constraints.anneal.enabled = False

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

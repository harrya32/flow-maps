"""Quick local config for sharp dive-gate interpolants.

The true interpolants move from a Gaussian source to a Gaussian target along the
x-axis, but briefly dive through a sample-specific gate blob near t=0.5 before
returning to the x-axis. Vanilla training has no explicit constraint loss here.
"""

from logging import config
import os

import ml_collections

experiments = [
    ("lsd", None, "convex"),
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    import jax

    del dataset_location  # Synthetic data generated on the fly.

    loss_type, psd_type, stopgrad_type = experiments[slurm_id % len(experiments)]

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
    config.problem.target = "dive_gate"
    config.problem.dataset_location = None
    config.problem.interp_type = "dive_gate"
    config.problem.interp_uses_labels = True
    config.problem.base = "dive_gate_source"
    config.problem.gaussian_scale = "adaptive"

    # Geometry: source/target are on the x-axis; gate samples form a region.
    config.problem.dive_gate_source_mean = [-3.0, 0.0]
    config.problem.dive_gate_target_mean = [3.0, 0.0]
    config.problem.dive_gate_source_std = 0.12
    config.problem.dive_gate_target_std = 0.12
    config.problem.dive_gate_depth = 0.85
    config.problem.dive_gate_jitter_std = [0.12, 0.07]
    config.problem.dive_gate_tau_down = 0.485
    config.problem.dive_gate_tau_mid = 0.500
    config.problem.dive_gate_tau_up = 0.515

    # Evaluation regions for the pathwise rule: hit A, B, then C.
    config.problem.gate_radii = [0.45, 0.30]
    config.problem.checkpoint_radii = [0.35, 0.24]

    # Optimization config.
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 2048
    config.optimization.diag_fraction = 0.95
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 20000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 2000
    config.optimization.schedule_type = "sqrt"

    # Logging config. Scalars are every step; plots are every 1k steps.
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 512
    config.logging.traj_plot_bs = 256
    config.logging.line_plot_bs = 64
    config.logging.line_plot_n_times = 51
    config.logging.multi_step_line_plot_bs = 64
    config.logging.multi_step_line_steps = [10, 25, 100]
    config.logging.euler_line_steps = [10, 25, 100]
    config.logging.scalar_freq = 1
    config.logging.progress_freq = 1
    config.logging.visual_freq = 1_000
    config.logging.save_freq = 1_000
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    config.logging.wandb_name = f"dive_gate_vanilla_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    config.logging.dive_gate = ml_collections.ConfigDict()
    config.logging.dive_gate.enabled = True
    config.logging.dive_gate.freq = config.logging.visual_freq
    config.logging.dive_gate.pre_checkpoint_center = [-0.35, 0.0]
    config.logging.dive_gate.pre_checkpoint_radii = config.problem.checkpoint_radii
    config.logging.dive_gate.gate_center = [0.0, -config.problem.dive_gate_depth]
    config.logging.dive_gate.gate_radii = config.problem.gate_radii
    config.logging.dive_gate.checkpoint_center = [0.9, 0.0]
    config.logging.dive_gate.checkpoint_radii = config.problem.checkpoint_radii
    config.logging.dive_gate.require_gate_hit = True

    # FID not relevant for low-dimensional synthetic trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # This is the vanilla baseline; constrained variants can turn this on later.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = False
    config.constraints.type = "dive_gate_path"
    config.constraints.path_mode = "flow_map"  # {"flow_map", "euler"}
    config.constraints.constraint_batch_size = 256
    config.constraints.constraint_batch_fraction = 1.0
    config.constraints.path_times = [
        0.0,
        0.25,
        0.40,
        0.44,
        0.46,
        0.48,
        0.50,
        0.52,
        0.54,
        0.60,
        0.65,
        0.75,
        1.0,
    ]
    config.constraints.path_n_times = len(config.constraints.path_times)
    config.constraints.euler_steps = 25
    config.constraints.weight = 5.0
    config.constraints.lambda_hit = 1.0
    config.constraints.lambda_hit_a = 1.0
    config.constraints.lambda_hit_b = 1.0
    config.constraints.lambda_hit_c = 1.0
    config.constraints.lambda_order = 1.0
    config.constraints.hit_loss = "miss"  # {"miss", "nll"}
    config.constraints.indicator_temperature = 0.08
    config.constraints.eps = 1e-6
    config.constraints.stage2_only = False
    config.constraints.pre_checkpoint_center = (
        config.logging.dive_gate.pre_checkpoint_center
    )
    config.constraints.pre_checkpoint_radii = (
        config.logging.dive_gate.pre_checkpoint_radii
    )
    config.constraints.gate_center = config.logging.dive_gate.gate_center
    config.constraints.gate_radii = config.logging.dive_gate.gate_radii
    config.constraints.checkpoint_center = config.logging.dive_gate.checkpoint_center
    config.constraints.checkpoint_radii = config.logging.dive_gate.checkpoint_radii

    config.constraints.anneal = ml_collections.ConfigDict()
    config.constraints.anneal.enabled = False

    # Network config.
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

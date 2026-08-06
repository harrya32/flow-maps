"""Quick local config for close non-crossing matched-gate trajectories.

Particles start near a common source, then branch to one of two close endpoints.
The interpolant for branch A must pass through midpoint gate A, and branch B
must pass through midpoint gate B.
"""

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
    config.training.seed = 42
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()

    # Problem config.
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 50_000
    config.problem.d = 2
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "matched_gates"
    config.problem.dataset_location = None
    config.problem.interp_type = "matched_gates"
    config.problem.interp_uses_labels = True
    config.problem.base = "matched_gates_source"
    config.problem.gaussian_scale = "adaptive"

    # Close, non-crossing branch geometry.
    config.problem.matched_gates_source_mean = [0.0, -2.0]
    config.problem.gate_midpoint_a = [-0.35, 0.0]
    config.problem.gate_midpoint_b = [0.35, 0.0]
    config.problem.gate_endpoint_a = [-0.45, 2.0]
    config.problem.gate_endpoint_b = [0.45, 2.0]
    config.problem.gate_tau_mid = 0.5
    config.problem.matched_gates_source_std = 0.12
    config.problem.matched_gates_endpoint_std = 0.12
    config.problem.source_radius = 0.18
    config.problem.gate_radius = 0.18
    config.problem.endpoint_radius = 0.22

    # Optimization config: deliberately small enough for quick local iteration.
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 2048
    config.optimization.diag_fraction = 0.75
    config.optimization.learning_rate = 3e-4
    config.optimization.clip = 1.0
    config.optimization.total_steps = 2_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 2_000
    config.optimization.schedule_type = "sqrt"

    # Logging config.
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 4_000
    config.logging.traj_plot_bs = 1_000
    config.logging.line_plot_bs = 300
    config.logging.line_plot_n_times = 41
    config.logging.multi_step_line_plot_bs = 160
    config.logging.multi_step_line_steps = [5, 10, 25]
    config.logging.euler_line_steps = [10, 25, 100]
    config.logging.scalar_freq = 50
    config.logging.progress_freq = 50
    config.logging.visual_freq = 250
    config.logging.save_freq = 1_000
    config.logging.wandb_project = "self-distill-flow-maps"

    method_str = f"{loss_type}_{psd_type}" if psd_type else loss_type
    config.logging.wandb_name = f"matched_gates_vanilla_{method_str}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    config.logging.matched_gates = ml_collections.ConfigDict()
    config.logging.matched_gates.enabled = True
    config.logging.matched_gates.freq = config.logging.visual_freq
    config.logging.matched_gates.source_radius = config.problem.source_radius
    config.logging.matched_gates.gate_radius = config.problem.gate_radius
    config.logging.matched_gates.endpoint_radius = config.problem.endpoint_radius
    config.logging.matched_gates.forbid_wrong_midpoint_first = True

    # FID not relevant for low-dimensional synthetic trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = None

    # Constraints are intentionally off for this vanilla baseline.
    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = False

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

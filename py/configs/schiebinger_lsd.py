"""
Nicholas M. Boffi
10/5/25

Lagrangian self-distillation on Schiebinger reprogramming data
using a low-dimensional (PCA) flow-map model.
"""

import os

import ml_collections

experiments = [
    ("lsd", None, "convex"),
]


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
) -> ml_collections.ConfigDict:
    # ensure jax.device_count works (consistent with other configs)
    import jax

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
    config.problem.n = 0  # populated after Schiebinger endpoint extraction
    config.problem.d = 5  # PCA dimension
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "schiebinger"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    config.problem.base = "schiebinger_first_timepoint"
    config.problem.gaussian_scale = "adaptive"

    # Schiebinger-specific fields used by common.datasets
    config.problem.schiebinger_filename = "reprogramming_schiebinger.h5ad"
    config.problem.subset_to_serum = True
    config.problem.embedding_key = "X_pca"
    config.problem.n_pcs = 5
    config.problem.time_key = "day"
    config.problem.whiten_pca = False
    config.problem.pca_random_state = 0
    config.problem.max_endpoint_train = 30000

    # optimization config (kept close to checker low-dimensional defaults)
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 100_000
    config.optimization.diag_fraction = 0.75
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 10.0
    config.optimization.total_steps = 250_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 35_000
    config.optimization.schedule_type = "sqrt"

    # logging config
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 25_000
    config.logging.visual_freq = 5_000
    config.logging.save_freq = 10_000
    config.logging.wandb_project = "self-distill-flow-maps"
    config.logging.wandb_name = "schiebinger_pca5_lsd"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name

    # FID not used for this non-image experiment
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = 0.9999

    # network config (paper checker-style MLP)
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 4
    config.network.n_neurons = 512
    config.network.output_dim = config.problem.d
    config.network.act = "gelu"
    config.network.use_residual = False
    config.network.use_weight = False
    config.network.use_bfloat16 = False
    config.network.rescale = 0.5

    # required but unused by MLP
    config.network.load_path = ""
    config.network.input_dims = (config.problem.d,)
    config.network.load_ema_fac = None
    config.network.img_resolution = None
    config.network.img_channels = None
    config.network.label_dim = None
    config.network.logvar_channels = None
    config.network.reset_optimizer = True
    config.network.unet_kwargs = None

    return config

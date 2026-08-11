"""Maizels PCA50 D3 -> D8 trajectory experiment.

The saved PCA50 CSV is used directly: no AnnData preprocessing or PCA fitting is
performed here. All variants use LSD; slurm IDs choose vanilla/bio-prior
coupling and flow-matching/flow-map training.
"""

import os

import ml_collections

variants = [
    # ID 0: vanilla flow matching.
    ("vanilla_flow_matching", "none", 1.0, False),
    # ID 1: vanilla flow map.
    ("vanilla_flow_map", "none", 0.75, False),
    # ID 2: bio-prior flow matching.
    ("bio_prior_flow_matching", "endpoint_interpolant", 1.0, False),
    # ID 3: bio-prior flow map.
    ("bio_prior_flow_map", "endpoint_interpolant", 0.75, False),
    # ID 4: bio-prior flow map with differentiable lineage constraint.
    ("bio_prior_constrained_flow_map", "endpoint_interpolant", 0.75, True),
]


def get_config(
    slurm_id: int, dataset_location: str = "", output_folder: str = ""
) -> ml_collections.ConfigDict:
    import jax

    variant_name, pair_mode, diag_fraction, use_lineage_constraint = variants[
        slurm_id % len(variants)
    ]
    loss_type = "lsd"
    psd_type = None
    stopgrad_type = "convex"

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
    config.training.seed = 1
    config.training.ema_facs = [0.999, 0.9999]
    config.training.ndevices = jax.device_count()
    #config.training.teacher_ema_factor = 0.999

    # Problem config.
    config.problem = ml_collections.ConfigDict()
    config.problem.n = 500_000
    config.problem.d = 50
    config.problem.image_dims = None
    config.problem.num_classes = None
    config.problem.target = "maizels_pca50"
    config.problem.dataset_location = dataset_location
    config.problem.interp_type = "linear"
    # Preserve source/target cell-type ids through batching for diagnostics only.
    config.problem.interp_uses_labels = True
    config.problem.base = "maizels_d3"
    config.problem.gaussian_scale = "adaptive"

    config.problem.source_time = "D3"
    config.problem.target_time = "D8"
    # These endpoint cells are excluded from training pair construction and
    # reserved for Maizels held-out trajectory diagnostics.
    config.problem.maizels_holdout_fraction = 0.2
    config.problem.maizels_holdout_n = 0
    config.problem.maizels_holdout_seed = 701
    config.problem.maizels_pair_mode = pair_mode
    config.problem.classifier_path = (
        "/mnt/pdata/hmka3/flow-maps/"
        "celltype_classifier_pca50.pt"
    )
    config.problem.n_interpolant_check_times = 50
    config.problem.classifier_prob_threshold = 0
    config.problem.classifier_margin_threshold = 0
    config.problem.classifier_batch_size = 8192
    config.problem.rejection_chunk_size = 50_000
    config.problem.rejection_max_candidates = 5_000_000
    config.problem.device_batching = True

    # Optimization config.
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 4096
    config.optimization.diag_fraction = diag_fraction
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 10.0
    config.optimization.total_steps = 50_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 10_000
    config.optimization.schedule_type = "sqrt"

    # Logging config. Generic low-d plots use PC1/PC2; Maizels diagnostics add
    # classifier-validity coloring on held-out D3/D8 endpoint cells.
    config.logging = ml_collections.ConfigDict()
    config.logging.plot_bs = 1024
    config.logging.traj_plot_bs = 512
    config.logging.line_plot_bs = 128
    config.logging.line_plot_n_times = 50
    config.logging.multi_step_line_plot_bs = 128
    config.logging.multi_step_line_steps = [10, 25, 100]
    config.logging.euler_line_steps = [10, 25, 100]
    config.logging.scalar_freq = 1
    config.logging.progress_freq = 1
    config.logging.visual_freq = 500
    config.logging.save_freq = 5_000
    config.logging.wandb_project = "self-distill-flow-maps"

    config.logging.wandb_name = f"maizels_pca50_{variant_name}"
    config.logging.wandb_entity = os.getenv("WANDB_ENTITY", "your-username")
    config.logging.output_folder = output_folder
    config.logging.output_name = config.logging.wandb_name
    config.logging.comparison_mode = variant_name

    config.logging.maizels = ml_collections.ConfigDict()
    config.logging.maizels.enabled = True
    config.logging.maizels.plot_bs = 512
    config.logging.maizels.path_n_times = 50
    config.logging.maizels.euler_n_steps = 50
    config.logging.maizels.flowmap_n_steps = 50
    config.logging.maizels.check_n_times = 50
    config.logging.maizels.pair_mode = "same_as_training"
    config.logging.maizels.plot_seed = 997
    config.logging.maizels.prob_threshold = config.problem.classifier_prob_threshold
    config.logging.maizels.margin_threshold = config.problem.classifier_margin_threshold
    config.logging.maizels.classifier_batch_size = config.problem.classifier_batch_size
    config.logging.maizels.validation_enabled = True
    config.logging.maizels.validation_bs = 1024
    config.logging.maizels.validation_seed = 2701
    config.logging.maizels.validation_pair_mode = "same_as_training"
    config.logging.maizels.distribution_eval_enabled = True
    config.logging.maizels.distribution_eval_source_pool = "auto"
    config.logging.maizels.distribution_eval_points_per_time = 1024
    config.logging.maizels.distribution_eval_max_timepoints = 0
    config.logging.maizels.distribution_eval_wasserstein_projections = 256
    config.logging.maizels.distribution_eval_euler_n_steps = (
        config.logging.maizels.euler_n_steps
    )
    config.logging.maizels.distribution_eval_flowmap_n_steps = 50
    config.logging.maizels.distribution_eval_mmd_bandwidths = []
    config.logging.maizels.distribution_eval_mmd_bandwidth_multipliers = [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ]

    # FID not relevant for PCA-space cellular trajectories.
    config.logging.fid_freq = 0
    config.logging.fid_stats_path = None
    config.logging.fid_n_samples = None
    config.logging.fid_batch_size = None
    config.logging.fid_n_steps_flow = None
    config.logging.fid_ema_factor = None
    config.logging.visual_ema_factor = 0.999

    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = use_lineage_constraint
    config.constraints.type = "maizels_lineage_path"
    config.constraints.path_mode = "direct"
    config.constraints.path_n_times = 10
    config.constraints.euler_steps = 10
    config.constraints.constraint_batch_size = 1024
    config.constraints.constraint_batch_fraction = 1.0
    config.constraints.weight = 1000.0
    config.constraints.lambda_start = 0.0
    config.constraints.lambda_transition = 1.0
    config.constraints.lambda_final = 0.0
    config.constraints.classifier_temperature = 1.0
    config.constraints.stage2_only = False

    # Network config.
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 4
    config.network.n_neurons = 256
    config.network.output_dim = 50
    config.network.act = "gelu"
    config.network.use_residual = False
    config.network.use_weight = False
    config.network.use_bfloat16 = False
    config.network.rescale = 1.0

    # Required but not used for MLP.
    config.network.load_path = ""
    config.network.input_dims = (50,)
    config.network.load_ema_fac = None
    config.network.img_resolution = None
    config.network.img_channels = None
    config.network.label_dim = None
    config.network.logvar_channels = None
    config.network.reset_optimizer = True
    config.network.unet_kwargs = None

    return config

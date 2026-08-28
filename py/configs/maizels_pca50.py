"""Maizels PCA50 endpoint or three-timepoint trajectory experiment.

The saved PCA50 CSV is used directly: no AnnData preprocessing or PCA fitting is
performed here. All variants use LSD; slurm IDs choose vanilla/bio-prior
coupling and flow-matching/flow-map training.
"""

from __future__ import annotations

import os
from pathlib import Path

import ml_collections


TIMEPOINTS = (
    "D3",
    "D3.2",
    "D3.4",
    "D3.6",
    "D3.8",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
)

SCHEDULES = {
    "d3_d8": ("D3", "D8"),
    "d3_d3p8_d8": ("D3", "D3.8", "D8"),
}


def _canonical_schedule(value: str | None) -> str:
    aliases = {
        "endpoints": "d3_d8",
        "d3-d8": "d3_d8",
        "three_timepoint": "d3_d3p8_d8",
        "three-timepoint": "d3_d3p8_d8",
        "d3-d3.8-d8": "d3_d3p8_d8",
    }
    value = str(value or "d3_d8").lower()
    value = aliases.get(value, value)
    if value not in SCHEDULES:
        raise ValueError(
            f"maizels_schedule must be one of {sorted(SCHEDULES)}, got {value!r}."
        )
    return value


def _canonical_time_mode(value: str | None) -> str:
    aliases = {
        "real": "real_time",
        "physical": "real_time",
        "physical_time": "real_time",
        "equal": "equal_time",
        "stage": "equal_time",
    }
    value = str(value or "real_time").lower()
    value = aliases.get(value, value)
    if value not in ("real_time", "equal_time"):
        raise ValueError(
            "maizels_time_mode must be 'real_time' or 'equal_time', "
            f"got {value!r}."
        )
    return value


def _timepoint_values(retained_timepoints, time_mode: str):
    if time_mode == "real_time":
        start, end = 3.0, 8.0
        return [(float(value[1:]) - start) / (end - start) for value in TIMEPOINTS]

    retained_indices = [TIMEPOINTS.index(value) for value in retained_timepoints]
    retained_values = [
        index / float(len(retained_timepoints) - 1)
        for index in range(len(retained_timepoints))
    ]
    values = [None] * len(TIMEPOINTS)
    for interval_index, (left, right) in enumerate(
        zip(retained_indices[:-1], retained_indices[1:])
    ):
        start_value = retained_values[interval_index]
        end_value = retained_values[interval_index + 1]
        for position in range(left, right + 1):
            fraction = (position - left) / float(right - left)
            values[position] = start_value + fraction * (end_value - start_value)
    if any(value is None for value in values):
        raise ValueError("Retained Maizels timepoints must span D3 through D8.")
    return values


def _resolve_classifier_path(schedule: str, classifier_path: str | None) -> str:
    if classifier_path:
        return str(Path(classifier_path).expanduser().resolve())
    env_path = os.getenv("MAIZELS_CLASSIFIER_PATH", "")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    repo_root = Path(__file__).resolve().parents[2]
    if schedule == "d3_d3p8_d8":
        return str(
            (
                repo_root
                / "outputs"
                / "maizels_classifier_d3_d3p8_d8"
                / "celltype_classifier_pca50_d3_d3p8_d8.pt"
            ).resolve()
        )
    return str((repo_root / "celltype_classifier_pca50.pt").resolve())


def _resolve_full_data_classifier_path() -> str:
    """Resolve the frozen all-timepoint classifier used only for evaluation."""
    env_path = os.getenv("MAIZELS_FULL_DATA_CLASSIFIER_PATH", "")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / "celltype_classifier_pca50.pt").resolve())


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
    # ID 5: bio-prior flow map with OT couplings.
    ("bio_prior_ot_flow_map", "ot_endpoint_interpolant", 0.75, False),
    # ID 6: bio-prior flow map with OT couplings and differentiable lineage constraint.
    ("bio_prior_ot_constrained_flow_map", "ot_endpoint_interpolant", 0.75, True),
    # ID 7: ot plain flow map.
    ("ot_flow_map", "ot_plain", 0.75, False),
]


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
    classifier_path: str | None = None,
    maizels_schedule: str | None = None,
    maizels_time_mode: str | None = None,
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
    config.training.seed = 0
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
    schedule = _canonical_schedule(
        maizels_schedule or os.getenv("MAIZELS_SCHEDULE", "d3_d8")
    )
    time_mode = _canonical_time_mode(
        maizels_time_mode or os.getenv("MAIZELS_TIME_MODE", "real_time")
    )
    retained_timepoints = SCHEDULES[schedule]

    config.problem.maizels_schedule = schedule
    config.problem.maizels_time_mode = time_mode
    config.problem.retained_timepoints = list(retained_timepoints)
    config.problem.evaluation_timepoints = [
        value for value in TIMEPOINTS if value not in retained_timepoints
    ]
    config.problem.timepoint_order = list(TIMEPOINTS)
    config.problem.timepoint_values = _timepoint_values(retained_timepoints, time_mode)
    config.problem.interp_type = (
        "time_rescaled_linear" if len(retained_timepoints) > 2 else "linear"
    )
    # Preserve source/target cell-type ids through batching for diagnostics only.
    config.problem.interp_uses_labels = True
    config.problem.pair_time_bounds_in_label = len(retained_timepoints) > 2
    config.problem.base = "maizels_d3"
    config.problem.gaussian_scale = "adaptive"

    config.problem.source_time = "D3"
    config.problem.target_time = "D8"
    # These endpoint cells are excluded from training pair construction and
    # reserved for Maizels held-out trajectory diagnostics.
    config.problem.maizels_holdout_fraction = 0.1
    config.problem.maizels_holdout_n = 0
    config.problem.maizels_holdout_seed = 701
    config.problem.maizels_pair_mode = pair_mode
    if maizels_ot_coupling is None:
        maizels_ot_coupling = os.getenv("MAIZELS_OT_COUPLING", "minibatch_ot")
    maizels_ot_coupling = {
        "global": "global_ot",
        "exact": "global_ot",
        "minibatch": "minibatch_ot",
    }.get(str(maizels_ot_coupling).lower(), str(maizels_ot_coupling).lower())
    if maizels_ot_coupling not in ("global_ot", "minibatch_ot"):
        raise ValueError(
            "maizels_ot_coupling must be 'global_ot' or 'minibatch_ot', "
            f"got {maizels_ot_coupling!r}."
        )
    config.problem.maizels_ot_coupling = maizels_ot_coupling
    config.problem.lineage_transition_mode = os.getenv(
        "MAIZELS_LINEAGE_TRANSITION_MODE",
        "descendant",
    )
    config.problem.classifier_path = _resolve_classifier_path(
        schedule,
        classifier_path,
    )
    config.problem.n_interpolant_check_times = 50
    config.problem.classifier_prob_threshold = 0
    config.problem.classifier_margin_threshold = 0
    config.problem.classifier_batch_size = 8192
    config.problem.rejection_chunk_size = 50_000
    config.problem.rejection_max_candidates = 5_000_000
    config.problem.ot_candidate_chunk_size = 50_000
    config.problem.ot_mass_tolerance = 1e-12
    config.problem.ot_drop_orphan_cells = True
    config.problem.ot_cache_enabled = True
    config.problem.ot_cache_dir = ""
    config.problem.ot_cache_version = "v1"
    config.problem.ot_progress_enabled = True
    config.problem.ot_verbose = True
    config.problem.device_batching = True

    # Optimization config.
    config.optimization = ml_collections.ConfigDict()
    config.optimization.bs = 128  # 4096
    # Dynamic OT couples each interval's share of the optimizer batch in one
    # problem. Its cost is raw squared Euclidean distance.
    config.problem.ot_minibatch_size = config.optimization.get_ref("bs")
    config.problem.ot_minibatch_max_resamples = 20
    config.problem.ot_minibatch_infeasible_fallback = "partial"
    config.optimization.diag_fraction = diag_fraction
    config.optimization.learning_rate = 1e-3
    config.optimization.clip = 10.0
    config.optimization.total_steps = 10_000
    config.optimization.total_samples = (
        config.optimization.bs * config.optimization.total_steps
    )
    config.optimization.decay_steps = 10_000
    config.optimization.schedule_type = "sqrt"

    config.optimization.early_stopping = ml_collections.ConfigDict()
    if early_stopping_patience is None:
        early_stopping_patience = int(
            os.getenv("MAIZELS_EARLY_STOPPING_PATIENCE", "10")
        )
    config.optimization.early_stopping.patience = int(early_stopping_patience)
    config.optimization.early_stopping.check_freq = int(
        os.getenv("MAIZELS_EARLY_STOPPING_CHECK_FREQ", "100")
    )
    config.optimization.early_stopping.min_delta = float(
        os.getenv("MAIZELS_EARLY_STOPPING_MIN_DELTA", "0.0")
    )
    config.optimization.early_stopping.warmup_steps = int(
        os.getenv("MAIZELS_EARLY_STOPPING_WARMUP_STEPS", "0")
    )
    config.optimization.early_stopping.metric = "validation_loss"
    config.optimization.early_stopping.mode = "min"
    config.optimization.early_stopping.save_best = True

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
    config.logging.visual_freq = 5000
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
    config.logging.maizels.lineage_transition_mode = "same_as_problem"
    # This evaluator is deliberately separate from the schedule-specific
    # classifier used to construct pairs and train the lineage constraint.
    config.logging.maizels.full_data_classifier_path = (
        _resolve_full_data_classifier_path()
    )
    config.logging.maizels.validation_enabled = True
    config.logging.maizels.validation_bs = 1024
    config.logging.maizels.validation_seed = 2701
    config.logging.maizels.validation_pair_mode = "same_as_training"
    # Hard lineage-violation diagnostics always follow held-out D3 cells over
    # the complete model trajectory from t=0 to t=1, including multi-interval
    # training runs. Omitted-day distribution evaluation remains interval-local.
    config.logging.maizels.trajectory_diagnostics_enabled = True
    # Trajectory-violation metrics use held-out D3 cells.
    # If this cap exceeds the holdout size, every held-out cell is used once.
    config.logging.maizels.trajectory_eval_source_pool = "heldout"
    config.logging.maizels.trajectory_eval_source_max_points = (
        config.logging.maizels.plot_bs
    )
    config.logging.maizels.trajectory_eval_seed = 2698
    config.logging.maizels.distribution_eval_enabled = True
    # Match the CITE/Multi population evaluation: push every source cell forward
    # exactly once. A source cap of 0 means the complete selected population.
    config.logging.maizels.distribution_eval_source_pool = "all" #auto
    config.logging.maizels.distribution_eval_source_max_points = 0
    config.logging.maizels.distribution_eval_points_per_time = 0
    config.logging.maizels.distribution_eval_max_timepoints = 0
    config.logging.maizels.distribution_eval_timepoints = list(
        config.problem.evaluation_timepoints
    )
    config.logging.maizels.distribution_eval_interval_local = (
        len(retained_timepoints) > 2
    )
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
    config.logging.visual_ema_factor = None

    config.constraints = ml_collections.ConfigDict()
    config.constraints.enabled = use_lineage_constraint
    config.constraints.type = "maizels_lineage_path"
    config.constraints.path_mode = "loss_points_nll" #velocity_loss_points_nll, loss_points_nll, direct
    config.constraints.path_n_times = 10
    config.constraints.euler_steps = 10
    config.constraints.constraint_batch_size = 32
    config.constraints.constraint_batch_fraction = 1.0
    config.constraints.weight = 1000.0
    config.constraints.lambda_start = 0.0
    config.constraints.lambda_transition = 1.0
    config.constraints.lambda_final = 0.0
    config.constraints.classifier_temperature = 1.0
    config.constraints.loss_point_entropy_weight = 0.01
    config.constraints.velocity_rollout_batch_size = 0
    config.constraints.velocity_rollout_reference_diag_fraction = 0.75
    config.constraints.velocity_rollout_max_step = 0.01
    config.constraints.velocity_rollout_max_steps = 0
    config.constraints.velocity_rollout_loss_scope = "endpoints" #path, endpoints
    config.constraints.lineage_transition_mode = "same_as_problem"
    config.constraints.stage2_only = False

    # Network config.
    config.network = ml_collections.ConfigDict()
    config.network.network_type = "mlp"
    config.network.n_hidden = 2 #4
    config.network.n_neurons = 1024 #256
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

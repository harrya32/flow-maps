"""LARRY in-vitro D2--D6 flow maps with held-out D4 evaluation.

The active jobs compare independent and minibatch-OT couplings, optional
endpoint/interpolant lineage filtering, and the differentiable lineage
constraint. Existing Slurm ids 0 and 1 remain unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import ml_collections

from common import larry
from configs.maizels_pca50 import get_config as _maizels_config


# (Maizels base id, pair mode, constraints enabled, path mode). The catalog
# remains complete so flow-matching variants can be activated later.
VARIANT_CATALOG = {
    "independent_flow_map": (1, "none", False, "loss_points_nll"),
    "independent_flow_matching": (0, "none", False, "loss_points_nll"),
    "bio_prior_flow_matching": (
        2,
        "endpoint_interpolant",
        False,
        "loss_points_nll",
    ),
    "bio_prior_flow_map": (3, "endpoint_interpolant", False, "loss_points_nll"),
    "bio_prior_constrained_flow_map": (
        4,
        "endpoint_interpolant",
        True,
        "loss_points_nll",
    ),
    "bio_prior_ot_flow_map": (
        5,
        "ot_endpoint_interpolant",
        False,
        "loss_points_nll",
    ),
    "bio_prior_ot_constrained_flow_map": (
        6,
        "ot_endpoint_interpolant",
        True,
        "loss_points_nll",
    ),
    "ot_flow_map": (7, "ot_plain", False, "loss_points_nll"),
    "bio_prior_ot_constrained_flow_matching": (
        8,
        "ot_endpoint_interpolant",
        True,
        "velocity_loss_points_nll",
    ),
}
ACTIVE_VARIANTS = (
    # Preserve the two original job ids.
    "independent_flow_map",
    "bio_prior_flow_map",
    # Add constrained independent and minibatch-OT flow maps.
    "bio_prior_constrained_flow_map",
    "ot_flow_map",
    "bio_prior_ot_flow_map",
    "bio_prior_ot_constrained_flow_map",
)


def get_hparam_sweep_spec(slurm_id: int) -> dict:
    if int(slurm_id) < 0 or int(slurm_id) >= len(ACTIVE_VARIANTS):
        raise ValueError(
            f"LARRY currently exposes Slurm ids 0--{len(ACTIVE_VARIANTS) - 1}."
        )
    name = ACTIVE_VARIANTS[int(slurm_id)]
    _, _, constrained, path_mode = VARIANT_CATALOG[name]
    return {
        "variant_name": name,
        "learning_rate": True,
        "constraint_weight": constrained,
        "entropy_weight": constrained and "nll" in path_mode,
    }


def _optional_path(value: str | None, env_name: str) -> str:
    selected = value or os.getenv(env_name, "")
    return str(Path(selected).expanduser().resolve()) if selected else ""


def _runtime_checkpoint_exists(path: str) -> bool:
    checkpoint = Path(path).expanduser()
    npz_path = (
        checkpoint if checkpoint.suffix == ".npz" else checkpoint.with_suffix(".npz")
    )
    return npz_path.is_file()


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
    classifier_path: str | None = None,
    full_data_classifier_path: str | None = None,
    learning_rate: float | None = None,
    constraint_weight: float | None = None,
    entropy_weight: float | None = None,
    seed: int | None = None,
    ot_minibatch_size: int | None = None,
    larry_representation: str = larry.PCA50_REPRESENTATION,
) -> ml_collections.ConfigDict:
    """Return an active LARRY flow-map configuration."""
    slurm_id = int(slurm_id)
    if slurm_id < 0 or slurm_id >= len(ACTIVE_VARIANTS):
        raise ValueError(
            f"LARRY currently exposes Slurm ids 0--{len(ACTIVE_VARIANTS) - 1}."
        )
    variant_name = ACTIVE_VARIANTS[slurm_id]
    base_id, pair_mode, constrained, path_mode = VARIANT_CATALOG[variant_name]
    representation = larry.canonical_representation(larry_representation)
    representation_dim = larry.representation_dim(representation)
    is_spring = representation == larry.SPRING2D_REPRESENTATION
    representation_label = "SPRING2D" if is_spring else "PCA50"

    resolved_dataset = larry.resolve_dataset_path(
        dataset_location,
        representation=representation,
    )
    # LARRY OT jobs are dynamic minibatch OT by default. An explicit launcher
    # override may still request the global solver for diagnostic comparisons.
    if maizels_ot_coupling is None:
        maizels_ot_coupling = "minibatch_ot"
    cfg = _maizels_config(
        base_id,
        str(resolved_dataset),
        output_folder,
        early_stopping_patience=early_stopping_patience,
        maizels_ot_coupling=maizels_ot_coupling,
        ot_minibatch_size=ot_minibatch_size,
        learning_rate=learning_rate,
        constraint_weight=constraint_weight,
        entropy_weight=entropy_weight,
        seed=seed,
    )

    # Dataset and D2--D6 training schedule. D4 is represented on the global
    # clock but is absent from every training and validation pair pool.
    cfg.problem.target = "larry_spring2d" if is_spring else "larry_pca50"
    cfg.problem.dataset_name = "larry_invitro"
    cfg.problem.lineage_dataset_name = (
        "larry_invitro_spring2d" if is_spring else "larry_invitro_hvg2000_pca50"
    )
    cfg.problem.dataset_location = str(resolved_dataset)
    cfg.problem.larry_filename = (
        larry.DEFAULT_SPRING_FILENAME if is_spring else larry.DEFAULT_FILENAME
    )
    cfg.problem.larry_representation = representation
    cfg.problem.larry_representation_key = larry.representation_key(representation)
    cfg.problem.representation_dim = representation_dim
    cfg.problem.n_hvgs = 0 if is_spring else larry.DEFAULT_N_HVGS
    # The shared classifier schema calls its input dimension n_pcs. For SPRING
    # this field is simply the two-dimensional state size.
    cfg.problem.n_pcs = representation_dim
    cfg.problem.n = 500_000
    cfg.problem.d = representation_dim
    cfg.problem.base = "larry_day2"
    cfg.problem.source_time = "D2"
    cfg.problem.target_time = "D6"
    cfg.problem.heldout_timepoint = "D4"
    cfg.problem.retained_timepoints = ["D2", "D6"]
    cfg.problem.evaluation_timepoints = ["D4"]
    cfg.problem.pca_fit_timepoints = [] if is_spring else ["D2", "D4", "D6"]
    cfg.problem.representation_fit_timepoints = ["D2", "D4", "D6"]
    cfg.problem.representation_is_published_layout = is_spring
    cfg.problem.hparam_val_times = []
    cfg.problem.timepoint_order = list(larry.TIMEPOINTS)
    cfg.problem.timepoint_values = [
        larry.NORMALIZED_TIMES[timepoint] for timepoint in larry.TIMEPOINTS
    ]
    cfg.problem.interp_type = "linear"
    cfg.problem.interp_uses_labels = True
    cfg.problem.pair_time_bounds_in_label = False

    # Keep the conventional 10% endpoint validation pools. Population
    # evaluation samples from the complete D2 pool, and clone evaluation uses
    # every clone-labelled D2 cell, including cells in this split.
    cfg.problem.maizels_holdout_fraction = 0.1
    cfg.problem.maizels_holdout_n = 0
    cfg.problem.maizels_holdout_seed = 701
    cfg.problem.maizels_pair_mode = pair_mode
    cfg.problem.pair_mode = cfg.problem.get_ref("maizels_pair_mode")
    # Match the Maizels prior filter: inspect 50 interior points and treat every
    # classifier prediction as informative. Endpoint annotations are checked
    # separately before the interpolant classifier is evaluated.
    cfg.problem.n_interpolant_check_times = 50
    cfg.problem.classifier_prob_threshold = 0
    cfg.problem.classifier_margin_threshold = 0
    cfg.constraints.enabled = constrained
    cfg.constraints.path_mode = path_mode

    cfg.problem.lineage_class_names = list(larry.CLASS_NAMES)
    cfg.problem.lineage_transition_edges = [
        list(edge) for edge in larry.TRANSITION_EDGES
    ]
    cfg.problem.lineage_unconstrained_class_names = []
    cfg.problem.lineage_transition_mode = os.getenv(
        "LARRY_LINEAGE_TRANSITION_MODE", "descendant"
    )

    training_classifier = _optional_path(
        classifier_path,
        "LARRY_CLASSIFIER_PATH",
    ) or str(
        larry.classifier_checkpoint_path(
            training_timepoints=("D2", "D6"),
            representation=representation,
            n_pcs=representation_dim,
        )
    )
    evaluation_classifier = _optional_path(
        full_data_classifier_path,
        "LARRY_FULL_DATA_CLASSIFIER_PATH",
    ) or str(
        larry.classifier_checkpoint_path(
            all_days=True,
            representation=representation,
            n_pcs=representation_dim,
        )
    )
    cfg.problem.flow_training_requires_classifier = (
        pair_mode in larry.CLASSIFIER_PAIR_MODES or constrained
    )
    training_classifier_available = _runtime_checkpoint_exists(training_classifier)
    evaluation_classifier_available = _runtime_checkpoint_exists(evaluation_classifier)
    if (
        cfg.problem.flow_training_requires_classifier
        and not training_classifier_available
    ):
        raise FileNotFoundError(
            f"LARRY {representation_label} variant {variant_name!r} requires "
            f"{training_classifier}. "
            "Run `conda run -n mfm_env python "
            "scripts/train_larry_celltype_classifiers.py"
            + (" --representation spring2d" if is_spring else "")
            + "` first, or provide "
            "--classifier_path."
        )
    cfg.problem.training_classifier_path = training_classifier
    cfg.problem.full_data_classifier_path = evaluation_classifier
    cfg.problem.training_classifier_available = training_classifier_available
    cfg.problem.full_data_classifier_available = evaluation_classifier_available
    cfg.problem.classifier_path = (
        training_classifier
        if cfg.problem.flow_training_requires_classifier
        else evaluation_classifier
    )

    cfg.network.output_dim = representation_dim
    cfg.network.input_dims = (representation_dim,)

    run_name = f"larry_invitro_{representation}_holdout_d4_{variant_name}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    cfg.logging.comparison_mode = variant_name

    cfg.logging.maizels.enabled = True
    cfg.logging.maizels.validation_enabled = True
    cfg.logging.maizels.distribution_eval_enabled = True
    cfg.logging.maizels.distribution_eval_timepoints = ["D4"]
    cfg.logging.maizels.distribution_eval_max_timepoints = 1
    cfg.logging.maizels.distribution_eval_interval_local = False
    cfg.logging.maizels.distribution_eval_source_pool = "all"
    # Exact POT EMD constructs a dense source-by-target cost matrix. Keep both
    # sides at 1,024 cells for periodic and final evaluation to bound memory.
    cfg.logging.maizels.distribution_eval_source_max_points = 1024
    cfg.logging.maizels.distribution_eval_points_per_time = 1024
    cfg.logging.maizels.distribution_eval_samplers = ["flowmap"]
    cfg.logging.maizels.distribution_eval_flowmap_n_steps = 50
    cfg.logging.maizels.distribution_eval_metrics = ["emd"]
    cfg.logging.maizels.distribution_eval_flow_batch_size = 4096
    cfg.logging.maizels.trajectory_eval_source_pool = "heldout"
    cfg.logging.maizels.trajectory_eval_samplers = ["flowmap"]
    cfg.logging.maizels.check_n_times = 50
    cfg.logging.maizels.full_data_classifier_path = evaluation_classifier
    cfg.logging.maizels.violation_metrics_available = evaluation_classifier_available
    cfg.logging.maizels.violation_metrics_unavailable_reason = (
        f"No LARRY {representation_label} all-days cell-type classifier has been "
        "trained; run "
        "`conda run -n mfm_env python "
        "scripts/train_larry_celltype_classifiers.py"
        + (" --representation spring2d" if is_spring else "")
        + "`."
        if not evaluation_classifier_available
        else ""
    )

    # Exact per-clone W1 is cheap because individual clones are small.  All D2
    # members are pushed once, and all observed D4 members form each target.
    cfg.logging.larry = ml_collections.ConfigDict()
    cfg.logging.larry.clone_wasserstein_enabled = True
    cfg.logging.larry.clone_source_time = "D2"
    cfg.logging.larry.clone_target_time = "D4"
    cfg.logging.larry.clone_min_source_cells = 1
    cfg.logging.larry.clone_min_target_cells = 1
    cfg.logging.larry.clone_sampler = "flowmap"
    cfg.logging.larry.clone_flowmap_n_steps = 50
    cfg.logging.larry.clone_flow_batch_size = 4096
    cfg.logging.larry.clone_max_cells_per_population = 0
    cfg.logging.larry.clone_eval_seed = cfg.training.seed + 3901

    return cfg

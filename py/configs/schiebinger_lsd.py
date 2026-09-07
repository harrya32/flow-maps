"""Schiebinger serum-reprogramming HVG-PCA trajectory experiments.

Select observed training days with ``schiebinger_train_times`` (or the
``SCHIEBINGER_TRAIN_TIMES`` environment variable).  The earliest and latest
selected days define the experiment window; omitted observations inside that
window are evaluation days, while observations outside it are ignored.

The slurm IDs mirror ``maizels_pca50``.  Biological transition filtering is
therefore active only for the endpoint-interpolant variants, and the lineage
loss is active only for the constrained variants.  Classifier-backed variants
automatically train a cached selected-days classifier before flow training.
An independent all-days classifier is used only for evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path

import ml_collections

from common import schiebinger
from configs.maizels_pca50 import get_config as _maizels_config


def _resolve_classifier_path(
    training_timepoints,
    classifier_path: str | None,
    n_pcs: int,
) -> str:
    if classifier_path:
        return str(Path(classifier_path).expanduser().resolve())
    env_path = os.getenv("SCHIEBINGER_CLASSIFIER_PATH", "")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str(
        schiebinger.classifier_checkpoint_path(
            training_timepoints=training_timepoints,
            n_pcs=n_pcs,
        )
    )


def _resolve_full_data_classifier_path(
    full_data_classifier_path: str | None,
    n_pcs: int,
) -> str:
    if full_data_classifier_path:
        return str(Path(full_data_classifier_path).expanduser().resolve())
    env_path = os.getenv("SCHIEBINGER_FULL_DATA_CLASSIFIER_PATH", "")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    return str(
        schiebinger.classifier_checkpoint_path(all_days=True, n_pcs=n_pcs)
    )


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    early_stopping_patience=None,
    maizels_ot_coupling=None,
    classifier_path: str | None = None,
    full_data_classifier_path: str | None = None,
    schiebinger_train_times: str | None = None,
    schiebinger_n_pcs: int = schiebinger.DEFAULT_N_PCS,
) -> ml_collections.ConfigDict:
    """Return a lineage-aware Schiebinger config for selected training days."""
    n_pcs = int(schiebinger_n_pcs)
    if n_pcs <= 0:
        raise ValueError("schiebinger_n_pcs must be positive.")
    training_timepoints = schiebinger.parse_training_timepoints(
        schiebinger_train_times
        if schiebinger_train_times is not None
        else os.getenv("SCHIEBINGER_TRAIN_TIMES", "0,18")
    )
    dataset_filename = schiebinger.pca_dataset_filename(n_pcs)
    resolved_dataset = schiebinger.resolve_dataset_path(
        dataset_location,
        filename=dataset_filename,
        n_pcs=n_pcs,
    )
    cfg = _maizels_config(
        slurm_id,
        str(resolved_dataset),
        output_folder,
        early_stopping_patience=early_stopping_patience,
        maizels_ot_coupling=maizels_ot_coupling,
    )
    variant_name = str(cfg.logging.comparison_mode)

    cfg.problem.target = "schiebinger"
    cfg.problem.dataset_name = "schiebinger_serum"
    cfg.problem.lineage_dataset_name = f"schiebinger_serum_hvg_pca{n_pcs}"
    cfg.problem.dataset_location = str(resolved_dataset)
    cfg.problem.schiebinger_filename = dataset_filename
    cfg.problem.subset_to_serum = False
    cfg.problem.embedding_key = "X_pca"
    cfg.problem.n_pcs = n_pcs
    cfg.problem.pca_random_state = cfg.training.seed
    cfg.problem.whiten_pca = False
    cfg.problem.time_key = "day"
    cfg.problem.cell_type_key = "cell_sets"

    cfg.problem.n = 500_000
    cfg.problem.d = n_pcs
    cfg.problem.base = "schiebinger_first_timepoint"
    source_time, target_time = training_timepoints[0], training_timepoints[-1]
    source_value, target_value = float(source_time), float(target_time)
    experiment_timepoints = [
        timepoint
        for timepoint in schiebinger.TIMEPOINTS
        if source_value <= float(timepoint) <= target_value
    ]
    cfg.problem.source_time = source_time
    cfg.problem.target_time = target_time
    cfg.problem.schiebinger_train_times = list(training_timepoints)
    cfg.problem.retained_timepoints = list(training_timepoints)
    cfg.problem.evaluation_timepoints = [
        timepoint
        for timepoint in experiment_timepoints
        if timepoint not in training_timepoints
    ]
    cfg.problem.timepoint_order = experiment_timepoints
    cfg.problem.timepoint_values = [
        (float(timepoint) - source_value) / (target_value - source_value)
        for timepoint in experiment_timepoints
    ]
    cfg.problem.interp_type = "time_rescaled_linear"
    cfg.problem.interp_uses_labels = True
    cfg.problem.pair_time_bounds_in_label = True
    cfg.problem.pair_mode = cfg.problem.get_ref("maizels_pair_mode")

    cfg.problem.schiebinger_holdout_fraction = 0.1
    cfg.problem.schiebinger_holdout_n = 0
    cfg.problem.schiebinger_holdout_seed = 701

    cfg.problem.lineage_transition_mode = os.getenv(
        "SCHIEBINGER_LINEAGE_TRANSITION_MODE", "descendant"
    )
    cfg.problem.lineage_class_names = list(schiebinger.CLASS_NAMES)
    cfg.problem.lineage_transition_edges = [
        list(edge) for edge in schiebinger.TRANSITION_EDGES
    ]
    cfg.problem.lineage_unconstrained_class_names = list(
        schiebinger.UNCONSTRAINED_CLASS_NAMES
    )
    training_classifier_path = _resolve_classifier_path(
        training_timepoints,
        classifier_path,
        n_pcs,
    )
    full_classifier_path = _resolve_full_data_classifier_path(
        full_data_classifier_path,
        n_pcs,
    )
    classifier_required = schiebinger.flow_training_requires_classifier(cfg)
    cfg.problem.flow_training_requires_classifier = classifier_required
    cfg.problem.training_classifier_path = training_classifier_path
    cfg.problem.full_data_classifier_path = full_classifier_path
    # Prior-free variants may point the generic diagnostics path at the
    # all-days model because this field is never consumed during training.
    cfg.problem.classifier_path = (
        training_classifier_path if classifier_required else full_classifier_path
    )
    cfg.problem.schiebinger_auto_train_classifiers = True
    cfg.problem.schiebinger_classifier_python = os.getenv(
        "SCHIEBINGER_CLASSIFIER_PYTHON", ""
    )
    cfg.problem.schiebinger_classifier_device = os.getenv(
        "SCHIEBINGER_CLASSIFIER_DEVICE", "cpu"
    )
    cfg.problem.schiebinger_classifier_batch_size = int(
        os.getenv("SCHIEBINGER_CLASSIFIER_BATCH_SIZE", "512")
    )
    cfg.problem.schiebinger_classifier_max_epochs = int(
        os.getenv("SCHIEBINGER_CLASSIFIER_MAX_EPOCHS", "100")
    )
    cfg.problem.schiebinger_classifier_patience = int(
        os.getenv("SCHIEBINGER_CLASSIFIER_PATIENCE", "20")
    )

    cfg.network.output_dim = n_pcs
    cfg.network.input_dims = (n_pcs,)

    cfg.optimization.early_stopping.patience = 30

    schedule = schiebinger.classifier_schedule_slug(training_timepoints)
    cfg.logging.wandb_name = (
        f"schiebinger_hvg_pca{n_pcs}_times_{schedule}_{variant_name}"
    )
    cfg.logging.output_name = cfg.logging.wandb_name
    cfg.logging.comparison_mode = variant_name

    cfg.logging.maizels.enabled = True
    cfg.logging.maizels.trajectory_diagnostics_enabled = True
    cfg.logging.maizels.distribution_eval_enabled = True
    cfg.logging.maizels.full_data_classifier_path = full_classifier_path
    cfg.logging.maizels.distribution_eval_timepoints = list(
        cfg.problem.evaluation_timepoints
    )
    cfg.logging.maizels.distribution_eval_source_pool = "auto"
    cfg.logging.maizels.distribution_eval_source_max_points = 0
    cfg.logging.maizels.distribution_eval_points_per_time = 0
    cfg.logging.maizels.distribution_eval_max_timepoints = 0
    cfg.logging.maizels.distribution_eval_interval_local = True

    return cfg

"""Three-timepoint CITE/Multi PCA100 flow-map experiments.

This is intentionally based on :mod:`configs.maizels_pca50`: slurm IDs select
the same vanilla/biological-prior, flow-matching/flow-map, and constrained
variants.  ``dataset_name`` selects CITE or Multi, while ``heldout_day`` selects
which internal population is excluded from training and used for trajectory
evaluation.

The original four timepoints are assigned the traditional MFM clock
``2 -> 0, 3 -> 1/3, 4 -> 2/3, 7 -> 1``.  Training pairs are balanced across
the two adjacent intervals remaining after one internal day is removed.
"""

from __future__ import annotations

import os
from pathlib import Path

import ml_collections

from common import cite_multi
from configs.maizels_pca50 import get_config as _maizels_config


def _resolve_classifier_path(dataset_name: str, classifier_path: str | None) -> str:
    if classifier_path:
        return str(Path(classifier_path).expanduser().resolve())
    env_path = os.getenv("CITE_MULTI_CLASSIFIER_PATH", "")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    repo_root = Path(__file__).resolve().parents[2]
    return str((repo_root / f"celltype_classifier_{dataset_name}_pca100.pt").resolve())


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    dataset_name: str | None = None,
    heldout_day: str | int | None = None,
    classifier_path: str | None = None,
) -> ml_collections.ConfigDict:
    """Return a Maizels-equivalent config for a CITE or Multi leave-one-day-out run."""
    dataset_name = cite_multi.canonical_dataset_name(
        dataset_name or os.getenv("CITE_MULTI_DATASET", "cite")
    )
    heldout_day = str(
        heldout_day
        if heldout_day is not None
        else os.getenv("CITE_MULTI_HELDOUT_DAY", "4")
    )
    if heldout_day not in ("3", "4"):
        raise ValueError("heldout_day must be 3 or 4.")

    cfg = _maizels_config(slurm_id, dataset_location, output_folder)
    variant_name = str(cfg.logging.comparison_mode)
    resolved_dataset = cite_multi.resolve_dataset_path(
        dataset_location,
        dataset_name,
    )

    cfg.problem.target = "cite_multi_pca100"
    cfg.problem.dataset_name = dataset_name
    cfg.problem.lineage_dataset_name = f"{dataset_name}_pca100"
    cfg.problem.dataset_location = str(resolved_dataset)
    cfg.problem.n = 500_000
    cfg.problem.d = 100
    cfg.problem.base = "cite_multi_day2"
    cfg.problem.source_time = "2"
    cfg.problem.target_time = "7"
    cfg.problem.heldout_timepoint = heldout_day
    cfg.problem.maizels_holdout_fraction = 0.1
    cfg.problem.cite_multi_train_fraction = 0.9
    cfg.problem.retained_timepoints = list(
        timepoint for timepoint in cite_multi.TIMEPOINTS if timepoint != heldout_day
    )
    cfg.problem.timepoint_order = list(cite_multi.TIMEPOINTS)
    cfg.problem.timepoint_values = [
        cite_multi.NORMALIZED_TIMES[timepoint]
        for timepoint in cite_multi.TIMEPOINTS
    ]
    cfg.problem.interp_type = "time_rescaled_linear"
    cfg.problem.interp_uses_labels = True
    cfg.problem.pair_time_bounds_in_label = True

    # Keep the Maizels-compatible name while exposing the generic spelling for
    # the new experiment. The shared reference keeps edits to either spelling
    # synchronized in derived configs and parameter sweeps.
    cfg.problem.pair_mode = cfg.problem.get_ref("maizels_pair_mode")
    # Couple the full optimizer batch rather than dividing it into smaller OT
    # blocks. Since a training batch is balanced over two retained intervals,
    # each interval gets one OT problem of size optimization.bs / 2.

    cfg.optimization.bs = 128

    cfg.problem.ot_minibatch_max_resamples = 20
    cfg.problem.ot_infeasible_fallback = "partial"
    cfg.problem.lineage_class_names = list(cite_multi.CLASS_NAMES)
    cfg.problem.lineage_transition_edges = [
        list(edge) for edge in cite_multi.TRANSITION_EDGES
    ]
    cfg.problem.classifier_path = _resolve_classifier_path(
        dataset_name,
        classifier_path,
    )

    # The complete Maizels logging suite remains enabled, but distributional
    # evaluation is restricted to the one day omitted from training.
    cfg.logging.wandb_name = (
        f"{dataset_name}_pca100_holdout_day{heldout_day}_{variant_name}"
    )
    cfg.logging.output_name = cfg.logging.wandb_name
    cfg.logging.comparison_mode = variant_name
    cfg.logging.maizels.distribution_eval_timepoints = [heldout_day]
    cfg.logging.maizels.distribution_eval_max_timepoints = 1
    # Keep the generic exact-EMD/MMD diagnostic bounded here. The MFM-compatible
    # evaluator below is the CITE/Multi full-population evaluation.
    cfg.logging.maizels.distribution_eval_source_pool = "auto"
    cfg.logging.maizels.distribution_eval_source_max_points = 1024
    cfg.logging.maizels.distribution_eval_points_per_time = 1024

    # Exact reproduction of the original CITE/Multi MFM test protocol. A slash
    # in ``mfm/test_EMD`` gives the metric its own W&B pane.
    cfg.logging.mfm = ml_collections.ConfigDict()
    cfg.logging.mfm.enabled = True
    cfg.logging.mfm.final_only = False
    cfg.logging.mfm.frequency = cfg.logging.get_ref("visual_freq")
    cfg.logging.mfm.euler_steps = 100
    cfg.logging.mfm.flowmap_steps = 100
    cfg.logging.mfm.max_points = 0
    cfg.logging.mfm.seed = cfg.training.seed + 2901

    cfg.network.output_dim = 100
    cfg.network.input_dims = (100,)
    cfg.network.n_neurons = 1024
    cfg.network.n_hidden = 2

    cfg.optimization.total_steps = 5_000


    cfg.constraints.loss_point_entropy_weight = 0.03
    cfg.constraints.weight = 10000.0

    cfg.logging.visual_ema_factor = None

    return cfg

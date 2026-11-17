"""Three-marginal Calderon experiment using the Maizels plain OT recipe.

Train at 4–6, 8–10 and 12–14 h; hold out 6–8 and 10–12 h. Filtering and loss
use the selected-times classifier, while violations use the all-times oracle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ml_collections import ConfigDict

from . import maizels_pca50
from common.calderon import (
    DEFAULT_DATASET, DEFAULT_TRAINING_MANIFEST, DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_SPLIT_MANIFEST, DEFAULT_TRAINING_EXPORT,
)


ROOT = Path(__file__).resolve().parents[2]


def get_config(
    dataset_location: str = "",
    output_folder: str = "",
    classifier_path: str = "",
    classifier_python: str = "",
    seed: int = 0,
    evaluation_classifier_path: str = "",
) -> ConfigDict:
    # ID 7 is Maizels's unconstrained minibatch-OT flow map: LSD, convex
    # stopgrad, 75% diagonal samples, RAdam, and the same native MLP.
    base = maizels_pca50.get_config(
        7, early_stopping_patience=10, maizels_ot_coupling="minibatch_ot",
        maizels_schedule="d3_d8", maizels_time_mode="real_time",
        hparam_val_times=(), seed=seed,
    )
    cfg = ConfigDict()
    for section in ("training", "optimization", "network"):
        cfg[section] = ConfigDict(base[section].to_dict())
    cfg.optimization.early_stopping.check_freq = 100
    cfg.optimization.early_stopping.warmup_steps = 0
    cfg.optimization.early_stopping.min_delta = 0.0
    cfg.constraints = ConfigDict({"enabled": False})
    cfg.network.output_dim = 100
    cfg.network.input_dims = (100,)

    cfg.problem = ConfigDict()
    cfg.problem.target = "calderon_pca100"
    cfg.problem.d = 100
    cfg.problem.image_dims = None
    cfg.problem.num_classes = None
    cfg.problem.base = "calderon_source"
    cfg.problem.interp_type = "time_rescaled_linear"
    # Pair labels carry tissue IDs and absolute interval bounds; the network
    # remains unconditional on tissue labels.
    cfg.problem.interp_uses_labels = True
    cfg.problem.pair_time_bounds_in_label = True
    cfg.problem.dataset_location = str(Path(dataset_location).expanduser().resolve()) if dataset_location else str(
        DEFAULT_DATASET
    )
    cfg.problem.cohort_branch_prefix = ""
    cfg.problem.retained_timepoints = ["4-6", "8-10", "12-14"]
    cfg.problem.evaluation_timepoints = ["6-8", "10-12"]
    cfg.problem.split_manifest_path = str(DEFAULT_SPLIT_MANIFEST)
    cfg.problem.source_time = "4-6"
    cfg.problem.target_time = "12-14"
    cfg.problem.source_hours = 5.0
    cfg.problem.target_hours = 13.0
    cfg.problem.validation_fraction = 0.1
    cfg.problem.test_fraction = 0.1
    cfg.problem.split_seed = 0
    cfg.problem.gaussian_scale = "adaptive"
    cfg.problem.maizels_pair_mode = "ot_plain"
    cfg.problem.maizels_ot_coupling = "minibatch_ot"
    cfg.problem.ot_minibatch_size = cfg.optimization.get_ref("bs")
    cfg.problem.ot_minibatch_max_resamples = 20
    cfg.problem.ot_minibatch_infeasible_fallback = "error"

    cfg.logging = ConfigDict()
    cfg.logging.output_folder = str(Path(output_folder).expanduser().resolve()) if output_folder else str(
        ROOT / "outputs" / f"calderon_symmetric_4_8_12_ot_flow_map_seed{seed}"
    )
    cfg.logging.scalar_freq = 100
    cfg.logging.save_freq = 5000
    cfg.logging.wandb_project = "self-distill-flow-maps"
    cfg.logging.wandb_name = f"calderon_symmetric_4_8_12_pca100_ot_flow_map_seed{seed}"

    cfg.evaluation = ConfigDict()
    cfg.evaluation.ema_factor = 0.999
    cfg.evaluation.validation_batches = 8
    cfg.evaluation.validation_seed = 2701
    # W1 snapshots are extracted independently at .25/.5/.75/1.0. The 50
    # violation checks use .02/.04/.../1.0 on the same composed rollout.
    cfg.evaluation.flowmap_n_steps = 100
    cfg.evaluation.checkpoint_times = [i / 50 for i in range(1, 51)]
    cfg.evaluation.validation_pair_mode = "ot_plain"
    cfg.evaluation.flow_batch_size = 256
    # Zero means every test source cell, pushed forward once.
    cfg.evaluation.source_max_points = 0
    cfg.evaluation.marginal_max_points = 2048
    cfg.evaluation.seed = 2698

    cfg.classifier = ConfigDict()
    cfg.classifier.manifest_path = str(Path(classifier_path).expanduser().resolve()) if classifier_path else str(
        DEFAULT_TRAINING_MANIFEST
    )
    cfg.classifier.evaluation_manifest_path = (
        str(Path(evaluation_classifier_path).expanduser().resolve())
        if evaluation_classifier_path else str(DEFAULT_EVALUATION_MANIFEST)
    )
    cfg.classifier.jax_export_path = str(DEFAULT_TRAINING_EXPORT)
    cfg.classifier.python = classifier_python or os.getenv("CALDERON_CLASSIFIER_PYTHON", sys.executable)
    cfg.classifier.allowed_pairs_path = str(ROOT / "docs/calderon2022/graph_data/broad_tissue_allowed_pairs.csv")
    cfg.classifier.batch_size = 4096
    cfg.classifier.threads = 4
    # Always report raw argmax violations as well as this confidence filter.
    cfg.classifier.prob_threshold = 0.9
    cfg.classifier.margin_threshold = 0.0
    return cfg

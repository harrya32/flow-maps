"""Calderon PCA100 versions of the three Maizels minibatch-OT SSFMs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import calderon_pca100, maizels_stochastic

ROOT = Path(__file__).resolve().parents[2]
# Calderon ID/name, corresponding Maizels stochastic ID.
VARIANTS = (("ot", 5), ("filtered", 3), ("constrained", 4))


def get_config(
    slurm_id: int = 0,
    dataset_location: str = "",
    output_folder: str = "",
    classifier_path: Optional[str] = None,
    evaluation_classifier_path: Optional[str] = None,
    classifier_export_path: Optional[str] = None,
    evaluation_classifier_export_path: Optional[str] = None,
    learning_rate: Optional[float] = None,
    constraint_weight: Optional[float] = None,
    entropy_weight: Optional[float] = None,
    diffusion_scale: Optional[float] = None,
    gamma_scale: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    seed: Optional[int] = None,
    total_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_pairs: Optional[int] = None,
):
    if not 0 <= int(slurm_id) < len(VARIANTS):
        raise ValueError("Calderon stochastic IDs are 0 (OT), 1 (filtered), and 2 (constrained).")
    variant, reference_id = VARIANTS[int(slurm_id)]
    cfg = maizels_stochastic.get_config(
        reference_id, learning_rate=learning_rate, constraint_weight=constraint_weight,
        entropy_weight=entropy_weight, diffusion_scale=diffusion_scale, gamma_scale=gamma_scale,
        early_stopping_patience=early_stopping_patience, seed=seed, total_steps=total_steps,
        batch_size=batch_size, n_pairs=n_pairs, maizels_schedule="d3_d3p8_d8", hparam_val_times=(),
    )
    data_cfg = calderon_pca100.get_config(
        dataset_location=dataset_location, classifier_path=classifier_path or "",
        evaluation_classifier_path=evaluation_classifier_path or "", seed=cfg.training.seed,
    )
    # Keep the stochastic optimizer, noise, network, OT, validation and logging
    # defaults. Only transplant the Calderon representation and metadata.
    for field in (
        "target", "d", "base", "dataset_location", "source_time", "target_time",
        "source_hours", "target_hours", "cohort_branch_prefix", "retained_timepoints",
        "evaluation_timepoints", "timepoint_order", "timepoint_values", "split_manifest_path",
        "split_protocol", "classifier_path",
    ):
        cfg.problem[field] = data_cfg.problem[field]
    cfg.problem.maizels_schedule = "symmetric_4_8_12"
    cfg.problem.validation_fraction = cfg.problem.maizels_holdout_fraction
    cfg.problem.test_fraction = 0.0
    cfg.problem.split_seed = cfg.problem.maizels_holdout_seed
    cfg.network.input_dims = (100,)
    cfg.network.output_dim = 100
    cfg.classifier = data_cfg.classifier
    if classifier_export_path:
        cfg.classifier.jax_export_path = str(Path(classifier_export_path).expanduser().resolve())
    if evaluation_classifier_export_path:
        cfg.classifier.evaluation_jax_export_path = str(Path(evaluation_classifier_export_path).expanduser().resolve())

    cfg.evaluation.protocol = "maizels"
    cfg.evaluation.max_source_points = 4096
    cfg.evaluation.max_target_points = 4096
    cfg.evaluation.include_direct = False
    cfg.evaluation.include_mmd = False
    cfg.logging.maizels.full_data_classifier_path = cfg.classifier.evaluation_manifest_path
    cfg.logging.maizels.distribution_eval_timepoints = list(cfg.problem.evaluation_timepoints)
    cfg.logging.maizels.distribution_eval_points_per_time = cfg.evaluation.max_target_points
    cfg.logging.maizels.distribution_eval_source_max_points = cfg.evaluation.max_source_points
    cfg.logging.output_folder = (str(Path(output_folder).expanduser().resolve()) if output_folder
                                 else str(ROOT / "outputs/calderon_stochastic"))
    original_prefix = "maizels_pca50_d3_d3p8_d8"
    cfg.logging.wandb_name = cfg.logging.wandb_name.replace(original_prefix, "calderon_symmetric_4_8_12_pca100", 1)
    cfg.logging.output_name = cfg.logging.wandb_name
    cfg.problem.stochastic_variant = variant
    cfg.problem.reference_stochastic_slurm_id = reference_id
    return cfg

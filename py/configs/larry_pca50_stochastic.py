"""Strong stochastic flow maps for LARRY in a configurable PCA space.

This exposes exactly the same nine variants and defaults as
``larry_spring2d_stochastic``; only the representation, dimensionality,
processed dataset, classifier checkpoints, and run-name prefix differ.
"""

from __future__ import annotations

from typing import Optional

import ml_collections

from common import larry
from .larry_spring2d_stochastic import VARIANTS
from .larry_spring2d_stochastic import get_config as _shared_get_config
from .larry_spring2d_stochastic import get_hparam_sweep_spec


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    classifier_path: Optional[str] = None,
    full_data_classifier_path: Optional[str] = None,
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
    ot_minibatch_size: Optional[int] = None,
    interpolant_check_times: Optional[int] = None,
    clone_samples_per_source: Optional[int] = None,
    clone_noise_draws: Optional[int] = None,
    clone_min_source_cells: Optional[int] = None,
    clone_min_target_cells: Optional[int] = None,
    larry_clone_labelled_only: bool = False,
    larry_n_pcs: Optional[int] = None,
) -> ml_collections.ConfigDict:
    """Return a LARRY PCA SSFM configuration (50 PCs by default)."""
    return _shared_get_config(
        slurm_id,
        dataset_location=dataset_location,
        output_folder=output_folder,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
        learning_rate=learning_rate,
        constraint_weight=constraint_weight,
        entropy_weight=entropy_weight,
        diffusion_scale=diffusion_scale,
        gamma_scale=gamma_scale,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        total_steps=total_steps,
        batch_size=batch_size,
        n_pairs=n_pairs,
        ot_minibatch_size=ot_minibatch_size,
        interpolant_check_times=interpolant_check_times,
        clone_samples_per_source=clone_samples_per_source,
        clone_noise_draws=clone_noise_draws,
        clone_min_source_cells=clone_min_source_cells,
        clone_min_target_cells=clone_min_target_cells,
        larry_representation=larry.PCA50_REPRESENTATION,
        larry_clone_labelled_only=larry_clone_labelled_only,
        larry_n_pcs=larry_n_pcs,
    )

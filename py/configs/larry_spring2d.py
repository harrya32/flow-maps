"""LARRY in-vitro flow maps in the published two-dimensional SPRING layout."""

from __future__ import annotations

from common import larry
from configs import larry_pca50


ACTIVE_VARIANTS = larry_pca50.ACTIVE_VARIANTS
VARIANT_CATALOG = larry_pca50.VARIANT_CATALOG
get_hparam_sweep_spec = larry_pca50.get_hparam_sweep_spec


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
    larry_clone_labelled_only: bool = False,
    larry_n_pcs: int | None = None,
):
    return larry_pca50.get_config(
        slurm_id,
        dataset_location,
        output_folder,
        early_stopping_patience=early_stopping_patience,
        maizels_ot_coupling=maizels_ot_coupling,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
        learning_rate=learning_rate,
        constraint_weight=constraint_weight,
        entropy_weight=entropy_weight,
        seed=seed,
        ot_minibatch_size=ot_minibatch_size,
        larry_representation=larry.SPRING2D_REPRESENTATION,
        larry_clone_labelled_only=larry_clone_labelled_only,
        larry_n_pcs=larry_n_pcs,
    )

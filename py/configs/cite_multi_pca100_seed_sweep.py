"""Seed-sweep wrapper for the CITE/Multi PCA100 experiments.

The shell launcher sets ``CITE_MULTI_SEED`` for each run. Keeping this in a
wrapper leaves the ordinary :mod:`configs.cite_multi_pca100` run names and
defaults unchanged while ensuring sweep checkpoints and W&B runs are unique.
"""

from __future__ import annotations

import os

from configs.cite_multi_pca100 import get_config as _base_get_config


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    dataset_name: str | None = None,
    heldout_day: str | int | None = None,
    cite_multi_time_mode: str | None = None,
    classifier_path: str | None = None,
    full_data_classifier_path: str | None = None,
    learning_rate: float | None = None,
):
    seed = int(os.getenv("CITE_MULTI_SEED", "0"))
    if seed < 0:
        raise ValueError("CITE_MULTI_SEED must be non-negative.")
    cfg = _base_get_config(
        slurm_id,
        dataset_location,
        output_folder,
        dataset_name=dataset_name,
        heldout_day=heldout_day,
        cite_multi_time_mode=cite_multi_time_mode,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
        learning_rate=learning_rate,
        seed=seed,
    )
    return cfg

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
    classifier_path: str | None = None,
    full_data_classifier_path: str | None = None,
):
    cfg = _base_get_config(
        slurm_id,
        dataset_location,
        output_folder,
        dataset_name=dataset_name,
        heldout_day=heldout_day,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
    )
    seed = int(os.getenv("CITE_MULTI_SEED", str(cfg.training.seed)))
    if seed < 0:
        raise ValueError("CITE_MULTI_SEED must be non-negative.")

    cfg.training.seed = seed
    cfg.logging.mfm.seed = seed + 2901
    run_name = f"{cfg.logging.wandb_name}_seed{seed}"
    cfg.logging.wandb_name = run_name
    cfg.logging.output_name = run_name
    return cfg

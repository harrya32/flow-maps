from pathlib import Path

import numpy as np

from common import larry
from scripts import train_cite_multi_celltype_classifiers as shared
from scripts import train_larry_celltype_classifiers as trainer


def test_default_run_trains_all_days_and_d2_d6_classifiers():
    args = trainer.parse_args([])
    runs = trainer.requested_runs(args)

    assert args.variants == ["all_days", "train_days_d2_d6"]
    assert [variant.key for variant, _ in runs] == args.variants
    assert (
        runs[0][1]
        == (
            larry.DEFAULT_CLASSIFIER_DIR
            / "celltype_classifier_larry_hvg2000_pca50_all_days.pt"
        ).resolve()
    )
    assert (
        runs[1][1]
        == (
            larry.DEFAULT_CLASSIFIER_DIR
            / "celltype_classifier_larry_hvg2000_pca50_train_days_d2_d6.pt"
        ).resolve()
    )
    assert tuple(sorted(larry.CLASS_NAMES)) == larry.CLASS_NAMES


def test_training_classifier_selects_exactly_d2_and_d6():
    data = {
        "n_cells": 7,
        "days": np.asarray(["D2", "D4", "D6", "D2", "D4", "D6", "D6"]),
    }
    selected = shared.select_variant_rows(
        np,
        data,
        trainer.CLASSIFIER_VARIANTS["train_days_d2_d6"],
    )

    np.testing.assert_array_equal(selected, np.asarray([0, 2, 3, 5, 6]))
    assert set(data["days"][selected]) == {"D2", "D6"}


def test_checkpoint_path_can_be_relocated(tmp_path: Path):
    path = trainer.checkpoint_path(
        tmp_path,
        "train_days_d2_d6",
        n_pcs=50,
        n_hvgs=2000,
    )

    assert path.parent == tmp_path.resolve()
    assert path.name == ("celltype_classifier_larry_hvg2000_pca50_train_days_d2_d6.pt")


def test_classifier_checkpoint_name_tracks_requested_pca_dimension(tmp_path: Path):
    path = trainer.checkpoint_path(
        tmp_path,
        "all_days",
        n_pcs=20,
        n_hvgs=2000,
    )

    assert path.name == "celltype_classifier_larry_hvg2000_pca20_all_days.pt"


def test_spring_classifier_uses_separate_two_dimensional_checkpoint_names():
    args = trainer.parse_args(["--representation", "spring2d"])
    runs = trainer.requested_runs(args)

    assert args.n_pcs == 2
    assert [path.name for _, path in runs] == [
        "celltype_classifier_larry_spring2d_all_days.pt",
        "celltype_classifier_larry_spring2d_train_days_d2_d6.pt",
    ]


def test_clone_labelled_classifiers_use_separate_checkpoint_names():
    args = trainer.parse_args(["--clone-labelled-only"])
    runs = trainer.requested_runs(args)

    assert [path.name for _, path in runs] == [
        "celltype_classifier_larry_clone_labelled_hvg2000_pca50_all_days.pt",
        (
            "celltype_classifier_larry_clone_labelled_hvg2000_pca50_"
            "train_days_d2_d6.pt"
        ),
    ]

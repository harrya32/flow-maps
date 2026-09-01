import numpy as np
import torch

from scripts import train_cite_multi_celltype_classifiers as classifiers


def test_default_run_trains_three_variants_for_each_dataset():
    args = classifiers.parse_args([])

    assert args.datasets == ["cite", "multi"]
    assert args.variants == ["all_days", "except_day3", "except_day4"]

    expected = {
        (
            dataset,
            variant,
        ): args.output_root
        / f"{dataset}-classifiers"
        / f"celltype_classifier_{dataset}_pca100_{variant}.pt"
        for dataset in args.datasets
        for variant in args.variants
    }
    for (dataset, variant), path in expected.items():
        assert (
            classifiers.checkpoint_path(
                args.output_root,
                dataset,
                args.n_pcs,
                variant,
            )
            == path
        )


def test_leave_one_day_out_variants_remove_only_the_requested_day():
    data = {
        "n_cells": 8,
        "days": np.asarray(["2", "3", "4", "7", "3", "4", "2", "7"]),
    }

    all_rows = classifiers.select_variant_rows(
        np,
        data,
        classifiers.CLASSIFIER_VARIANTS["all_days"],
    )
    no_day3 = classifiers.select_variant_rows(
        np,
        data,
        classifiers.CLASSIFIER_VARIANTS["except_day3"],
    )
    no_day4 = classifiers.select_variant_rows(
        np,
        data,
        classifiers.CLASSIFIER_VARIANTS["except_day4"],
    )

    np.testing.assert_array_equal(all_rows, np.arange(8))
    assert set(data["days"][no_day3]) == {"2", "4", "7"}
    assert set(data["days"][no_day4]) == {"2", "3", "7"}
    assert "3" not in data["days"][no_day3]
    assert "4" not in data["days"][no_day4]


def test_classifier_split_is_stratified_ninety_ten():
    class_names = ["BP", "EryP", "HSC", "MasP", "MkP", "MoP", "NeuP"]
    labels = np.repeat(class_names, 10)
    features = np.arange(70 * 3, dtype=np.float32).reshape(70, 3)

    _, prepared, class_counts = classifiers.make_loaders(
        np,
        torch,
        features,
        labels,
        class_names,
        seed=0,
        batch_size=16,
        num_workers=0,
        device=torch.device("cpu"),
    )

    assert len(prepared["split_indices"]["train"]) == 63
    assert len(prepared["split_indices"]["validation"]) == 7
    np.testing.assert_array_equal(class_counts, np.full(7, 10))
    np.testing.assert_array_equal(
        np.bincount(prepared["split_y"]["validation"], minlength=7),
        np.ones(7, dtype=int),
    )

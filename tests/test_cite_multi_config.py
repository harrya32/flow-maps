from pathlib import Path

from configs import cite_multi_pca100


def test_training_and_evaluation_use_separate_dataset_specific_classifiers():
    for dataset_name in ("cite", "multi"):
        for heldout_day in ("3", "4"):
            cfg = cite_multi_pca100.get_config(
                0,
                dataset_name=dataset_name,
                heldout_day=heldout_day,
            )

            classifier_dir = f"{dataset_name}-classifiers"
            assert Path(cfg.problem.classifier_path).parent.name == classifier_dir
            assert cfg.problem.classifier_path.endswith(
                f"celltype_classifier_{dataset_name}_pca100_"
                f"except_day{heldout_day}.pt"
            )
            assert (
                Path(cfg.logging.maizels.full_data_classifier_path).parent.name
                == classifier_dir
            )
            assert cfg.logging.maizels.full_data_classifier_path.endswith(
                f"celltype_classifier_{dataset_name}_pca100_all_days.pt"
            )
            assert (
                cfg.logging.maizels.full_data_classifier_path
                != cfg.problem.classifier_path
            )


def test_training_and_evaluation_classifier_overrides_are_independent(tmp_path):
    training_path = tmp_path / "training.pt"
    evaluation_path = tmp_path / "evaluation.pt"
    cfg = cite_multi_pca100.get_config(
        0,
        dataset_name="cite",
        heldout_day="3",
        classifier_path=str(training_path),
        full_data_classifier_path=str(evaluation_path),
    )

    assert cfg.problem.classifier_path == str(training_path.resolve())
    assert cfg.logging.maizels.full_data_classifier_path == str(
        evaluation_path.resolve()
    )

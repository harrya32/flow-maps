from pathlib import Path

import numpy as np

from common import schiebinger
from common import schiebinger_classifier_training as automatic
from configs import schiebinger_lsd
from scripts import train_cite_multi_celltype_classifiers as shared
from scripts import train_schiebinger_celltype_classifiers as trainer


def test_default_manual_run_requests_only_all_days_classifier():
    args = trainer.parse_args([])
    runs = trainer.requested_runs(args)

    assert len(runs) == 1
    variant, checkpoint = runs[0]
    assert variant.key == "all_days"
    assert variant.usage == "flow_evaluation_only"
    assert checkpoint.parent.name == "schiebinger_classifiers"
    assert checkpoint.name == (
        "celltype_classifier_schiebinger_hvg1479_pca5_all_days.pt"
    )


def test_selected_days_variant_keeps_exact_requested_schedule():
    args = trainer.parse_args(
        ["--train-times", "0,3,8.25,18", "--all-days"]
    )
    runs = trainer.requested_runs(args)
    schedule_variant, schedule_path = runs[1]

    assert schedule_variant.included_days == ("0", "3", "8.25", "18")
    assert schedule_variant.key == "train_days_0_3_8p25_18"
    assert schedule_path.name == (
        "celltype_classifier_schiebinger_hvg1479_"
        "pca5_train_days_0_3_8p25_18.pt"
    )


def test_manual_classifier_paths_include_arbitrary_pca_dimension():
    args = trainer.parse_args(
        ["--all-days", "--train-times", "0,3,18", "--n-pcs", "12"]
    )
    runs = trainer.requested_runs(args)

    assert runs[0][1].name == (
        "celltype_classifier_schiebinger_hvg1479_pca12_all_days.pt"
    )
    assert runs[1][1].name == (
        "celltype_classifier_schiebinger_hvg1479_pca12_train_days_0_3_18.pt"
    )

    data = {
        "n_cells": 6,
        "days": np.asarray(["0", "0.5", "3", "8.25", "9", "18"]),
    }
    selected = shared.select_variant_rows(np, data, schedule_variant)
    np.testing.assert_array_equal(selected, np.asarray([0, 2, 3, 5]))


def _complete_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    path.with_suffix(".npz").touch()


def test_automatic_training_builds_all_days_and_required_schedule(tmp_path):
    cfg = schiebinger_lsd.get_config(
        2,
        schiebinger_train_times="0,3,8.25,18",
    )
    full_path = tmp_path / "full.pt"
    training_path = tmp_path / "selected.pt"
    cfg.logging.maizels.full_data_classifier_path = str(full_path)
    cfg.problem.full_data_classifier_path = str(full_path)
    cfg.problem.training_classifier_path = str(training_path)
    cfg.problem.classifier_path = str(training_path)
    commands = []

    def runner(command, *, check, cwd):
        commands.append((command, check, cwd))
        _complete_checkpoint(full_path)
        _complete_checkpoint(training_path)

    automatic.ensure_schiebinger_classifiers(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )

    assert len(commands) == 1
    command, check, cwd = commands[0]
    assert check
    assert Path(cwd).name == "flow-maps"
    assert command[0] == "python-for-test"
    assert "--all-days" in command
    assert "--train-times" in command
    assert command[command.index("--train-times") + 1] == "0,3,8.25,18"

    # The complete pair is reused and does not launch another training process.
    automatic.ensure_schiebinger_classifiers(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )
    assert len(commands) == 1


def test_prior_free_flow_only_prepares_all_days_evaluation_model(tmp_path):
    cfg = schiebinger_lsd.get_config(0)
    full_path = tmp_path / "full.pt"
    training_path = tmp_path / "unused-selected.pt"
    cfg.logging.maizels.full_data_classifier_path = str(full_path)
    cfg.problem.full_data_classifier_path = str(full_path)
    cfg.problem.training_classifier_path = str(training_path)
    cfg.problem.classifier_path = str(full_path)
    commands = []

    def runner(command, *, check, cwd):
        commands.append(command)
        _complete_checkpoint(full_path)

    automatic.ensure_schiebinger_classifiers(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )

    assert len(commands) == 1
    assert "--all-days" in commands[0]
    assert "--train-times" not in commands[0]
    assert not training_path.exists()


def test_cached_full_model_is_not_requested_again_when_schedule_is_missing(
    tmp_path,
):
    cfg = schiebinger_lsd.get_config(
        2,
        schiebinger_train_times="0,3,8.25,18",
    )
    full_path = tmp_path / "full.pt"
    training_path = tmp_path / "selected.pt"
    _complete_checkpoint(full_path)
    cfg.logging.maizels.full_data_classifier_path = str(full_path)
    cfg.problem.full_data_classifier_path = str(full_path)
    cfg.problem.training_classifier_path = str(training_path)
    cfg.problem.classifier_path = str(training_path)
    commands = []

    def runner(command, *, check, cwd):
        commands.append(command)
        _complete_checkpoint(training_path)

    automatic.ensure_schiebinger_classifiers(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )

    assert len(commands) == 1
    assert "--all-days" not in commands[0]
    assert "--train-times" in commands[0]

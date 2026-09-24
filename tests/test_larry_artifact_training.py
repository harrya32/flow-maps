from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common import larry
from common import larry_artifact_training as automatic


def _config(tmp_path: Path, *, clone_labelled_only: bool = True, n_pcs: int = 50):
    data_dir = tmp_path / "data"
    classifier_dir = tmp_path / "classifiers"
    dataset_path = data_dir / larry.pca_dataset_filename(
        n_pcs=n_pcs,
        clone_labelled_only=clone_labelled_only,
    )
    training_path = larry.classifier_checkpoint_path(
        training_timepoints=("D2", "D6"),
        classifier_dir=classifier_dir,
        n_pcs=n_pcs,
        clone_labelled_only=clone_labelled_only,
    )
    full_path = larry.classifier_checkpoint_path(
        all_days=True,
        classifier_dir=classifier_dir,
        n_pcs=n_pcs,
        clone_labelled_only=clone_labelled_only,
    )
    return SimpleNamespace(
        problem=SimpleNamespace(
            target="larry_pca50",
            larry_clone_labelled_only=clone_labelled_only,
            larry_auto_prepare_artifacts=True,
            larry_representation="pca50",
            dataset_location=str(dataset_path),
            larry_raw_data_dir=str(data_dir),
            n_hvgs=2000,
            n_pcs=n_pcs,
            larry_artifact_seed=0,
            larry_classifier_batch_size=512,
            larry_classifier_max_epochs=100,
            larry_classifier_patience=20,
            larry_classifier_device="cpu",
            training_classifier_path=str(training_path),
            full_data_classifier_path=str(full_path),
            training_classifier_available=False,
            full_data_classifier_available=False,
        ),
        logging=SimpleNamespace(
            maizels=SimpleNamespace(
                violation_metrics_available=False,
                violation_metrics_unavailable_reason="missing",
            )
        ),
    )


def test_automatic_clone_artifacts_are_built_once_and_then_reused(
    monkeypatch, tmp_path
):
    cfg = _config(tmp_path)
    commands = []
    monkeypatch.setattr(
        automatic,
        "_validate_pca_dataset",
        lambda cfg, path: None,
    )

    def runner(command, *, check, cwd):
        commands.append((command, check, cwd))
        if Path(command[1]) == automatic.PCA_BUILDER:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch()
            return
        output_dir = Path(command[command.index("--output-dir") + 1])
        variant_start = command.index("--variants") + 1
        variants = command[variant_start:]
        for variant in variants:
            path = automatic._canonical_classifier_path(cfg, variant, output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.with_suffix(".npz").touch()

    automatic.ensure_larry_artifacts(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )

    assert len(commands) == 2
    assert "--clone-labelled-only" in commands[0][0]
    assert set(
        commands[1][0][commands[1][0].index("--variants") + 1 :]
    ) == {"all_days", "train_days_d2_d6"}
    assert cfg.problem.training_classifier_available
    assert cfg.problem.full_data_classifier_available
    assert cfg.logging.maizels.violation_metrics_available

    automatic.ensure_larry_artifacts(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )
    assert len(commands) == 2


def test_npz_only_clone_artifacts_do_not_launch_preprocessing(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    dataset_path = Path(cfg.problem.dataset_location)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.touch()
    Path(cfg.problem.training_classifier_path).with_suffix(".npz").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(cfg.problem.training_classifier_path).with_suffix(".npz").touch()
    Path(cfg.problem.full_data_classifier_path).with_suffix(".npz").touch()
    monkeypatch.setattr(
        automatic,
        "_validate_pca_dataset",
        lambda cfg, path: None,
    )

    def runner(*args, **kwargs):
        raise AssertionError("Cached clone-only artifacts must be reused.")

    automatic.ensure_larry_artifacts(cfg, runner=runner)

    assert cfg.problem.training_classifier_available
    assert cfg.problem.full_data_classifier_available


def test_automatic_nondefault_pca_artifacts_include_requested_dimension(
    monkeypatch, tmp_path
):
    cfg = _config(tmp_path, clone_labelled_only=False, n_pcs=20)
    commands = []
    monkeypatch.setattr(automatic, "_validate_pca_dataset", lambda cfg, path: None)

    def runner(command, *, check, cwd):
        commands.append(command)
        if Path(command[1]) == automatic.PCA_BUILDER:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.touch()
            return
        output_dir = Path(command[command.index("--output-dir") + 1])
        variants = command[command.index("--variants") + 1 :]
        for variant in variants:
            path = automatic._canonical_classifier_path(cfg, variant, output_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(".npz").touch()

    automatic.ensure_larry_artifacts(
        cfg,
        runner=runner,
        python_executable="python-for-test",
    )

    assert len(commands) == 2
    assert commands[0][commands[0].index("--n-pcs") + 1] == "20"
    assert "--clone-labelled-only" not in commands[0]
    assert commands[1][commands[1].index("--n-pcs") + 1] == "20"
    assert "--clone-labelled-only" not in commands[1]


def test_clone_dataset_validation_rejects_unlabelled_cells(monkeypatch, tmp_path):
    cfg = _config(tmp_path)
    monkeypatch.setattr(
        automatic.larry,
        "load_dataset",
        lambda *args, **kwargs: {
            "clone_ids": np.asarray([2, -1, 8]),
            "timepoints": np.asarray(["D2", "D4", "D6"]),
        },
    )

    with pytest.raises(ValueError, match="without clone assignments"):
        automatic._validate_pca_dataset(
            cfg,
            Path(cfg.problem.dataset_location),
        )

from argparse import Namespace
from pathlib import Path
import sys

import numpy as np
import pytest


pytest.importorskip("pytorch_lightning")

REPO_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = REPO_ROOT / "metric-flow-matching"
if str(MFM_ROOT) not in sys.path:
    sys.path.insert(0, str(MFM_ROOT))

from common import cite_multi, maizels  # noqa: E402
from mfm.dataloaders import trajectory_data  # noqa: E402
from mfm.flow_matchers import cite_multi_eval  # noqa: E402
from mfm.flow_matchers import flow_net_train, maizels_eval  # noqa: E402
from mfm.train.main import phase_accelerator  # noqa: E402
from mfm.train.train_utils import create_callbacks  # noqa: E402


def _eval_args(**updates):
    values = {
        "data_name": "cite",
        "dim": 100,
        "whiten": False,
        "cite_multi_eval_enabled": True,
        "cite_multi_eval_classifier_path": "",
        "cite_multi_eval_check_times": 50,
        "cite_multi_eval_every_n_steps": 500,
        "cite_multi_eval_source_max_points": 0,
        "cite_multi_eval_prob_threshold": 0.0,
        "cite_multi_eval_margin_threshold": 0.0,
        "cite_multi_eval_classifier_batch_size": 8192,
        "cite_multi_eval_lineage_transition_mode": "descendant",
    }
    values.update(updates)
    return Namespace(**values)


@pytest.mark.parametrize("data_name", ["cite", "multi"])
def test_auto_geopath_accelerator_avoids_apple_mps(monkeypatch, data_name):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    args = Namespace(
        accelerator="gpu",
        geopath_accelerator="auto",
        flow_accelerator="auto",
        data_type="scrna",
        data_name=data_name,
    )

    assert phase_accelerator(args, "geopath") == "cpu"
    assert phase_accelerator(args, "flow") == "gpu"


def test_default_classifier_is_dataset_specific_all_days_variant():
    for dataset_name in ("cite", "multi"):
        path = cite_multi_eval.default_all_days_classifier_path(dataset_name)
        assert path.parent == REPO_ROOT / f"{dataset_name}-classifiers"
        assert path.name == (f"celltype_classifier_{dataset_name}_pca100_all_days.pt")


def test_evaluation_requires_the_classifier_coordinate_system():
    cite_multi_eval.validate_cite_multi_evaluation_args(_eval_args())
    with pytest.raises(ValueError, match="dim=100"):
        cite_multi_eval.validate_cite_multi_evaluation_args(_eval_args(dim=50))
    with pytest.raises(ValueError, match="whiten=false"):
        cite_multi_eval.validate_cite_multi_evaluation_args(_eval_args(whiten=True))


def test_evaluation_population_uses_heldout_day2_without_replacement():
    source_x = np.arange(60, dtype=np.float32).reshape(6, 10)
    source_types = np.asarray(["BP", "EryP", "HSC", "MasP", "MkP", "MoP"])
    target_x = np.full((4, 10), 7.0, dtype=np.float32)
    datamodule = Namespace(
        timepoint_splits={
            "2": {"holdout_x": source_x, "holdout_types": source_types},
            "7": {
                "holdout_x": target_x,
                "holdout_types": np.asarray(["HSC"] * 4),
            },
        }
    )

    all_x, all_types, actual_target = cite_multi_eval.evaluation_population(
        datamodule, max_points=0, seed=7
    )
    np.testing.assert_array_equal(all_x, source_x)
    np.testing.assert_array_equal(all_types, source_types)
    np.testing.assert_array_equal(actual_target, target_x)

    capped_x, capped_types, _ = cite_multi_eval.evaluation_population(
        datamodule, max_points=3, seed=7
    )
    assert capped_x.shape[0] == 3
    assert np.unique(capped_x[:, 0]).size == 3
    source_type_by_first_value = dict(zip(source_x[:, 0], source_types))
    assert capped_types.tolist() == [
        source_type_by_first_value[value] for value in capped_x[:, 0]
    ]


def test_path_scoring_uses_classifier_order_and_cite_multi_lineage(monkeypatch):
    classifier_classes = ["HSC", "BP", "EryP", "MasP", "MkP", "MoP", "NeuP"]
    captured = {}

    monkeypatch.setattr(
        cite_multi_eval.maizels,
        "load_classifier",
        lambda path: (object(), classifier_classes, np.zeros(100), np.ones(100)),
    )

    def fake_check(**kwargs):
        captured.update(kwargs)
        return {
            "valid": np.asarray([True, False]),
            "n_confident": 4,
            "n_points": 4,
        }

    monkeypatch.setattr(
        cite_multi_eval.maizels, "check_paths_with_classifier", fake_check
    )
    result = cite_multi_eval.score_paths_with_all_days_classifier(
        np.zeros((2, 2, 100), dtype=np.float32),
        np.asarray(["HSC", "BP"]),
        classifier_path="classifier.pt",
        prob_threshold=0.0,
        margin_threshold=0.0,
        classifier_batch_size=32,
        lineage_transition_mode="descendant",
    )

    np.testing.assert_array_equal(captured["start_type_ids"], [0, 1])
    assert captured["transition_edges"] == cite_multi.TRANSITION_EDGES
    assert result["valid"].tolist() == [True, False]


def test_pt_classifier_loader_infers_pca_dimension(tmp_path):
    import torch

    path = tmp_path / "classifier.pt"
    model = maizels._make_celltype_mlp(3, 2)
    torch.save(
        {
            "metadata": {
                "class_names": ["A", "B"],
                "scaler_mean": np.zeros(3, dtype=np.float32),
                "scaler_scale": np.ones(3, dtype=np.float32),
            },
            "model_state_dict": model.state_dict(),
        },
        path,
    )

    loaded, classes, mean, scale = maizels.load_classifier(path)
    assert classes == ["A", "B"]
    assert loaded.net[0].in_features == 3
    assert mean.shape == scale.shape == (3,)


def test_classifier_metrics_include_shared_final_comparison_namespace():
    metrics = cite_multi_eval.classifier_metric_values(
        {
            "valid": np.asarray([True, False, False, True]),
            "n_confident": 6,
            "n_points": 8,
        },
        source_count=4,
    )
    assert metrics["cite_multi/eval_source_count"] == 4
    assert (
        metrics["cite_multi/full_data_classifier/model_euler_invalid_trajectory_pct"]
        == 50.0
    )
    assert (
        metrics["maizels/full_data_classifier/model_euler_invalid_trajectory_pct"]
        == 50.0
    )
    assert metrics["cite_multi/full_data_classifier/confident_point_pct"] == 75.0


def test_temporal_data_split_preserves_cell_type_alignment(monkeypatch):
    import torch

    days = np.repeat(np.asarray(["2", "3", "4", "7"]), 6)
    cell_types = np.asarray([cite_multi.CLASS_NAMES[ii % 7] for ii in range(24)])
    x = np.zeros((24, 100), dtype=np.float32)
    x[:, 0] = np.arange(24)

    def fake_load(path, max_dim, *, return_cell_types):
        assert return_cell_types
        return x, days, np.asarray(["2", "3", "4", "7"]), cell_types

    monkeypatch.setattr(trajectory_data, "custom_load_dataset", fake_load)
    torch.manual_seed(11)
    datamodule = trajectory_data.TemporalDataModule(
        _eval_args(
            data_type="scrna",
            data_path="unused.h5ad",
            batch_size=2,
            split_ratios=[0.5, 0.5],
        )
    )

    expected = dict(zip(x[:, 0], cell_types))
    for pool in datamodule.timepoint_splits.values():
        assert pool["types"].tolist() == [expected[value] for value in pool["x"][:, 0]]
        assert pool["holdout_types"].tolist() == [
            expected[value] for value in pool["holdout_x"][:, 0]
        ]


def test_flow_callbacks_include_cite_multi_evaluator(tmp_path):
    classifier_path = tmp_path / "all_days.pt"
    classifier_path.touch()
    args = _eval_args(
        data_type="scrna",
        working_dir=str(tmp_path),
        patience=2,
        cite_multi_eval_classifier_path=str(classifier_path),
    )
    callbacks = create_callbacks(
        args,
        phase="flow",
        data_type="scrna",
        run_id="test-run",
        datamodule=Namespace(),
    )
    assert any(
        isinstance(callback, cite_multi_eval.CiteMultiEvaluationCallback)
        for callback in callbacks
    )


def test_callback_logs_final_all_days_lineage_metrics(monkeypatch, tmp_path):
    import torch

    classifier_path = tmp_path / "all_days.pt"
    classifier_path.touch()
    source_x = np.arange(400, dtype=np.float32).reshape(4, 100)
    datamodule = Namespace(
        timepoint_splits={
            "2": {
                "holdout_x": source_x,
                "holdout_types": np.asarray(["HSC", "BP", "EryP", "NeuP"]),
            },
            "7": {
                "holdout_x": source_x + 1.0,
                "holdout_types": np.asarray(["HSC", "BP", "EryP", "NeuP"]),
            },
        }
    )
    args = _eval_args(
        cite_multi_eval_classifier_path=str(classifier_path),
        cite_multi_eval_check_times=3,
        seed_current=42,
    )
    callback = cite_multi_eval.CiteMultiEvaluationCallback(args, datamodule)

    captured_paths = {}

    def fake_score(paths, source_types, **kwargs):
        captured_paths["shape"] = paths.shape
        captured_paths["edges"] = cite_multi.TRANSITION_EDGES
        return {
            "valid": np.asarray([True, False, True, False]),
            "n_confident": 12,
            "n_points": 12,
        }

    logged = {}
    monkeypatch.setattr(
        cite_multi_eval, "score_paths_with_all_days_classifier", fake_score
    )
    monkeypatch.setattr(cite_multi_eval.wandb, "Image", lambda figure: "image")
    monkeypatch.setattr(
        cite_multi_eval.wandb, "log", lambda metrics: logged.update(metrics)
    )

    class ZeroFlow(torch.nn.Module):
        def forward(self, time, x):
            return torch.zeros_like(x)

    trainer = Namespace(
        global_step=100,
        is_global_zero=True,
        checkpoint_callback=Namespace(best_model_score=torch.tensor(0.25)),
    )
    module = Namespace(flow_net=ZeroFlow(), device=torch.device("cpu"))
    callback._evaluate(trainer, module, force=True, final_best=True)

    assert captured_paths["shape"] == (4, 3, 100)
    assert logged["final_eval/euler_invalid_trajectory_pct"] == 50.0
    assert (
        logged["final_eval/full_data_classifier/euler_invalid_trajectory_pct"] == 50.0
    )
    assert logged["final_eval/best_validation_loss"] == 0.25


def test_blockwise_rbf_mmd_matches_dense_maizels_formula():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 4))
    y = rng.normal(size=(5, 4))

    bandwidth = maizels_eval._median_bandwidth(x, y, np.random.default_rng(23))
    bandwidths = bandwidth * np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    xx = maizels_eval._sqdist(x, x)
    yy = maizels_eval._sqdist(y, y)
    xy = maizels_eval._sqdist(x, y)
    expected = np.mean(
        [
            np.exp(-xx / (2.0 * bw * bw)).mean()
            + np.exp(-yy / (2.0 * bw * bw)).mean()
            - 2.0 * np.exp(-xy / (2.0 * bw * bw)).mean()
            for bw in bandwidths
        ]
    )

    actual = maizels_eval.rbf_mmd2(x, y, np.random.default_rng(23), block_size=3)
    assert actual == pytest.approx(max(float(expected), 0.0), abs=1e-12)


def test_cite_multi_test_step_logs_emd_and_rbf_mmd(monkeypatch):
    import torch

    class FakeNeuralODE:
        def __init__(self, *args, **kwargs):
            pass

        def trajectory(self, x, t_span):
            return torch.stack([x for _ in t_span])

    monkeypatch.setattr(flow_net_train, "NeuralODE", FakeNeuralODE)
    monkeypatch.setattr(
        flow_net_train, "wasserstein_distance", lambda *args, **kwargs: 1.5
    )
    monkeypatch.setattr(flow_net_train, "rbf_mmd2", lambda *args, **kwargs: 0.25)

    args = Namespace(
        flow_optimizer="adam",
        flow_lr=1e-3,
        flow_weight_decay=0.0,
        whiten=False,
        working_dir=".",
        data_type="scrna",
        seed_current=42,
    )
    model = flow_net_train.FlowNetTrainTrajectory(
        flow_matcher=Namespace(),
        flow_net=torch.nn.Linear(100, 100),
        skipped_time_points=[1],
        args=args,
    )
    model.timesteps = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    model._trainer = Namespace(
        datamodule=Namespace(
            data_type="scrna",
            data_name="cite",
            timepoint_splits={key: {} for key in ("2", "3", "4", "7")},
        )
    )
    logged = {}
    monkeypatch.setattr(
        model,
        "log",
        lambda name, value, **kwargs: logged.update({name: float(value)}),
    )

    batch = [torch.zeros((2, 100)) for _ in range(4)]
    model.test_step(batch, 0)

    assert logged["test_EMD"] == 1.5
    assert logged["mfm/test_EMD"] == 1.5
    assert logged["mfm/test_rbf_MMD2"] == 0.25
    assert logged["distribution_eval/3_euler_rbf_mmd2"] == 0.25
    assert logged["distribution_eval/euler_rbf_mmd2_mean"] == 0.25
    assert logged["final_eval/euler_mean_rbf_mmd2"] == 0.25

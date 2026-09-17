from pathlib import Path

import numpy as np
import pytest

from common import cite_multi
from common import cite_multi_stochastic_eval
from configs import cite_multi_stochastic
from launchers import cite_multi_stochastic as cite_multi_stochastic_launcher
from launchers import maizels_stochastic as shared_launcher
from scripts import run_cite_multi_stochastic_multiseed


def test_config_exposes_all_six_ssfm_variants_for_both_datasets():
    expected = (
        ("none", False, False),
        ("endpoint", False, False),
        ("endpoint", True, False),
        ("ot_endpoint", False, True),
        ("ot_endpoint", True, True),
        ("ot_plain", False, True),
    )
    for dataset_name in ("cite", "multi"):
        for slurm_id, (pair_mode, constrained, minibatch_ot) in enumerate(expected):
            cfg = cite_multi_stochastic.get_config(
                slurm_id,
                dataset_name=dataset_name,
                heldout_day="4",
                total_steps=20,
                batch_size=8,
                n_pairs=24,
            )
            assert cfg.problem.maizels_pair_mode == pair_mode
            assert cfg.problem.pair_mode == pair_mode
            assert cfg.problem.n_interpolant_check_times == 0
            assert bool(cfg.constraints.enabled) is constrained
            assert cite_multi.uses_minibatch_ot_config(cfg) is minibatch_ot
            assert cfg.problem.d == 100
            assert cfg.evaluation.max_source_points == 0
            assert cfg.evaluation.max_target_points == 0
            assert cfg.evaluation.flowmap_n_steps == 100
            assert cfg.optimization.early_stopping.check_freq == 100
            assert cfg.optimization.early_stopping.patience == 10


def test_config_uses_dataset_specific_training_and_full_data_classifiers():
    cfg = cite_multi_stochastic.get_config(
        2, dataset_name="multi", heldout_day="3"
    )
    assert cfg.problem.classifier_path.endswith(
        "multi-classifiers/celltype_classifier_multi_pca100_except_day3.pt"
    )
    assert cfg.logging.maizels.full_data_classifier_path.endswith(
        "multi-classifiers/celltype_classifier_multi_pca100_all_days.pt"
    )
    assert cfg.constraints.path_mode == "direct_offdiagonal_endpoint_nll"
    assert cfg.constraints.path_n_times == 2


def test_irrelevant_constraint_overrides_are_rejected():
    with pytest.raises(ValueError, match="constrained stochastic variant"):
        cite_multi_stochastic.get_config(0, constraint_weight=1.0)
    with pytest.raises(ValueError, match="constrained stochastic variant"):
        cite_multi_stochastic.get_config(3, entropy_weight=0.1)


def test_launcher_builds_requested_dataset_and_runtime_overrides():
    args = cite_multi_stochastic_launcher.parse_args(
        [
            "--slurm_id",
            "4",
            "--dataset_name",
            "multi",
            "--heldout_day",
            "3",
            "--visual_frequency",
            "250",
            "--eval_noise_draws",
            "2",
            "--lineage_eval_max_points",
            "64",
        ]
    )
    cfg = cite_multi_stochastic_launcher._build_config(args)
    assert cfg.problem.dataset_name == "multi"
    assert cfg.problem.heldout_timepoint == "3"
    assert cfg.problem.maizels_pair_mode == "ot_endpoint"
    assert cfg.constraints.enabled
    assert cfg.logging.visual_freq == 250
    assert cfg.evaluation.n_noise_draws == 2
    assert cfg.evaluation.lineage_max_source_points == 64


def test_multiseed_runner_parses_axes_and_forwards_relevant_options(tmp_path):
    assert run_cite_multi_stochastic_multiseed.parse_datasets("all") == (
        "cite",
        "multi",
    )
    assert run_cite_multi_stochastic_multiseed.parse_heldout_days("4,3") == (
        "4",
        "3",
    )
    args = run_cite_multi_stochastic_multiseed.parse_args(
        [
            "--dataset-location",
            "/tmp/data",
            "--constraint-weight",
            "12",
            "--entropy-weight",
            "0.03",
            "--lineage-eval-max-points",
            "64",
        ]
    )
    common = {
        "args": args,
        "dataset_name": "multi",
        "heldout_day": "3",
        "slurm_id": 4,
        "seed": 7,
        "output_folder": tmp_path / "checkpoints",
        "metrics_path": tmp_path / "metrics.json",
    }
    unconstrained = run_cite_multi_stochastic_multiseed.build_command(
        constrained=False, **common
    )
    constrained = run_cite_multi_stochastic_multiseed.build_command(
        constrained=True, **common
    )
    assert unconstrained[unconstrained.index("--dataset_name") + 1] == "multi"
    assert unconstrained[unconstrained.index("--heldout_day") + 1] == "3"
    assert "--constraint_weight" not in unconstrained
    assert "--entropy_weight" not in unconstrained
    assert constrained[constrained.index("--constraint_weight") + 1] == "12.0"
    assert constrained[constrained.index("--entropy_weight") + 1] == "0.03"
    assert constrained[constrained.index("--lineage_eval_max_points") + 1] == "64"


def test_shared_launcher_uses_cite_multi_minibatch_ot_backend(monkeypatch):
    cfg = cite_multi_stochastic.get_config(
        3, total_steps=10, batch_size=8, n_pairs=20
    )
    expected = {
        "x0": np.zeros((8, 3), dtype=np.float32),
        "x1": np.ones((8, 3), dtype=np.float32),
        "label": np.zeros((8, 4), dtype=np.float32),
    }
    calls = []

    def fake_couple(cfg, pools, n_pairs, *, seed, pair_mode):
        calls.append((pools, n_pairs, seed, pair_mode))
        return expected, {"coupling": "dynamic_minibatch_ot"}

    monkeypatch.setattr(
        cite_multi, "couple_minibatch_ot_timepoint_pools", fake_couple
    )
    pools = {"timepoints": {}, "intervals": ()}
    batch = shared_launcher._sample_batch(
        pools,
        np.random.default_rng(7),
        8,
        cfg,
        cite_multi,
    )

    assert len(calls) == 1
    assert calls[0][0] is pools
    assert calls[0][1] == 8
    assert calls[0][3] == "ot_endpoint"
    np.testing.assert_allclose(np.asarray(batch["x1"]), 1.0)


def test_distribution_eval_uses_preceding_day_full_populations_and_two_samplers(
    monkeypatch,
):
    cfg = cite_multi_stochastic.get_config(
        0, heldout_day="4", total_steps=10, batch_size=8, n_pairs=20
    )
    pools = {
        "3": {"x": np.zeros((3, 2), dtype=np.float32)},
        "4": {"x": np.full((5, 2), 4.0, dtype=np.float32)},
    }
    emd_shapes = []
    monkeypatch.setattr(
        cite_multi_stochastic_eval.cite_multi,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    monkeypatch.setattr(
        cite_multi_stochastic_eval.shared_eval,
        "sample_pushforward",
        lambda model, params, x, *args, **kwargs: x + 1.0,
    )
    monkeypatch.setattr(
        cite_multi_stochastic_eval.shared_eval,
        "sample_composed_pushforward",
        lambda model, params, x, *args, **kwargs: x + 2.0,
    )

    def fake_emd(prediction, actual):
        emd_shapes.append((prediction.shape[0], actual.shape[0]))
        return float(np.mean(prediction))

    monkeypatch.setattr(
        cite_multi_stochastic_eval.wasserstein, "exact_emd", fake_emd
    )
    monkeypatch.setattr(
        cite_multi_stochastic_eval.shared_eval,
        "rbf_mmd2",
        lambda *args, **kwargs: 0.0,
    )

    metrics, plot_data = cite_multi_stochastic_eval.distribution_metrics(
        object(), {}, cfg, n_noise_draws=1
    )

    assert emd_shapes == [(3, 5), (3, 5)]
    assert metrics["mfm/test_EMD"] == 1.0
    assert metrics["mfm/test_EMD_direct_ssfm"] == 1.0
    assert metrics["mfm/test_EMD_flowmap"] == 2.0
    assert metrics["final_eval/day4_direct_emd"] == 1.0
    assert metrics["final_eval/day4_flowmap_emd"] == 2.0
    assert [values.shape[0] for values in plot_data["4"]] == [5, 3, 3]


def test_lineage_eval_uses_heldout_day2_cells_and_both_classifiers(
    monkeypatch, tmp_path
):
    cfg = cite_multi_stochastic.get_config(
        0, total_steps=10, batch_size=8, n_pairs=20
    )
    schedule_pt = tmp_path / "three_day.pt"
    full_pt = tmp_path / "all_days.pt"
    schedule_pt.with_suffix(".npz").touch()
    full_pt.with_suffix(".npz").touch()
    cfg.problem.classifier_path = str(schedule_pt)
    cfg.logging.maizels.full_data_classifier_path = str(full_pt)
    cfg.evaluation.n_noise_draws = 1
    cfg.evaluation.lineage_max_source_points = 2
    pools = {
        "2": {
            "holdout_x": np.asarray(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
            ),
            "holdout_types": np.full(3, "HSC", dtype=object),
        },
        "7": {"holdout_x": np.asarray([[8.0, 8.0]], dtype=np.float32)},
    }
    seen_classifiers = []
    monkeypatch.setattr(
        cite_multi_stochastic_eval.cite_multi,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )

    def fake_path(model, params, x, *args, **kwargs):
        assert x.shape == (2, 2)
        assert kwargs["n_steps"] == 100
        return np.repeat(x[:, None, :], 100, axis=1)

    monkeypatch.setattr(
        cite_multi_stochastic_eval.shared_eval, "sample_composed_path", fake_path
    )

    def fake_check(paths, start_type_ids, classifier_path, **kwargs):
        del paths, start_type_ids, kwargs
        seen_classifiers.append(Path(classifier_path).name)
        valid = (
            np.asarray([True, True])
            if Path(classifier_path).name == "all_days.npz"
            else np.asarray([True, False])
        )
        return {"valid": valid}

    monkeypatch.setattr(
        cite_multi_stochastic_eval.cite_multi,
        "check_paths_with_classifier",
        fake_check,
    )

    metrics = cite_multi_stochastic_eval.lineage_metrics(object(), {}, cfg)
    assert seen_classifiers == ["three_day.npz", "all_days.npz"]
    assert metrics["final_eval/lineage_eval_source_count"] == 2.0
    assert metrics["final_eval/flowmap_valid_trajectory_pct"] == 50.0
    assert (
        metrics["final_eval/full_data_classifier/stochastic_path_valid_fraction"]
        == 1.0
    )

    seen_classifiers.clear()
    plot_data = cite_multi_stochastic_eval.full_data_trajectory_plot_data(
        object(), {}, cfg
    )
    assert seen_classifiers == ["all_days.npz"]
    assert plot_data["paths"].shape == (2, 101, 2)
    np.testing.assert_array_equal(plot_data["valid"], [True, True])

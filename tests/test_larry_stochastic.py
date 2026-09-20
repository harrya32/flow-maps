from pathlib import Path

import numpy as np
import pytest

from common import larry
from common import larry_stochastic_eval
from configs import larry_pca50
from configs import larry_spring2d_stochastic
from launchers import larry_stochastic as larry_stochastic_launcher
from launchers import maizels_stochastic as shared_launcher


@pytest.fixture(autouse=True)
def available_larry_classifiers(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )
    monkeypatch.setattr(
        larry_spring2d_stochastic,
        "_checkpoint_exists",
        lambda path: True,
    )


def test_config_exposes_all_seven_larry_spring_ssfm_variants():
    expected = (
        ("none", False, False),
        ("endpoint", False, False),
        ("endpoint", True, False),
        ("ot_endpoint", False, True),
        ("ot_endpoint", True, True),
        ("ot_plain", False, True),
        ("ot_endpoint", True, True),
    )
    for slurm_id, (pair_mode, constrained, minibatch_ot) in enumerate(expected):
        cfg = larry_spring2d_stochastic.get_config(
            slurm_id,
            total_steps=20,
            batch_size=8,
            n_pairs=24,
        )
        assert cfg.problem.target == "larry_spring2d"
        assert cfg.problem.d == 2
        assert cfg.problem.maizels_pair_mode == pair_mode
        assert cfg.problem.pair_mode == pair_mode
        assert cfg.problem.n_interpolant_check_times == 0
        assert bool(cfg.constraints.enabled) is constrained
        assert larry.uses_minibatch_ot(cfg) is minibatch_ot
        assert cfg.problem.ot_minibatch_size == 32
        assert cfg.evaluation.max_source_points == 1024
        assert cfg.evaluation.max_target_points == 1024
        assert cfg.evaluation.flowmap_n_steps == 50
        assert cfg.evaluation.clone_target_times == ["D4", "D6"]
        assert cfg.evaluation.clone_samples_per_source == 32
        assert cfg.evaluation.clone_n_noise_draws == 3
        expected_path_mode = (
            "stochastic_rollout_endpoint_nll"
            if slurm_id == 6
            else "direct_offdiagonal_endpoint_nll"
        )
        if constrained:
            assert cfg.constraints.path_mode == expected_path_mode
        assert cfg.ssfm.local_fraction == (1.0 if slurm_id == 6 else 0.75)


def test_config_uses_training_classifier_for_constraints_and_all_days_for_eval():
    cfg = larry_spring2d_stochastic.get_config(2)

    assert Path(cfg.problem.classifier_path).name == (
        "celltype_classifier_larry_spring2d_train_days_d2_d6.pt"
    )
    assert Path(cfg.logging.maizels.full_data_classifier_path).name == (
        "celltype_classifier_larry_spring2d_all_days.pt"
    )
    assert cfg.problem.flow_training_requires_classifier


def test_launcher_applies_clone_and_runtime_overrides():
    args = larry_stochastic_launcher.parse_args(
        [
            "--slurm_id",
            "4",
            "--total_steps",
            "20",
            "--batch_size",
            "8",
            "--n_pairs",
            "24",
            "--ot_minibatch_size",
            "4",
            "--clone_samples_per_source",
            "7",
            "--clone_noise_draws",
            "2",
            "--clone_target_times",
            "D4",
            "--clone_eval_batch_size",
            "64",
            "--eval_flowmap_steps",
            "9",
        ]
    )
    cfg = larry_stochastic_launcher._build_config(args)

    assert cfg.problem.maizels_pair_mode == "ot_endpoint"
    assert cfg.problem.ot_minibatch_size == 4
    assert cfg.evaluation.clone_samples_per_source == 7
    assert cfg.evaluation.clone_n_noise_draws == 2
    assert cfg.evaluation.clone_target_times == ["D4"]
    assert cfg.evaluation.clone_batch_size == 64
    assert cfg.evaluation.flowmap_n_steps == 9
    assert cfg.evaluation.lineage_n_steps == 9
    assert cfg.evaluation.clone_flowmap_n_steps == 9


def test_shared_launcher_uses_larry_minibatch_ot_backend(monkeypatch):
    cfg = larry_spring2d_stochastic.get_config(
        3, total_steps=10, batch_size=8, n_pairs=20
    )
    expected = {
        "x0": np.zeros((8, 2), dtype=np.float32),
        "x1": np.ones((8, 2), dtype=np.float32),
        "label": np.zeros((8, 4), dtype=np.float32),
    }
    calls = []

    def fake_couple(cfg, pools, n_pairs, *, seed, pair_mode):
        calls.append((pools, n_pairs, seed, pair_mode))
        return expected, {"coupling": "dynamic_minibatch_ot"}

    monkeypatch.setattr(larry, "couple_minibatch_ot_timepoint_pools", fake_couple)
    pools = {"timepoints": {}, "intervals": ()}
    batch = shared_launcher._sample_batch(
        pools,
        np.random.default_rng(7),
        8,
        cfg,
        larry,
    )

    assert len(calls) == 1
    assert calls[0][0] is pools
    assert calls[0][1] == 8
    assert calls[0][3] == "ot_endpoint"
    np.testing.assert_allclose(np.asarray(batch["x1"]), 1.0)


def test_population_eval_uses_only_composed_stochastic_maps(monkeypatch):
    cfg = larry_spring2d_stochastic.get_config(
        0, total_steps=10, batch_size=8, n_pairs=20
    )
    cfg.evaluation.n_noise_draws = 2
    pools = {
        "D2": {"x": np.zeros((3, 2), dtype=np.float32)},
        "D4": {"x": np.full((5, 2), 4.0, dtype=np.float32)},
    }
    calls = []
    monkeypatch.setattr(
        larry_stochastic_eval.larry,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )

    def fake_composed(model, params, source, *args, **kwargs):
        calls.append((source.shape, kwargs["n_steps"]))
        return source + 2.0

    monkeypatch.setattr(
        larry_stochastic_eval.shared_eval,
        "sample_composed_pushforward",
        fake_composed,
    )
    monkeypatch.setattr(
        larry_stochastic_eval.wasserstein,
        "exact_emd",
        lambda prediction, target: float(np.mean(prediction)),
    )
    monkeypatch.setattr(
        larry_stochastic_eval.shared_eval,
        "rbf_mmd2",
        lambda *args, **kwargs: 0.0,
    )

    metrics, plot_data = larry_stochastic_eval.distribution_metrics(object(), {}, cfg)

    assert calls == [((3, 2), 50), ((3, 2), 50)]
    assert metrics["mfm/test_EMD"] == 2.0
    assert metrics["final_eval/d2_to_d4_flowmap_emd"] == 2.0
    assert plot_data["D4"][0].shape == (5, 2)
    assert plot_data["D4"][1].shape == (3, 2)


def test_local_only_population_eval_uses_euler_maruyama(monkeypatch):
    cfg = larry_spring2d_stochastic.get_config(
        6, total_steps=10, batch_size=8, n_pairs=20
    )
    cfg.evaluation.n_noise_draws = 1
    pools = {
        "D2": {"x": np.zeros((3, 2), dtype=np.float32)},
        "D4": {"x": np.full((5, 2), 4.0, dtype=np.float32)},
    }
    monkeypatch.setattr(
        larry_stochastic_eval.larry,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    monkeypatch.setattr(
        larry_stochastic_eval.shared_eval,
        "sample_composed_pushforward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local-only evaluation composed a flow map")
        ),
    )
    seen = []

    def fake_em(model, params, source, *args, **kwargs):
        seen.append(kwargs["n_steps"])
        return source + 3.0

    monkeypatch.setattr(
        larry_stochastic_eval.shared_eval,
        "sample_euler_maruyama_pushforward",
        fake_em,
    )
    monkeypatch.setattr(
        larry_stochastic_eval.wasserstein,
        "exact_emd",
        lambda prediction, target: float(np.mean(prediction)),
    )
    monkeypatch.setattr(
        larry_stochastic_eval.shared_eval,
        "rbf_mmd2",
        lambda *args, **kwargs: 0.0,
    )

    metrics, _ = larry_stochastic_eval.distribution_metrics(object(), {}, cfg)

    assert seen == [50]
    assert metrics["mfm/test_EMD_euler_maruyama"] == 3.0
    assert metrics["final_eval/d2_to_d4_euler_maruyama_emd"] == 3.0
    assert "final_eval/d2_to_d4_flowmap_emd" not in metrics


def test_clone_wasserstein_repeats_each_source_and_uses_original_cell_weights(
    monkeypatch,
):
    cfg = larry_spring2d_stochastic.get_config(
        0,
        total_steps=10,
        batch_size=8,
        n_pairs=20,
        clone_samples_per_source=3,
        clone_noise_draws=2,
    )
    cfg.evaluation.clone_target_times = ["D4"]
    data = {
        "x": np.asarray(
            [[0.0], [2.0], [10.0], [1.0], [3.0], [12.0], [14.0]],
            dtype=np.float32,
        ),
        "timepoints": np.asarray(
            ["D2", "D2", "D2", "D4", "D4", "D4", "D4"], dtype=object
        ),
        "clone_ids": np.asarray([0, 0, 1, 0, 0, 1, 1], dtype=np.int64),
    }
    seen = []
    monkeypatch.setattr(
        larry_stochastic_eval.larry,
        "all_timepoint_data",
        lambda *args, **kwargs: data,
    )

    def identity_samples(model, params, source, *args, **kwargs):
        seen.append((source.copy(), kwargs["n_steps"], kwargs["batch_size"]))
        return source

    monkeypatch.setattr(
        larry_stochastic_eval,
        "_composed_samples_in_batches",
        identity_samples,
    )
    monkeypatch.setattr(
        larry_stochastic_eval.wasserstein,
        "exact_emd",
        lambda prediction, target: abs(
            float(np.mean(prediction)) - float(np.mean(target))
        ),
    )

    metrics = larry_stochastic_eval.clone_wasserstein_metrics(object(), {}, cfg)

    assert len(seen) == 2
    assert seen[0][0].shape == (9, 1)
    assert seen[0][1] == 50
    assert metrics["final_eval/clone_d2_to_d4_eligible_clone_count"] == 2
    assert metrics["final_eval/clone_d2_to_d4_source_cell_count"] == 3
    assert metrics["final_eval/clone_d2_to_d4_target_cell_count"] == 4
    assert metrics["final_eval/clone_d2_to_d4_generated_sample_count_per_draw"] == 9
    assert metrics["final_eval/clone_d2_to_d4_samples_per_source"] == 3
    assert metrics["final_eval/clone_d2_to_d4_noise_draw_count"] == 2
    assert metrics["final_eval/clone_d2_to_d4_wasserstein_macro"] == pytest.approx(2.0)
    assert metrics[
        "final_eval/clone_d2_to_d4_wasserstein_source_weighted"
    ] == pytest.approx(5.0 / 3.0)
    assert metrics[
        "final_eval/clone_d2_to_d4_wasserstein_target_weighted"
    ] == pytest.approx(2.0)
    assert metrics[
        "final_eval/clone_d2_to_d4_wasserstein_macro_std_over_noise"
    ] == pytest.approx(0.0)


def test_local_only_clone_eval_uses_euler_maruyama_batches(monkeypatch):
    cfg = larry_spring2d_stochastic.get_config(
        6,
        total_steps=10,
        batch_size=8,
        n_pairs=20,
        clone_samples_per_source=2,
        clone_noise_draws=1,
    )
    cfg.evaluation.clone_target_times = ["D4"]
    data = {
        "x": np.asarray([[0.0], [2.0], [1.0], [3.0]], dtype=np.float32),
        "timepoints": np.asarray(["D2", "D2", "D4", "D4"], dtype=object),
        "clone_ids": np.asarray([0, 0, 0, 0], dtype=np.int64),
    }
    monkeypatch.setattr(
        larry_stochastic_eval.larry,
        "all_timepoint_data",
        lambda *args, **kwargs: data,
    )
    monkeypatch.setattr(
        larry_stochastic_eval,
        "_composed_samples_in_batches",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("local-only clone evaluation composed a flow map")
        ),
    )
    seen = []

    def fake_em(model, params, source, *args, **kwargs):
        seen.append(kwargs["n_steps"])
        return source

    monkeypatch.setattr(
        larry_stochastic_eval, "_euler_maruyama_samples_in_batches", fake_em
    )
    monkeypatch.setattr(
        larry_stochastic_eval.wasserstein,
        "exact_emd",
        lambda prediction, target: abs(
            float(np.mean(prediction)) - float(np.mean(target))
        ),
    )

    metrics = larry_stochastic_eval.clone_wasserstein_metrics(object(), {}, cfg)

    assert seen == [50]
    assert metrics["final_eval/clone_d2_to_d4_available"] == 1.0

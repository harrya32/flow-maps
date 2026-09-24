from pathlib import Path

import numpy as np
import pytest

from common import larry
from common import larry_stochastic_eval
from configs import larry_pca50
from configs import larry_pca50_stochastic
from configs import larry_spring2d_stochastic
from launchers import larry_stochastic as larry_stochastic_launcher
from launchers import maizels_stochastic as shared_launcher
from scripts import run_larry_stochastic_multiseed


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


def test_config_exposes_all_nine_larry_spring_ssfm_variants():
    expected = (
        ("none", False, False),
        ("endpoint", False, False),
        ("endpoint", True, False),
        ("ot_endpoint", False, True),
        ("ot_endpoint", True, True),
        ("ot_plain", False, True),
        ("ot_endpoint", True, True),
        ("ot_endpoint_interpolant", False, True),
        ("ot_endpoint_interpolant", True, True),
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
        assert cfg.problem.n_interpolant_check_times == (
            50 if pair_mode == "ot_endpoint_interpolant" else 0
        )
        assert bool(cfg.constraints.enabled) is constrained
        assert larry.uses_minibatch_ot(cfg) is minibatch_ot
        assert cfg.problem.ot_minibatch_size == 32
        assert cfg.evaluation.max_source_points == 1024
        assert cfg.evaluation.max_target_points == 1024
        assert cfg.evaluation.flowmap_n_steps == 50
        assert cfg.evaluation.clone_target_times == ["D4", "D6"]
        assert cfg.evaluation.clone_samples_per_source == 32
        assert cfg.evaluation.clone_n_noise_draws == 3
        assert cfg.evaluation.clone_min_source_cells == 1
        assert cfg.evaluation.clone_min_target_cells == 10
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


def test_pca50_stochastic_config_only_changes_larry_representation():
    spring = larry_spring2d_stochastic.get_config(
        4, total_steps=20, batch_size=8, n_pairs=24
    )
    pca = larry_pca50_stochastic.get_config(4, total_steps=20, batch_size=8, n_pairs=24)

    assert pca.problem.target == "larry_pca50"
    assert pca.problem.larry_representation == "pca50"
    assert pca.problem.larry_representation_key == "X_pca"
    assert pca.problem.d == 50
    assert tuple(pca.network.input_dims) == (50,)
    assert pca.network.output_dim == 50
    assert Path(pca.problem.dataset_location).name == (
        "stateFate_inVitro_hvg2000_pca50.h5ad"
    )
    assert Path(pca.problem.classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca50_train_days_d2_d6.pt"
    )
    assert Path(pca.logging.maizels.full_data_classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca50_all_days.pt"
    )
    assert pca.logging.output_name.startswith("larry_pca50_holdout_d4_")
    assert not pca.evaluation.evaluate_instantaneous_and_ema

    assert pca.problem.maizels_pair_mode == spring.problem.maizels_pair_mode
    assert pca.optimization.to_dict() == spring.optimization.to_dict()
    assert pca.ssfm.to_dict() == spring.ssfm.to_dict()
    assert pca.constraints.to_dict() == spring.constraints.to_dict()


def test_pca50_stochastic_config_supports_clone_labelled_subset(tmp_path):
    cfg = larry_pca50_stochastic.get_config(
        4,
        dataset_location=str(tmp_path),
        total_steps=20,
        batch_size=8,
        n_pairs=24,
        larry_clone_labelled_only=True,
    )

    assert cfg.problem.larry_clone_labelled_only
    assert Path(cfg.problem.dataset_location).name == (
        "stateFate_inVitro_clone_labelled_hvg2000_pca50.h5ad"
    )
    assert "clone_labelled" in cfg.logging.output_name


def test_pca_stochastic_config_supports_arbitrary_dimension(tmp_path):
    cfg = larry_pca50_stochastic.get_config(
        4,
        dataset_location=str(tmp_path),
        total_steps=20,
        batch_size=8,
        n_pairs=24,
        larry_n_pcs=20,
    )

    assert cfg.problem.n_pcs == 20
    assert cfg.problem.d == 20
    assert tuple(cfg.network.input_dims) == (20,)
    assert cfg.network.output_dim == 20
    assert "pca20" in cfg.logging.output_name


def test_final_evaluation_separates_instantaneous_and_ema_metrics(tmp_path):
    cfg = larry_spring2d_stochastic.get_config(
        0, total_steps=20, batch_size=8, n_pairs=24
    )
    cfg.evaluation.evaluate_instantaneous_and_ema = True
    calls = []

    class FakeEvaluationBackend:
        @staticmethod
        def final_evaluation(model, params, cfg, output_dir):
            del model, cfg
            calls.append((params, Path(output_dir)))
            value = float(params)
            return {
                "final_eval/evaluation_mean_emd": value,
                "mfm/test_EMD": value + 1.0,
            }

    metrics = shared_launcher._evaluate_best_parameters(
        object(),
        2.0,
        3.0,
        cfg,
        tmp_path,
        FakeEvaluationBackend,
    )

    assert calls == [
        (2.0, tmp_path / "final_eval"),
        (3.0, tmp_path / "final_eval_EMA"),
    ]
    assert metrics == {
        "final_eval/evaluation_mean_emd": 2.0,
        "mfm/test_EMD": 3.0,
        "final_eval_EMA/evaluation_mean_emd": 3.0,
        "mfm_EMA/test_EMD": 4.0,
    }


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
            "--clone_min_source_cells",
            "3",
            "--clone_min_target_cells",
            "5",
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
    assert cfg.evaluation.clone_min_source_cells == 3
    assert cfg.evaluation.clone_min_target_cells == 5
    assert cfg.evaluation.clone_target_times == ["D4"]
    assert cfg.evaluation.clone_batch_size == 64
    assert cfg.evaluation.flowmap_n_steps == 9
    assert cfg.evaluation.lineage_n_steps == 9
    assert cfg.evaluation.clone_flowmap_n_steps == 9


def test_launcher_propagates_clone_labelled_training_setting(tmp_path):
    args = larry_stochastic_launcher.parse_args(
        [
            "--cfg_path",
            "configs.larry_pca50_stochastic",
            "--slurm_id",
            "0",
            "--dataset_location",
            str(tmp_path),
            "--larry_clone_labelled_only",
            "--larry_n_pcs",
            "20",
            "--total_steps",
            "20",
            "--batch_size",
            "8",
            "--n_pairs",
            "24",
        ]
    )
    cfg = larry_stochastic_launcher._build_config(args)

    assert cfg.problem.larry_clone_labelled_only
    assert cfg.problem.n_pcs == 20


def test_multiseed_runner_builds_selected_larry_setting_commands(tmp_path):
    args = run_larry_stochastic_multiseed.parse_args(
        [
            "--slurm_ids",
            "0,4",
            "--seeds",
            "2,7",
            "--dataset_location",
            str(tmp_path / "data"),
            "--constraint_weight",
            "6",
            "--entropy_weight",
            "0.02",
            "--ot_minibatch_size",
            "12",
            "--clone_samples_per_source",
            "9",
            "--clone_noise_draws",
            "2",
            "--clone_min_source_cells",
            "3",
            "--clone_min_target_cells",
            "5",
            "--clone_target_times",
            "D4,D6",
            "--larry_clone_labelled_only",
            "--larry_n_pcs",
            "20",
            "--python",
            "/env/python",
        ]
    )
    assert run_larry_stochastic_multiseed.parse_slurm_ids("0,4", 9) == (0, 4)

    baseline = run_larry_stochastic_multiseed.build_command(
        args,
        slurm_id=0,
        seed=2,
        constrained=False,
        minibatch_ot=False,
        interpolant_filter=False,
        output_folder=tmp_path / "baseline",
        metrics_path=tmp_path / "baseline.json",
    )
    constrained_ot = run_larry_stochastic_multiseed.build_command(
        args,
        slurm_id=4,
        seed=7,
        constrained=True,
        minibatch_ot=True,
        interpolant_filter=False,
        output_folder=tmp_path / "constrained",
        metrics_path=tmp_path / "constrained.json",
    )

    assert baseline[0] == "/env/python"
    assert baseline[1].endswith("py/launchers/larry_stochastic.py")
    assert baseline[baseline.index("--slurm_id") + 1] == "0"
    assert baseline[baseline.index("--seed") + 1] == "2"
    assert "--constraint_weight" not in baseline
    assert "--entropy_weight" not in baseline
    assert "--ot_minibatch_size" not in baseline
    assert baseline[baseline.index("--clone_samples_per_source") + 1] == "9"
    assert baseline[baseline.index("--clone_min_source_cells") + 1] == "3"
    assert baseline[baseline.index("--clone_min_target_cells") + 1] == "5"
    assert baseline[baseline.index("--clone_target_times") + 1] == "D4,D6"
    assert "--larry_clone_labelled_only" in baseline
    assert baseline[baseline.index("--larry_n_pcs") + 1] == "20"

    assert constrained_ot[constrained_ot.index("--slurm_id") + 1] == "4"
    assert constrained_ot[constrained_ot.index("--seed") + 1] == "7"
    assert constrained_ot[constrained_ot.index("--constraint_weight") + 1] == "6.0"
    assert constrained_ot[constrained_ot.index("--entropy_weight") + 1] == "0.02"
    assert constrained_ot[constrained_ot.index("--ot_minibatch_size") + 1] == "12"


def test_interpolant_filter_override_is_scoped_to_matched_ablation(tmp_path):
    args = run_larry_stochastic_multiseed.parse_args(
        [
            "--interpolant_check_times",
            "12",
            "--ot_minibatch_size",
            "8",
            "--python",
            "/env/python",
        ]
    )
    filtered = run_larry_stochastic_multiseed.build_command(
        args,
        slurm_id=7,
        seed=0,
        constrained=False,
        minibatch_ot=True,
        interpolant_filter=True,
        output_folder=tmp_path / "filtered",
        metrics_path=tmp_path / "filtered.json",
    )
    endpoint_only = run_larry_stochastic_multiseed.build_command(
        args,
        slurm_id=3,
        seed=0,
        constrained=False,
        minibatch_ot=True,
        interpolant_filter=False,
        output_folder=tmp_path / "endpoint",
        metrics_path=tmp_path / "endpoint.json",
    )

    assert filtered[filtered.index("--interpolant_check_times") + 1] == "12"
    assert "--interpolant_check_times" not in endpoint_only
    assert (
        larry_spring2d_stochastic.get_config(
            7, interpolant_check_times=12
        ).problem.n_interpolant_check_times
        == 12
    )
    with pytest.raises(ValueError, match="relevant only"):
        larry_spring2d_stochastic.get_config(3, interpolant_check_times=12)


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
    assert metrics["final_eval/evaluation_mean_emd"] == 2.0
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
    assert metrics["final_eval/evaluation_mean_emd"] == 3.0
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
        clone_min_target_cells=1,
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
        clone_min_target_cells=1,
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

"""Check stochastic default parity, data isolation, filtering and evaluation."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import flax.serialization
import jax
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "tests"))

from common import calderon, calderon_logging, calderon_stochastic_data as data_backend
from common import calderon_stochastic_eval as evaluation, maizels_stochastic_eval as shared_eval
from configs import calderon_stochastic, maizels_stochastic
from launchers import calderon_stochastic as launcher, maizels_stochastic as shared_launcher
from test_calderon_three_time_flow import fixture


@pytest.mark.parametrize("index", range(3))
def test_defaults_match_the_corresponding_maizels_stochastic_variant(index):
    cfg = calderon_stochastic.get_config(index)
    reference = maizels_stochastic.get_config(calderon_stochastic.VARIANTS[index][1], hparam_val_times=())
    for section in ("training", "optimization", "ssfm", "constraints"):
        assert cfg[section].to_dict() == reference[section].to_dict()
    for key, value in reference.evaluation.items():
        if key not in ("max_source_points", "max_target_points"):
            assert cfg.evaluation[key] == value
    for key in ("visual_freq", "scalar_freq", "progress_freq", "save_freq", "wandb_project", "wandb_entity"):
        assert cfg.logging[key] == reference.logging[key]
    for key in ("ot_minibatch_size", "ot_minibatch_max_resamples", "ot_minibatch_infeasible_fallback",
                "classifier_batch_size", "classifier_prob_threshold", "classifier_margin_threshold"):
        assert cfg.problem[key] == reference.problem[key]
    assert cfg.evaluation.max_source_points == cfg.evaluation.max_target_points == 4096
    assert not cfg.evaluation.include_direct
    assert not cfg.evaluation.include_mmd
    assert cfg.problem.retained_timepoints == ["4-6", "8-10", "12-14"]
    assert cfg.problem.evaluation_timepoints == ["6-8", "10-12"]
    assert cfg.problem.n_interpolant_check_times == 0
    assert cfg.classifier.manifest_path == str(calderon.DEFAULT_TRAINING_MANIFEST)
    assert cfg.logging.maizels.full_data_classifier_path == str(calderon.DEFAULT_EVALUATION_MANIFEST)
    assert not cfg.training.conditional


def test_cli_keeps_constraint_and_evaluation_overrides_separate():
    args = launcher.parse_args(["--variant", "constrained", "--constraint-weight", "100", "--seed", "3",
                               "--eval-max-points", "2048", "--eval-noise-draws", "2"])
    cfg = launcher._build_config(args)
    assert cfg.constraints.weight == 100
    assert cfg.training.seed == 3
    assert cfg.evaluation.max_source_points == cfg.evaluation.max_target_points == 2048
    assert cfg.evaluation.n_noise_draws == 2
    assert cfg.logging.maizels.distribution_eval_source_max_points == 2048
    with pytest.raises(ValueError, match="constrained stochastic variant"):
        launcher._build_config(launcher.parse_args(["--variant", "ot", "--constraint-weight", "100"]))
    with pytest.raises(ValueError, match="at least 0"):
        launcher._build_config(launcher.parse_args(["--marginal-cap", "-1"]))


def toy_data(tmp_path, index):
    old, _, _ = fixture(tmp_path)
    cfg = calderon_stochastic.get_config(index, dataset_location=old.problem.dataset_location,
        classifier_path=old.classifier.manifest_path, evaluation_classifier_path=old.classifier.evaluation_manifest_path,
        total_steps=2, batch_size=16)
    cfg.problem.split_manifest_path = old.problem.split_manifest_path
    cfg.classifier.allowed_pairs_path = old.classifier.allowed_pairs_path
    data = calderon.load_endpoint_experiment(cfg)
    calderon_logging.register(cfg, data)
    return cfg, data


@pytest.mark.parametrize("index", range(3))
def test_ot_is_recoupled_with_balanced_intervals_and_disjoint_holdouts(tmp_path, index):
    cfg, data = toy_data(tmp_path, index)
    pools, stats = data_backend.make_minibatch_ot_training_pools(cfg)
    assert stats["stored_endpoint_cells"] == sum(len(data.splits[w]["train"]) for w in data.training_times)
    assert set(pools["timepoints"]) == {"4-6", "8-10", "12-14"}
    rng = np.random.default_rng(10)
    first = shared_launcher._sample_batch(pools, rng, 16, cfg, data_backend)
    second = shared_launcher._sample_batch(pools, rng, 16, cfg, data_backend)
    assert not np.array_equal(first["x0"], second["x0"])
    allowed = calderon.load_allowed_pairs(cfg.classifier.allowed_pairs_path, data.class_names, allow_extra_classes=True)
    for batch in (first, second):
        labels = np.asarray(batch["label"])
        assert labels.shape == (16, 4)
        for source, target in zip(data.training_times[:-1], data.training_times[1:]):
            selected = labels[:, 2] == data.normalized_time(source)
            assert selected.sum() == 8
            assert np.all(labels[selected, 3] == data.normalized_time(target))
            for field, window in (("x0", source), ("x1", target)):
                values = np.asarray(batch[field])[selected, 0]
                assert set(values).issubset(set(data.pool(window, "train")["x"][:, 0]))
                assert not set(values).intersection(data.pool(window, "validation")["x"][:, 0])
        if index:
            assert allowed[labels[:, 0].astype(int), labels[:, 1].astype(int)].all()
    val, val_stats = data_backend.make_validation_pair_pool(cfg, 16)
    for source, target in zip(data.training_times[:-1], data.training_times[1:]):
        selected = val["label"][:, 2] == data.normalized_time(source)
        for field, window in (("x0", source), ("x1", target)):
            assert set(val[field][selected, 0]).issubset(set(data.pool(window, "validation")["x"][:, 0]))
    assert all(s["interpolant_checked"] == 0 for s in val_stats["intervals"])


def test_zero_mass_endpoint_mask_is_not_relaxed(tmp_path):
    cfg, data = toy_data(tmp_path, 1)
    pools, _ = data_backend.make_minibatch_ot_training_pools(cfg)
    # A -> C is absent from the supplied relation; do not add a closure.
    for pool in pools["timepoints"].values():
        pool["type_ids"] = np.zeros(len(pool["x"]), dtype=np.int32)
    pools["timepoints"]["8-10"]["type_ids"][:] = 2
    cfg.problem.ot_minibatch_max_resamples = 2
    with pytest.raises(RuntimeError):
        data_backend.couple_minibatch_ot_timepoint_pools(cfg, pools, 16, seed=1)


def large_pools():
    return {window: {"x": np.full((5000, 100), value, dtype=np.float32)}
            for window, value in (("4-6", 0), ("6-8", 5), ("8-10", 10), ("10-12", 15), ("12-14", 20))}


def test_intermediate_eval_caps_both_populations_and_restarts_from_each_observed_time(monkeypatch):
    cfg = calderon_stochastic.get_config()
    pools, calls, costs = large_pools(), [], []
    monkeypatch.setattr(data_backend, "timepoint_pool_splits", lambda *args, **kwargs: pools)
    def sample(model, params, source, start, end, key, **kwargs):
        calls.append((float(source[0, 0]), start, end, source.shape, kwargs.get("n_steps")))
        return source + 2
    def forbidden(*args, **kwargs):
        pytest.fail("Calderon evaluation must not compute direct samples or MMD.")
    monkeypatch.setattr(shared_eval, "sample_pushforward", forbidden)
    monkeypatch.setattr(shared_eval, "sample_composed_pushforward", sample)
    def emd(prediction, target):
        costs.append((prediction.shape, target.shape))
        return float(prediction.mean())
    monkeypatch.setattr(shared_eval.wasserstein, "exact_emd", emd)
    monkeypatch.setattr(shared_eval, "rbf_mmd2", forbidden)
    metrics, plots = evaluation.distribution_metrics(object(), {}, cfg)
    assert costs == [((4096, 100), (4096, 100))] * 6
    assert {(v, start, end) for v, start, end, _, _ in calls} == {(0., 0., .25), (10., .5, .75)}
    assert all(steps == 50 for _, _, _, _, steps in calls if steps is not None)
    assert metrics["final_eval/flowmap_mean_emd_test_times"] == 7.
    assert metrics["final_eval/ssfm_mean_emd_test_times"] == 7.
    assert not any("direct" in key or "mmd" in key for key in metrics)
    assert set(plots) == {"6-8", "10-12"}
    assert all(len(values) == 2 for values in plots.values())


def test_removing_direct_evaluation_preserves_composed_noise_draws(monkeypatch):
    cfg, pools = calderon_stochastic.get_config(), large_pools()
    seen = []
    monkeypatch.setattr(data_backend, "timepoint_pool_splits", lambda *args, **kwargs: pools)
    monkeypatch.setattr(shared_eval, "sample_pushforward", lambda model, params, source, *args, **kwargs: source)
    def compose(model, params, source, start, end, key, **kwargs):
        seen.append(np.asarray(key).copy())
        return source
    monkeypatch.setattr(shared_eval, "sample_composed_pushforward", compose)
    monkeypatch.setattr(shared_eval.wasserstein, "exact_emd", lambda *args: 1.)
    shared_eval.distribution_metrics(object(), {}, cfg, data_backend=data_backend, include_mmd=False)
    old_keys = np.stack(seen)
    seen.clear()
    evaluation.distribution_metrics(object(), {}, cfg)
    np.testing.assert_array_equal(np.stack(seen), old_keys)


def test_periodic_plots_sample_only_composed_maps_and_label_them(monkeypatch, tmp_path):
    import matplotlib.axes
    cfg, pools, titles = calderon_stochastic.get_config(), large_pools(), []
    monkeypatch.setattr(data_backend, "timepoint_pool_splits", lambda *args, **kwargs: pools)
    def forbidden(*args, **kwargs):
        pytest.fail("Periodic Calderon plots must not sample a direct map or calculate MMD.")
    monkeypatch.setattr(shared_eval, "sample_pushforward", forbidden)
    monkeypatch.setattr(shared_eval, "rbf_mmd2", forbidden)
    monkeypatch.setattr(shared_eval, "sample_composed_pushforward", lambda model, params, source, *args, **kwargs: source + 1)
    original = matplotlib.axes.Axes.set_title
    def title(axis, label, *args, **kwargs):
        titles.append(label)
        return original(axis, label, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", title)
    plots = evaluation.pushforward_plot_data(object(), {}, cfg, max_points=8)
    assert all(len(populations) == 2 for populations in plots.values())
    image_path = tmp_path / "composed_only.png"
    evaluation.save_pushforward_plot(plots, image_path, flowmap_n_steps=50, euler_maruyama=False)
    assert image_path.is_file()
    assert "Composed SSFM (50 steps)" in titles
    assert not any("Direct" in value or "Euler" in value for value in titles)


def test_observed_eval_restarts_real_populations_and_composes_the_full_rollout(monkeypatch):
    cfg, pools = calderon_stochastic.get_config(), large_pools()
    calls, shapes = [], []
    monkeypatch.setattr(data_backend, "timepoint_pool_splits", lambda *args, **kwargs: pools)
    def sample(model, params, source, start, end, key, **kwargs):
        calls.append((float(source[0, 0]), start, end, kwargs["n_steps"]))
        return source + 1
    monkeypatch.setattr(shared_eval, "sample_composed_pushforward", sample)
    def emd(prediction, target):
        shapes.append((len(prediction), len(target)))
        return float(prediction.mean())
    monkeypatch.setattr(shared_eval.wasserstein, "exact_emd", emd)
    metrics = evaluation.observed_distribution_metrics(object(), {}, cfg, n_noise_draws=1)
    assert calls == [(0., 0., .5, 50), (10., .5, 1., 50), (1., .5, 1., 50)]
    assert shapes == [(4096, 4096)] * 3
    assert metrics["final_eval/observed_flowmap_mean_emd"] == 6.
    assert metrics["final_eval/day4-6_to_day12-14_rollout_flowmap_emd"] == 2.


def test_violations_use_holdout_cells_manifest_classifiers_and_three_noise_draws(monkeypatch, tmp_path):
    cfg = calderon_stochastic.get_config()
    training, oracle = tmp_path / "training.json", tmp_path / "oracle.json"
    training.write_text("{}"); oracle.write_text("{}")
    cfg.problem.classifier_path = str(training)
    cfg.logging.maizels.full_data_classifier_path = str(oracle)
    pools = {"4-6": {"x": np.full((1000, 100), 99., dtype=np.float32),
        "holdout_x": np.full((700, 100), 6., dtype=np.float32), "holdout_types": np.array(["A"] * 700)}}
    monkeypatch.setattr(data_backend, "timepoint_pool_splits", lambda *args, **kwargs: pools)
    monkeypatch.setattr(data_backend, "class_names", lambda cfg: ("A", "B"))
    paths, classifiers = [], []
    def sample(model, params, x, start, end, key, cfg, **kwargs):
        assert x.shape == (512, 100) and np.all(x == 6.)
        assert (start, end, kwargs["n_steps"]) == (0., 1., 50)
        paths.append(key)
        return np.repeat(x[:, None], 50, axis=1)
    monkeypatch.setattr(shared_eval, "sample_evaluation_path", sample)
    def check(path, source_ids, classifier_path, **kwargs):
        assert kwargs["prob_threshold"] == kwargs["margin_threshold"] == 0
        assert np.all(source_ids == 0)
        classifiers.append(classifier_path)
        return {"valid": np.ones(512, dtype=bool) if classifier_path == training else np.arange(512) % 2 == 0}
    monkeypatch.setattr(data_backend, "check_paths_with_classifier", check)
    metrics = evaluation.lineage_metrics(object(), {}, cfg)
    assert len(paths) == 3 and classifiers == [training, oracle] * 3
    assert metrics["final_eval/lineage_eval_source_count"] == 512
    assert metrics["final_eval/full_data_classifier/flowmap_invalid_trajectory_pct"] == 50.


def test_checkpoint_eval_restores_json_sequences_and_saved_ema(monkeypatch, tmp_path):
    cfg = calderon_stochastic.get_config()
    # Older checkpoints predate the reduced reporting flags.
    del cfg.evaluation.include_direct
    del cfg.evaluation.include_mmd
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    (tmp_path / "data_summary.json").write_text(json.dumps({"provenance": {"dataset_sha256": "fixture"}}))
    payload = {"config": cfg.to_dict(), "state": {"params": {"weight": np.array([1.], dtype=np.float32)}},
               "ema_params": {"weight": np.array([2.], dtype=np.float32)}, "pc_scale": np.ones(100, dtype=np.float32),
               "best_step": np.array(10, dtype=np.int32), "best_validation_loss": np.array(1., dtype=np.float32)}
    checkpoint = tmp_path / "best.msgpack"
    checkpoint.write_bytes(flax.serialization.to_bytes(payload))
    original = checkpoint.read_bytes()
    monkeypatch.setattr(launcher, "_prepare_data", lambda cfg: SimpleNamespace(provenance={"dataset_sha256": "fixture"}))
    monkeypatch.setattr(shared_launcher, "_create_model_and_state", lambda *args: (object(), None, None, None))
    def evaluate(model, instantaneous, ema, restored, result_dir, backend):
        assert list(restored.problem.retained_timepoints) == ["4-6", "8-10", "12-14"]
        assert restored.evaluation.max_source_points == restored.evaluation.max_target_points == 17
        assert not restored.evaluation.include_direct and not restored.evaluation.include_mmd
        np.testing.assert_array_equal(instantaneous["weight"], [1.])
        np.testing.assert_array_equal(ema["weight"], [2.])
        return {"final_eval/flowmap_mean_emd": 3.}
    monkeypatch.setattr(shared_launcher, "_evaluate_best_parameters", evaluate)
    assert launcher.main(["--stage", "evaluate", "--checkpoint", str(checkpoint), "--marginal-cap", "17"]) == 0
    assert checkpoint.read_bytes() == original
    info = json.loads((tmp_path / "evaluations/w1_17/evaluation_info.json").read_text())
    assert info["parameters"] == "EMA"

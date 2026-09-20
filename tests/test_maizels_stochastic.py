import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common import maizels_stochastic_interpolant
from common import maizels_stochastic_eval
from common import maizels_stochastic_training
from common import ssfm_brownian
from common import stochastic_flow_map
from configs import maizels_stochastic
from launchers import maizels_stochastic as maizels_stochastic_launcher
from scripts import run_maizels_stochastic_multiseed
from scripts import sweep_maizels_stochastic_hparams


def test_stochastic_config_has_seven_isolated_variants():
    standard = maizels_stochastic.get_config(
        0, total_steps=10, batch_size=8, n_pairs=20
    )
    prior = maizels_stochastic.get_config(1, total_steps=10, batch_size=8, n_pairs=20)
    constrained = maizels_stochastic.get_config(
        2,
        total_steps=10,
        batch_size=8,
        n_pairs=20,
        diffusion_scale=0.35,
    )
    bio_ot = maizels_stochastic.get_config(3, total_steps=10, batch_size=8, n_pairs=20)
    constrained_bio_ot = maizels_stochastic.get_config(
        4, total_steps=10, batch_size=8, n_pairs=20
    )
    plain_ot = maizels_stochastic.get_config(
        5, total_steps=10, batch_size=8, n_pairs=20
    )
    rollout_constrained_bio_ot = maizels_stochastic.get_config(
        6, total_steps=10, batch_size=8, n_pairs=20
    )

    assert standard.problem.maizels_pair_mode == "none"
    assert standard.problem.n_interpolant_check_times == 0
    assert not standard.constraints.enabled
    assert prior.problem.maizels_pair_mode == "endpoint"
    assert prior.problem.n_interpolant_check_times == 0
    assert not prior.constraints.enabled
    assert constrained.problem.maizels_pair_mode == "endpoint"
    assert constrained.constraints.enabled
    assert constrained.constraints.path_mode == "direct_offdiagonal_endpoint_nll"
    assert constrained.constraints.path_n_times == 2
    assert constrained.ssfm.diffusion_scale == 0.35
    assert list(constrained.problem.hparam_val_times) == ["D3.4", "D6"]
    assert constrained.optimization.early_stopping.metric == "validation_loss"
    assert constrained.optimization.early_stopping.check_freq == 100
    assert constrained.optimization.early_stopping.patience == 10
    assert constrained.evaluation.max_source_points == 0
    assert constrained.evaluation.max_target_points == 0
    assert constrained.evaluation.flowmap_n_steps == 50
    assert constrained.evaluation.lineage_max_source_points == 512
    assert constrained.evaluation.lineage_n_steps == 50
    assert constrained.logging.visual_freq == 5_000
    assert bio_ot.problem.maizels_pair_mode == "ot_endpoint"
    assert bio_ot.problem.n_interpolant_check_times == 0
    assert bio_ot.problem.maizels_ot_coupling == "minibatch_ot"
    assert not bio_ot.constraints.enabled
    assert constrained_bio_ot.problem.maizels_pair_mode == "ot_endpoint"
    assert constrained_bio_ot.constraints.enabled
    assert plain_ot.problem.maizels_pair_mode == "ot_plain"
    assert plain_ot.problem.maizels_ot_coupling == "minibatch_ot"
    assert rollout_constrained_bio_ot.problem.maizels_pair_mode == "ot_endpoint"
    assert rollout_constrained_bio_ot.constraints.enabled
    assert (
        rollout_constrained_bio_ot.constraints.path_mode
        == "stochastic_rollout_endpoint_nll"
    )
    assert rollout_constrained_bio_ot.constraints.stochastic_rollout_max_step == 0.05
    assert rollout_constrained_bio_ot.constraints.stochastic_rollout_batch_size == 2
    assert maizels_stochastic.get_hparam_sweep_spec(2)["diffusion_scale"]
    assert maizels_stochastic.get_hparam_sweep_spec(3)["minibatch_ot"]


def test_constraint_overrides_are_rejected_for_unconstrained_variants():
    with pytest.raises(ValueError, match="constrained stochastic variant"):
        maizels_stochastic.get_config(0, constraint_weight=1.0)
    with pytest.raises(ValueError, match="constrained stochastic variant"):
        maizels_stochastic.get_config(1, entropy_weight=0.1)


def test_stochastic_launcher_overrides_visual_frequency():
    args = maizels_stochastic_launcher.parse_args(
        ["--slurm_id", "0", "--visual_frequency", "250"]
    )
    cfg = maizels_stochastic_launcher._build_config(args)
    assert cfg.logging.visual_freq == 250


def test_stochastic_minibatch_ot_is_recoupled_for_each_batch(monkeypatch):
    cfg = maizels_stochastic.get_config(3, total_steps=10, batch_size=8, n_pairs=20)
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
        maizels_stochastic_launcher.maizels,
        "couple_minibatch_ot_timepoint_pools",
        fake_couple,
    )
    pools = {"timepoints": {}, "intervals": ()}
    batch = maizels_stochastic_launcher._sample_batch(
        pools, np.random.default_rng(7), 8, cfg
    )

    assert len(calls) == 1
    assert calls[0][0] is pools
    assert calls[0][1] == 8
    assert calls[0][3] == "ot_endpoint"
    np.testing.assert_allclose(np.asarray(batch["x1"]), 1.0)


def test_stochastic_sweep_includes_diffusion_and_only_relevant_constraints():
    plain = sweep_maizels_stochastic_hparams.build_grid(
        maizels_stochastic.get_hparam_sweep_spec(0),
        learning_rates=(1e-3,),
        diffusion_scales=(0.1, 0.2),
        constraint_weights=(1.0, 10.0),
        entropy_weights=(0.0, 0.01),
    )
    constrained = sweep_maizels_stochastic_hparams.build_grid(
        maizels_stochastic.get_hparam_sweep_spec(2),
        learning_rates=(1e-3,),
        diffusion_scales=(0.1, 0.2),
        constraint_weights=(1.0, 10.0),
        entropy_weights=(0.0, 0.01),
    )

    assert len(plain) == 2
    assert all(row["constraint_weight"] is None for row in plain)
    assert len(constrained) == 8
    assert {row["diffusion_scale"] for row in constrained} == {0.1, 0.2}


def test_stochastic_multiseed_parses_selected_or_all_slurm_ids():
    assert run_maizels_stochastic_multiseed.parse_slurm_ids("all", 7) == tuple(range(7))
    assert run_maizels_stochastic_multiseed.parse_slurm_ids("0,3,6", 7) == (
        0,
        3,
        6,
    )
    with pytest.raises(
        run_maizels_stochastic_multiseed.argparse.ArgumentTypeError,
        match="duplicate",
    ):
        run_maizels_stochastic_multiseed.parse_slurm_ids("1,1", 7)


def test_stochastic_multiseed_forwards_constraints_only_when_relevant(tmp_path):
    args = run_maizels_stochastic_multiseed.parse_args(
        [
            "--dataset-location",
            "/tmp/data.csv.gz",
            "--constraint-weight",
            "12",
            "--entropy-weight",
            "0.03",
        ]
    )
    common = {
        "args": args,
        "slurm_id": 3,
        "seed": 7,
        "output_folder": tmp_path / "checkpoints",
        "metrics_path": tmp_path / "metrics.json",
    }
    unconstrained = run_maizels_stochastic_multiseed.build_command(
        constrained=False, **common
    )
    constrained = run_maizels_stochastic_multiseed.build_command(
        constrained=True, **common
    )
    assert "--constraint_weight" not in unconstrained
    assert "--entropy_weight" not in unconstrained
    assert constrained[constrained.index("--constraint_weight") + 1] == "12.0"
    assert constrained[constrained.index("--entropy_weight") + 1] == "0.03"


def test_feature_scale_uses_population_standard_deviation():
    arrays = (
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
    )
    expected = np.std(np.concatenate(arrays, axis=0), axis=0)
    np.testing.assert_allclose(
        maizels_stochastic_launcher._feature_scale(arrays),
        expected,
        rtol=1e-6,
    )


def test_brownian_coefficients_and_chen_combination_have_expected_shapes():
    left = ssfm_brownian.sample_legendre_coefficients(
        jax.random.PRNGKey(1),
        jnp.full((7,), 0.2),
        n_coefficients=3,
        data_dim=5,
    )
    right = ssfm_brownian.sample_legendre_coefficients(
        jax.random.PRNGKey(2),
        jnp.full((7,), 0.2),
        n_coefficients=3,
        data_dim=5,
    )
    combined = ssfm_brownian.chen_combine_halves(left, right)

    assert left.shape == (7, 3, 5)
    assert combined.shape == left.shape
    np.testing.assert_allclose(
        np.asarray(ssfm_brownian.brownian_increment(combined)),
        np.asarray(left[:, 0] + right[:, 0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_interpolant_preserves_endpoints_and_has_finite_drift():
    x0 = jnp.asarray([[1.0, -2.0]])
    x1 = jnp.asarray([[4.0, 2.0]])
    epsilon = jnp.asarray([[0.5, -0.5]])
    scale = jnp.asarray([2.0, 3.0])

    start = maizels_stochastic_interpolant.stochastic_interpolant_target(
        x0,
        x1,
        jnp.asarray([0.0]),
        epsilon,
        jnp.asarray([0.4]),
        scale,
        gamma_scale=0.2,
        diffusion_scale=0.2,
    )
    end = maizels_stochastic_interpolant.stochastic_interpolant_target(
        x0,
        x1,
        jnp.asarray([1.0]),
        epsilon,
        jnp.asarray([0.4]),
        scale,
        gamma_scale=0.2,
        diffusion_scale=0.2,
    )

    np.testing.assert_allclose(np.asarray(start.state), np.asarray(x0), atol=1e-6)
    np.testing.assert_allclose(np.asarray(end.state), np.asarray(x1), atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(start.drift)))
    assert bool(jnp.all(jnp.isfinite(end.drift)))
    np.testing.assert_allclose(np.asarray(start.diffusion), 0.0, atol=1e-6)


def test_ssfm_is_identity_on_zero_length_intervals():
    model = stochastic_flow_map.StrongStochasticFlowMapMLP(
        data_dim=4,
        n_coefficients=3,
        hidden_dim=16,
        n_hidden=1,
        uncertainty_hidden_dim=8,
    )
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 4))
    coefficients = jnp.zeros((3, 3, 4))
    times = jnp.full((3,), 0.4)
    variables = model.init(jax.random.PRNGKey(1), times, times, x, coefficients)
    prediction, _ = model.apply(variables, times, times, x, coefficients)
    np.testing.assert_allclose(np.asarray(prediction), np.asarray(x), atol=1e-6)


def test_stochastic_rollout_uses_variable_em_steps_and_backpropagates():
    class AdditiveLocalMap:
        @staticmethod
        def apply(variables, s, t, x, coefficients):
            del coefficients
            rate = variables["params"]["rate"]
            prediction = x + (t - s)[:, None] * rate
            return prediction, jnp.zeros_like(s)

    x_start = jnp.zeros((2, 2), dtype=jnp.float32)
    start = jnp.asarray([0.0, 0.1], dtype=jnp.float32)
    end = jnp.asarray([0.12, 0.31], dtype=jnp.float32)

    def objective(rate):
        paths, transition_mask = maizels_stochastic_training.stochastic_rollout_paths(
            AdditiveLocalMap(),
            {"rate": rate},
            x_start,
            start,
            end,
            jax.random.PRNGKey(17),
            n_coefficients=3,
            data_dim=2,
            max_step=0.1,
            max_steps=4,
        )
        return jnp.sum(paths[:, -1, :]), (paths, transition_mask)

    (value, (paths, transition_mask)), gradient = jax.value_and_grad(
        objective, has_aux=True
    )(jnp.asarray(0.5, dtype=jnp.float32))

    assert paths.shape == (2, 5, 2)
    np.testing.assert_array_equal(
        np.asarray(transition_mask),
        np.asarray([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.float32),
    )
    np.testing.assert_allclose(np.asarray(value), 0.33, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(gradient), 0.66, rtol=1e-6)


def test_composed_pushforward_reapplies_the_flow_map():
    class MultiplicativeMap:
        @staticmethod
        def apply(variables, s, t, x, coefficients):
            del variables, coefficients
            prediction = x + (t - s)[:, None] * x
            return prediction, jnp.zeros_like(s)

    source = np.ones((3, 2), dtype=np.float32)
    direct = maizels_stochastic_eval.sample_pushforward(
        MultiplicativeMap(),
        {},
        source,
        0.0,
        1.0,
        jax.random.PRNGKey(0),
        n_coefficients=3,
    )
    composed = maizels_stochastic_eval.sample_composed_pushforward(
        MultiplicativeMap(),
        {},
        source,
        0.0,
        1.0,
        jax.random.PRNGKey(1),
        n_coefficients=3,
        n_steps=2,
    )

    np.testing.assert_allclose(direct, 2.0)
    np.testing.assert_allclose(composed, 2.25)


def test_stochastic_distribution_eval_uses_full_populations_and_both_samplers(
    monkeypatch,
):
    cfg = maizels_stochastic.get_config(0, total_steps=10, batch_size=8, n_pairs=20)
    cfg.problem.evaluation_timepoints = ["D3.4"]
    cfg.problem.hparam_val_times = ["D3.4"]
    source = np.zeros((3, 2), dtype=np.float32)
    target = np.full((5, 2), 4.0, dtype=np.float32)
    pools = {
        "D3": {"x": source},
        "D3.4": {"x": target},
    }
    emd_shapes = []

    monkeypatch.setattr(
        maizels_stochastic_eval.maizels,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    monkeypatch.setattr(
        maizels_stochastic_eval,
        "sample_pushforward",
        lambda model, params, x, *args, **kwargs: x + 1.0,
    )
    monkeypatch.setattr(
        maizels_stochastic_eval,
        "sample_composed_pushforward",
        lambda model, params, x, *args, **kwargs: x + 2.0,
    )

    def fake_emd(prediction, actual):
        emd_shapes.append((prediction.shape[0], actual.shape[0]))
        return float(np.mean(prediction))

    monkeypatch.setattr(maizels_stochastic_eval.wasserstein, "exact_emd", fake_emd)
    monkeypatch.setattr(
        maizels_stochastic_eval, "rbf_mmd2", lambda *args, **kwargs: 0.0
    )

    metrics, plot_data = maizels_stochastic_eval.distribution_metrics(
        object(), {}, cfg, n_noise_draws=1
    )

    assert emd_shapes == [(3, 5), (3, 5)]
    assert metrics["final_eval/D3p4_direct_emd"] == 1.0
    assert metrics["final_eval/D3p4_flowmap_emd"] == 2.0
    assert metrics["final_eval/direct_mean_emd_hparam_val_times"] == 1.0
    assert metrics["final_eval/flowmap_mean_emd_hparam_val_times"] == 2.0
    assert metrics["final_eval/ssfm_mean_emd_hparam_val_times"] == 1.0
    assert [values.shape[0] for values in plot_data["D3.4"]] == [5, 3, 3]


def test_lineage_eval_uses_heldout_d3_cells_and_both_classifiers(monkeypatch, tmp_path):
    cfg = maizels_stochastic.get_config(0, total_steps=10, batch_size=8, n_pairs=20)
    schedule_pt = tmp_path / "three_day.pt"
    full_pt = tmp_path / "all_days.pt"
    schedule_pt.with_suffix(".npz").touch()
    full_pt.with_suffix(".npz").touch()
    cfg.problem.classifier_path = str(schedule_pt)
    cfg.logging.maizels.full_data_classifier_path = str(full_pt)
    cfg.evaluation.n_noise_draws = 1
    cfg.evaluation.lineage_max_source_points = 2

    pools = {
        "D3": {
            "x": np.full((4, 2), 99.0, dtype=np.float32),
            "types": np.full(4, "NMP", dtype=object),
            "holdout_x": np.asarray(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32
            ),
            "holdout_types": np.full(3, "NMP", dtype=object),
        },
        "D8": {"holdout_x": np.asarray([[8.0, 8.0], [9.0, 9.0]], dtype=np.float32)},
    }
    seen_classifiers = []

    monkeypatch.setattr(
        maizels_stochastic_eval.maizels,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )

    def fake_path(model, params, x, *args, **kwargs):
        assert x.shape == (2, 2)
        assert not np.any(x == 99.0)
        assert kwargs["n_steps"] == 50
        return np.repeat(x[:, None, :], 50, axis=1)

    monkeypatch.setattr(maizels_stochastic_eval, "sample_composed_path", fake_path)

    def fake_check(paths, start_type_ids, classifier_path, **kwargs):
        del paths, start_type_ids, kwargs
        seen_classifiers.append(classifier_path.name)
        valid = (
            np.asarray([True, True])
            if classifier_path.name == "all_days.npz"
            else np.asarray([True, False])
        )
        return {"valid": valid}

    monkeypatch.setattr(
        maizels_stochastic_eval.maizels,
        "check_paths_with_classifier",
        fake_check,
    )

    metrics = maizels_stochastic_eval.lineage_metrics(object(), {}, cfg)

    assert seen_classifiers == ["three_day.npz", "all_days.npz"]
    assert metrics["final_eval/lineage_eval_source_count"] == 2.0
    assert (
        metrics["final_eval/schedule_classifier/stochastic_path_valid_fraction"] == 0.5
    )
    assert (
        metrics["final_eval/full_data_classifier/stochastic_path_valid_fraction"] == 1.0
    )
    assert metrics["final_eval/flowmap_valid_trajectory_pct"] == 50.0
    assert not hasattr(maizels_stochastic_eval, "semigroup_metrics")

    seen_classifiers.clear()
    plot_data = maizels_stochastic_eval.full_data_trajectory_plot_data(
        object(), {}, cfg
    )
    assert seen_classifiers == ["all_days.npz"]
    assert plot_data["paths"].shape == (2, 51, 2)
    np.testing.assert_array_equal(plot_data["valid"], [True, True])


def test_plain_ssfm_loss_has_finite_gradients():
    cfg = maizels_stochastic.get_config(0, total_steps=10, batch_size=8, n_pairs=20)
    cfg.problem.d = 4
    cfg.ssfm.hidden_dim = 16
    cfg.ssfm.n_hidden = 1
    cfg.ssfm.uncertainty_hidden_dim = 8
    model = stochastic_flow_map.StrongStochasticFlowMapMLP(
        data_dim=4,
        n_coefficients=3,
        hidden_dim=16,
        n_hidden=1,
        uncertainty_hidden_dim=8,
    )
    variables = model.init(
        jax.random.PRNGKey(1),
        jnp.zeros((8,)),
        jnp.ones((8,)),
        jnp.zeros((8, 4)),
        jnp.zeros((8, 3, 4)),
    )
    labels = jnp.tile(jnp.asarray([[0.0, 1.0, 0.0, 0.16]]), (8, 1))
    batch = {
        "x0": jax.random.normal(jax.random.PRNGKey(2), (8, 4)),
        "x1": jax.random.normal(jax.random.PRNGKey(3), (8, 4)),
        "label": labels,
    }
    loss_fn = maizels_stochastic_training.make_loss_fn(model, cfg, jnp.ones((4,)))
    (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        variables["params"],
        variables["params"],
        batch,
        jax.random.PRNGKey(4),
        jnp.asarray(1),
    )

    assert bool(jnp.isfinite(loss))
    assert all(
        bool(jnp.all(jnp.isfinite(value))) for value in jax.tree_util.tree_leaves(grads)
    )


def test_constrained_ssfm_reuses_direct_student_prediction(monkeypatch):
    cfg = maizels_stochastic.get_config(2, total_steps=10, batch_size=8, n_pairs=20)
    cfg.problem.d = 2

    class CountingMap:
        def __init__(self):
            self.calls = []

        def apply(self, variables, s, t, x, coefficients):
            del t, coefficients
            parameter_set = variables["params"]
            self.calls.append(parameter_set)
            offset = 1.0 if parameter_set == "student" else 2.0
            return x + offset, jnp.zeros_like(s)

    seen = {}

    def fake_lineage_loss(path, labels, classifier, **kwargs):
        del labels, classifier, kwargs
        seen["path"] = np.asarray(path)
        zero = jnp.asarray(0.0, dtype=path.dtype)
        return zero, {
            "transition_invalid_mass": zero,
            "transition_valid_mass": zero,
            "final_entropy_loss": zero,
        }

    monkeypatch.setattr(
        maizels_stochastic_training,
        "lineage_loss_for_path",
        fake_lineage_loss,
    )
    model = CountingMap()
    labels = jnp.tile(jnp.asarray([[0.0, 1.0, 0.0, 0.16]]), (8, 1))
    batch = {
        "x0": jax.random.normal(jax.random.PRNGKey(11), (8, 2)),
        "x1": jax.random.normal(jax.random.PRNGKey(12), (8, 2)),
        "label": labels,
    }
    loss_fn = maizels_stochastic_training.make_loss_fn(
        model,
        cfg,
        jnp.ones((2,)),
        classifier=object(),
    )

    loss, _ = loss_fn(
        "student",
        "ema",
        batch,
        jax.random.PRNGKey(13),
        jnp.asarray(1),
    )

    assert bool(jnp.isfinite(loss))
    assert model.calls == ["student", "student", "ema", "ema"]
    assert seen["path"].shape == (2, 2, 2)
    np.testing.assert_allclose(
        seen["path"][:, 1] - seen["path"][:, 0],
        1.0,
    )

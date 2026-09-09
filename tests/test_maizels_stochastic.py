import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common import maizels_stochastic_interpolant
from common import maizels_stochastic_training
from common import ssfm_brownian
from common import stochastic_flow_map
from configs import maizels_stochastic
from launchers import maizels_stochastic as maizels_stochastic_launcher
from scripts import sweep_maizels_stochastic_hparams


def test_stochastic_config_has_three_isolated_variants():
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

    assert standard.problem.maizels_pair_mode == "none"
    assert not standard.constraints.enabled
    assert prior.problem.maizels_pair_mode == "endpoint"
    assert not prior.constraints.enabled
    assert constrained.problem.maizels_pair_mode == "endpoint"
    assert constrained.constraints.enabled
    assert constrained.ssfm.diffusion_scale == 0.35
    assert list(constrained.problem.hparam_val_times) == ["D3.4", "D6"]
    assert maizels_stochastic.get_hparam_sweep_spec(2)["diffusion_scale"]


def test_constraint_overrides_are_rejected_for_unconstrained_variants():
    with pytest.raises(ValueError, match="stochastic Slurm ID 2"):
        maizels_stochastic.get_config(0, constraint_weight=1.0)
    with pytest.raises(ValueError, match="stochastic Slurm ID 2"):
        maizels_stochastic.get_config(1, entropy_weight=0.1)


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

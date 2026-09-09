"""Losses and classifier setup for direct Maizels strong flow-map training."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from . import maizels
from . import maizels_stochastic_interpolant as stochastic_interpolant
from . import ssfm_brownian


def setup_lineage_classifier(cfg) -> Dict[str, Any]:
    """Load the frozen schedule classifier used by the constrained variant."""
    params, class_names, scaler_mean, scaler_scale = maizels.load_jax_classifier_params(
        cfg.problem.classifier_path
    )
    transition_mode = maizels.lineage_transition_mode_from_config(cfg)
    return {
        "params": params,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "invalid_transition": jnp.asarray(
            maizels.lineage_invalid_transition_matrix(
                class_names,
                transition_mode=transition_mode,
            ),
            dtype=jnp.float32,
        ),
        "canonical_to_classifier": jnp.asarray(
            maizels.classifier_index_lookup(class_names),
            dtype=jnp.int32,
        ),
    }


def lineage_loss_for_path(
    path: jnp.ndarray,
    labels: jnp.ndarray,
    classifier: Dict[str, Any],
    *,
    temperature: float,
    lambda_start: float,
    lambda_transition: float,
    lambda_final: float,
    entropy_weight: float,
) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Return differentiable classifier-NLL lineage loss for ``[B,T,D]`` paths."""
    flat = path.reshape((-1, path.shape[-1]))
    logits = maizels.jax_classifier_logits(
        classifier["params"],
        classifier["scaler_mean"],
        classifier["scaler_scale"],
        flat,
    )
    probs = jax.nn.softmax(
        logits / jnp.maximum(jnp.asarray(temperature, logits.dtype), 1e-6),
        axis=-1,
    ).reshape((path.shape[0], path.shape[1], -1))
    target_type_ids = labels[:, 1] if float(lambda_final) > 0.0 else None
    terms = maizels.lineage_soft_terms_from_probs(
        probs,
        labels[:, 0],
        classifier["invalid_transition"],
        classifier["canonical_to_classifier"],
        target_type_ids=target_type_ids,
    )
    loss = (
        float(lambda_start) * terms["start_valid_nll_loss"]
        + float(lambda_transition) * terms["transition_valid_nll_loss"]
        + float(lambda_final) * terms["final_valid_nll_loss"]
        + float(entropy_weight) * terms["final_entropy_loss"]
    )
    return jnp.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6), terms


def _sample_horizon(
    key: jnp.ndarray,
    shape,
    lower: float,
    upper: jnp.ndarray,
    log_uniform: bool,
) -> jnp.ndarray:
    lower_value = jnp.asarray(lower, dtype=jnp.float32)
    upper_value = jnp.maximum(jnp.asarray(upper, dtype=jnp.float32), lower_value)
    uniform = jax.random.uniform(key, shape, dtype=jnp.float32)
    if log_uniform:
        return jnp.exp(
            jnp.log(lower_value)
            + uniform * (jnp.log(upper_value) - jnp.log(lower_value))
        )
    return lower_value + uniform * (upper_value - lower_value)


def _uncertainty_weighted_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    uncertainty: jnp.ndarray,
    horizon: jnp.ndarray,
) -> jnp.ndarray:
    raw = jnp.mean(jnp.square(prediction - jax.lax.stop_gradient(target)), axis=-1)
    raw = raw / jnp.maximum(horizon, 1e-6)
    uncertainty = jnp.clip(uncertainty, -10.0, 10.0)
    return jnp.mean(jnp.exp(-uncertainty) * raw + uncertainty)


def make_loss_fn(
    model,
    cfg,
    pc_scale: jnp.ndarray,
    classifier: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Create the direct local-target plus SSFM semigroup objective.

    The EMA model is an internal consistency target, not a separately trained
    SDE teacher.  The constrained variant adds its classifier loss to a
    differentiable two-half-step path from the current model.
    """
    eta = float(cfg.ssfm.local_fraction)
    dt = float(cfg.ssfm.local_step_fraction)
    eps = float(cfg.ssfm.time_eps_fraction)
    max_horizon = float(cfg.ssfm.max_horizon_fraction)
    n_coefficients = int(cfg.ssfm.n_coefficients)
    data_dim = int(cfg.problem.d)
    gamma_scale = float(cfg.ssfm.gamma_scale)
    diffusion_scale = float(cfg.ssfm.diffusion_scale)
    log_uniform = bool(cfg.ssfm.log_uniform_horizon)
    curriculum = bool(cfg.ssfm.horizon_curriculum)
    curriculum_steps = max(
        1.0,
        float(cfg.ssfm.horizon_curriculum_fraction)
        * float(cfg.optimization.total_steps),
    )
    curriculum_power = float(cfg.ssfm.horizon_curriculum_power)
    constraints_enabled = bool(cfg.constraints.enabled)
    if constraints_enabled and classifier is None:
        raise ValueError("The constrained SSFM variant requires a classifier.")

    def apply(params, s, t, x, coefficients):
        return model.apply({"params": params}, s, t, x, coefficients)

    def interpolant_state(x0, x1, tau, duration, key):
        epsilon = jax.random.normal(key, x0.shape, dtype=x0.dtype)
        return stochastic_interpolant.stochastic_interpolant_target(
            x0,
            x1,
            tau,
            epsilon,
            duration,
            pc_scale,
            gamma_scale=gamma_scale,
            diffusion_scale=diffusion_scale,
        )

    def loss_fn(params, ema_params, batch, key, step):
        x0 = batch["x0"]
        x1 = batch["x1"]
        labels = batch["label"]
        if x0.shape[0] < 2:
            raise ValueError("SSFM training batches require at least two examples.")
        n_local = max(1, min(x0.shape[0] - 1, int(x0.shape[0] * eta)))
        local = (x0[:n_local], x1[:n_local], labels[:n_local])
        semigroup = (x0[n_local:], x1[n_local:], labels[n_local:])
        key_local, key_semigroup = jax.random.split(key)

        def local_loss(batch_values, local_key):
            lx0, lx1, llabels = batch_values
            key_h, key_tau, key_epsilon, key_brownian = jax.random.split(local_key, 4)
            duration = jnp.maximum(llabels[:, 3] - llabels[:, 2], 1e-6)
            h_fraction = _sample_horizon(
                key_h,
                (lx0.shape[0],),
                dt / 2.0,
                jnp.asarray(dt),
                False,
            )
            tau = eps + jax.random.uniform(
                key_tau, (lx0.shape[0],), dtype=lx0.dtype
            ) * jnp.maximum(1.0 - 2.0 * eps - h_fraction, 1e-6)
            targets = interpolant_state(lx0, lx1, tau, duration, key_epsilon)
            horizon = h_fraction * duration
            s = llabels[:, 2] + tau * duration
            t = s + horizon
            coefficients = ssfm_brownian.sample_legendre_coefficients(
                key_brownian,
                horizon,
                n_coefficients=n_coefficients,
                data_dim=data_dim,
            )
            target = (
                targets.state
                + horizon[:, None] * targets.drift
                + ssfm_brownian.brownian_increment(coefficients) * targets.diffusion
            )
            prediction, uncertainty = apply(params, s, t, targets.state, coefficients)
            return _uncertainty_weighted_loss(prediction, target, uncertainty, horizon)

        def semigroup_loss(batch_values, semigroup_key):
            dx0, dx1, dlabels = batch_values
            (
                key_h,
                key_tau,
                key_epsilon,
                key_left,
                key_right,
            ) = jax.random.split(semigroup_key, 5)
            duration = jnp.maximum(dlabels[:, 3] - dlabels[:, 2], 1e-6)
            if curriculum:
                progress = jnp.clip(step / curriculum_steps, 0.0, 1.0)
                progress = progress**curriculum_power
                horizon_max = dt * (max_horizon / dt) ** progress
            else:
                progress = jnp.asarray(1.0, dtype=dx0.dtype)
                horizon_max = jnp.asarray(max_horizon, dtype=dx0.dtype)
            h_fraction = _sample_horizon(
                key_h,
                (dx0.shape[0],),
                dt,
                horizon_max,
                log_uniform,
            )
            tau = eps + jax.random.uniform(
                key_tau, (dx0.shape[0],), dtype=dx0.dtype
            ) * jnp.maximum(1.0 - 2.0 * eps - h_fraction, 1e-6)
            targets = interpolant_state(dx0, dx1, tau, duration, key_epsilon)
            horizon = h_fraction * duration
            s = dlabels[:, 2] + tau * duration
            midpoint = s + 0.5 * horizon
            t = s + horizon
            left_coefficients = ssfm_brownian.sample_legendre_coefficients(
                key_left,
                0.5 * horizon,
                n_coefficients=n_coefficients,
                data_dim=data_dim,
            )
            right_coefficients = ssfm_brownian.sample_legendre_coefficients(
                key_right,
                0.5 * horizon,
                n_coefficients=n_coefficients,
                data_dim=data_dim,
            )
            full_coefficients = ssfm_brownian.chen_combine_halves(
                left_coefficients, right_coefficients
            )
            prediction, uncertainty = apply(
                params, s, t, targets.state, full_coefficients
            )
            ema_mid, _ = apply(
                ema_params,
                s,
                midpoint,
                targets.state,
                left_coefficients,
            )
            ema_target, _ = apply(
                ema_params,
                midpoint,
                t,
                ema_mid,
                right_coefficients,
            )
            consistency = _uncertainty_weighted_loss(
                prediction,
                ema_target,
                uncertainty,
                horizon,
            )

            lineage_loss = jnp.asarray(0.0, dtype=consistency.dtype)
            lineage_terms = {
                "transition_invalid_mass": lineage_loss,
                "transition_valid_mass": lineage_loss,
                "final_entropy_loss": lineage_loss,
            }
            if constraints_enabled:
                constraint_bs = min(
                    int(cfg.constraints.constraint_batch_size), dx0.shape[0]
                )
                current_mid, _ = apply(
                    params,
                    s[:constraint_bs],
                    midpoint[:constraint_bs],
                    targets.state[:constraint_bs],
                    left_coefficients[:constraint_bs],
                )
                current_end, _ = apply(
                    params,
                    midpoint[:constraint_bs],
                    t[:constraint_bs],
                    current_mid,
                    right_coefficients[:constraint_bs],
                )
                path = jnp.stack(
                    [targets.state[:constraint_bs], current_mid, current_end],
                    axis=1,
                )
                lineage_loss, lineage_terms = lineage_loss_for_path(
                    path,
                    dlabels[:constraint_bs],
                    classifier,
                    temperature=float(cfg.constraints.classifier_temperature),
                    lambda_start=float(cfg.constraints.lambda_start),
                    lambda_transition=float(cfg.constraints.lambda_transition),
                    lambda_final=float(cfg.constraints.lambda_final),
                    entropy_weight=float(cfg.constraints.loss_point_entropy_weight),
                )
            return consistency, lineage_loss, lineage_terms, horizon_max, progress

        local_value = local_loss(local, key_local)
        (
            distill_value,
            lineage_value,
            lineage_terms,
            horizon_max,
            horizon_progress,
        ) = semigroup_loss(semigroup, key_semigroup)
        ssfm_value = eta * local_value + (1.0 - eta) * distill_value
        total = ssfm_value + float(cfg.constraints.weight) * lineage_value
        metrics = {
            "loss": total,
            "loss/ssfm": ssfm_value,
            "loss/local": local_value,
            "loss/semigroup": distill_value,
            "loss/lineage_unweighted": lineage_value,
            "loss/lineage_weighted": float(cfg.constraints.weight) * lineage_value,
            "lineage/transition_invalid_mass": lineage_terms["transition_invalid_mass"],
            "lineage/transition_valid_mass": lineage_terms["transition_valid_mass"],
            "lineage/final_entropy": lineage_terms["final_entropy_loss"],
            "training/horizon_max_fraction": horizon_max,
            "training/horizon_progress": horizon_progress,
        }
        return total, metrics

    return loss_fn

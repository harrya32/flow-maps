"""Stochastic-interpolant targets for the direct Maizels SSFM experiment."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class InterpolantTarget(NamedTuple):
    """State and local Itô coefficients at one stochastic-interpolant time."""

    state: jnp.ndarray
    velocity: jnp.ndarray
    score: jnp.ndarray
    diffusion: jnp.ndarray
    drift: jnp.ndarray


def feature_scale(x0: jnp.ndarray, x1: jnp.ndarray) -> jnp.ndarray:
    """Return a stable per-PC standard deviation for noise calibration."""
    values = jnp.concatenate([jnp.asarray(x0), jnp.asarray(x1)], axis=0)
    scale = jnp.std(values, axis=0)
    return jnp.maximum(scale, jnp.asarray(1e-4, dtype=values.dtype))


def stochastic_interpolant_target(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    tau: jnp.ndarray,
    epsilon: jnp.ndarray,
    interval_length: jnp.ndarray,
    pc_scale: jnp.ndarray,
    *,
    gamma_scale: float,
    diffusion_scale: float,
) -> InterpolantTarget:
    """Build an endpoint-preserving stochastic interpolant and its SDE target.

    ``tau`` is local to each retained interval and lies in ``(0, 1)``.  The
    bridge noise uses ``gamma(tau) = gamma_scale * sin(pi * tau)``.  The
    diagonal SDE diffusion is scaled as ``sqrt(sin(pi*tau) / interval_length)``;
    it therefore vanishes at retained endpoints and its score correction stays
    finite there.
    """
    x0 = jnp.asarray(x0)
    x1 = jnp.asarray(x1)
    tau = jnp.asarray(tau, dtype=x0.dtype)
    epsilon = jnp.asarray(epsilon, dtype=x0.dtype)
    interval_length = jnp.asarray(interval_length, dtype=x0.dtype)
    pc_scale = jnp.maximum(jnp.asarray(pc_scale, dtype=x0.dtype), 1e-4)

    if x0.shape != x1.shape or x0.shape != epsilon.shape:
        raise ValueError(
            "x0, x1, and epsilon must have identical shapes; got "
            f"{x0.shape}, {x1.shape}, and {epsilon.shape}."
        )
    if float(gamma_scale) <= 0.0:
        raise ValueError("gamma_scale must be positive.")
    if float(diffusion_scale) < 0.0:
        raise ValueError("diffusion_scale must be non-negative.")

    duration = jnp.maximum(interval_length, 1e-6)
    sin_tau = jnp.maximum(jnp.sin(jnp.pi * tau), 0.0)
    safe_sin_tau = jnp.maximum(sin_tau, 1e-6)
    cos_tau = jnp.cos(jnp.pi * tau)
    noise_scale = float(gamma_scale) * sin_tau[..., None] * pc_scale

    state = (1.0 - tau[..., None]) * x0 + tau[..., None] * x1 + noise_scale * epsilon
    velocity = (
        x1 - x0 + float(gamma_scale) * jnp.pi * cos_tau[..., None] * pc_scale * epsilon
    ) / duration[..., None]
    safe_noise_scale = float(gamma_scale) * safe_sin_tau[..., None] * pc_scale
    score = -epsilon / safe_noise_scale
    diffusion = (
        float(diffusion_scale) * pc_scale * jnp.sqrt(sin_tau / duration)[..., None]
    )
    drift = velocity + 0.5 * jnp.square(diffusion) * score
    return InterpolantTarget(
        state=state,
        velocity=velocity,
        score=score,
        diffusion=diffusion,
        drift=drift,
    )

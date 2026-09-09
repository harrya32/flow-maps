"""Flax implementation of a strong stochastic flow map for vector data."""

from __future__ import annotations

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from . import ssfm_brownian


def _kernel_init(scale: float = 1.0) -> Callable:
    # ``distribution='normal'`` also avoids the truncated-normal ``erf`` issue
    # in the experimental JAX Metal backend used by this project.
    return nn.initializers.variance_scaling(
        scale=scale,
        mode="fan_avg",
        distribution="normal",
    )


def _time_features(s: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    s = jnp.asarray(s, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)
    h = t - s
    return jnp.stack(
        [
            s,
            t,
            h,
            jnp.log(jnp.maximum(h, 1e-6)),
            jnp.sin(2.0 * jnp.pi * s),
            jnp.cos(2.0 * jnp.pi * s),
            jnp.sin(2.0 * jnp.pi * t),
            jnp.cos(2.0 * jnp.pi * t),
        ],
        axis=-1,
    )


class VectorMLP(nn.Module):
    output_dim: int
    hidden_dim: int
    n_hidden: int
    final_kernel_scale: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for index in range(int(self.n_hidden)):
            x = nn.Dense(
                self.hidden_dim,
                kernel_init=_kernel_init(),
                name=f"hidden_{index}",
            )(x)
            x = nn.silu(x)
        return nn.Dense(
            self.output_dim,
            kernel_init=_kernel_init(self.final_kernel_scale),
            bias_init=nn.initializers.zeros,
            name="output",
        )(x)


class StrongStochasticFlowMapMLP(nn.Module):
    """Interval-conditioned SSFM with learned drift and diffusion components.

    The parameterization mirrors the official implementation:

    ``x_t = x_s + (t-s) * drift + W * diffusion``.

    Drift sees the state and all Brownian coefficients. Diffusion sees the two
    times and Brownian coefficients but not the state, matching the additive
    diffusion used for the Maizels stochastic interpolant.
    """

    data_dim: int
    n_coefficients: int = 3
    hidden_dim: int = 512
    n_hidden: int = 3
    rescale: float = 1.0
    uncertainty_hidden_dim: int = 64

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,
        t: jnp.ndarray,
        x_s: jnp.ndarray,
        coefficients: jnp.ndarray,
    ):
        if coefficients.shape[-2:] != (self.n_coefficients, self.data_dim):
            raise ValueError(
                "Expected Brownian coefficients with trailing shape "
                f"({self.n_coefficients}, {self.data_dim}), got "
                f"{coefficients.shape[-2:]}."
            )
        if x_s.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected states with dimension {self.data_dim}, got {x_s.shape[-1]}."
            )

        time_features = _time_features(s, t)
        leading_shape = coefficients.shape[:-2]
        coefficient_features = coefficients.reshape(
            leading_shape + (self.n_coefficients * self.data_dim,)
        )
        scale = jnp.asarray(max(float(self.rescale), 1e-6), dtype=x_s.dtype)

        drift_input = jnp.concatenate(
            [x_s / scale, time_features, coefficient_features], axis=-1
        )
        diffusion_input = jnp.concatenate(
            [time_features, coefficient_features], axis=-1
        )
        drift = scale * VectorMLP(
            output_dim=self.data_dim,
            hidden_dim=self.hidden_dim,
            n_hidden=self.n_hidden,
            name="drift",
        )(drift_input)
        diffusion = scale * VectorMLP(
            output_dim=self.data_dim,
            hidden_dim=self.hidden_dim,
            n_hidden=max(1, self.n_hidden - 1),
            name="diffusion",
        )(diffusion_input)

        uncertainty_features = VectorMLP(
            output_dim=1,
            hidden_dim=self.uncertainty_hidden_dim,
            n_hidden=1,
            final_kernel_scale=1e-4,
            name="uncertainty",
        )(time_features)
        uncertainty = jnp.squeeze(uncertainty_features, axis=-1)

        h = jnp.asarray(t, dtype=x_s.dtype) - jnp.asarray(s, dtype=x_s.dtype)
        x_t = (
            x_s
            + h[..., None] * drift
            + ssfm_brownian.brownian_increment(coefficients) * diffusion
        )
        return x_t, uncertainty


def ema_update(params, ema_params, decay: float):
    """Update an EMA parameter tree."""
    return jax.tree_util.tree_map(
        lambda value, ema: decay * ema + (1.0 - decay) * value,
        params,
        ema_params,
    )

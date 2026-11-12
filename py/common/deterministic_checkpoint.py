"""Batched composition and Euler sampling of a deterministic Flax flow map."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def pushforward(
    net,
    params,
    source,
    *,
    start_time,
    end_time,
    n_steps,
    batch_size,
    euler=False,
):
    """Take exactly N equal steps from the observed source to the target time."""
    if n_steps <= 0 or batch_size <= 0:
        raise ValueError("n_steps and batch_size must be positive.")
    if end_time < start_time:
        raise ValueError("end_time must be greater than or equal to start_time.")
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2 or not len(source):
        raise ValueError(
            "Sampling requires a non-empty two-dimensional source population."
        )
    if end_time == start_time:
        return source.copy()
    times = jnp.linspace(start_time, end_time, n_steps + 1, dtype=jnp.float32)

    @jax.jit
    def run_batch(x):
        def step(current, interval):
            left, right = interval
            if euler:
                velocity = jax.vmap(
                    lambda point: net.apply(
                        params,
                        left,
                        point,
                        label=None,
                        train=False,
                        calc_weight=False,
                        method="calc_b",
                    )
                )(current)
                result = current + (right - left) * velocity
            else:
                result = jax.vmap(
                    lambda point: net.apply(
                        params,
                        left,
                        right,
                        point,
                        label=None,
                        train=False,
                        calc_weight=False,
                        return_X_and_phi=False,
                    )
                )(current)
            return result, None

        result, _ = jax.lax.scan(step, x, (times[:-1], times[1:]))
        return result

    pieces = [
        np.asarray(
            jax.device_get(run_batch(jnp.asarray(source[left : left + batch_size]))),
            dtype=np.float32,
        )
        for left in range(0, len(source), batch_size)
    ]
    prediction = np.concatenate(pieces, axis=0)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite deterministic flow-map predictions.")
    return prediction

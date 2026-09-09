"""Brownian polynomial coefficients for strong stochastic flow maps.

This is a small Flax/JAX-compatible port of the shifted-Legendre coefficient
construction used by ``ssfm/experiments/sde/sde.py``.  Keeping it here avoids
requiring the reference repository's newer Equinox/Diffrax/JAX environment.

For an interval of length ``h``, coefficient ``n`` is an independent centred
Gaussian with variance ``h / (2 n + 1)``.  The first coefficient is the
Brownian increment.  ``chen_combine_halves`` combines coefficients on two
equal adjacent half intervals into coefficients on their union.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


MAX_COEFFICIENTS = 4

_C_MATRIX = jnp.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [-1.0 / 2.0, 1.0 / 2.0, 0.0, 0.0],
        [0.0, -3.0 / 4.0, 1.0 / 4.0, 0.0],
        [1.0 / 8.0, 3.0 / 8.0, -5.0 / 8.0, 1.0 / 8.0],
    ],
    dtype=jnp.float32,
)
_SIGN_MATRIX = jnp.asarray(
    [
        [(-1.0) ** (n + m) for m in range(MAX_COEFFICIENTS)]
        for n in range(MAX_COEFFICIENTS)
    ],
    dtype=jnp.float32,
)


def _validate_n_coefficients(n_coefficients: int) -> int:
    n_coefficients = int(n_coefficients)
    if not 1 <= n_coefficients <= MAX_COEFFICIENTS:
        raise ValueError(
            f"n_coefficients must be in [1, {MAX_COEFFICIENTS}], "
            f"got {n_coefficients}."
        )
    return n_coefficients


def sample_legendre_coefficients(
    key: jnp.ndarray,
    interval_length: jnp.ndarray,
    *,
    n_coefficients: int,
    data_dim: int,
) -> jnp.ndarray:
    """Sample shifted-Legendre Brownian coefficients.

    ``interval_length`` may be scalar or batched.  The returned shape is
    ``interval_length.shape + (n_coefficients, data_dim)``.
    """
    n_coefficients = _validate_n_coefficients(n_coefficients)
    interval_length = jnp.asarray(interval_length, dtype=jnp.float32)
    if interval_length.ndim > 1:
        raise ValueError("interval_length must be scalar or one-dimensional.")

    indices = jnp.arange(n_coefficients, dtype=interval_length.dtype)
    variances = interval_length[..., None] / (2.0 * indices + 1.0)
    std = jnp.sqrt(jnp.maximum(variances, 0.0))[..., :, None]
    noise = jax.random.normal(
        key,
        shape=interval_length.shape + (n_coefficients, int(data_dim)),
        dtype=interval_length.dtype,
    )
    return std * noise


def chen_combine_halves(left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
    """Combine coefficients from two equal adjacent Brownian intervals.

    The final two axes must be ``(n_coefficients, data_dim)``.  Leading batch
    axes are preserved.
    """
    left = jnp.asarray(left)
    right = jnp.asarray(right)
    if left.shape != right.shape:
        raise ValueError(
            f"left and right coefficient shapes differ: {left.shape} vs {right.shape}."
        )
    if left.ndim < 2:
        raise ValueError("Brownian coefficients require coefficient and data axes.")
    n_coefficients = _validate_n_coefficients(left.shape[-2])
    c = _C_MATRIX[:n_coefficients, :n_coefficients].astype(left.dtype)
    signed_c = (c * _SIGN_MATRIX[:n_coefficients, :n_coefficients]).astype(left.dtype)
    return jnp.einsum("ij,...jd->...id", c, left) + jnp.einsum(
        "ij,...jd->...id", signed_c, right
    )


def brownian_increment(coefficients: jnp.ndarray) -> jnp.ndarray:
    """Return the Brownian increment, i.e. the zeroth coefficient."""
    coefficients = jnp.asarray(coefficients)
    if coefficients.ndim < 2:
        raise ValueError("Brownian coefficients require coefficient and data axes.")
    return coefficients[..., 0, :]

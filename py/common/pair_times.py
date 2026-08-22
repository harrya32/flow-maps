"""Helpers for datasets whose paired examples occupy sub-intervals of [0, 1]."""

from __future__ import annotations

import jax.numpy as jnp


TIME_START_COLUMN = 2
TIME_END_COLUMN = 3


def enabled(cfg) -> bool:
    """Whether pair labels contain ``[source_type, target_type, t0, t1]``."""
    problem = getattr(cfg, "problem", None)
    return bool(getattr(problem, "pair_time_bounds_in_label", False))


def bounds(label, *, dtype=None):
    """Extract a scalar or batched pair interval from its label."""
    if label is None or label.shape[-1] <= TIME_END_COLUMN:
        raise ValueError(
            "Time-aware pairs require labels with columns "
            "[source_type, target_type, t_start, t_end]."
        )
    t_start = label[..., TIME_START_COLUMN]
    t_end = label[..., TIME_END_COLUMN]
    if dtype is not None:
        t_start = jnp.asarray(t_start, dtype=dtype)
        t_end = jnp.asarray(t_end, dtype=dtype)
    return t_start, t_end


def local_to_global(local_time, label, *, dtype=None):
    """Map a local time in [0, 1] into the interval stored in ``label``."""
    t_start, t_end = bounds(label, dtype=dtype)
    local_time = jnp.asarray(local_time, dtype=dtype)
    return t_start + (t_end - t_start) * local_time


def global_to_local(global_time, label, *, dtype=None):
    """Map an absolute time into the pair-local interpolation coordinate."""
    t_start, t_end = bounds(label, dtype=dtype)
    global_time = jnp.asarray(global_time, dtype=dtype)
    duration = jnp.maximum(t_end - t_start, jnp.asarray(1e-8, dtype=t_start.dtype))
    return (global_time - t_start) / duration

"""Exact empirical Wasserstein metrics shared by lineage evaluations."""

from __future__ import annotations

import numpy as np


def exact_emd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_iter_max: int = 10_000_000,
) -> float:
    """Return exact uniform-mass W1 using Euclidean ground cost.

    This matches the ``test_EMD`` calculation used by metric-flow-matching.
    The dense cost matrix is assembled in place to limit peak memory.
    """
    try:
        import ot as pot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Exact EMD evaluation requires POT (pip install POT==0.9.3)."
        ) from exc

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("EMD inputs must be two-dimensional sample arrays.")
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("EMD cannot be computed on an empty population.")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            "EMD populations must have the same feature dimension, got "
            f"{x.shape[1]} and {y.shape[1]}."
        )

    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    cost = x @ y.T
    cost *= np.float32(-2.0)
    cost += x_norm
    cost += y_norm
    np.maximum(cost, np.float32(0.0), out=cost)
    np.sqrt(cost, out=cost)

    return float(
        pot.emd2(
            pot.unif(x.shape[0]),
            pot.unif(y.shape[0]),
            cost,
            numItermax=int(num_iter_max),
        )
    )

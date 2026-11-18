"""Exact empirical Wasserstein metrics shared by lineage evaluations."""

from __future__ import annotations

from contextlib import contextmanager
import tempfile

import numpy as np


@contextmanager
def _emd_cost_matrix(x, y, *, max_in_memory_pairs=20_000_000, row_block_size=256):
    """Preserve the Euclidean cost while keeping large matrices on disk."""
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T

    def fill(block, norms):
        block *= np.float32(-2.)
        block += norms
        block += y_norm
        np.maximum(block, np.float32(0.), out=block)
        np.sqrt(block, out=block)
        return block

    if len(x) * len(y) <= max_in_memory_pairs:
        yield fill(x @ y.T, x_norm)
        return
    # POT expects a contiguous float64 matrix. Building it by rows avoids
    # holding both a full float32 matrix and its float64 conversion in RAM.
    with tempfile.TemporaryFile() as scratch:
        cost = np.memmap(scratch, dtype=np.float64, mode="w+", shape=(len(x), len(y)))
        try:
            for start in range(0, len(x), row_block_size):
                end = start + row_block_size
                cost[start:end] = fill(x[start:end] @ y.T, x_norm[start:end])
            yield cost
        finally:
            cost._mmap.close()


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

    with _emd_cost_matrix(x, y) as cost:
        return float(
            pot.emd2(
                pot.unif(x.shape[0]),
                pot.unif(y.shape[0]),
                cost,
                numItermax=int(num_iter_max),
            )
        )

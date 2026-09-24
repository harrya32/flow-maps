"""Diagnose whether a synthetic process exposes lineage-blind OT shortcuts.

The useful regime for this benchmark is deliberately narrow: oracle paths
should almost always respect the declared lineage, while an unconstrained OT
coupling/interpolant should contain a measurable number of forbidden paths.
This module reports both quantities before any neural model is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG, load_config
from .dataset import exact_marginal, hard_labels, simulate
from .diffusion import integrated_variance_numpy, resolve_prior
from .models import make_ot_plan, valid_transition_matrix


def _invalidity(paths: np.ndarray, dataset: dict[str, Any]) -> dict[str, float]:
    """Return irreversible-transition rates for ``path x time x coordinate``."""
    labels = hard_labels(paths, dataset)
    allowed = valid_transition_matrix().cpu().numpy().astype(bool)
    forbidden = ~allowed[labels[:, :-1], labels[:, 1:]]
    return {
        "consecutive_forbidden_rate": float(forbidden.mean()),
        "any_forbidden_path_rate": float(forbidden.any(axis=1).mean()),
    }


def _bridge_mean_paths(source, target, left, right, times, prior):
    cumulative = integrated_variance_numpy(left, times, prior)
    total = integrated_variance_numpy(left, right, prior)
    fraction = cumulative / np.maximum(total, 1e-15)
    return (
        source[:, None, :]
        + fraction[None, :, :] * (target - source)[:, None, :]
    )


def _stochastic_bridge_paths(
    source,
    target,
    left,
    right,
    times,
    prior,
    replicates,
    rng,
):
    """Draw exact paths of the additive Gaussian bridge on a fixed grid."""
    cumulative = integrated_variance_numpy(left, times, prior)
    total = integrated_variance_numpy(left, right, prior)
    fraction = cumulative / np.maximum(total, 1e-15)
    increment_variance = integrated_variance_numpy(times[:-1], times[1:], prior)
    increments = rng.normal(
        size=(int(replicates), len(source), len(times) - 1, 2)
    ) * np.sqrt(increment_variance)[None, None, :, :]
    brownian = np.concatenate(
        [
            np.zeros((int(replicates), len(source), 1, 2), dtype=np.float64),
            np.cumsum(increments, axis=2),
        ],
        axis=2,
    )
    mean = (
        source[None, :, None, :]
        + fraction[None, None, :, :]
        * (target - source)[None, :, None, :]
    )
    bridge = mean + brownian - fraction[None, None, :, :] * brownian[:, :, -1:, :]
    return bridge.reshape(-1, len(times), 2)


def _paired_endpoints(source, target, prior, left, right):
    plan, values = make_ot_plan(source, target, prior, left, right)
    rows, columns = np.nonzero(plan)
    order = np.argsort(rows)
    return source[rows[order]], target[columns[order]], values


def _interval_diagnostics(
    source,
    target,
    left,
    right,
    prior,
    dataset,
    grid_size,
    bridge_replicates,
    rng,
):
    source, target, coupling = _paired_endpoints(
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        prior,
        float(left),
        float(right),
    )
    allowed = valid_transition_matrix().cpu().numpy().astype(bool)
    source_labels = hard_labels(source, dataset)
    target_labels = hard_labels(target, dataset)
    endpoint_forbidden = ~allowed[source_labels, target_labels]
    times = np.linspace(float(left), float(right), int(grid_size), dtype=np.float64)
    linear_fraction = ((times - left) / (right - left))[None, :, None]
    linear = source[:, None, :] + linear_fraction * (target - source)[:, None, :]
    mean_bridge = _bridge_mean_paths(source, target, left, right, times, prior)
    stochastic = _stochastic_bridge_paths(
        source,
        target,
        left,
        right,
        times,
        prior,
        bridge_replicates,
        rng,
    )
    return {
        **coupling,
        "pair_count": int(len(source)),
        "endpoint_forbidden_rate": float(endpoint_forbidden.mean()),
        "linear_interpolant": _invalidity(linear, dataset),
        "bridge_mean_interpolant": _invalidity(mean_bridge, dataset),
        "stochastic_bridge_interpolant": _invalidity(stochastic, dataset),
    }


def diagnose_config(
    config: dict[str, Any],
    *,
    sample_count: int = 800,
    truth_path_count: int = 3000,
    grid_size: int = 101,
    bridge_replicates: int = 4,
    seed: int = 1701,
):
    dataset = config["dataset"]
    prior_name = next(iter(config["priors"]))
    prior = resolve_prior(config["priors"][prior_name], dataset)
    rng = np.random.default_rng(int(seed))
    times = np.asarray(dataset["training_times"], dtype=np.float64)
    marginals = [exact_marginal(time, sample_count, rng, dataset) for time in times]
    intervals = {}
    for index, (left, right) in enumerate(zip(times[:-1], times[1:])):
        intervals[f"{left:g}->{right:g}"] = _interval_diagnostics(
            marginals[index],
            marginals[index + 1],
            left,
            right,
            prior,
            dataset,
            grid_size,
            bridge_replicates,
            np.random.default_rng(int(seed) + 100 + index),
        )

    truth_times = np.linspace(
        float(dataset["start_time"]),
        float(dataset["end_time"]),
        max(2, int(round((dataset["end_time"] - dataset["start_time"]) * (grid_size - 1))) + 1),
    )
    truth_initial = exact_marginal(
        float(dataset["start_time"]), truth_path_count, rng, dataset
    )
    truth_paths = simulate(
        truth_initial,
        float(dataset["start_time"]),
        truth_times,
        int(seed) + 500,
        dataset,
    )
    observation_times = np.arange(
        float(dataset["start_time"]),
        float(dataset["end_time"]) + 0.5 * float(dataset["snapshot_step"]),
        float(dataset["snapshot_step"]),
    )
    observation_paths = simulate(
        truth_initial,
        float(dataset["start_time"]),
        observation_times,
        int(seed) + 500,
        dataset,
    )
    label_proportions = {}
    for time_value, marginal in zip(times, marginals):
        counts = np.bincount(hard_labels(marginal, dataset), minlength=5)
        label_proportions[f"{time_value:g}"] = (counts / counts.sum()).tolist()
    return {
        "prior": prior_name,
        "sample_count": int(sample_count),
        "truth_path_count": int(truth_path_count),
        "grid_size_per_unit": int(grid_size - 1),
        "bridge_replicates": int(bridge_replicates),
        "training_time_label_proportions": label_proportions,
        "oracle_paths_dense_grid": _invalidity(truth_paths, dataset),
        "oracle_paths_observation_grid": _invalidity(observation_paths, dataset),
        "intervals": intervals,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-count", type=int, default=800)
    parser.add_argument("--truth-path-count", type=int, default=3000)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--bridge-replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1701)
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    result = diagnose_config(
        load_config(arguments.config),
        sample_count=arguments.sample_count,
        truth_path_count=arguments.truth_path_count,
        grid_size=arguments.grid_size,
        bridge_replicates=arguments.bridge_replicates,
        seed=arguments.seed,
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text)
        print(arguments.output)


if __name__ == "__main__":
    main()

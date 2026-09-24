"""Configurable curved one-to-two-to-three Gaussian-mixture process.

The module contains the oracle process used only for dataset construction and
evaluation. Training code consumes exported snapshots and a public diffusion
specification, never oracle paths or density functions.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .diffusion import instantaneous_variance_numpy


CLASS_NAMES = ("root", "lower", "upper", "upper_down", "upper_up")
BRANCH_NAMES = ("lower", "upper_down", "upper_up")


def smooth_step(time, left: float, right: float):
    time = np.asarray(time, dtype=np.float64)
    unit = np.clip((time - left) / (right - left), 0.0, 1.0)
    return unit * unit * (3.0 - 2.0 * unit), 6.0 * unit * (1.0 - unit) / (right - left)


def _ramps(time, settings: dict[str, Any]):
    return {
        name: smooth_step(time, float(bounds[0]), float(bounds[1]))
        for name, bounds in settings["ramps"].items()
    }


def means_and_derivatives(time, settings: dict[str, Any]):
    """Return the three component means and their time derivatives."""
    time = np.asarray(time, dtype=np.float64)
    ramps = _ramps(time, settings)
    phenotype, phenotype_dt = [], []
    for branch in BRANCH_NAMES:
        coefficients = settings["phenotype_coefficients"][branch]
        value = sum(float(coefficient) * ramps[name][0] for name, coefficient in coefficients.items())
        derivative = sum(float(coefficient) * ramps[name][1] for name, coefficient in coefficients.items())
        phenotype.append(value)
        phenotype_dt.append(derivative)
    phenotype = np.stack(phenotype, axis=-1)
    phenotype_dt = np.stack(phenotype_dt, axis=-1)
    polynomial = settings["progress_mean"]
    progress = (
        float(polynomial["constant"])
        + float(polynomial["linear"]) * time
        + float(polynomial["quadratic"]) * time**2
    )
    progress_dt = float(polynomial["linear"]) + 2.0 * float(polynomial["quadratic"]) * time
    progress = np.broadcast_to(progress[..., None], phenotype.shape).copy()
    progress_dt = np.broadcast_to(progress_dt[..., None], phenotype.shape).copy()
    # Optional branch-specific offsets let process variants make Euclidean OT
    # ambiguous without changing the default benchmark geometry.
    for branch_index, branch in enumerate(BRANCH_NAMES):
        for ramp_name, coefficient in settings.get("progress_coefficients", {}).get(branch, {}).items():
            progress[..., branch_index] += float(coefficient) * ramps[ramp_name][0]
            progress_dt[..., branch_index] += float(coefficient) * ramps[ramp_name][1]
    scale = float(settings["coordinate_scale"])
    return (
        np.stack([progress, phenotype], axis=-1) * scale,
        np.stack([progress_dt, phenotype_dt], axis=-1) * scale,
    )


def std_and_derivative(time, settings: dict[str, Any]):
    """Smoothly interpolate component-independent diagonal standard deviations."""
    time = np.asarray(time, dtype=np.float64)
    knots = np.asarray(settings["std_knots"], dtype=np.float64)
    widths = np.asarray(settings["std_values"], dtype=np.float64)
    sigma = np.broadcast_to(widths[0], (*time.shape, 2)).copy()
    derivative = np.zeros_like(sigma)
    for index in range(len(knots) - 1):
        step, step_dt = smooth_step(time, knots[index], knots[index + 1])
        delta = widths[index + 1] - widths[index]
        sigma += step[..., None] * delta
        derivative += step_dt[..., None] * delta
    scale = float(settings["coordinate_scale"])
    return sigma * scale, derivative * scale


def density_fields(points, time, settings: dict[str, Any]):
    """Return log density, score, and probability-flow velocity."""
    points = np.asarray(points, dtype=np.float64)
    means, mean_dt = means_and_derivatives(time, settings)
    std, std_dt = std_and_derivative(time, settings)
    weights = np.asarray(settings["mixture_weights"], dtype=np.float64)
    delta = points[..., None, :] - means
    logits = np.log(weights) - 0.5 * np.sum((delta / std[..., None, :]) ** 2, axis=-1)
    normalizer = logsumexp(logits, axis=-1)
    responsibilities = np.exp(logits - normalizer[..., None])
    score = np.sum(
        responsibilities[..., None] * (-delta / std[..., None, :] ** 2), axis=-2
    )
    component_velocity = mean_dt + std_dt[..., None, :] / std[..., None, :] * delta
    velocity = np.sum(responsibilities[..., None] * component_velocity, axis=-2)
    log_density = normalizer - np.log(2.0 * np.pi) - np.sum(np.log(std), axis=-1)
    return log_density, score, velocity


def drift(points, time, settings: dict[str, Any]):
    _, score, velocity = density_fields(points, time, settings)
    variance = instantaneous_variance_numpy(time, settings["true_diffusion"])
    return velocity + 0.5 * variance * score


def exact_marginal(time: float, count: int, rng: np.random.Generator, settings: dict[str, Any]):
    means, _ = means_and_derivatives(time, settings)
    std, _ = std_and_derivative(time, settings)
    components = rng.choice(3, size=count, p=settings["mixture_weights"])
    return means[components] + std * rng.normal(size=(count, 2))


def hard_labels(points, settings: dict[str, Any]):
    labels = settings["labels"]
    x, y = np.moveaxis(np.asarray(points) / float(settings["coordinate_scale"]), -1, 0)
    return np.where(
        x < float(labels["first_x_boundary"]),
        0,
        np.where(
            y < float(labels["lower_y_boundary"]),
            1,
            np.where(
                x < float(labels["second_x_boundary"]),
                2,
                np.where(y < float(labels["children_y_boundary"]), 3, 4),
            ),
        ),
    ).astype(np.int64)


def terminal_probabilities(points, settings: dict[str, Any]):
    means, _ = means_and_derivatives(float(settings["end_time"]), settings)
    std, _ = std_and_derivative(float(settings["end_time"]), settings)
    delta = np.asarray(points)[..., None, :] - means
    logits = np.log(settings["mixture_weights"]) - 0.5 * np.sum((delta / std) ** 2, axis=-1)
    return np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))


def simulate(initial, start: float, times, seed: int, settings: dict[str, Any], *, steps_per_unit=None):
    """Simulate the single-state Markov SDE with left-point Euler-Maruyama."""
    times = np.asarray(times, dtype=np.float64)
    if np.any(np.diff(times) < 0) or times[0] < start - 1e-10:
        raise ValueError("Output times must be ordered and no earlier than start.")
    rate = int(settings["steps_per_unit"] if steps_per_unit is None else steps_per_unit)
    rng = np.random.default_rng(seed)
    state = np.asarray(initial, dtype=np.float64).copy()
    output = []
    current = float(start)
    diffusion = settings["true_diffusion"]
    for target in times:
        count = max(0, int(np.ceil((target - current) * rate - 1e-9)))
        if count:
            dt = (target - current) / count
            for index in range(count):
                time_value = current + index * dt
                variance = instantaneous_variance_numpy(time_value, diffusion)
                state += drift(state, time_value, settings) * dt
                state += np.sqrt(dt * variance) * rng.normal(size=state.shape)
        output.append(state.astype(np.float32).copy())
        current = float(target)
    return np.stack(output, axis=1)


def snapshot_times(settings: dict[str, Any]) -> np.ndarray:
    start = float(settings["start_time"])
    end = float(settings["end_time"])
    step = float(settings["snapshot_step"])
    count = int(round((end - start) / step))
    times = start + step * np.arange(count + 1)
    if not np.isclose(times[-1], end):
        raise ValueError("snapshot_step must divide the configured time interval.")
    return np.round(times, 10)


def prepare_dataset(output: Path, settings: dict[str, Any]):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rng = np.random.default_rng(int(settings["seed"]))
    times = snapshot_times(settings)
    particles = int(settings["oracle_particles"])
    cells = int(settings["cells_per_time"])
    if cells > particles:
        raise ValueError("cells_per_time cannot exceed oracle_particles.")
    initial = exact_marginal(times[0], particles, rng, settings)
    paths = simulate(initial, times[0], times, int(settings["seed"]) + 1, settings)
    indices = np.stack([rng.choice(particles, cells, replace=False) for _ in times])
    snapshots = np.stack([paths[index, time_index] for time_index, index in enumerate(indices)])
    labels = hard_labels(snapshots, settings)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float32)[labels]
    np.savez_compressed(
        output / "observed_snapshots.npz",
        times=times,
        points=snapshots,
        labels=labels,
        one_hot=one_hot,
    )
    np.savez_compressed(
        output / "oracle_paths.npz",
        times=times,
        paths=paths,
        snapshot_indices=indices,
        labels=hard_labels(paths, settings),
    )
    selected = []
    for training_time in settings["training_times"]:
        matches = np.flatnonzero(np.isclose(times, training_time))
        if len(matches) != 1:
            raise ValueError(f"Training time {training_time} is not on the snapshot grid.")
        selected.append(int(matches[0]))
    selected = np.asarray(selected, dtype=np.int64)
    np.savez_compressed(
        output / "training_data.npz",
        times=times[selected],
        points=snapshots[selected],
        labels=labels[selected],
        one_hot=one_hot[selected],
    )
    manifest = {
        "status": "completed",
        "dataset": settings,
        "class_names": list(CLASS_NAMES),
        "generation_seconds": time.perf_counter() - started,
        "training_time_indices": selected.tolist(),
        "supervision": "Only unpaired snapshots and hard labels at configured training_times.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_dataset(output, snapshots, times, paths, settings)
    return manifest


def plot_dataset(output, snapshots, times, paths, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ["#527DB5", "#248E89", "#C78B24", "#8865B8", "#CE6173"]
    fig, axis = plt.subplots(figsize=(10, 7))
    x_limits = (-0.05, 2.55)
    y_limits = (-0.80, 1.32)
    gx, gy = np.meshgrid(np.linspace(*x_limits, 500), np.linspace(*y_limits, 400))
    grid = np.stack([gx, gy], axis=-1)
    axis.pcolormesh(
        gx,
        gy,
        hard_labels(grid, settings),
        cmap=ListedColormap(colors),
        alpha=0.09,
        shading="auto",
        vmin=0,
        vmax=4,
    )
    for time_value in settings["training_times"]:
        index = int(np.flatnonzero(np.isclose(times, time_value))[0])
        sample = snapshots[index]
        label = hard_labels(sample, settings)
        axis.scatter(*sample.T, c=label, cmap=ListedColormap(colors), s=3, alpha=0.22, vmin=0, vmax=4)
    for path in paths[: min(80, len(paths))]:
        axis.plot(*path.T, color="#536273", lw=0.5, alpha=0.13)
    dense_time = np.linspace(settings["start_time"], settings["end_time"], 501)
    means, _ = means_and_derivatives(dense_time, settings)
    for index, color in enumerate([colors[1], colors[3], colors[4]]):
        axis.plot(*means[:, index].T, color=color, lw=2)
    label_settings = settings["labels"]
    scale = float(settings["coordinate_scale"])
    for boundary in (label_settings["first_x_boundary"], label_settings["second_x_boundary"]):
        axis.axvline(float(boundary) * scale, ls="--", color="#6B7280", lw=1)
    axis.axhline(float(label_settings["lower_y_boundary"]) * scale, ls="--", color="#6B7280", lw=1)
    axis.axhline(float(label_settings["children_y_boundary"]) * scale, ls="--", color="#6B7280", lw=1)
    axis.set(xlim=x_limits, ylim=y_limits, xlabel="developmental progress", ylabel="phenotype")
    axis.set_title("Configurable curved-v5 synthetic process")
    axis.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output / "dataset.png", dpi=170)
    plt.close(fig)

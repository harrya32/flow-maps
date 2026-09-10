"""Evaluation utilities for the isolated Maizels stochastic experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import maizels
from . import ssfm_brownian
from . import wasserstein


def _sample_indices(rng: np.random.Generator, n: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=maximum, replace=False)).astype(np.int64)


def _time_tag(timepoint: str) -> str:
    return str(timepoint).replace(".", "p")


def sample_pushforward(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
) -> np.ndarray:
    """Apply one direct SSFM draw to a population."""
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    n = x_jax.shape[0]
    s = jnp.full((n,), float(start_time), dtype=x_jax.dtype)
    t = jnp.full((n,), float(end_time), dtype=x_jax.dtype)
    horizon = t - s
    coefficients = ssfm_brownian.sample_legendre_coefficients(
        key,
        horizon,
        n_coefficients=n_coefficients,
        data_dim=x_jax.shape[-1],
    )
    prediction, _ = model.apply({"params": params}, s, t, x_jax, coefficients)
    return np.asarray(jax.device_get(prediction), dtype=np.float32)


def sample_composed_pushforward(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Compose stochastic flow maps over equal subintervals.

    Each subinterval receives an independent Brownian segment. This is the
    stochastic analogue of the deterministic experiment's multi-step
    ``flowmap`` sampler; it never evaluates an Euler discretization of a
    velocity field.
    """
    prediction, _ = _sample_composed(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=False,
    )
    return prediction


def sample_composed_path(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Return the states after every step of a composed stochastic map."""
    _, path = _sample_composed(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=True,
    )
    return path


def _sample_composed(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
    return_path: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run equal-subinterval stochastic maps, optionally retaining the path."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    n = x_jax.shape[0]
    times = jnp.linspace(
        float(start_time), float(end_time), n_steps + 1, dtype=x_jax.dtype
    )
    step_keys = jax.random.split(key, n_steps)

    def step(current, inputs):
        left, right, step_key = inputs
        s = jnp.full((n,), left, dtype=x_jax.dtype)
        t = jnp.full((n,), right, dtype=x_jax.dtype)
        coefficients = ssfm_brownian.sample_legendre_coefficients(
            step_key,
            t - s,
            n_coefficients=n_coefficients,
            data_dim=x_jax.shape[-1],
        )
        prediction, _ = model.apply({"params": params}, s, t, current, coefficients)
        return prediction, prediction if return_path else None

    prediction, states = jax.lax.scan(
        step,
        x_jax,
        (times[:-1], times[1:], step_keys),
    )
    prediction_np = np.asarray(jax.device_get(prediction), dtype=np.float32)
    if states is None:
        return prediction_np, None
    path_np = np.asarray(jax.device_get(jnp.swapaxes(states, 0, 1)), dtype=np.float32)
    return prediction_np, path_np


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * (x @ y.T)
    )
    return np.maximum(result, 0.0)


def rbf_mmd2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Biased RBF MMD² with a reproducible median-distance bandwidth."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y], axis=0)
    indices = _sample_indices(rng, combined.shape[0], 512)
    calibration = _squared_distances(combined[indices], combined[indices])
    positive = calibration[calibration > 0]
    bandwidth2 = float(np.median(positive)) if positive.size else 1.0
    bandwidth2 = max(bandwidth2, 1e-8)

    def kernel(a, b):
        return np.exp(-_squared_distances(a, b) / (2.0 * bandwidth2))

    value = kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
    return float(max(value, 0.0))


def _interval_source(cfg, timepoint: str) -> Tuple[str, str]:
    return maizels.retained_interval_for_timepoint(cfg, timepoint)


def _heldout_timepoints(cfg) -> Sequence[str]:
    return tuple(str(value) for value in cfg.problem.evaluation_timepoints)


def _mean_or_absent(metrics: Dict[str, float], key: str, values: Iterable[float]):
    values = list(values)
    if values:
        metrics[key] = float(np.mean(values))


def distribution_metrics(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_source_points: Optional[int] = None,
    max_target_points: Optional[int] = None,
    timepoints: Optional[Sequence[str]] = None,
    n_noise_draws: Optional[int] = None,
) -> Tuple[
    Dict[str, float],
    Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Compare direct and composed SSFM samples with omitted-day populations."""
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_source_points = int(
        cfg.evaluation.max_source_points
        if max_source_points is None
        else max_source_points
    )
    max_target_points = int(
        cfg.evaluation.max_target_points
        if max_target_points is None
        else max_target_points
    )
    n_draws = int(
        cfg.evaluation.n_noise_draws if n_noise_draws is None else n_noise_draws
    )
    if n_draws <= 0:
        raise ValueError("evaluation.n_noise_draws must be positive.")
    flowmap_n_steps = int(cfg.evaluation.flowmap_n_steps)
    if flowmap_n_steps <= 0:
        raise ValueError("evaluation.flowmap_n_steps must be positive.")

    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    val_times = set(str(value) for value in cfg.problem.hparam_val_times)
    samplers = ("direct", "flowmap")
    all_emd = {name: [] for name in samplers}
    val_emd = {name: [] for name in samplers}
    test_emd = {name: [] for name in samplers}
    metrics: Dict[str, float] = {}
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    selected_timepoints = (
        _heldout_timepoints(cfg)
        if timepoints is None
        else tuple(str(value) for value in timepoints)
    )
    unavailable = sorted(set(selected_timepoints) - set(_heldout_timepoints(cfg)))
    if unavailable:
        raise ValueError(
            f"Requested stochastic evaluation times are not held out: {unavailable}."
        )
    for timepoint in selected_timepoints:
        source_time, _ = _interval_source(cfg, timepoint)
        source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
        target_all = np.asarray(pools[timepoint]["x"], dtype=np.float32)
        source = source_all[
            _sample_indices(rng, source_all.shape[0], max_source_points)
        ]
        target = target_all[
            _sample_indices(rng, target_all.shape[0], max_target_points)
        ]
        start = maizels.normalized_time(source_time, cfg)
        end = maizels.normalized_time(timepoint, cfg)
        draw_emd = {name: [] for name in samplers}
        draw_mmd = {name: [] for name in samplers}
        first_prediction = {}
        for draw in range(n_draws):
            key, direct_key, flowmap_key = jax.random.split(key, 3)
            predictions = {
                "direct": sample_pushforward(
                    model,
                    params,
                    source,
                    start,
                    end,
                    direct_key,
                    n_coefficients=int(cfg.ssfm.n_coefficients),
                ),
                "flowmap": sample_composed_pushforward(
                    model,
                    params,
                    source,
                    start,
                    end,
                    flowmap_key,
                    n_coefficients=int(cfg.ssfm.n_coefficients),
                    n_steps=flowmap_n_steps,
                ),
            }
            for sampler, prediction in predictions.items():
                if sampler not in first_prediction:
                    first_prediction[sampler] = prediction
                draw_emd[sampler].append(wasserstein.exact_emd(prediction, target))
                draw_mmd[sampler].append(rbf_mmd2(prediction, target, seed=seed + draw))

        tag = _time_tag(timepoint)
        for sampler in samplers:
            emd = float(np.mean(draw_emd[sampler]))
            metrics[f"final_eval/{tag}_{sampler}_emd"] = emd
            metrics[f"final_eval/{tag}_{sampler}_emd_std_over_noise"] = float(
                np.std(draw_emd[sampler])
            )
            metrics[f"final_eval/{tag}_{sampler}_rbf_mmd2"] = float(
                np.mean(draw_mmd[sampler])
            )
            all_emd[sampler].append(emd)
            (val_emd if timepoint in val_times else test_emd)[sampler].append(emd)

        # Backwards-compatible names from the initial direct-only evaluator.
        metrics[f"final_eval/{tag}_ssfm_emd"] = metrics[f"final_eval/{tag}_direct_emd"]
        metrics[f"final_eval/{tag}_ssfm_emd_std_over_noise"] = metrics[
            f"final_eval/{tag}_direct_emd_std_over_noise"
        ]
        metrics[f"final_eval/{tag}_ssfm_rbf_mmd2"] = metrics[
            f"final_eval/{tag}_direct_rbf_mmd2"
        ]
        plot_data[timepoint] = (
            target,
            first_prediction["direct"],
            first_prediction["flowmap"],
        )

    for sampler in samplers:
        _mean_or_absent(metrics, f"final_eval/{sampler}_mean_emd", all_emd[sampler])
        _mean_or_absent(
            metrics,
            f"final_eval/{sampler}_mean_emd_hparam_val_times",
            val_emd[sampler],
        )
        _mean_or_absent(
            metrics,
            f"final_eval/{sampler}_mean_emd_test_times",
            test_emd[sampler],
        )

    # Preserve the former aggregate keys as aliases for direct sampling.
    for suffix in ("mean_emd", "mean_emd_hparam_val_times", "mean_emd_test_times"):
        direct_key = f"final_eval/direct_{suffix}"
        if direct_key in metrics:
            metrics[f"final_eval/ssfm_{suffix}"] = metrics[direct_key]
    return metrics, plot_data


def pushforward_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample plotting populations without running EMD or MMD calculations."""
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_points = int(cfg.logging.maizels.plot_bs if max_points is None else max_points)
    flowmap_n_steps = int(cfg.evaluation.flowmap_n_steps)
    if max_points < 0:
        raise ValueError("Plot max_points must be non-negative (zero means all).")
    if flowmap_n_steps <= 0:
        raise ValueError("evaluation.flowmap_n_steps must be positive.")

    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    plot_data = {}
    for timepoint in _heldout_timepoints(cfg):
        source_time, _ = _interval_source(cfg, timepoint)
        source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
        target_all = np.asarray(pools[timepoint]["x"], dtype=np.float32)
        source = source_all[_sample_indices(rng, source_all.shape[0], max_points)]
        target = target_all[_sample_indices(rng, target_all.shape[0], max_points)]
        start = maizels.normalized_time(source_time, cfg)
        end = maizels.normalized_time(timepoint, cfg)
        key, direct_key, flowmap_key = jax.random.split(key, 3)
        direct = sample_pushforward(
            model,
            params,
            source,
            start,
            end,
            direct_key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
        )
        flowmap = sample_composed_pushforward(
            model,
            params,
            source,
            start,
            end,
            flowmap_key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=flowmap_n_steps,
        )
        plot_data[timepoint] = (target, direct, flowmap)
    return plot_data


def lineage_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Score composed paths from held-out D3 cells with both classifiers."""
    seed = int(cfg.evaluation.seed + 211 if seed is None else seed)
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source = pools[str(cfg.problem.source_time)]
    source_x_all = np.asarray(source["holdout_x"], dtype=np.float32)
    source_types_all = np.asarray(source["holdout_types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(
        rng,
        source_x_all.shape[0],
        int(cfg.evaluation.lineage_max_source_points),
    )
    source_x = source_x_all[indices]
    if source_x.shape[0] == 0:
        return {}
    type_to_id = maizels.class_to_id_map(maizels.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types_all[indices]],
        dtype=np.int32,
    )
    key = jax.random.PRNGKey(seed)
    classifiers = {
        "schedule_classifier": Path(cfg.problem.classifier_path),
        "full_data_classifier": Path(cfg.logging.maizels.full_data_classifier_path),
    }
    classifiers = {
        name: path if path.suffix == ".npz" else path.with_suffix(".npz")
        for name, path in classifiers.items()
    }
    available_classifiers = {}
    for name, path in classifiers.items():
        if path.is_file():
            available_classifiers[name] = path
        else:
            print(
                f"Skipping {name} stochastic lineage metrics because the "
                f"NumPy classifier checkpoint is absent: {path}"
            )
    if not available_classifiers:
        return {}

    n_steps = int(cfg.evaluation.lineage_n_steps)
    if n_steps <= 0:
        raise ValueError("evaluation.lineage_n_steps must be positive.")
    valid_fractions = {name: [] for name in available_classifiers}
    for _ in range(int(cfg.evaluation.n_noise_draws)):
        key, draw_key = jax.random.split(key)
        path = sample_composed_path(
            model,
            params,
            source_x,
            maizels.normalized_time(cfg.problem.source_time, cfg),
            maizels.normalized_time(cfg.problem.target_time, cfg),
            draw_key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=n_steps,
        )
        for name, classifier_path in available_classifiers.items():
            validity = maizels.check_paths_with_classifier(
                path,
                source_ids,
                classifier_path,
                prob_threshold=float(cfg.problem.classifier_prob_threshold),
                margin_threshold=float(cfg.problem.classifier_margin_threshold),
                classifier_batch_size=int(cfg.problem.classifier_batch_size),
                lineage_transition_mode=maizels.lineage_transition_mode_from_config(
                    cfg
                ),
            )
            valid_fractions[name].append(float(np.mean(validity["valid"])))

    metrics = {"final_eval/lineage_eval_source_count": float(source_x.shape[0])}
    for name, values in valid_fractions.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        prefix = f"final_eval/{name}"
        metrics[f"{prefix}/stochastic_path_valid_fraction"] = mean
        metrics[f"{prefix}/stochastic_path_valid_fraction_std_over_noise"] = std
        metrics[f"{prefix}/flowmap_valid_trajectory_pct"] = 100.0 * mean
        metrics[f"{prefix}/flowmap_invalid_trajectory_pct"] = 100.0 * (1.0 - mean)
        metrics[f"{prefix}/flowmap_valid_trajectory_pct_std_over_noise"] = 100.0 * std

    # Match the deterministic metric convention: unprefixed validity uses the
    # schedule-specific classifier; the all-day classifier has its own prefix.
    if "schedule_classifier" in valid_fractions:
        metrics["final_eval/stochastic_path_valid_fraction"] = metrics[
            "final_eval/schedule_classifier/stochastic_path_valid_fraction"
        ]
        metrics["final_eval/stochastic_path_valid_fraction_std_over_noise"] = metrics[
            "final_eval/schedule_classifier/stochastic_path_valid_fraction_std_over_noise"
        ]
        metrics["final_eval/flowmap_valid_trajectory_pct"] = metrics[
            "final_eval/schedule_classifier/flowmap_valid_trajectory_pct"
        ]
        metrics["final_eval/flowmap_invalid_trajectory_pct"] = metrics[
            "final_eval/schedule_classifier/flowmap_invalid_trajectory_pct"
        ]
    return metrics


def save_pushforward_plot(
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    flowmap_n_steps: int,
) -> None:
    """Save actual, direct, and composed SSFM populations for held-out days."""
    if not plot_data:
        return
    import matplotlib.pyplot as plt

    timepoints = list(plot_data)
    figure, axes = plt.subplots(
        len(timepoints),
        3,
        figsize=(12.0, max(2.6 * len(timepoints), 3.0)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, timepoint in enumerate(timepoints):
        actual, direct, flowmap = plot_data[timepoint]
        axes[row, 0].scatter(actual[:, 0], actual[:, 1], s=3, alpha=0.35)
        axes[row, 1].scatter(direct[:, 0], direct[:, 1], s=3, alpha=0.35)
        axes[row, 2].scatter(flowmap[:, 0], flowmap[:, 1], s=3, alpha=0.35)
        axes[row, 0].set_ylabel(timepoint)
        axes[row, 0].set_title("Actual" if row == 0 else "")
        axes[row, 1].set_title("Direct SSFM" if row == 0 else "")
        axes[row, 2].set_title(
            f"Composed SSFM ({flowmap_n_steps} steps)" if row == 0 else ""
        )
    axes[-1, 0].set_xlabel("PC1")
    axes[-1, 1].set_xlabel("PC1")
    axes[-1, 2].set_xlabel("PC1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def final_evaluation(model, params, cfg, output_dir: Path) -> Dict[str, float]:
    """Run distribution and lineage evaluation for one checkpoint."""
    distribution, plot_data = distribution_metrics(model, params, cfg)
    metrics = dict(distribution)
    metrics.update(lineage_metrics(model, params, cfg))
    if bool(cfg.evaluation.save_plot):
        save_pushforward_plot(
            plot_data,
            Path(output_dir) / "heldout_pushforwards.png",
            flowmap_n_steps=int(cfg.evaluation.flowmap_n_steps),
        )
    return metrics

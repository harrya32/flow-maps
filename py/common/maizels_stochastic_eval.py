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
) -> Tuple[Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Compare direct stochastic pushforwards with every omitted-day population."""
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

    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    val_times = set(str(value) for value in cfg.problem.hparam_val_times)
    all_emd = []
    val_emd = []
    test_emd = []
    metrics: Dict[str, float] = {}
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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
        draw_emd = []
        draw_mmd = []
        first_prediction = None
        for draw in range(n_draws):
            key, draw_key = jax.random.split(key)
            prediction = sample_pushforward(
                model,
                params,
                source,
                start,
                end,
                draw_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
            )
            if first_prediction is None:
                first_prediction = prediction
            draw_emd.append(wasserstein.exact_emd(prediction, target))
            draw_mmd.append(rbf_mmd2(prediction, target, seed=seed + draw))
        emd = float(np.mean(draw_emd))
        tag = _time_tag(timepoint)
        metrics[f"final_eval/{tag}_ssfm_emd"] = emd
        metrics[f"final_eval/{tag}_ssfm_emd_std_over_noise"] = float(np.std(draw_emd))
        metrics[f"final_eval/{tag}_ssfm_rbf_mmd2"] = float(np.mean(draw_mmd))
        all_emd.append(emd)
        (val_emd if timepoint in val_times else test_emd).append(emd)
        plot_data[timepoint] = (target, first_prediction)

    _mean_or_absent(metrics, "final_eval/ssfm_mean_emd", all_emd)
    _mean_or_absent(
        metrics,
        "final_eval/ssfm_mean_emd_hparam_val_times",
        val_emd,
    )
    _mean_or_absent(metrics, "final_eval/ssfm_mean_emd_test_times", test_emd)
    return metrics, plot_data


def semigroup_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Measure direct-vs-two-half-step consistency using Chen-coupled noise."""
    seed = int(cfg.evaluation.seed + 101 if seed is None else seed)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    values = []
    metrics = {}
    retained = tuple(str(value) for value in cfg.problem.retained_timepoints)
    for left, right in zip(retained[:-1], retained[1:]):
        source_all = np.asarray(pools[left]["x"], dtype=np.float32)
        source = source_all[
            _sample_indices(
                rng,
                source_all.shape[0],
                int(cfg.evaluation.max_source_points),
            )
        ]
        x = jnp.asarray(source)
        n = x.shape[0]
        s_value = maizels.normalized_time(left, cfg)
        t_value = maizels.normalized_time(right, cfg)
        midpoint_value = 0.5 * (s_value + t_value)
        half_horizon = jnp.full((n,), 0.5 * (t_value - s_value), dtype=jnp.float32)
        key, left_key, right_key = jax.random.split(key, 3)
        left_coefficients = ssfm_brownian.sample_legendre_coefficients(
            left_key,
            half_horizon,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            data_dim=int(cfg.problem.d),
        )
        right_coefficients = ssfm_brownian.sample_legendre_coefficients(
            right_key,
            half_horizon,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            data_dim=int(cfg.problem.d),
        )
        full_coefficients = ssfm_brownian.chen_combine_halves(
            left_coefficients, right_coefficients
        )
        s = jnp.full((n,), s_value, dtype=jnp.float32)
        midpoint = jnp.full((n,), midpoint_value, dtype=jnp.float32)
        t = jnp.full((n,), t_value, dtype=jnp.float32)
        direct, _ = model.apply({"params": params}, s, t, x, full_coefficients)
        middle, _ = model.apply({"params": params}, s, midpoint, x, left_coefficients)
        composed, _ = model.apply(
            {"params": params}, midpoint, t, middle, right_coefficients
        )
        defect = float(jax.device_get(jnp.mean(jnp.square(direct - composed))))
        metrics[f"final_eval/{_time_tag(left)}_to_{_time_tag(right)}_semigroup_mse"] = (
            defect
        )
        values.append(defect)
    _mean_or_absent(metrics, "final_eval/semigroup_mse_mean", values)
    return metrics


def lineage_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Evaluate complete stochastic paths with the frozen all-day classifier."""
    configured_classifier = Path(cfg.logging.maizels.full_data_classifier_path)
    classifier_path = (
        configured_classifier
        if configured_classifier.suffix == ".npz"
        else configured_classifier.with_suffix(".npz")
    )
    if not classifier_path.is_file():
        print(
            "Skipping stochastic lineage evaluation because the NumPy "
            f"classifier checkpoint is absent: {classifier_path}"
        )
        return {}
    seed = int(cfg.evaluation.seed + 211 if seed is None else seed)
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source = pools[str(cfg.problem.source_time)]
    source_x_all = np.asarray(source["x"], dtype=np.float32)
    source_types_all = np.asarray(source["types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(
        rng, source_x_all.shape[0], int(cfg.evaluation.max_source_points)
    )
    source_x = source_x_all[indices]
    type_to_id = maizels.class_to_id_map(maizels.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types_all[indices]],
        dtype=np.int32,
    )
    key = jax.random.PRNGKey(seed)
    valid_fractions = []
    timepoints = tuple(str(value) for value in cfg.problem.timepoint_order)
    for _ in range(int(cfg.evaluation.n_noise_draws)):
        current = source_x
        path = [current]
        for left, right in zip(timepoints[:-1], timepoints[1:]):
            key, draw_key = jax.random.split(key)
            current = sample_pushforward(
                model,
                params,
                current,
                maizels.normalized_time(left, cfg),
                maizels.normalized_time(right, cfg),
                draw_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
            )
            path.append(current)
        validity = maizels.check_paths_with_classifier(
            np.stack(path, axis=1),
            source_ids,
            classifier_path,
            prob_threshold=0.0,
            margin_threshold=0.0,
            lineage_transition_mode=maizels.lineage_transition_mode_from_config(cfg),
        )
        valid_fractions.append(float(np.mean(validity["valid"])))
    return {
        "final_eval/stochastic_path_valid_fraction": float(np.mean(valid_fractions)),
        "final_eval/stochastic_path_valid_fraction_std_over_noise": float(
            np.std(valid_fractions)
        ),
    }


def save_pushforward_plot(
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Save actual-vs-pushforward PC1/PC2 panels for held-out days."""
    if not plot_data:
        return
    import matplotlib.pyplot as plt

    timepoints = list(plot_data)
    figure, axes = plt.subplots(
        len(timepoints),
        2,
        figsize=(8.0, max(2.6 * len(timepoints), 3.0)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, timepoint in enumerate(timepoints):
        actual, prediction = plot_data[timepoint]
        axes[row, 0].scatter(actual[:, 0], actual[:, 1], s=3, alpha=0.35)
        axes[row, 1].scatter(prediction[:, 0], prediction[:, 1], s=3, alpha=0.35)
        axes[row, 0].set_ylabel(timepoint)
        axes[row, 0].set_title("Actual" if row == 0 else "")
        axes[row, 1].set_title("SSFM pushforward" if row == 0 else "")
    axes[-1, 0].set_xlabel("PC1")
    axes[-1, 1].set_xlabel("PC1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def final_evaluation(model, params, cfg, output_dir: Path) -> Dict[str, float]:
    """Run distribution, semigroup, and lineage evaluation for one checkpoint."""
    distribution, plot_data = distribution_metrics(model, params, cfg)
    metrics = dict(distribution)
    metrics.update(semigroup_metrics(model, params, cfg))
    metrics.update(lineage_metrics(model, params, cfg))
    if bool(cfg.evaluation.save_plot):
        save_pushforward_plot(
            plot_data,
            Path(output_dir) / "heldout_pushforwards.png",
        )
    return metrics

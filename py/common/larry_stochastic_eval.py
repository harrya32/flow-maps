"""Evaluation for LARRY strong stochastic flow maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import jax
import numpy as np

from . import larry
from . import maizels
from . import maizels_stochastic_eval as shared_eval
from . import wasserstein


def _sample_indices(rng: np.random.Generator, n: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=maximum, replace=False)).astype(np.int64)


def _classifier_npz(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def _composed_samples_in_batches(
    model,
    params,
    source: np.ndarray,
    start_time: float,
    end_time: float,
    key,
    *,
    n_coefficients: int,
    n_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Sample composed SSFM endpoints without materializing one huge JAX batch."""
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Stochastic source samples must be a two-dimensional array.")
    if source.shape[0] == 0:
        return source.copy()
    batch_size = source.shape[0] if int(batch_size) <= 0 else int(batch_size)
    starts = tuple(range(0, source.shape[0], batch_size))
    keys = jax.random.split(key, len(starts))
    predictions = []
    for chunk_index, start in enumerate(starts):
        chunk = source[start : start + batch_size]
        valid = chunk.shape[0]
        # Keeping the last batch at the same shape avoids an extra JAX compile.
        if valid < batch_size and source.shape[0] > batch_size:
            padding = np.repeat(chunk[-1:], batch_size - valid, axis=0)
            model_input = np.concatenate([chunk, padding], axis=0)
        else:
            model_input = chunk
        prediction = shared_eval.sample_composed_pushforward(
            model,
            params,
            model_input,
            start_time,
            end_time,
            keys[chunk_index],
            n_coefficients=n_coefficients,
            n_steps=n_steps,
        )
        predictions.append(np.asarray(prediction[:valid], dtype=np.float32))
    return np.concatenate(predictions, axis=0)


def _euler_maruyama_samples_in_batches(
    model,
    params,
    source: np.ndarray,
    start_time: float,
    end_time: float,
    key,
    *,
    n_coefficients: int,
    n_steps: int,
    batch_size: int,
) -> np.ndarray:
    """Roll local EM endpoints without materializing one huge JAX batch."""
    source = np.asarray(source, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Stochastic source samples must be a two-dimensional array.")
    if source.shape[0] == 0:
        return source.copy()
    batch_size = source.shape[0] if int(batch_size) <= 0 else int(batch_size)
    starts = tuple(range(0, source.shape[0], batch_size))
    keys = jax.random.split(key, len(starts))
    predictions = []
    for chunk_index, start in enumerate(starts):
        chunk = source[start : start + batch_size]
        valid = chunk.shape[0]
        if valid < batch_size and source.shape[0] > batch_size:
            padding = np.repeat(chunk[-1:], batch_size - valid, axis=0)
            model_input = np.concatenate([chunk, padding], axis=0)
        else:
            model_input = chunk
        prediction = shared_eval.sample_euler_maruyama_pushforward(
            model,
            params,
            model_input,
            start_time,
            end_time,
            keys[chunk_index],
            n_coefficients=n_coefficients,
            n_steps=n_steps,
        )
        predictions.append(np.asarray(prediction[:valid], dtype=np.float32))
    return np.concatenate(predictions, axis=0)


def distribution_metrics(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_source_points: Optional[int] = None,
    max_target_points: Optional[int] = None,
    n_noise_draws: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Evaluate valid D2-to-D4 stochastic samples against held-out D4."""
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
    use_euler_maruyama = shared_eval.uses_euler_maruyama_evaluation(cfg)
    n_steps = int(cfg.evaluation.flowmap_n_steps)
    if not use_euler_maruyama and n_steps <= 0:
        raise ValueError("evaluation.flowmap_n_steps must be positive.")

    pools = larry.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source_time = larry.format_timepoint(cfg.problem.source_time)
    target_time = larry.format_timepoint(cfg.problem.heldout_timepoint)
    rng = np.random.default_rng(seed)
    source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
    target_all = np.asarray(pools[target_time]["x"], dtype=np.float32)
    source = source_all[_sample_indices(rng, source_all.shape[0], max_source_points)]
    target = target_all[_sample_indices(rng, target_all.shape[0], max_target_points)]
    start = larry.normalized_time(source_time, cfg)
    end = larry.normalized_time(target_time, cfg)
    key = jax.random.PRNGKey(seed)
    emd_values = []
    mmd_values = []
    first_prediction = None
    for draw in range(n_draws):
        key, draw_key = jax.random.split(key)
        if use_euler_maruyama:
            prediction = shared_eval.sample_euler_maruyama_pushforward(
                model,
                params,
                source,
                start,
                end,
                draw_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=shared_eval.euler_maruyama_n_steps(cfg),
            )
        else:
            prediction = shared_eval.sample_composed_pushforward(
                model,
                params,
                source,
                start,
                end,
                draw_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=n_steps,
            )
        if first_prediction is None:
            first_prediction = prediction
        emd_values.append(wasserstein.exact_emd(prediction, target))
        mmd_values.append(shared_eval.rbf_mmd2(prediction, target, seed=seed + draw))

    tag = f"{source_time.lower()}_to_{target_time.lower()}"
    sampler = "euler_maruyama" if use_euler_maruyama else "flowmap"
    mean_emd = float(np.mean(emd_values))
    metrics = {
        "final_eval/source_count": float(source.shape[0]),
        "final_eval/target_count": float(target.shape[0]),
        f"final_eval/{tag}_{sampler}_emd": mean_emd,
        f"final_eval/{tag}_{sampler}_emd_std_over_noise": float(np.std(emd_values)),
        f"final_eval/{tag}_{sampler}_rbf_mmd2": float(np.mean(mmd_values)),
        f"final_eval/{sampler}_mean_emd": mean_emd,
        "final_eval/evaluation_mean_emd": mean_emd,
        # Preserve the sampler-neutral benchmark key.
        "mfm/test_EMD": mean_emd,
        f"mfm/test_EMD_{sampler}": mean_emd,
    }
    print(
        "Held-out LARRY stochastic EMD evaluation: "
        f"{source_time}->{target_time}, source_n={source.shape[0]}, "
        f"target_n={target.shape[0]}, draws={n_draws}, "
        f"{sampler}_EMD={mean_emd:.8g}"
    )
    return metrics, {target_time: (target, first_prediction)}


def _clone_target_times(cfg) -> Tuple[str, ...]:
    values = tuple(
        larry.format_timepoint(value) for value in cfg.evaluation.clone_target_times
    )
    if not values:
        raise ValueError("evaluation.clone_target_times cannot be empty.")
    invalid = sorted(set(values) - set(larry.TIMEPOINTS))
    if invalid:
        raise ValueError(f"Unknown LARRY clone target times: {invalid}.")
    source = larry.format_timepoint(cfg.evaluation.clone_source_time)
    if source in values:
        raise ValueError("A clone target time must differ from its source time.")
    return values


def _clone_target_groups(
    target_x: np.ndarray,
    target_ids: np.ndarray,
    eligible: np.ndarray,
    *,
    maximum: int,
    rng: np.random.Generator,
) -> Dict[int, np.ndarray]:
    groups = {}
    for clone_id in eligible.tolist():
        values = np.asarray(target_x[target_ids == clone_id], dtype=np.float32)
        if maximum > 0 and values.shape[0] > maximum:
            indices = _sample_indices(rng, values.shape[0], maximum)
            values = values[indices]
        groups[int(clone_id)] = values
    return groups


def _repeated_clone_sources(
    source_x: np.ndarray,
    source_ids: np.ndarray,
    eligible: np.ndarray,
    samples_per_source: int,
) -> Tuple[np.ndarray, Dict[int, slice], np.ndarray]:
    blocks = []
    spans = {}
    source_counts = []
    cursor = 0
    for clone_id in eligible.tolist():
        values = np.asarray(source_x[source_ids == clone_id], dtype=np.float32)
        source_counts.append(values.shape[0])
        repeated = np.repeat(values, int(samples_per_source), axis=0)
        blocks.append(repeated)
        spans[int(clone_id)] = slice(cursor, cursor + repeated.shape[0])
        cursor += repeated.shape[0]
    return np.concatenate(blocks, axis=0), spans, np.asarray(source_counts)


def _clone_wasserstein_for_target(
    model,
    params,
    cfg,
    data,
    target_time: str,
    *,
    seed: int,
) -> Dict[str, float]:
    source_time = larry.format_timepoint(cfg.evaluation.clone_source_time)
    target_time = larry.format_timepoint(target_time)
    clone_ids = np.asarray(data["clone_ids"], dtype=np.int64)
    source_mask = (data["timepoints"] == source_time) & (clone_ids >= 0)
    target_mask = (data["timepoints"] == target_time) & (clone_ids >= 0)
    source_x = np.asarray(data["x"][source_mask], dtype=np.float32)
    source_ids = clone_ids[source_mask]
    target_x = np.asarray(data["x"][target_mask], dtype=np.float32)
    target_ids = clone_ids[target_mask]

    min_source = max(1, int(cfg.evaluation.clone_min_source_cells))
    min_target = max(1, int(cfg.evaluation.clone_min_target_cells))
    source_unique, source_counts = np.unique(source_ids, return_counts=True)
    target_unique, target_counts = np.unique(target_ids, return_counts=True)
    source_count = dict(zip(source_unique.tolist(), source_counts.tolist()))
    target_count = dict(zip(target_unique.tolist(), target_counts.tolist()))
    eligible = np.asarray(
        sorted(
            clone_id
            for clone_id in set(source_count).intersection(target_count)
            if source_count[clone_id] >= min_source
            and target_count[clone_id] >= min_target
        ),
        dtype=np.int64,
    )
    tag = f"{source_time.lower()}_to_{target_time.lower()}"
    prefix = f"final_eval/clone_{tag}"
    if eligible.size == 0:
        return {
            f"{prefix}_available": 0.0,
            f"{prefix}_eligible_clone_count": 0.0,
        }

    samples_per_source = int(cfg.evaluation.clone_samples_per_source)
    n_draws = int(cfg.evaluation.clone_n_noise_draws)
    n_steps = int(cfg.evaluation.clone_flowmap_n_steps)
    batch_size = int(cfg.evaluation.clone_batch_size)
    use_euler_maruyama = shared_eval.uses_euler_maruyama_evaluation(cfg)
    if (
        samples_per_source <= 0
        or n_draws <= 0
        or (not use_euler_maruyama and n_steps <= 0)
        or batch_size <= 0
    ):
        raise ValueError(
            "Clone samples, noise draws, flow-map steps, and batch size must be "
            "positive."
        )

    rng = np.random.default_rng(seed)
    target_groups = _clone_target_groups(
        target_x,
        target_ids,
        eligible,
        maximum=int(cfg.evaluation.clone_max_target_cells),
        rng=rng,
    )
    repeated_source, source_spans, evaluated_source_counts = _repeated_clone_sources(
        source_x,
        source_ids,
        eligible,
        samples_per_source,
    )
    evaluated_target_counts = np.asarray(
        [target_groups[int(clone_id)].shape[0] for clone_id in eligible],
        dtype=np.float64,
    )
    source_weights = evaluated_source_counts.astype(np.float64)
    key = jax.random.PRNGKey(seed)
    distances_by_draw = []
    for _ in range(n_draws):
        key, draw_key = jax.random.split(key)
        sampler_args = (
            model,
            params,
            repeated_source,
            larry.normalized_time(source_time, cfg),
            larry.normalized_time(target_time, cfg),
            draw_key,
        )
        if use_euler_maruyama:
            prediction = _euler_maruyama_samples_in_batches(
                *sampler_args,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=shared_eval.euler_maruyama_n_steps(cfg),
                batch_size=batch_size,
            )
        else:
            prediction = _composed_samples_in_batches(
                *sampler_args,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=n_steps,
                batch_size=batch_size,
            )
        distances_by_draw.append(
            [
                wasserstein.exact_emd(
                    prediction[source_spans[int(clone_id)]],
                    target_groups[int(clone_id)],
                )
                for clone_id in eligible
            ]
        )

    distances = np.asarray(distances_by_draw, dtype=np.float64)
    mean_per_clone = np.mean(distances, axis=0)
    draw_macro = np.mean(distances, axis=1)
    draw_median = np.median(distances, axis=1)
    draw_source_weighted = np.average(distances, axis=1, weights=source_weights)
    draw_target_weighted = np.average(
        distances, axis=1, weights=evaluated_target_counts
    )
    metrics = {
        f"{prefix}_available": 1.0,
        f"{prefix}_wasserstein_macro": float(np.mean(mean_per_clone)),
        f"{prefix}_wasserstein_macro_std_over_noise": float(np.std(draw_macro)),
        f"{prefix}_wasserstein_median": float(np.median(mean_per_clone)),
        f"{prefix}_wasserstein_median_std_over_noise": float(np.std(draw_median)),
        f"{prefix}_wasserstein_source_weighted": float(
            np.average(mean_per_clone, weights=source_weights)
        ),
        f"{prefix}_wasserstein_source_weighted_std_over_noise": float(
            np.std(draw_source_weighted)
        ),
        f"{prefix}_wasserstein_target_weighted": float(
            np.average(mean_per_clone, weights=evaluated_target_counts)
        ),
        f"{prefix}_wasserstein_target_weighted_std_over_noise": float(
            np.std(draw_target_weighted)
        ),
        f"{prefix}_per_clone_noise_std_mean": float(np.mean(np.std(distances, axis=0))),
        f"{prefix}_eligible_clone_count": float(eligible.size),
        f"{prefix}_source_cell_count": float(source_weights.sum()),
        f"{prefix}_target_cell_count": float(evaluated_target_counts.sum()),
        f"{prefix}_generated_sample_count_per_draw": float(repeated_source.shape[0]),
        f"{prefix}_samples_per_source": float(samples_per_source),
        f"{prefix}_noise_draw_count": float(n_draws),
    }
    print(
        "Stochastic clone-conditioned Wasserstein: "
        f"{source_time}->{target_time}, clones={eligible.size}, "
        f"source_cells={int(source_weights.sum())}, "
        f"target_cells={int(evaluated_target_counts.sum())}, "
        f"samples/source={samples_per_source}, draws={n_draws}, "
        f"macro_W1={metrics[f'{prefix}_wasserstein_macro']:.8g}"
    )
    return metrics


def clone_wasserstein_metrics(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate stochastic clone-conditional distributions at D4 and D6."""
    if not bool(cfg.evaluation.clone_wasserstein_enabled):
        return {}
    base_seed = int(cfg.evaluation.clone_seed if seed is None else seed)
    data = larry.all_timepoint_data(
        cfg.problem.dataset_location,
        representation=str(cfg.problem.larry_representation),
        n_pcs=int(cfg.problem.n_pcs),
    )
    metrics = {}
    for index, target_time in enumerate(_clone_target_times(cfg)):
        metrics.update(
            _clone_wasserstein_for_target(
                model,
                params,
                cfg,
                data,
                target_time,
                seed=base_seed + 1009 * index,
            )
        )
    return metrics


def _available_lineage_classifiers(cfg) -> Dict[str, Path]:
    configured = {
        "schedule_classifier": _classifier_npz(cfg.problem.classifier_path),
        "full_data_classifier": _classifier_npz(
            cfg.logging.maizels.full_data_classifier_path
        ),
    }
    available = {}
    for name, path in configured.items():
        if path.is_file():
            available[name] = path
        else:
            print(
                f"Skipping {name} stochastic lineage metrics because the "
                f"NumPy classifier checkpoint is absent: {path}"
            )
    return available


def _lineage_source(cfg, seed: int, maximum: int):
    pools = larry.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source_time = larry.format_timepoint(cfg.problem.source_time)
    source_pool = pools[source_time]
    source_all = np.asarray(source_pool["holdout_x"], dtype=np.float32)
    source_types = np.asarray(source_pool["holdout_types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(rng, source_all.shape[0], maximum)
    type_to_id = maizels.class_to_id_map(larry.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types[indices]],
        dtype=np.int32,
    )
    return pools, source_all[indices], source_ids


def lineage_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Score valid stochastic D2-to-D6 paths with both classifiers."""
    seed = int(cfg.evaluation.seed + 211 if seed is None else seed)
    _, source_x, source_ids = _lineage_source(
        cfg, seed, int(cfg.evaluation.lineage_max_source_points)
    )
    if source_x.shape[0] == 0:
        return {}
    classifiers = _available_lineage_classifiers(cfg)
    if not classifiers:
        return {
            "final_eval/violation_metrics_available": 0.0,
            "final_eval/violation_metrics_skipped_no_classifier": 1.0,
        }
    use_euler_maruyama = shared_eval.uses_euler_maruyama_evaluation(cfg)
    n_steps = int(cfg.evaluation.lineage_n_steps)
    if not use_euler_maruyama and n_steps <= 0:
        raise ValueError("evaluation.lineage_n_steps must be positive.")
    sampler_tag = "euler_maruyama" if use_euler_maruyama else "flowmap"

    key = jax.random.PRNGKey(seed)
    valid_fractions = {name: [] for name in classifiers}
    for _ in range(int(cfg.evaluation.n_noise_draws)):
        key, draw_key = jax.random.split(key)
        path = shared_eval.sample_evaluation_path(
            model,
            params,
            source_x,
            larry.normalized_time(cfg.problem.source_time, cfg),
            larry.normalized_time(cfg.problem.target_time, cfg),
            draw_key,
            cfg,
            n_steps=n_steps,
        )
        for name, classifier_path in classifiers.items():
            validity = larry.check_paths_with_classifier(
                path,
                source_ids,
                classifier_path,
                prob_threshold=float(cfg.problem.classifier_prob_threshold),
                margin_threshold=float(cfg.problem.classifier_margin_threshold),
                classifier_batch_size=int(cfg.problem.classifier_batch_size),
                lineage_transition_mode=larry.lineage_transition_mode_from_config(cfg),
            )
            valid_fractions[name].append(float(np.mean(validity["valid"])))

    metrics = {
        "final_eval/violation_metrics_available": 1.0,
        "final_eval/lineage_eval_source_count": float(source_x.shape[0]),
    }
    for name, values in valid_fractions.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        prefix = f"final_eval/{name}"
        metrics[f"{prefix}/stochastic_path_valid_fraction"] = mean
        metrics[f"{prefix}/stochastic_path_valid_fraction_std_over_noise"] = std
        metrics[f"{prefix}/{sampler_tag}_valid_trajectory_pct"] = 100.0 * mean
        metrics[f"{prefix}/{sampler_tag}_invalid_trajectory_pct"] = 100.0 * (1.0 - mean)
        metrics[f"{prefix}/{sampler_tag}_valid_trajectory_pct_std_over_noise"] = (
            100.0 * std
        )

    # The all-days classifier is the held-out evaluator used for the headline
    # violation metric. Fall back to the D2/D6 classifier only if unavailable.
    primary = (
        "full_data_classifier"
        if "full_data_classifier" in valid_fractions
        else "schedule_classifier"
    )
    metrics["final_eval/stochastic_path_valid_fraction"] = metrics[
        f"final_eval/{primary}/stochastic_path_valid_fraction"
    ]
    metrics["final_eval/stochastic_path_valid_fraction_std_over_noise"] = metrics[
        f"final_eval/{primary}/stochastic_path_valid_fraction_std_over_noise"
    ]
    metrics[f"final_eval/{sampler_tag}_valid_trajectory_pct"] = metrics[
        f"final_eval/{primary}/{sampler_tag}_valid_trajectory_pct"
    ]
    metrics[f"final_eval/{sampler_tag}_invalid_trajectory_pct"] = metrics[
        f"final_eval/{primary}/{sampler_tag}_invalid_trajectory_pct"
    ]
    return metrics


def pushforward_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Sample one valid D2-to-D4 population for plotting."""
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_points = int(cfg.logging.maizels.plot_bs if max_points is None else max_points)
    pools = larry.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source_time = larry.format_timepoint(cfg.problem.source_time)
    target_time = larry.format_timepoint(cfg.problem.heldout_timepoint)
    rng = np.random.default_rng(seed)
    source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
    target_all = np.asarray(pools[target_time]["x"], dtype=np.float32)
    source = source_all[_sample_indices(rng, source_all.shape[0], max_points)]
    target = target_all[_sample_indices(rng, target_all.shape[0], max_points)]
    sampler_args = (
        model,
        params,
        source,
        larry.normalized_time(source_time, cfg),
        larry.normalized_time(target_time, cfg),
        jax.random.PRNGKey(seed),
    )
    if shared_eval.uses_euler_maruyama_evaluation(cfg):
        prediction = shared_eval.sample_euler_maruyama_pushforward(
            *sampler_args,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=shared_eval.euler_maruyama_n_steps(cfg),
        )
    else:
        prediction = shared_eval.sample_composed_pushforward(
            *sampler_args,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=int(cfg.evaluation.flowmap_n_steps),
        )
    return {target_time: (target, prediction)}


def save_pushforward_plot(
    plot_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    flowmap_n_steps: int,
    euler_maruyama: bool = False,
    euler_maruyama_n_steps: int = 50,
) -> None:
    """Plot actual and generated populations in the first two coordinates."""
    if not plot_data:
        return
    import matplotlib.pyplot as plt

    timepoints = list(plot_data)
    figure, axes = plt.subplots(
        len(timepoints),
        2,
        figsize=(8.0, max(3.2 * len(timepoints), 3.2)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    coordinate_labels = None
    for row, timepoint in enumerate(timepoints):
        actual, prediction = plot_data[timepoint]
        if coordinate_labels is None:
            coordinate_labels = (
                ("SPRING-1", "SPRING-2") if actual.shape[-1] == 2 else ("PC1", "PC2")
            )
        axes[row, 0].scatter(actual[:, 0], actual[:, 1], s=4, alpha=0.35)
        axes[row, 1].scatter(prediction[:, 0], prediction[:, 1], s=4, alpha=0.35)
        axes[row, 0].set_title(f"Actual {timepoint}")
        axes[row, 1].set_title(
            f"{timepoint}: "
            + (
                f"Euler--Maruyama rollout ({int(euler_maruyama_n_steps)} steps)"
                if euler_maruyama
                else f"Composed SSFM ({flowmap_n_steps} steps)"
            )
        )
    for axis in axes[-1]:
        axis.set_xlabel(coordinate_labels[0])
    for axis in axes[:, 0]:
        axis.set_ylabel(coordinate_labels[1])
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def full_data_trajectory_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
):
    """Generate valid D2-to-D6 paths and all-days-classifier validity."""
    classifier_path = _classifier_npz(cfg.logging.maizels.full_data_classifier_path)
    if not classifier_path.is_file():
        return None
    seed = int(cfg.evaluation.seed + 503 if seed is None else seed)
    plot_max = int(cfg.logging.maizels.plot_bs)
    pools, source, source_ids = _lineage_source(cfg, seed, plot_max)
    if source.shape[0] == 0:
        return None
    target = np.asarray(
        pools[larry.format_timepoint(cfg.problem.target_time)]["holdout_x"],
        dtype=np.float32,
    )
    use_euler_maruyama = shared_eval.uses_euler_maruyama_evaluation(cfg)
    generated = shared_eval.sample_evaluation_path(
        model,
        params,
        source,
        larry.normalized_time(cfg.problem.source_time, cfg),
        larry.normalized_time(cfg.problem.target_time, cfg),
        jax.random.PRNGKey(seed),
        cfg,
        n_steps=int(cfg.evaluation.lineage_n_steps),
    )
    validity = larry.check_paths_with_classifier(
        generated,
        source_ids,
        classifier_path,
        prob_threshold=float(cfg.problem.classifier_prob_threshold),
        margin_threshold=float(cfg.problem.classifier_margin_threshold),
        classifier_batch_size=int(cfg.problem.classifier_batch_size),
        lineage_transition_mode=larry.lineage_transition_mode_from_config(cfg),
    )
    return {
        "paths": np.concatenate([source[:, None, :], generated], axis=1),
        "valid": np.asarray(validity["valid"], dtype=bool),
        "source": source,
        "target": target,
        "sampler_label": (
            "Euler--Maruyama rollout"
            if use_euler_maruyama
            else "Composed SSFM trajectories"
        ),
    }


def save_full_data_trajectory_plot(plot_data, output_path: Path) -> None:
    """Plot the first two trajectory coordinates by classifier validity."""
    if plot_data is None:
        return
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    paths = np.asarray(plot_data["paths"], dtype=np.float32)
    valid = np.asarray(plot_data["valid"], dtype=bool)
    source = np.asarray(plot_data["source"], dtype=np.float32)
    target = np.asarray(plot_data["target"], dtype=np.float32)

    def segments(selected):
        selected_paths = paths[selected, :, :2]
        if selected_paths.shape[0] == 0:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack(
            [selected_paths[:, :-1], selected_paths[:, 1:]], axis=2
        ).reshape((-1, 2, 2))

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.scatter(
        target[:, 0],
        target[:, 1],
        s=5,
        alpha=0.18,
        color="tab:blue",
        label="D6 holdout",
    )
    for selected, color, width, alpha, label in (
        (valid, "black", 0.4, 0.25, "valid"),
        (~valid, "crimson", 0.8, 0.7, "invalid"),
    ):
        selected_segments = segments(selected)
        if selected_segments.shape[0]:
            axis.add_collection(
                LineCollection(
                    selected_segments,
                    colors=color,
                    linewidths=width,
                    alpha=alpha,
                    label=label,
                )
            )
    axis.scatter(
        source[:, 0],
        source[:, 1],
        s=8,
        alpha=0.6,
        color="tab:green",
        label="D2 holdout",
    )
    axis.autoscale_view()
    coordinate_labels = (
        ("SPRING-1", "SPRING-2") if source.shape[-1] == 2 else ("PC1", "PC2")
    )
    axis.set_xlabel(coordinate_labels[0])
    axis.set_ylabel(coordinate_labels[1])
    axis.set_title(
        f"{plot_data.get('sampler_label', 'Composed SSFM trajectories')}: "
        f"{100.0 * float(np.mean(valid)):.1f}% lineage-valid"
    )
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def final_evaluation(model, params, cfg, output_dir: Path) -> Dict[str, float]:
    """Run population, lineage, and stochastic clone evaluation."""
    distribution, plot_data = distribution_metrics(model, params, cfg)
    metrics = dict(distribution)
    metrics.update(lineage_metrics(model, params, cfg))
    metrics.update(clone_wasserstein_metrics(model, params, cfg))
    if bool(cfg.evaluation.save_plot):
        save_pushforward_plot(
            plot_data,
            Path(output_dir) / "heldout_pushforwards.png",
            flowmap_n_steps=int(cfg.evaluation.flowmap_n_steps),
            euler_maruyama=shared_eval.uses_euler_maruyama_evaluation(cfg),
            euler_maruyama_n_steps=shared_eval.euler_maruyama_n_steps(cfg),
        )
        trajectory_data = full_data_trajectory_plot_data(model, params, cfg)
        if trajectory_data is not None:
            save_full_data_trajectory_plot(
                trajectory_data,
                Path(output_dir) / "heldout_d2_trajectory_validity.png",
            )
    return metrics

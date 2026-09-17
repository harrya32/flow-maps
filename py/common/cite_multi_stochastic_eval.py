"""Evaluation for direct CITE/Multi strong stochastic flow maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import numpy as np

from . import cite_multi
from . import maizels
from . import maizels_stochastic_eval as shared_eval
from . import wasserstein


def _sample_indices(rng: np.random.Generator, n: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=maximum, replace=False)).astype(np.int64)


def _heldout_interval(cfg) -> Tuple[str, str]:
    order = [str(value) for value in cfg.problem.timepoint_order]
    heldout = str(cfg.problem.heldout_timepoint)
    try:
        index = order.index(heldout)
    except ValueError as exc:
        raise ValueError(f"Held-out day {heldout!r} is not in {order}.") from exc
    if index <= 0:
        raise ValueError("Held-out evaluation requires a preceding observation day.")
    return order[index - 1], heldout


def _classifier_npz(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.suffix == ".npz" else path.with_suffix(".npz")


def distribution_metrics(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_source_points: Optional[int] = None,
    max_target_points: Optional[int] = None,
    n_noise_draws: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate direct and composed samples at the omitted CITE/Multi day."""
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

    source_day, heldout_day = _heldout_interval(cfg)
    pools = cite_multi.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    source_all = np.asarray(pools[source_day]["x"], dtype=np.float32)
    target_all = np.asarray(pools[heldout_day]["x"], dtype=np.float32)
    source = source_all[
        _sample_indices(rng, source_all.shape[0], max_source_points)
    ]
    target = target_all[
        _sample_indices(rng, target_all.shape[0], max_target_points)
    ]
    start_time = cite_multi.normalized_time(source_day)
    end_time = cite_multi.normalized_time(heldout_day)
    key = jax.random.PRNGKey(seed)
    emd_values = {"direct": [], "flowmap": []}
    mmd_values = {"direct": [], "flowmap": []}
    first_prediction = {}
    for draw in range(n_draws):
        key, direct_key, flowmap_key = jax.random.split(key, 3)
        predictions = {
            "direct": shared_eval.sample_pushforward(
                model,
                params,
                source,
                start_time,
                end_time,
                direct_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
            ),
            "flowmap": shared_eval.sample_composed_pushforward(
                model,
                params,
                source,
                start_time,
                end_time,
                flowmap_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=flowmap_n_steps,
            ),
        }
        for sampler, prediction in predictions.items():
            first_prediction.setdefault(sampler, prediction)
            emd_values[sampler].append(wasserstein.exact_emd(prediction, target))
            mmd_values[sampler].append(
                shared_eval.rbf_mmd2(prediction, target, seed=seed + draw)
            )

    tag = f"day{heldout_day.replace('.', 'p')}"
    metrics: Dict[str, float] = {
        "final_eval/source_count": float(source.shape[0]),
        "final_eval/target_count": float(target.shape[0]),
    }
    for sampler in ("direct", "flowmap"):
        mean_emd = float(np.mean(emd_values[sampler]))
        metrics[f"final_eval/{tag}_{sampler}_emd"] = mean_emd
        metrics[f"final_eval/{tag}_{sampler}_emd_std_over_noise"] = float(
            np.std(emd_values[sampler])
        )
        metrics[f"final_eval/{tag}_{sampler}_rbf_mmd2"] = float(
            np.mean(mmd_values[sampler])
        )
        metrics[f"final_eval/{sampler}_mean_emd"] = mean_emd

    # Keep the original benchmark namespace while naming both stochastic
    # samplers explicitly.  The unsuffixed key is the native direct SSFM draw.
    metrics["mfm/test_EMD"] = metrics[f"final_eval/{tag}_direct_emd"]
    metrics["mfm/test_EMD_direct_ssfm"] = metrics["mfm/test_EMD"]
    metrics["mfm/test_EMD_flowmap"] = metrics[f"final_eval/{tag}_flowmap_emd"]
    metrics[f"final_eval/{tag}_ssfm_emd"] = metrics["mfm/test_EMD"]
    metrics["final_eval/ssfm_mean_emd"] = metrics["mfm/test_EMD"]
    print(
        "Held-out stochastic EMD evaluation: "
        f"{source_day}->{heldout_day}, t={start_time:g}->{end_time:g}, "
        f"source_n={source.shape[0]}, target_n={target.shape[0]}, "
        f"draws={n_draws}, direct_EMD={metrics['mfm/test_EMD']:.8g}, "
        f"flowmap_EMD={metrics['mfm/test_EMD_flowmap']:.8g}"
    )
    return metrics, {
        heldout_day: (
            target,
            first_prediction["direct"],
            first_prediction["flowmap"],
        )
    }


def pushforward_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Sample the held-out population plot without calculating EMD."""
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_points = int(cfg.logging.maizels.plot_bs if max_points is None else max_points)
    if max_points < 0:
        raise ValueError("Plot max_points must be non-negative (zero means all).")
    source_day, heldout_day = _heldout_interval(cfg)
    pools = cite_multi.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    source_all = np.asarray(pools[source_day]["x"], dtype=np.float32)
    target_all = np.asarray(pools[heldout_day]["x"], dtype=np.float32)
    source = source_all[_sample_indices(rng, source_all.shape[0], max_points)]
    target = target_all[_sample_indices(rng, target_all.shape[0], max_points)]
    _, direct_key, flowmap_key = jax.random.split(jax.random.PRNGKey(seed), 3)
    start_time = cite_multi.normalized_time(source_day)
    end_time = cite_multi.normalized_time(heldout_day)
    direct = shared_eval.sample_pushforward(
        model,
        params,
        source,
        start_time,
        end_time,
        direct_key,
        n_coefficients=int(cfg.ssfm.n_coefficients),
    )
    flowmap = shared_eval.sample_composed_pushforward(
        model,
        params,
        source,
        start_time,
        end_time,
        flowmap_key,
        n_coefficients=int(cfg.ssfm.n_coefficients),
        n_steps=int(cfg.evaluation.flowmap_n_steps),
    )
    return {heldout_day: (target, direct, flowmap)}


def _lineage_source(cfg, seed: int):
    pools = cite_multi.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source_day = str(cfg.problem.source_time)
    source_pool = pools[source_day]
    source_all = np.asarray(source_pool["holdout_x"], dtype=np.float32)
    source_types_all = np.asarray(source_pool["holdout_types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(
        rng,
        source_all.shape[0],
        int(cfg.evaluation.lineage_max_source_points),
    )
    type_to_id = maizels.class_to_id_map(cite_multi.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types_all[indices]],
        dtype=np.int32,
    )
    return pools, source_all[indices], source_ids


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


def lineage_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Score full day-2-to-day-7 paths with both CITE/Multi classifiers."""
    seed = int(cfg.evaluation.seed + 211 if seed is None else seed)
    _, source_x, source_ids = _lineage_source(cfg, seed)
    if source_x.shape[0] == 0:
        return {}
    classifiers = _available_lineage_classifiers(cfg)
    if not classifiers:
        return {}
    n_steps = int(cfg.evaluation.lineage_n_steps)
    if n_steps <= 0:
        raise ValueError("evaluation.lineage_n_steps must be positive.")

    key = jax.random.PRNGKey(seed)
    valid_fractions = {name: [] for name in classifiers}
    for _ in range(int(cfg.evaluation.n_noise_draws)):
        key, draw_key = jax.random.split(key)
        path = shared_eval.sample_composed_path(
            model,
            params,
            source_x,
            cite_multi.normalized_time(cfg.problem.source_time),
            cite_multi.normalized_time(cfg.problem.target_time),
            draw_key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=n_steps,
        )
        for name, classifier_path in classifiers.items():
            validity = cite_multi.check_paths_with_classifier(
                path,
                source_ids,
                classifier_path,
                prob_threshold=float(cfg.problem.classifier_prob_threshold),
                margin_threshold=float(cfg.problem.classifier_margin_threshold),
                classifier_batch_size=int(cfg.problem.classifier_batch_size),
                lineage_transition_mode=cite_multi.lineage_transition_mode_from_config(
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


def full_data_trajectory_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
):
    """Generate held-out day-2 trajectories and full-classifier validity."""
    classifier_path = _classifier_npz(
        cfg.logging.maizels.full_data_classifier_path
    )
    if not classifier_path.is_file():
        print(
            "Skipping stochastic validity trajectory plot because the full-data "
            f"NumPy classifier checkpoint is absent: {classifier_path}"
        )
        return None
    seed = int(cfg.evaluation.seed + 503 if seed is None else seed)
    pools, source, source_ids = _lineage_source(cfg, seed)
    if source.shape[0] == 0:
        return None
    target = np.asarray(
        pools[str(cfg.problem.target_time)]["holdout_x"], dtype=np.float32
    )
    generated = shared_eval.sample_composed_path(
        model,
        params,
        source,
        cite_multi.normalized_time(cfg.problem.source_time),
        cite_multi.normalized_time(cfg.problem.target_time),
        jax.random.PRNGKey(seed),
        n_coefficients=int(cfg.ssfm.n_coefficients),
        n_steps=int(cfg.evaluation.lineage_n_steps),
    )
    validity = cite_multi.check_paths_with_classifier(
        generated,
        source_ids,
        classifier_path,
        prob_threshold=float(cfg.problem.classifier_prob_threshold),
        margin_threshold=float(cfg.problem.classifier_margin_threshold),
        classifier_batch_size=int(cfg.problem.classifier_batch_size),
        lineage_transition_mode=cite_multi.lineage_transition_mode_from_config(cfg),
    )
    return {
        "paths": np.concatenate([source[:, None, :], generated], axis=1),
        "valid": np.asarray(validity["valid"], dtype=bool),
        "source": source,
        "target": target,
        "source_day": str(cfg.problem.source_time),
        "target_day": str(cfg.problem.target_time),
    }


def save_full_data_trajectory_plot(plot_data, output_path: Path) -> None:
    """Plot composed trajectories by full-data-classifier validity."""
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
        label=f"held-out day {plot_data['target_day']}",
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
        label=f"held-out day {plot_data['source_day']}",
    )
    axis.autoscale_view()
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(
        "Composed SSFM trajectories: "
        f"{100.0 * float(np.mean(valid)):.1f}% lineage-valid\n"
        "(full-data classifier)"
    )
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


save_pushforward_plot = shared_eval.save_pushforward_plot


def final_evaluation(model, params, cfg, output_dir: Path) -> Dict[str, float]:
    """Run held-out-population and full-trajectory evaluation."""
    distribution, plot_data = distribution_metrics(model, params, cfg)
    metrics = dict(distribution)
    metrics.update(lineage_metrics(model, params, cfg))
    if bool(cfg.evaluation.save_plot):
        save_pushforward_plot(
            plot_data,
            Path(output_dir) / "heldout_pushforwards.png",
            flowmap_n_steps=int(cfg.evaluation.flowmap_n_steps),
        )
        trajectory_data = full_data_trajectory_plot_data(model, params, cfg)
        if trajectory_data is not None:
            save_full_data_trajectory_plot(
                trajectory_data,
                Path(output_dir) / "heldout_day2_trajectory_validity.png",
            )
    return metrics

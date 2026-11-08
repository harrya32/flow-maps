"""Shared raw-PCA distribution distances and classifier lineage checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import maizels, wasserstein
from mfm.flow_matchers.maizels_eval import (
    MaizelsEvaluationCallback,
    omitted_day_emd_aggregate_metrics,
    rbf_mmd2,
)
from otpfm_baseline.evaluation import classifier_paths as cite_classifier_paths
from otpfm_baseline.evaluation import evaluate as evaluate_cite_multi

from . import REPO_ROOT


def classifier_paths(args, heldout_day):
    if args.dataset_name != "maizels":
        return cite_classifier_paths(
            args.dataset_name,
            heldout_day,
            classifier_path=args.classifier_path,
            full_data_classifier_path=args.full_data_classifier_path,
        )
    name = "celltype_classifier_pca50.pt"
    if args.maizels_schedule == "d3_d3p8_d8":
        name = "celltype_classifier_pca50_d3_d3p8_d8.pt"
    observed = REPO_ROOT / name
    if not observed.is_file() and not observed.with_suffix(".npz").is_file():
        observed = REPO_ROOT / "outputs" / "maizels_classifier_d3_d3p8_d8" / name
    paths = {
        "observed_days": (
            Path(args.classifier_path).expanduser()
            if args.classifier_path
            else observed
        ),
        "all_days": (
            Path(args.full_data_classifier_path).expanduser()
            if args.full_data_classifier_path
            else REPO_ROOT / "celltype_classifier_pca50.pt"
        ),
    }
    for path in paths.values():
        if not path.is_file() and not path.with_suffix(".npz").is_file():
            raise FileNotFoundError(f"Maizels evaluation classifier not found: {path}")
    return {key: path.resolve() for key, path in paths.items()}


def _subset(x, cap, rng):
    indices = np.arange(len(x))
    if 0 < cap < len(x):
        indices = rng.choice(len(x), cap, replace=False)
    return np.asarray(x, np.float32)[indices], indices


def _path_figure(paths, source, target, valid, *, model_label="Cell-MNN"):
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for index in range(min(128, len(paths))):
        ax.plot(
            paths[index, :, 0],
            paths[index, :, 1],
            color="#2ca02c" if valid[index] else "#d62728",
            alpha=0.3,
            lw=0.8,
        )
    ax.scatter(source[:, 0], source[:, 1], s=5, c="black", label="Held-out D3")
    ax.scatter(target[:, 0], target[:, 1], s=5, c="#1f77b4", label="Observed D8")
    ax.set(
        xlabel="PC1",
        ylabel="PC2",
        title=f"{model_label} paths (green valid, red invalid)",
    )
    ax.legend()
    return fig


def observed_metrics(data, sampler, *, seed, max_points, primary_sampler="direct"):
    """Existing final observed-interval and carried source-to-final EMDs."""
    rng = np.random.default_rng(
        int(seed) + (1701 if data.dataset_name == "maizels" else 2901)
    )
    pools, retained = data.timepoint_splits, data.retained
    sources = {
        day: _subset(pools[day]["x"], max_points, rng)[0] for day in retained[:-1]
    }
    targets = {
        day: _subset(pools[day]["x"], max_points, rng)[0] for day in retained[1:]
    }
    metrics = {}

    def time_tag(day):
        tag = day.replace(".", "p")
        return tag if day.startswith("D") else f"day{tag}"

    for mode in ("direct", "flowmap", "euler"):
        emds, rollout = [], sources[retained[0]]
        for index, (left, right) in enumerate(zip(retained[:-1], retained[1:])):
            prediction = sampler.sample(
                sources[left],
                start_time=data.time_map[left],
                end_time=data.time_map[right],
                sampler=mode,
                n_steps=50,
            )
            emd = wasserstein.exact_emd(prediction, targets[right])
            tag = f"{time_tag(left)}_to_{time_tag(right)}"
            metrics[f"final_eval/observed_{tag}_{mode}_emd"] = emd
            if mode == primary_sampler:
                metrics[f"final_eval/observed_{tag}_evaluation_emd"] = emd
            emds.append(emd)
            rollout = (
                prediction
                if index == 0
                else sampler.sample(
                    rollout,
                    start_time=data.time_map[left],
                    end_time=data.time_map[right],
                    sampler=mode,
                    n_steps=50,
                )
            )
        metrics[f"final_eval/observed_{mode}_mean_emd"] = float(np.mean(emds))
        tag = f"{time_tag(retained[0])}_to_{time_tag(retained[-1])}"
        metrics[f"final_eval/{tag}_rollout_{mode}_emd"] = wasserstein.exact_emd(
            rollout, targets[retained[-1]]
        )
    metrics["final_eval/observed_evaluation_mean_emd"] = metrics[
        f"final_eval/observed_{primary_sampler}_mean_emd"
    ]
    metrics[f"final_eval/{tag}_rollout_evaluation_emd"] = metrics[
        f"final_eval/{tag}_rollout_{primary_sampler}_emd"
    ]
    return metrics


def evaluate(
    data,
    sampler,
    classifiers,
    *,
    seed,
    max_points=0,
    make_plot=True,
    include_observed=False,
    primary_sampler="direct",
    metric_prefix="cellmnn",
    model_label="Cell-MNN",
    lineage_mode="descendant",
    prob_threshold=0.0,
    margin_threshold=0.0,
    classifier_batch_size=8192,
    flow_steps=50,
    check_times=50,
):
    if data.dataset_name != "maizels":
        metrics, figure = evaluate_cite_multi(
            data,
            sampler,
            classifiers,
            seed=seed,
            max_points=max_points,
            make_plot=make_plot,
        )
        # Preserve MFM's Euler/local aliases and also report the paper's native
        # analytical prediction from the actual preceding population.
        previous = "2" if data.heldout_day == "3" else "3"
        rng = np.random.default_rng(int(seed) + 2901)
        source, _ = _subset(data.timepoint_splits[previous]["x"], max_points, rng)
        target, _ = _subset(
            data.timepoint_splits[data.heldout_day]["x"], max_points, rng
        )
        prediction = sampler.sample(
            source,
            start_time=data.time_map[previous],
            end_time=data.time_map[data.heldout_day],
            sampler="direct",
        )
        metrics["cellmnn/test_EMD"] = wasserstein.exact_emd(prediction, target)
        metrics["cellmnn/test_rbf_MMD2"] = rbf_mmd2(prediction, target, rng)
        metrics["test_EMD"] = metrics["cellmnn/test_EMD"]
        if include_observed:
            metrics.update(
                observed_metrics(data, sampler, seed=seed, max_points=max_points)
            )
        return metrics, {} if figure is None else {"lineage_trajectories": figure}

    pools = data.timepoint_splits
    omitted = [day for day in maizels.TIMEPOINTS if day not in data.retained]
    rng = np.random.default_rng(int(seed) + 1701)
    populations = []
    for day in omitted:
        left, _ = maizels.retained_interval_for_timepoint(data.cfg, day)
        source, _ = _subset(pools[left]["x"], max_points, rng)
        actual, _ = _subset(pools[day]["x"], max_points, rng)
        populations.append((day, left, source, actual))
    validation_rng = np.random.default_rng(int(seed) + 1802)
    observed = []
    for left, right in zip(data.retained[:-1], data.retained[1:]):
        source, _ = _subset(pools[left]["holdout_x"], max_points, validation_rng)
        target, _ = _subset(pools[right]["holdout_x"], max_points, validation_rng)
        observed.append(
            (left, right, source, target, wasserstein.exact_emd(source, target))
        )
    path_source, indices = _subset(
        pools["D3"]["holdout_x"], max_points, np.random.default_rng(2698)
    )
    mapping = maizels.class_to_id_map()
    source_types = np.asarray(
        [mapping[str(value)] for value in pools["D3"]["holdout_types"][indices]],
        np.int32,
    )
    metrics = {
        "maizels/eval_source_count": len(path_source),
        "final_eval/source_count": len(path_source),
        "lineage_eval/violation_metrics_available": 1.0,
    }
    figures = {}
    for mode in ("direct", "flowmap", "euler"):
        emds, mmds, rows = [], [], []
        for day, left, source, actual in populations:
            prediction = sampler.sample(
                source,
                start_time=data.time_map[left],
                end_time=data.time_map[day],
                sampler=mode,
                n_steps=flow_steps,
            )
            emd, mmd = wasserstein.exact_emd(prediction, actual), rbf_mmd2(
                prediction, actual, rng
            )
            tag = day.replace(".", "p")
            metrics[f"distribution_eval/{tag}_{mode}_emd"] = emd
            metrics[f"distribution_eval/{tag}_{mode}_rbf_mmd2"] = mmd
            emds.append(emd)
            mmds.append(mmd)
            rows.append((day, actual, prediction))
        metrics[f"distribution_eval/{mode}_emd_mean"] = float(np.mean(emds))
        metrics[f"distribution_eval/{mode}_rbf_mmd2_mean"] = float(np.mean(mmds))
        metrics[f"final_eval/{mode}_mean_emd"] = float(np.mean(emds))
        metrics[f"final_eval/{mode}_mean_rbf_mmd2"] = float(np.mean(mmds))
        aggregates = omitted_day_emd_aggregate_metrics(
            mode, omitted, emds, ("D3.4", "D6")
        )
        metrics.update(aggregates)
        metrics.update(
            {
                key.replace("distribution_eval/", "final_eval/"): value
                for key, value in aggregates.items()
            }
        )
        val_emds, relative = [], []
        for left, right, source, target, baseline in observed:
            prediction = sampler.sample(
                source,
                start_time=data.time_map[left],
                end_time=data.time_map[right],
                sampler=mode,
                n_steps=flow_steps,
            )
            emd = wasserstein.exact_emd(prediction, target)
            val_emds.append(emd)
            relative.append(emd / max(baseline, 1e-12))
            tag = right.replace(".", "p")
            metrics[f"validation_distribution/{tag}_{mode}_emd"] = emd
            metrics[f"validation_distribution/{tag}_source_target_emd"] = baseline
        metrics[f"validation_distribution/{mode}_emd_mean"] = float(np.mean(val_emds))
        metrics[f"validation_distribution/{mode}_relative_emd_mean"] = float(
            np.mean(relative)
        )
        metrics[f"final_eval/validation_{mode}_emd_mean"] = float(np.mean(val_emds))
        metrics[f"final_eval/validation_{mode}_relative_emd_mean"] = float(
            np.mean(relative)
        )
        paths = sampler.sample(
            path_source, sampler=mode, n_steps=check_times, paths=True
        )
        for classifier_name, path in classifiers.items():
            validity = maizels.check_paths_with_classifier(
                paths[:, 1:],
                source_types,
                classifier_path=path,
                prob_threshold=prob_threshold,
                margin_threshold=margin_threshold,
                classifier_batch_size=classifier_batch_size,
                lineage_transition_mode=lineage_mode,
            )
            valid = np.asarray(validity["valid"], dtype=bool)
            prefix = (
                "maizels"
                if classifier_name == "observed_days"
                else "maizels/full_data_classifier"
            )
            final_prefix = (
                "final_eval"
                if classifier_name == "observed_days"
                else "final_eval/full_data_classifier"
            )
            for kind, percent in (
                ("valid", 100.0 * float(np.mean(valid))),
                ("invalid", 100.0 * float(np.mean(~valid))),
            ):
                metrics[f"{prefix}/model_{mode}_{kind}_trajectory_pct"] = percent
                metrics[f"{final_prefix}/{mode}_{kind}_trajectory_pct"] = percent
            if make_plot and mode == primary_sampler and classifier_name == "all_days":
                figures["lineage_trajectories"] = _path_figure(
                    paths,
                    path_source,
                    pools["D8"]["holdout_x"],
                    valid,
                    model_label=model_label,
                )
        if make_plot and mode == primary_sampler:
            figures["intermediate_distributions"] = (
                MaizelsEvaluationCallback._distribution_figure(rows)
            )
    metrics[f"{metric_prefix}/test_EMD"] = metrics[
        f"final_eval/{primary_sampler}_mean_emd"
    ]
    metrics["test_EMD"] = metrics[f"{metric_prefix}/test_EMD"]
    if include_observed:
        metrics.update(
            observed_metrics(
                data,
                sampler,
                seed=seed,
                max_points=max_points,
                primary_sampler=primary_sampler,
            )
        )
    return metrics, figures

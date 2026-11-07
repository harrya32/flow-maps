"""Match the constrained CITE/Multi population and lineage evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common import cite_multi, wasserstein
from mfm.flow_matchers.cite_multi_eval import (
    CiteMultiEvaluationCallback,
    score_paths_with_all_days_classifier,
)
from mfm.flow_matchers.maizels_eval import rbf_mmd2

from . import REPO_ROOT


def classifier_paths(
    dataset_name, heldout_day, *, classifier_path="", full_data_classifier_path=""
):
    root = REPO_ROOT / f"{dataset_name}-classifiers"
    paths = {
        "observed_days": (
            Path(classifier_path).expanduser()
            if classifier_path
            else root
            / f"celltype_classifier_{dataset_name}_pca100_except_day{heldout_day}.pt"
        ),
        "all_days": (
            Path(full_data_classifier_path).expanduser()
            if full_data_classifier_path
            else root / f"celltype_classifier_{dataset_name}_pca100_all_days.pt"
        ),
    }
    for path in paths.values():
        if not path.is_file() and not path.with_suffix(".npz").is_file():
            raise FileNotFoundError(
                f"CITE/Multi evaluation classifier not found: {path}"
            )
    return {key: path.resolve() for key, path in paths.items()}


def _subset(values, max_points, rng):
    indices = np.arange(len(values))
    if 0 < max_points < len(values):
        indices = rng.choice(len(values), size=max_points, replace=False)
    return values[indices], indices


def evaluate(
    data,
    model_sampler,
    classifiers,
    *,
    seed=42,
    max_points=0,
    flow_steps=50,
    check_times=50,
    mfm_steps=100,
    prob_threshold=0.0,
    margin_threshold=0.0,
    classifier_batch_size=8192,
    lineage_mode="descendant",
    make_plot=True,
):
    """Return benchmark metric names and an optional PC1/PC2 validity figure."""
    pools = data.timepoint_splits
    heldout = data.heldout_day
    rng = np.random.default_rng(int(seed) + 1701)
    source, _ = _subset(pools["2"]["x"], max_points, rng)
    target, _ = _subset(pools[heldout]["x"], max_points, rng)
    # Generic diagnostics: all D2 cells -> omitted day, 50 uniform steps.
    metrics = {}
    for sampler in ("direct", "flowmap", "euler"):
        prediction = model_sampler.sample(
            source, end_time=data.time_map[heldout], n_steps=flow_steps, sampler=sampler
        )
        emd = wasserstein.exact_emd(prediction, target)
        mmd2 = rbf_mmd2(prediction, target, rng)
        metrics.update(
            {
                f"distribution_eval/{heldout}_{sampler}_emd": emd,
                f"distribution_eval/{heldout}_{sampler}_rbf_mmd2": mmd2,
                f"distribution_eval/{sampler}_emd_mean": emd,
                f"distribution_eval/{sampler}_rbf_mmd2_mean": mmd2,
                f"final_eval/{sampler}_mean_emd": emd,
                f"final_eval/{sampler}_mean_rbf_mmd2": mmd2,
            }
        )

    # MFM-compatible diagnostic: the real preceding population -> omitted day.
    local_rng = np.random.default_rng(int(seed) + 2901)
    previous = cite_multi.TIMEPOINTS[cite_multi.TIMEPOINTS.index(heldout) - 1]
    local_source, _ = _subset(pools[previous]["x"], max_points, local_rng)
    local_target, _ = _subset(pools[heldout]["x"], max_points, local_rng)
    for sampler, key in (
        ("euler", "mfm/test_EMD"),
        ("flowmap", "mfm/test_EMD_flowmap"),
    ):
        prediction = model_sampler.sample(
            local_source,
            start_time=data.time_map[previous],
            end_time=data.time_map[heldout],
            n_steps=mfm_steps,
            sampler=sampler,
        )
        metrics[key] = wasserstein.exact_emd(prediction, local_target)
    metrics["test_EMD"] = metrics["mfm/test_EMD"]

    # Lineage checks: held-out D2 cells over the complete D2 -> D7 trajectory.
    source_pool = pools["2"]
    path_source, indices = _subset(
        source_pool["holdout_x"], max_points, np.random.default_rng(2698)
    )
    source_types = source_pool["holdout_types"][indices]
    metrics["maizels/eval_source_count"] = len(path_source)
    metrics["final_eval/source_count"] = len(path_source)
    metrics["lineage_eval/violation_metrics_available"] = 1.0
    figure = None
    for sampler in ("direct", "flowmap", "euler"):
        paths = model_sampler.sample(
            path_source, n_steps=check_times, sampler=sampler, paths=True
        )
        for classifier_name, path in classifiers.items():
            validity = score_paths_with_all_days_classifier(
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
            for kind, percentage in (
                ("valid", 100.0 * float(np.mean(valid))),
                ("invalid", 100.0 * float(np.mean(~valid))),
            ):
                metrics[f"{prefix}/model_{sampler}_{kind}_trajectory_pct"] = percentage
                metrics[f"{final_prefix}/{sampler}_{kind}_trajectory_pct"] = percentage
            if make_plot and sampler == "flowmap" and classifier_name == "all_days":
                figure = CiteMultiEvaluationCallback._trajectory_figure(
                    paths,
                    path_source,
                    pools["7"]["holdout_x"],
                    valid,
                )
    return metrics, figure

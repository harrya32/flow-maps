#!/usr/bin/env python3
"""Audit and report proportional label holdouts in the full Calderon atlas."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd

from scripts.prepare_calderon_lineage_data import file_hash
from scripts.prepare_calderon_all_times_data import VARIANT
from scripts.train_calderon_lineage_classifiers import scores


def read(path):
    return json.loads(Path(path).read_text())


def audit(data_dir, output_dir):
    preparation = read(data_dir/f"preprocessing_{VARIANT}.json")
    assert preparation["hvg_fit"] == preparation["pca_fit"] == "all_cells"
    assert preparation["n_cells"] == preparation["n_fit_cells"] == 314387
    h5ad = data_dir/f"lineage_assigned_hvg3000_pca100_{VARIANT}.h5ad"
    transform = data_dir/f"preprocessing_hvg3000_pca100_{VARIANT}.npz"
    assert file_hash(h5ad) == preparation["h5ad_sha256"]
    assert file_hash(transform) == preparation["preprocessing_npz_sha256"]
    source_dir = Path(preparation["source_directory"])
    for name, digest in preparation["source_hashes"].items():
        assert file_hash(source_dir/name) == digest, name
    for name, digest in read(output_dir/"implementation_hashes.json").items():
        assert file_hash(ROOT/name) == digest, name
    cells = pd.read_csv(data_dir/"cell_splits.csv.gz")
    assert cells.cell_id.is_unique and len(cells) == 314387
    sizes = {name: int(cells.classifier_split.eq(name).sum()) for name in ("train", "validation", "test")}
    assert sizes == preparation["split_sizes"] and sum(sizes.values()) == len(cells)
    with np.load(transform, allow_pickle=False) as fitted:
        np.testing.assert_array_equal(fitted["fit_indices"], np.arange(len(cells)))
        for name in sizes:
            np.testing.assert_array_equal(fitted[f"classifier_{name}_indices"], np.flatnonzero(cells.classifier_split.eq(name)))
    split = read(data_dir/"split_validation.json")
    assert split["status"] == "passed" and split["all_classes_and_time_windows_present_in_every_split"]
    selection = read(output_dir/"selection.json")
    complete = read(output_dir/"evaluation_complete.json")
    digest = file_hash(output_dir/"selection.json")
    assert complete["selection_sha256"] == digest and complete["selection_used_validation_only"]
    assert selection["selection_frozen_before_new_test_evaluation"]
    assert selection["source_h5ad_sha256"] == preparation["h5ad_sha256"]
    results = pd.read_csv(output_dir/"reports/results.csv")
    selected = results.loc[results.selected].copy()
    assert set(selected.target) == {"branch_id", "broad_tissue"} and len(selected) == 2
    test = cells.loc[cells.classifier_split.eq("test")]
    for group in selection["groups"]:
        target, chosen = group["target"], group["selected"]
        assert not group["test_used_for_selection"]
        for candidate in group["candidates"]:
            assert file_hash(candidate["checkpoint"]) == candidate["checkpoint_sha256"]
            metadata = read(Path(candidate["checkpoint"]).with_name(f"{target}_validation.json"))
            history = pd.read_csv(Path(candidate["checkpoint"]).parent/"history.csv")
            maximum = history[f"{target}_validation_macro_f1"].max()
            assert np.isclose(metadata["best_validation_macro_f1_during_fit"], maximum, rtol=0, atol=1e-12)
        record = read(output_dir/"reports"/f"{VARIANT}_{target}_{chosen}_metrics.json")
        manifest = read(output_dir/"models"/f"{VARIANT}_{target}_selected.json")
        assert manifest["selection_sha256"] == record["selection_sha256"] == digest
        with np.load(output_dir/"reports"/f"{target}_selected_test_probabilities.npz", allow_pickle=False) as predictions:
            probabilities, truth = predictions["probabilities"], predictions["truth"]
            assert probabilities.shape == (31439, len(group["class_names"]))
            assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-5, rtol=0)
            np.testing.assert_array_equal(predictions["cell_ids"], test.cell_id)
            np.testing.assert_array_equal(np.asarray(group["class_names"])[truth], test[target])
            counts = cells.loc[cells.classifier_split.eq("train"), target].value_counts().reindex(group["class_names"]).to_numpy()
            recalculated = scores(truth, probabilities, counts)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss"):
                assert np.isclose(recalculated[metric], record["test"][metric], rtol=0, atol=1e-12)
        assert record["test"]["n_present_classes"] == len(group["class_names"])
    result = {"status": "passed", "all_cell_unsupervised_fit_verified": True,
        "classifier_label_splits_disjoint": True, "all_classes_and_times_present": True,
        "selected_checkpoints_use_validation_only": True, "test_metrics_recomputed": True,
        "selected_manifests_reloaded_and_verified": complete["selected_manifests_verified"],
        "source_counts_and_previous_representations_unchanged": True,
        "n_cells": len(cells), "split_sizes": sizes, "selection_sha256": digest,
        "representation_fit_includes_validation_test_expressions": True,
        "max_absolute_proportion_difference_pp": split["max_absolute_proportion_difference_pp"]}
    (output_dir/"results_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    return preparation, selection, results, selected, result


def figure(data_dir, output_dir, selected, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    targets = ("branch_id", "broad_tissue")
    colors = ("#256F9D", "#388977", "#B96935")
    for offset, metric, name, color in zip((-.26, 0, .26), ("accuracy", "balanced_accuracy", "macro_f1"),
                                           ("Accuracy", "Mean class recall", "Macro-F1"), colors):
        values = [100*selected.loc[selected.target.eq(t), metric].iloc[0] for t in targets]
        bars = axes[0, 0].bar(np.arange(2)+offset, values, .24, color=color, label=name)
        axes[0, 0].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    axes[0, 0].set_xticks([0, 1], ["53 branch IDs", "24 tissue labels"])
    axes[0, 0].set(ylabel="Held-out score (%)", ylim=(0, 100), title="A  Validation-selected classifiers")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    for target, color, name in zip(targets, colors, ("Branch IDs", "Tissue labels")):
        method = selected.loc[selected.target.eq(target), "method"].iloc[0]
        frame = pd.read_csv(output_dir/"reports"/f"{VARIANT}_{target}_{method}_by_window.csv")
        hours = frame.window.map(lambda w: int(w.split("-")[0])+1)
        axes[0, 1].plot(hours, 100*frame.accuracy, "o-", color=color, label=name)
    axes[0, 1].set(xlabel="Inferred-age midpoint (h)", ylabel="Held-out accuracy (%)", ylim=(0, 100),
                   xticks=np.arange(1, 18, 2), title="B  Test cells from every time window")
    axes[0, 1].legend(loc="lower left", fontsize=8)
    proportions = pd.read_csv(data_dir/"split_proportions.csv")
    for ax, target, title in zip(axes[1], targets, ("C  Branch proportions", "D  Tissue proportions")):
        frame = proportions.loc[proportions.target.eq(target)]
        maximum = max(frame.dataset_pct.max(), frame.validation_pct.max(), frame.test_pct.max())*1.1
        ax.plot([0, maximum], [0, maximum], color="#888888", linewidth=1, linestyle="--")
        ax.scatter(frame.dataset_pct, frame.validation_pct, s=36, marker="o", facecolors="none", edgecolors=colors[0], label="Validation")
        ax.scatter(frame.dataset_pct, frame.test_pct, s=30, marker="+", color=colors[1], label="Test")
        ax.set(xlabel="Class proportion in full dataset (%)", ylabel="Class proportion in holdout (%)",
               xlim=(0, maximum), ylim=(0, maximum), title=title)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Calderon all-time atlas · HVG3000/PCA100 fitted on all 314,387 cells\n31,439 label-held-out test cells · proportional class split", fontsize=13)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination/f"all_times_classification.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def report(data_dir, output_dir, report_dir=ROOT/"docs/calderon2022"):
    data_dir, output_dir, report_dir = Path(data_dir).resolve(), Path(output_dir).resolve(), Path(report_dir).resolve()
    preparation, selection, results, selected, verification = audit(data_dir, output_dir)
    lines = ["# Calderon classifiers across all timepoints", "",
        "HVG selection and PCA were refitted on **all 314,387 lineage-assigned cells**. "
        "Every inferred-age window from 0–18 h contributes; no time window is reserved. "
        "The classifiers use the new **3,000 HVGs and PCA100** representation, with a shared branch-stratified 80/10/10 label split.", "",
        "| Target | Selected predictor | Test cells | Accuracy | Mean per-class recall | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _, row in selected.iterrows():
        group = next(g for g in selection["groups"] if g["target"] == row.target)
        method = "+".join(group["ensemble_members"]) if row.method == "ensemble" else row.method
        lines.append(f"| {'Branch IDs (53)' if row.target == 'branch_id' else 'Broad tissues (24)'} | {method} | "
                     f"{int(row.test_cells):,} | **{row.accuracy:.2%}** | {row.balanced_accuracy:.2%} | {row.macro_f1:.2%} |")
    lines += ["", "![Held-out performance and split proportions](figures/all_times_classification.png)", "",
        "## Data and evaluation scope", "",
        "The training, validation and test sets contain **251,509 / 31,439 / 31,439 cells**. "
        "All 53 branches, all 24 tissues and all nine time windows are present in every split. "
        "Branches map deterministically to tissues, so branch stratification also preserves tissue proportions.", "",
        "| Label target | Largest validation deviation from full-dataset share | Largest test deviation |",
        "| --- | ---: | ---: |"]
    for target, name in (("branch_id", "Branches"), ("broad_tissue", "Tissues")):
        delta = verification["max_absolute_proportion_difference_pp"][target]
        lines.append(f"| {name} | {delta['validation']:.5f} percentage points | {delta['test']:.5f} percentage points |")
    lines += ["", "The proportions follow the dataset's class frequencies. They are not uniform across classes. "
        "`split_proportions.csv` provides every class count/share and the time-window breakdown.", "",
        "This is **transductive label-held-out accuracy**: validation/test labels are excluded from supervised training, "
        "while their expression profiles enter the requested all-cell HVG/PCA fit. "
        "The classifier's additional PC standardizer uses training cells only. Model/ensemble selection and temperature "
        "calibration use validation labels only, and choices are frozen before test scores are evaluated. "
        "These are random cell holdouts from the same atlas, measuring recovery of the released computational annotations.", "",
        "Only cells with released lineage assignments are included. The 18–20 h cells lack those assignments. "
        "The classifiers receive PCA100 only; age, cluster and annotation metadata are not predictor inputs.", "",
        "## Representation and training", "",
        "Seurat v3 variance stabilization (LOESS span 0.3, genes detected in at least three cells) selects 3,000 HVGs from "
        "the original sparse integer counts. Each cell is normalized to 10,000 using its complete 23,932-feature library, "
        "then log1p transformed. Gene scaling and exact covariance PCA fit every labelled cell. "
        f"PCA100 explains **{preparation['explained_variance_ratio']:.2%}** of scaled HVG variance. "
        f"A 256-cell dense projection check had maximum absolute error **{preparation['max_projection_error_256_cells']:.3g}**.", "",
        "Nine new neural networks cover five established recipes: wide, stronger class weighting, residual, residual with "
        "mixup/noise/label smoothing/EMA, and joint branch+tissue prediction with hierarchical consistency. "
        "Training uses AdamW, clipping at 5, five-epoch warmup and cosine decay, batch size 2,048, a 160-epoch limit and "
        "30-epoch early-stopping patience. Each target selects its validation macro-F1 checkpoint. "
        "Calibrated ensembles contain at most three members and require a validation macro-F1 gain of 0.0005 per addition. "
        "This is one initialization/split seed.", "",
        "| Target | Recipe | Validation macro-F1 | Test accuracy | Test macro-F1 | Selected |",
        "| --- | --- | ---: | ---: | ---: | --- |"]
    for _, row in results.iterrows():
        lines.append(f"| {row.target} | {row.method} | {row.validation_macro_f1:.2%} | {row.accuracy:.2%} | {row.macro_f1:.2%} | {'yes' if row.selected else ''} |")
    lines += ["", "## Classifier confidence", "",
        "| Target | Test coverage at probability ≥0.9 | Accuracy among those cells | Test ECE |",
        "| --- | ---: | ---: | ---: |"]
    for _, row in selected.iterrows():
        metrics = read(output_dir/"reports"/f"{VARIANT}_{row.target}_{row.method}_metrics.json")["test"]
        subset = metrics["confidence_subsets"]["0.9"]
        accuracy = "n/a" if subset["accuracy"] is None else f"{subset['accuracy']:.2%}"
        lines.append(f"| {row.target} | {subset['coverage']:.2%} | {accuracy} | {metrics['expected_calibration_error_15bins']:.4f} |")
    lines += ["", "## Artifacts and reproduction", "",
        f"Dataset: `{data_dir / f'lineage_assigned_hvg3000_pca100_{VARIANT}.h5ad'}`.", "",
        f"Classifier directory: `{output_dir}`. `models/` contains a selected manifest for each target and "
        "`search/` contains reloadable PyTorch checkpoints and validation curves. `reports/` contains all scores, "
        "per-class confusion/recall, per-window performance and selected test probabilities. "
        "The manifests require this new PCA space. Existing flow models were fitted in their earlier PCA spaces.", "",
        "The audit recomputes both selected test metrics, checks the full-cell PCA fitting indices, verifies disjoint label splits, "
        "class proportions, validation-only selection, frozen checkpoint hashes and probability reloads. "
        "Original counts and both earlier latent datasets are unchanged. The three all-time data checks and five shared neural-network "
        "checks passed. `results_validation.json`, `evaluation_complete.json` and `implementation_hashes.json` record verification.", "",
        "```sh", "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /opt/anaconda3/envs/maizels2023aa/bin/python \\",
        "  scripts/prepare_calderon_all_times_data.py", "",
        "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /usr/bin/caffeinate -i \\",
        "  /opt/anaconda3/envs/maizels2023aa/bin/python scripts/train_calderon_all_times_classifiers.py --device mps", "```", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/"all_times_classification.md").write_text("\n".join(lines))
    figure(data_dir, output_dir, selected, report_dir/"figures")
    print(f"Report: {report_dir / 'all_times_classification.md'}", flush=True)
    print(selected.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"datasets/calderon2022/all_times")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"calderon-classifiers/all_times")
    args = parser.parse_args()
    report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit selected-time tissue classification and complete temporal holdouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd

from scripts.prepare_calderon_lineage_data import file_hash
from scripts.prepare_calderon_three_time_data import VARIANT, TRAINING_WINDOWS, INTERMEDIATE_WINDOWS
from scripts.train_calderon_three_time_classifier import TARGET, dataset, evaluation_indices, split_hash
from scripts.train_calderon_lineage_classifiers import scores

POPULATIONS = ("selected_time_test", "intermediate_6_8", "intermediate_10_12", "intermediate_pooled")
NAMES = {"selected_time_test": "Held-out cells at 4–6, 8–10 and 12–14 h",
         "intermediate_6_8": "Entire 6–8 h marginal",
         "intermediate_10_12": "Entire 10–12 h marginal",
         "intermediate_pooled": "Both intermediate marginals pooled"}


def read(path):
    return json.loads(Path(path).read_text())


def prefix(output_dir, population, method):
    return output_dir / "reports" / f"{VARIANT}_{population}_{TARGET}_{method}"


def audit(data_dir, output_dir):
    path, preparation, (_, obs, _, indices), table = dataset(SimpleNamespace(data_dir=data_dir))
    selection_path = output_dir / "selection.json"
    selection = read(selection_path)
    complete = read(output_dir / "evaluation_complete.json")
    digest = file_hash(selection_path)
    assert complete["status"] == "passed" and complete["selection_sha256"] == digest
    assert selection["selection_frozen_before_test_and_intermediate_evaluation"]
    assert complete["selection_used_selected_time_validation_only"]
    assert complete["intermediate_labels_excluded_from_training_and_selection"]
    assert complete["test_cells_excluded_from_training_and_selection"]
    group = selection["group"]
    classes = group["class_names"]
    assert len(classes) == 23 and "Blastoderm" not in classes
    assert group["cell_and_split_sha256"] == split_hash(obs)
    assert not group["test_used_for_selection"] and not group["intermediate_used_for_selection"]
    manifest = output_dir / "models" / f"{VARIANT}_{TARGET}_selected.json"
    assert file_hash(manifest) == complete["selected_manifest_sha256"]
    for name, expected in read(output_dir / "implementation_hashes.json").items():
        assert file_hash(ROOT / name) == expected, name
    reference = output_dir / "reference_oracle_hashes.json"
    if reference.exists():
        for name, expected in read(reference).items():
            assert file_hash(name) == expected, name
    for candidate in group["candidates"]:
        checkpoint = Path(candidate["checkpoint"])
        assert file_hash(checkpoint) == candidate["checkpoint_sha256"]
        metadata = read(checkpoint.with_name(f"{TARGET}_validation.json"))
        history = pd.read_csv(checkpoint.parent / "history.csv")
        assert np.isclose(metadata["best_validation_macro_f1_during_fit"], history[f"{TARGET}_validation_macro_f1"].max(), atol=1e-12, rtol=0)
        assert metadata["cell_and_split_sha256"] == group["cell_and_split_sha256"]
        assert metadata["source_h5ad"] == str(path)
    all_records = read(output_dir / "reports/all_evaluation_metrics.json")
    selected = {record["population"]: record for record in all_records if record["selected"]}
    assert set(selected) == set(POPULATIONS)
    counts = obs.iloc[indices["train"]][TARGET].astype(str).value_counts().reindex(classes).to_numpy()
    populations = evaluation_indices(table)
    lookup = table.set_index("source_pca_row")
    verified = []
    for population in POPULATIONS:
        record = selected[population]
        expected_rows = populations[population]
        with np.load(output_dir / "reports" / f"{population}_selected_probabilities.npz", allow_pickle=False) as saved:
            probabilities, truth = saved["probabilities"], saved["truth"]
            assert probabilities.shape == (len(expected_rows), len(classes))
            assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)
            np.testing.assert_array_equal(saved["source_pca_rows"], expected_rows)
            np.testing.assert_array_equal(saved["cell_ids"], lookup.loc[expected_rows, "cell_id"])
            np.testing.assert_array_equal(np.asarray(classes)[truth], lookup.loc[expected_rows, TARGET])
            assert saved["selection_sha256"].item() == digest
            recomputed = scores(truth, probabilities, counts)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1", "log_loss"):
                assert np.isclose(recomputed[metric], record["test"][metric], atol=1e-12, rtol=0)
        assert record["selection_sha256"] == digest and not record["test_used_for_selection"]
        verified.append({"population": population, "n_cells": len(expected_rows),
                         "n_present_classes": record["test"]["n_present_classes"], "metrics_recomputed": True})
    assert selected["selected_time_test"]["test"]["n_cells"] == 16221
    assert selected["intermediate_6_8"]["test"]["n_cells"] == 30876
    assert selected["intermediate_10_12"]["test"]["n_cells"] == 21448
    assert selected["intermediate_pooled"]["test"]["n_cells"] == 52324
    result = {"status": "passed", "training_windows": list(TRAINING_WINDOWS),
        "intermediate_windows": list(INTERMEDIATE_WINDOWS), "split_sizes": preparation["split_sizes"],
        "cell_and_temporal_holdouts_disjoint_from_training_and_selection": True,
        "selection_frozen_before_test_and_intermediate_evaluation": True,
        "selection_sha256": digest, "verified_populations": verified,
        "selected_predictor_reload_verifications": complete["verified_populations"],
        "source_pca_unchanged_and_reused": True, "reference_oracle_unchanged": True,
        "validation_and_test_expression_in_global_pca_fit": True}
    (output_dir / "results_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return preparation, group, all_records, selected, result


def window_scores(output_dir, group, selected):
    stem = prefix(output_dir, "selected_time_test", group["selected"])
    frame = pd.read_csv(stem.with_name(stem.name + "_by_window.csv"))
    frame["evaluation_kind"] = "Held-out cells"
    for population, window in (("intermediate_6_8", "6-8"), ("intermediate_10_12", "10-12")):
        row = selected[population]["test"]
        frame = pd.concat([frame, pd.DataFrame([{"window": window, "evaluation_kind": "Entire held-out marginal",
            **{key: row[key] for key in ("n_cells", "n_present_classes", "accuracy", "balanced_accuracy", "macro_f1")}}])], ignore_index=True)
    return frame.sort_values("window", key=lambda value: value.map(lambda time: int(time.split("-")[0])))


def figures(output_dir, group, selected, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = ["Selected-time\ntest cells", "6–8 h\nentire marginal", "10–12 h\nentire marginal", "Intermediate\npooled"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for offset, metric, name, color in zip((-.25, 0, .25), ("accuracy", "balanced_accuracy", "macro_f1"),
                                           ("Accuracy", "Mean class recall", "Macro-F1"), ("#276B9C", "#388977", "#B96935")):
        values = [100 * selected[population]["test"][metric] for population in POPULATIONS]
        bars = axes[0].bar(np.arange(4) + offset, values, width=.23, label=name, color=color)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=7.5, padding=3)
    axes[0].set(xticks=range(4), xticklabels=labels, ylabel="Score (%)", ylim=(0, 100),
                title="A  Frozen predictor on cell and temporal holdouts")
    axes[0].legend(loc="upper left", fontsize=8)
    by_window = window_scores(output_dir, group, selected)
    colors = ["#276B9C" if window in TRAINING_WINDOWS else "#B96935" for window in by_window.window]
    bars = axes[1].bar(np.arange(5), 100 * by_window.accuracy, color=colors, width=.65)
    axes[1].bar_label(bars, fmt="%.2f%%", fontsize=9, padding=3)
    labels = [f"{window} h\nn={int(n):,}" for window, n in zip(by_window.window, by_window.n_cells)]
    axes[1].set(xticks=range(5), xticklabels=labels, ylim=(0, 100), ylabel="Accuracy (%)",
                title="B  Blue: held-out cells · orange: unseen windows")
    fig.suptitle("Calderon tissue classifier trained at 4–6, 8–10 and 12–14 h\nPCA100 fixed from the all-times atlas · validation selection at training windows only", fontsize=12)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination / f"three_time_classifier.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    classes = group["class_names"]
    values, support = [], []
    for population in POPULATIONS[:3]:
        frame = pd.read_csv(prefix(output_dir, population, group["selected"]).with_name(
            prefix(output_dir, population, group["selected"]).name + "_per_class.csv"), index_col=0).reindex(classes)
        values.append(frame.recall.to_numpy())
        support.append(frame.support.to_numpy())
    array, counts = np.stack(values, axis=1), np.stack(support, axis=1)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#E9ECEF")
    fig, ax = plt.subplots(figsize=(8.7, 9), constrained_layout=True)
    image = ax.imshow(np.ma.masked_where(counts == 0, array), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set(yticks=range(len(classes)), yticklabels=classes, xticks=range(3),
           xticklabels=["Selected-time test", "6–8 h", "10–12 h"], title="Recall by tissue · same validation-selected predictor")
    ax.tick_params(axis="y", labelsize=9)
    for i in range(len(classes)):
        for j in range(3):
            ax.text(j, i, f"{array[i, j]:.0%}  (n={int(counts[i, j])})" if counts[i, j] else "absent",
                    ha="center", va="center", fontsize=8, color="white" if array[i, j] > .55 and counts[i, j] else "#32383E")
    fig.colorbar(image, ax=ax, shrink=.75, label="Recall")
    for extension in ("png", "svg"):
        fig.savefig(destination / f"three_time_classifier_recall.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def report(data_dir, output_dir, report_dir=ROOT / "docs/calderon2022"):
    data_dir, output_dir, report_dir = Path(data_dir).resolve(), Path(output_dir).resolve(), Path(report_dir).resolve()
    preparation, group, all_records, selected, validation = audit(data_dir, output_dir)
    sizes = preparation["split_sizes"]
    method = "+".join(group["ensemble_members"]) if group["selected"] == "ensemble" else group["selected"]
    lines = ["# Tissue classifier trained at three symmetric timepoints", "",
        "The constraint classifier receives supervised training data only from **4–6, 8–10 and 12–14 h**. "
        "Its **23 tissue labels**, including `Unknown`, cover every tissue present within the retained 4–14 h interval. "
        "The **entire 6–8 and 10–12 h marginals** are excluded from classifier training, validation, calibration and model selection.", "",
        f"The selected predictor is **{method}**, chosen using selected-time validation macro-F1. "
        "The following test scores were computed only after selection was frozen.", "",
        "| Evaluation population | Cells | Present tissues | Accuracy | Mean class recall | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for population in POPULATIONS:
        row = selected[population]["test"]
        lines.append(f"| {NAMES[population]} | {row['n_cells']:,} | {row['n_present_classes']} | "
                     f"**{row['accuracy']:.2%}** | {row['balanced_accuracy']:.2%} | {row['macro_f1']:.2%} |")
    lines += ["", "Mean class recall and macro-F1 average the classes present in each evaluated population. "
        "The output vocabulary always contains 23 labels, even when a particular evaluation window contains fewer labels.", "",
        "![Cell and temporal holdout performance](figures/three_time_classifier.png)", "",
        "## Splits and representation", "",
        f"The three selected windows contain **162,206 cells**, split into **{sizes['train']:,} training / "
        f"{sizes['validation']:,} validation / {sizes['test']:,} test cells**. "
        "Stratification by inferred-age window and branch ID preserves tissue frequencies through the deterministic branch-to-tissue mapping. "
        "Every selected-time tissue and auxiliary branch class is represented in all three cell splits. "
        "The intermediate holdout contains **30,876 cells at 6–8 h** and **21,448 at 10–12 h**.", "",
        "The **existing all-times HVG3000/PCA100 representation** is retained exactly; no HVG/PCA refit is performed. "
        "Its unsupervised fit includes all 314,387 lineage-assigned expression profiles, including temporal and cell holdouts. "
        "The classifier's additional PC standardizer is fitted on selected-time training cells only. "
        "Age, cluster IDs and annotation fields are excluded from predictor inputs. Outside-range cells are excluded from this classifier experiment.", "",
        "The selected-time cache contains PC coordinates and annotations, with its expression matrix omitted. "
        "Every cached coordinate was checked against the original PCA by source row and cell ID.", "",
        "## Accuracy by time window", "",
        "| Window | Evaluation cells | Cells | Accuracy | Mean class recall | Macro-F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for _, row in window_scores(output_dir, group, selected).iterrows():
        lines.append(f"| {row.window} h | {row.evaluation_kind} | {int(row.n_cells):,} | "
                     f"{row.accuracy:.2%} | {row.balanced_accuracy:.2%} | {row.macro_f1:.2%} |")
    lines += ["",
        "## Training and validation selection", "",
        "Five established recipes were retrained: wide, stronger class weighting, residual, residual with mixup/noise/label smoothing/EMA, "
        "and joint branch+tissue prediction. The joint recipe uses **46 selected-time branch labels** as an auxiliary target; its tissue head is evaluated here. "
        "No auxiliary supervision comes from intermediate windows. The models use the existing AdamW, five-epoch warmup/cosine schedule, "
        "batch size 2,048, gradient clipping at 5, a 160-epoch limit and 30-epoch early-stopping patience. "
        "Checkpoint selection, scalar temperature calibration and greedy ensembles of at most three members use selected-time validation cells only. "
        "This is one initialization and split seed.", "",
        "| Recipe | Validation accuracy | Validation macro-F1 | Selected-time test accuracy | Intermediate pooled accuracy | Selected |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    methods = list(dict.fromkeys(record["method"] for record in all_records))
    for recipe in methods:
        cell_record = next(record for record in all_records if record["method"] == recipe and record["population"] == "selected_time_test")
        temporal = next(record for record in all_records if record["method"] == recipe and record["population"] == "intermediate_pooled")
        lines.append(f"| {recipe} | {cell_record['validation']['accuracy']:.2%} | {cell_record['validation']['macro_f1']:.2%} | "
                     f"{cell_record['test']['accuracy']:.2%} | {temporal['test']['accuracy']:.2%} | {'yes' if cell_record['selected'] else ''} |")
    lines += ["", "## Classifier confidence", "",
        "| Evaluation population | Coverage at probability ≥0.9 | Accuracy among those cells | ECE (15 bins) |",
        "| --- | ---: | ---: | ---: |"]
    for population in POPULATIONS:
        row = selected[population]["test"]
        confidence = row["confidence_subsets"]["0.9"]
        accuracy = "n/a" if confidence["accuracy"] is None else f"{confidence['accuracy']:.2%}"
        lines.append(f"| {NAMES[population]} | {confidence['coverage']:.2%} | {accuracy} | {row['expected_calibration_error_15bins']:.4f} |")
    lines += ["", "## Per-tissue performance", "",
        "![Per-tissue recall and support](figures/three_time_classifier_recall.png)", "",
        "The per-class CSV files include precision, recall, F1, test support and training-cell support. "
        "Grey cells mark labels absent from the evaluation population. Confusion matrices and per-window metrics are also saved.", "",
        "## Artifacts and verification", "",
        f"Classifier directory: `{output_dir}`. The selected manifest is "
        f"`models/{VARIANT}_{TARGET}_selected.json`, with reloadable neural checkpoints and frozen calibration. "
        "`reports/` contains every candidate score, selected test probabilities, cell IDs and source PCA rows. "
        "The selected probability vectors were reproduced after manifest reload for all four evaluation populations.", "",
        "The audit recomputes all selected metrics, checks disjoint supervised/temporal holdouts, frozen selection and checkpoint hashes, "
        "and verifies unchanged global PCA and reference-oracle artifacts. The four temporal-split checks passed. "
        "`results_validation.json`, `evaluation_complete.json` and `implementation_hashes.json` record verification.", "",
        "The existing all-times ensemble is preserved for reference ('oracle') violation tracking. This newly trained classifier has separate weights "
        "and a 23-label vocabulary that omits `Blastoderm`; graph masks and reference predictions should be aligned by label name when integrating the flow experiment.", "",
        "```sh", "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/envs/maizels2023aa/bin/python \\",
        "  scripts/prepare_calderon_three_time_data.py", "",
        "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 /usr/bin/caffeinate -i \\",
        "  /opt/anaconda3/envs/maizels2023aa/bin/python scripts/train_calderon_three_time_classifier.py --device mps", "```", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    destination = report_dir / "three_time_classifier.md"
    destination.write_text("\n".join(lines))
    figures(output_dir, group, selected, report_dir / "figures")
    print(f"Report: {destination}", flush=True)
    print(json.dumps({population: {key: selected[population]['test'][key] for key in ('n_cells', 'accuracy', 'balanced_accuracy', 'macro_f1')}
                      for population in POPULATIONS}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / f"datasets/calderon2022/{VARIANT}")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"calderon-classifiers/{VARIANT}")
    args = parser.parse_args()
    report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

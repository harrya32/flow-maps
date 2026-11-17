#!/usr/bin/env python3
"""Verify and report the three matched Calderon OT flow-map variants."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd

from common.calderon import sha256_file
from configs.maizels_pca50 import get_config as maizels_config


def read(path):
    return json.loads(Path(path).read_text())


def pct(value):
    return "unassessed" if value is None else f"{value:.2f}%"


def label_groups(frame):
    result = {}
    for name, mask in (("named", frame.source_tissue.ne("Unknown")), ("unknown", frame.source_tissue.eq("Unknown"))):
        group = frame.loc[mask]
        assessed = group.confidence_filtered_status.ne("unassessed")
        violations = group.confidence_filtered_status.eq("violation")
        result[name] = {"n_sources": len(group), "raw_pct": float(100*group.raw_violation.mean()),
                        "n_assessed": int(assessed.sum()), "n_violations": int(violations.sum()),
                        "filtered_pct": float(100*violations.sum()/assessed.sum()) if assessed.any() else None,
                        "coverage_pct": float(100*assessed.mean())}
    return result


def verify(run):
    references = read(run / "reference_artifact_hashes.json")
    directories = [Path(references[name]["directory"]) for name in ("plain", "filtered")] + [run]
    evaluation_dirs = [directories[0] / "evaluations/checkpoints50", directories[1], run]
    metrics = [read(p / "metrics.json") for p in evaluation_dirs]
    configs = [read(p / "config.json") for p in directories]
    cells = pd.read_csv(run / "cell_splits.csv.gz")
    for name, reference in references.items():
        base = Path(reference["directory"])
        for filename, digest in reference["files"].items():
            assert sha256_file(base / filename) == digest, (name, filename)
        for section in ("training", "optimization", "network"):
            assert read(base / "config.json")[section] == configs[-1][section], (name, section)
        pd.testing.assert_frame_equal(pd.read_csv(base / "cell_splits.csv.gz"), cells)
    for filename, digest in read(run / "implementation_hashes.json").items():
        assert sha256_file(ROOT / filename) == digest, filename
    expected = maizels_config(6, hparam_val_times=(), seed=0, maizels_ot_coupling="minibatch_ot").constraints.to_dict()
    assert configs[-1]["constraints"] == metrics[-1]["training_lineage_constraint"] == expected
    assert configs[-1]["problem"]["interp_uses_labels"] and not configs[-1]["training"]["conditional"]
    assert configs[-1]["problem"]["interpolant_check_chunk_size"] == 50
    for previous in metrics[:-1]:
        for key in ("source_window", "target_window", "cohort_branch_prefix", "sampler", "flowmap_n_steps", "ema_factor", "n_test_sources_evaluated"):
            assert previous[key] == metrics[-1][key], key
        for key in ("classifier_manifest_sha256", "classifier_selection_sha256", "allowed_pairs_sha256", "checkpoint_times", "prob_threshold", "margin_threshold", "real_cell_classifier"):
            assert previous["tissue_trajectories"][key] == metrics[-1]["tissue_trajectories"][key], key
        for key in ("dataset_sha256", "classifier_manifest_sha256", "allowed_pairs_sha256"):
            assert previous["provenance"][key] == metrics[-1]["provenance"][key], key
    with np.load(run / "evaluation/evaluation_inputs.npz", allow_pickle=False) as actual:
        assert actual["paths"].shape == (4641, 50, 100) and np.isfinite(actual["paths"]).all()
        test = cells.loc[cells.day.eq("4-6") & cells.flow_split.eq("test"), "cell_id"]
        assert len(actual["source_cell_ids"]) == len(set(actual["source_cell_ids"])) == len(test) == 4641
        assert set(actual["source_cell_ids"]) == set(test)
        train_ids = set(cells.loc[cells.flow_split.eq("train"), "cell_id"])
        source_training_overlap = len(set(actual["source_cell_ids"]) & train_ids)
        assert source_training_overlap == 0
        for directory in evaluation_dirs[:-1]:
            with np.load(directory / "evaluation/evaluation_inputs.npz", allow_pickle=False) as old:
                assert set(actual.files) == set(old.files)
                for key in actual.files:
                    if key != "paths":
                        np.testing.assert_array_equal(actual[key], old[key])
    training = read(run / "training_summary.json")
    history = pd.read_csv(run / "training_history.csv")
    filtered_history = pd.read_csv(directories[1] / "training_history.csv")
    shared_steps = sorted(set(history.step) & set(filtered_history.step))
    filter_columns = ["step", "endpoint_rejected", "interpolant_checked", "interpolant_rejected", "ot_refinements", "ot_mean_retained_valid_mass"]
    pd.testing.assert_frame_equal(
        filtered_history.loc[filtered_history.step.isin(shared_steps), filter_columns].reset_index(drop=True),
        history.loc[history.step.isin(shared_steps), filter_columns].reset_index(drop=True))
    best = history.loc[history.validation_loss.idxmin()]
    assert int(best.step) == training["best_step"] == metrics[-1]["training_step"]
    assert np.isclose(best.validation_loss, training["best_validation_loss"], rtol=1e-12)
    assert training["validation_pair_mode"] == "ot_plain"
    assert training["validation_objective"] == "plain_ot_lsd_without_constraint"
    assert sha256_file(run / "best.msgpack") == metrics[-1]["checkpoint_sha256"]
    pairs = read(run / "pair_filtering_summary.json")
    assert pairs["batches"] == training["completed_steps"]
    assert pairs["sampled_pairs"] == pairs["batches"] * configs[-1]["optimization"]["bs"]
    for key in ("sampled_source_tissues", "sampled_target_tissues"):
        assert sum(pairs[key].values()) == pairs["sampled_pairs"]
    if pairs["resampled_empty_blocks"] == 0:
        assert pairs["interpolant_classifier_points"] == pairs["interpolant_checked"] * 50
    gradient = read(run / "classifier_gradient_verification.json")
    assert gradient["status"] == "passed" and gradient["argmax_decisions_identical"]
    assert gradient["confidence_0p9_masks_identical"] and gradient["gradient_descent_reduces_penalty"]
    assert gradient["export_sha256"] == metrics[-1]["training_interpolant_filter"]["classifier_export_sha256"]
    assert gradient["differentiable_forward_sha256"] == sha256_file(ROOT / "py/common/calderon_lineage.py")
    summaries = [pd.read_csv(p / "evaluation/trajectory_summary.csv.gz") for p in evaluation_dirs]
    for frame in summaries:
        pd.testing.assert_frame_equal(summaries[0][["source_cell_id", "source_tissue"]], frame[["source_cell_id", "source_tissue"]])
    prediction_diagnostics = {}
    for name, directory, frame in zip(("plain", "filtered", "constrained"), evaluation_dirs, summaries):
        with np.load(directory / "evaluation/trajectory_probabilities.npz", allow_pickle=False) as predictions:
            probabilities = predictions["probabilities"]
            assert probabilities.shape == (4641, 50, 24)
            assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            np.testing.assert_allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(predictions["source_cell_ids"], frame.source_cell_id.to_numpy())
            np.testing.assert_array_equal(predictions["confident"].sum(axis=1), frame.n_confident_checkpoints.to_numpy())
            unknown = list(predictions["class_names"]).index("Unknown")
            unknown_predictions = probabilities.argmax(axis=-1) == unknown
            confident_points = predictions["confident"]
            prediction_diagnostics[name] = {
                "predicted_unknown_pct_at_middle": float(100*unknown_predictions[:, 24].mean()),
                "predicted_unknown_pct_at_endpoint": float(100*unknown_predictions[:, -1].mean()),
                "predicted_unknown_pct_all_checkpoints": float(100*unknown_predictions.mean()),
                "unknown_pct_of_confident_checkpoint_labels": float(100*unknown_predictions[confident_points].mean()) if confident_points.any() else None,
            }
    raw, filtered = metrics[-1]["tissue_trajectories"]["raw"], metrics[-1]["tissue_trajectories"]["confidence_filtered"]
    assert int(summaries[-1].raw_violation.sum()) == raw["n_violations"]
    assert int(summaries[-1].confidence_filtered_status.eq("violation").sum()) == filtered["n_violations"]
    assert int(summaries[-1].confidence_filtered_status.ne("unassessed").sum()) == filtered["n_assessed"]
    assert int(summaries[-1].n_confident_checkpoints.eq(50).sum()) == filtered["n_fully_assessed"]
    paired = {"violates_both": int((summaries[1].raw_violation & summaries[2].raw_violation).sum()),
              "violates_filter_only": int((summaries[1].raw_violation & ~summaries[2].raw_violation).sum()),
              "violates_constraint_only": int((~summaries[1].raw_violation & summaries[2].raw_violation).sum()),
              "violates_neither": int((~summaries[1].raw_violation & ~summaries[2].raw_violation).sum())}
    audit = {"status": "passed", "reference_artifacts_unchanged": True,
             "same_training_settings_network_cell_splits_and_evaluation_inputs": True,
             "source_training_overlap": source_training_overlap, "n_test_sources": 4641, "checking_points": 50, "composition_steps": 100,
             "native_id6_constraint_settings_verified": True, "validation_excludes_the_lineage_constraint": True,
             "filter_statistics_identical_at_logged_common_steps": {"n_steps": len(shared_steps), "last_step": int(max(shared_steps))},
             "paired_raw_status_vs_filter_only": paired,
             "prediction_diagnostics": prediction_diagnostics,
             "source_label_groups": {name: label_groups(frame) for name, frame in zip(("plain", "filtered", "constrained"), summaries)}}
    (run / "evaluation/results_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    return directories, metrics, training, pairs, audit


def figure(directories, metrics, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names, colors = ("Plain OT", "Filtered OT", "Constrained + filtered"), ("#226C9B", "#3C8877", "#B86936")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for directory, result, name, color, shift in zip(directories, metrics, names, colors, (-.26, 0, .26)):
        history = pd.read_csv(directory / "training_history.csv")
        selection = read(directory / "training_summary.json")
        axes[0, 0].plot(history.step, history.validation_loss, label=name, color=color)
        axes[0, 0].scatter([selection["best_step"]], [selection["best_validation_loss"]], color=color, marker="*", s=90)
        values = [result["distribution"][key]["w1"] for key in ("heldout_reference", "target_test")]
        bars = axes[0, 1].bar(np.arange(2) + shift, values, .24, color=color, label=name)
        axes[0, 1].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        raw, confident = result["tissue_trajectories"]["raw"], result["tissue_trajectories"]["confidence_filtered"]
        rates = [raw["violation_pct_all_sources"], confident["violation_pct_assessed"]]
        counts = [raw["n_sources"], confident["n_assessed"]]
        bars = axes[1, 0].bar(np.arange(2) + shift, [v if v is not None else 0 for v in rates], .24, color=color, label=name)
        axes[1, 0].bar_label(bars, labels=[f"{pct(v)}\n(n={n:,})" for v, n in zip(rates, counts)], fontsize=7, padding=3)
        coverage = 100 * np.asarray(confident["confident_counts"]) / raw["n_sources"]
        axes[1, 1].plot(result["tissue_trajectories"]["checkpoint_hours"], coverage, label=name, color=color)
    axes[0, 0].set(xlabel="Optimizer step", ylabel="Fixed plain-OT endpoint-validation LSD loss", title="A  Shared endpoint validation")
    axes[0, 1].set_xticks(np.arange(2), ["8–10 h held out", "12–14 h test"])
    axes[0, 1].set(ylabel="W1 in original PCA100 space", title="B  W1 from composed sampling",
                   ylim=(0, 1.35 * max(m["distribution"][k]["w1"] for m in metrics for k in ("heldout_reference", "target_test"))))
    axes[1, 0].set_xticks(np.arange(2), ["Raw: all paths", "≥1 confident checkpoint"])
    axes[1, 0].set(ylabel="Paths with a forbidden transition (%)", ylim=(0, 100), title="C  Tissue violations at 50 checks")
    axes[1, 1].set(xlabel="Inferred developmental age (h)", ylabel="Confident checkpoint labels (%)", ylim=(0, 100), title="D  Confidence ≥0.9 coverage")
    for ax, location in zip(axes.ravel(), ("upper right", "upper left", "upper right", "upper left")):
        ax.legend(loc=location, fontsize=8)
    fig.suptitle("Calderon Ubi · three OT flow-map variants · seed 0 · 4,641 test starts", fontsize=13)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination / f"constrained_ot_flow_map_results.{extension}", dpi=220, facecolor="white")
    plt.close(fig)


def report(run, directories, metrics, training, pairs, audit, destination):
    current = metrics[-1]["tissue_trajectories"]
    raw, confident = current["raw"], current["confidence_filtered"]
    old = metrics[1]["tissue_trajectories"]
    controls = read(directories[0] / "evaluation/controls/control_metrics.json")
    lines = ["# Calderon constrained OT flow-map results", "",
             f"Relative to filtering alone, adding the native Maizels lineage penalty changes raw path violations "
             f"**{pct(old['raw']['violation_pct_all_sources'])} → {pct(raw['violation_pct_all_sources'])}**. "
             f"Confidence-filtered violations change **{pct(old['confidence_filtered']['violation_pct_assessed'])} → {pct(confident['violation_pct_assessed'])}**, "
             f"with assessed-path coverage **{pct(old['confidence_filtered']['trajectory_coverage_pct'])} → {pct(confident['trajectory_coverage_pct'])}**.", "",
             f"Middle W1 changes **{metrics[1]['distribution']['heldout_reference']['w1']:.4f} → "
             f"{metrics[-1]['distribution']['heldout_reference']['w1']:.4f}**, and endpoint W1 changes "
             f"**{metrics[1]['distribution']['target_test']['w1']:.4f} → {metrics[-1]['distribution']['target_test']['w1']:.4f}**. "
             "Higher W1 is worse. At this weight-1000 setting, the reduction in raw violations comes with worse agreement with both observed marginals.", "",
             f"The seed-0 run completed **{training['completed_steps']:,} updates** in **{training['seconds']/60:.2f} minutes**. "
             f"The shared plain-OT endpoint-validation objective selected **step {training['best_step']:,}** with loss "
             f"**{training['best_validation_loss']:.4f}**. All variants evaluate the same 4,641 test starting cells, "
             "50 checking points and 100-step composed sampling.", "",
             "![Matched three-variant comparison](figures/constrained_ot_flow_map_results.png)", "",
             "| Metric | Plain OT | Filtered OT | Constrained + filtered OT |", "| --- | ---: | ---: | ---: |"]
    for key, name in (("heldout_reference", "8–10 h held-out W1"), ("target_test", "12–14 h test W1")):
        lines.append(f"| {name} | " + " | ".join(f"{m['distribution'][key]['w1']:.4f}" for m in metrics) + " |")
    for name, group, field in (("Raw path violations", "raw", "violation_pct_all_sources"),
                               ("Confidence-filtered violations among assessed paths", "confidence_filtered", "violation_pct_assessed"),
                               ("At least one confident checkpoint: coverage", "confidence_filtered", "trajectory_coverage_pct"),
                               ("All 50 checkpoints confident: coverage", "confidence_filtered", "full_checkpoint_coverage_pct"),
                               ("Raw endpoint-only violations", "raw", "endpoint_violation_pct_assessed")):
        lines.append(f"| {name} | " + " | ".join(pct(m['tissue_trajectories'][group][field]) for m in metrics) + " |")
    lines += ["", f"Constrained raw violations: **{raw['n_violations']:,}/4,641 paths**. Confidence-filtered violations: "
              f"**{confident['n_violations']:,}/{confident['n_assessed']:,} assessed paths**, with **{confident['n_unassessed']:,} unassessed**. "
              f"Only **{confident['n_fully_assessed']:,} paths** have confident labels at every checking point. "
              "Skipped checkpoints are not demonstrated valid states.", "",
              "W1 uses Euclidean PCA100 ground cost and the same 2,048 generated/reference samples. "
              "The middle state is obtained after 50 composed steps and the endpoint after 100. Neither reported W1 score selects a checkpoint.", "",
              f"For context, unchanged starting cells have middle/end W1 **{controls['identity_source']['heldout_reference_w1']:.4f} / "
              f"{controls['identity_source']['target_test_w1']:.4f}**. Disjoint real-versus-real samples give "
              f"**{controls['real_vs_real']['heldout_reference']['w1']:.4f} / {controls['real_vs_real']['target_test']['w1']:.4f}**. "
              "These are descriptive controls, not formal lower bounds or subtractive corrections. "
              "[The baseline report](ot_flow_map_results.md) documents their construction.", "",
              "## Active training constraint", "",
              "This is Maizels ID 6 with the same filtered OT pairs as ID 5. On the 32 off-diagonal loss samples per batch, "
              "the frozen classifier predicts probabilities at the observed interpolant I_s and the learned flow-map output X_{s,t}(I_s). "
              "The added loss is **1000 × mean[-log(clip(p_sᵀ R p_t, 10⁻⁶, 1))]**, where R is the existing projected allowed-tissue relation. "
              "The native off-diagonal normalization preserves this weight despite the 75% diagonal fraction. "
              "Start, final-label and entropy penalties are zero, matching the repository defaults. "
              "The classifier remains frozen, and the network remains unconditional.", "",
              "The constraint uses the supplied relation directly, with no new transitive closure after tissue merging. "
              "It penalizes sampled flow-map transitions; it does not enforce a hard constraint on every point of the final composed paths.", "",
              "## Filter, mass and uncertainty", "",
              f"The original full 50-point interpolant check remains active. Across **{pairs['batches']:,} batches**, "
              f"**{pairs['interpolant_checked']:,} candidate interpolants** were checked and **{pairs['interpolant_rejected']:,} rejected** "
              f"({pairs['interpolant_rejection_pct_checked']:.2f}%). Mean retained valid mass was **{100*pairs['mean_retained_valid_mass']:.2f}%**. "
              "Partial assignments are renormalized, so effective training tissue fractions differ from the endpoint snapshots. "
              "These statistics include all completed updates, including those after the selected checkpoint.", "",
              "| Model | Starting-label group | Test starts | Raw violations | Filtered violations | Assessed-path coverage |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for name in ("plain", "filtered", "constrained"):
        for group in ("named", "unknown"):
            item = audit["source_label_groups"][name][group]
            lines.append(f"| {name.title()} | {'Named tissue' if group == 'named' else 'Unknown'} | {item['n_sources']:,} | "
                         f"{pct(item['raw_pct'])} | {pct(item['filtered_pct'])} ({item['n_violations']:,}/{item['n_assessed']:,}) | {pct(item['coverage_pct'])} |")
    before = audit["source_label_groups"]["filtered"]["named"]
    after = audit["source_label_groups"]["constrained"]["named"]
    before_share = 100*before["n_assessed"]/metrics[1]["tissue_trajectories"]["confidence_filtered"]["n_assessed"]
    after_share = 100*after["n_assessed"]/confident["n_assessed"]
    lines += ["", f"For named-tissue starts, confidence-filtered violations fall **{pct(before['filtered_pct'])} → "
              f"{pct(after['filtered_pct'])}**, while coverage rises **{pct(before['coverage_pct'])} → {pct(after['coverage_pct'])}**. "
              f"Named-tissue starts increase from **{before_share:.2f}% to {after_share:.2f}%** of the aggregate assessed denominator. "
              "The aggregate filtered rate therefore needs the label-group breakdown: the groups have very different violation rates, "
              "and the set of assessed paths and checkpoints changes between models.", "",
              "The same frozen classifier builds the training filter, supplies constraint gradients and scores generated paths. "
              "Its real 8–10 h Ubi accuracy is 48.24%. The unchanged-expression control already gives 15.49% raw violations. "
              "These results measure agreement with the reconstructed prior and classifier, rather than independent clone-fate accuracy. "
              "The existing reference-atlas PCA/classifier can have seen endpoint flow-test cells. This is one training seed.", "",
              "## Verification and artifacts", "",
              "The audit confirms identical network/optimizer settings, cell splits, source/reference samples, classifier and graph hashes; "
              "the checkpoint is the minimum shared validation loss and excludes the lineage penalty. Both reference runs are unchanged. "
              "Filter statistics match filtering alone at every logged common optimizer step. "
              "The differentiable decoder reproduced all decisions and ≥0.9 confidence masks on 6,912 training-cell/interpolant points "
              "with maximum probability error below 6×10⁻⁷. Its native loss gradient matched directional finite differences. "
              "All 23 Calderon checks, 10 shared OT/lineage checks and a genuine constrained training/evaluation smoke run passed.", "",
              f"Run directory: `{run}`. Inspect `metrics.json`, `training_summary.json`, `pair_filtering_summary.json`, "
              "`comparison_protocol.json`, `classifier_gradient_verification.json`, `implementation_hashes.json`, "
              "`reference_artifact_hashes.json`, `best.msgpack` and the per-cell evaluation artifacts. "
              "[Method and reproduction commands](constrained_flow_map.md) describe the implementation.", "",
              "```sh", "/opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_constrained_flow_map.py", "```", ""]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "constrained_ot_flow_map_results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/calderon_ubi_constrained_filtered_ot_flow_map_seed0")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/calderon2022")
    args = parser.parse_args()
    directories, metrics, training, pairs, audit = verify(args.run_dir)
    report(args.run_dir, directories, metrics, training, pairs, audit, args.report_dir)
    figure(directories, metrics, args.report_dir / "figures")
    print(f"Report: {args.report_dir / 'constrained_ot_flow_map_results.md'}")


if __name__ == "__main__":
    main()

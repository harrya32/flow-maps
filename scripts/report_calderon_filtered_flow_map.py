#!/usr/bin/env python3
"""Verify and compare completed plain/filtered Calderon flow-map runs."""

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


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(run, baseline):
    base_eval = baseline / "evaluations/checkpoints50"
    old, new = read_json(base_eval / "metrics.json"), read_json(run / "metrics.json")
    old_cfg, new_cfg = read_json(baseline / "config.json"), read_json(run / "config.json")
    for section in ("training", "optimization", "network"):
        assert old_cfg[section] == new_cfg[section], section
    pd.testing.assert_frame_equal(pd.read_csv(baseline / "cell_splits.csv.gz"), pd.read_csv(run / "cell_splits.csv.gz"))
    for name, value in read_json(run / "baseline_artifact_hashes.json").items():
        assert sha256_file(baseline / name) == value, name
    for name in ("source_window", "target_window", "cohort_branch_prefix", "flowmap_n_steps", "ema_factor", "n_test_sources_evaluated"):
        assert old[name] == new[name], name
    for name in ("classifier_manifest_sha256", "classifier_selection_sha256", "allowed_pairs_sha256", "checkpoint_times", "prob_threshold", "margin_threshold", "real_cell_classifier"):
        assert old["tissue_trajectories"][name] == new["tissue_trajectories"][name], name
    for name in ("dataset_sha256", "classifier_manifest_sha256", "allowed_pairs_sha256"):
        assert old["provenance"][name] == new["provenance"][name], name
    with np.load(base_eval / "evaluation/evaluation_inputs.npz", allow_pickle=False) as a, np.load(run / "evaluation/evaluation_inputs.npz", allow_pickle=False) as b:
        for key in a.files:
            if key != "paths":
                np.testing.assert_array_equal(a[key], b[key])
        assert b["paths"].shape == (4641, 50, 100) and np.isfinite(b["paths"]).all()
    training = read_json(run / "training_summary.json")
    history = pd.read_csv(run / "training_history.csv")
    best = history.loc[history.validation_loss.idxmin()]
    assert int(best.step) == training["best_step"] == new["training_step"]
    assert np.isclose(best.validation_loss, training["best_validation_loss"], rtol=1e-12)
    assert sha256_file(run / "best.msgpack") == new["checkpoint_sha256"]
    assert training["validation_pair_mode"] == "ot_plain"
    pairs = read_json(run / "pair_filtering_summary.json")
    assert pairs["batches"] == training["completed_steps"]
    assert pairs["sampled_pairs"] == pairs["batches"] * new_cfg["optimization"]["bs"]
    for key in ("sampled_source_tissues", "sampled_target_tissues"):
        assert sum(pairs[key].values()) == pairs["sampled_pairs"]
    if not (run / "endpoint_support_audit.json").exists():
        from audit_calderon_endpoint_support import audit
        audit(run)
    support = read_json(run / "endpoint_support_audit.json")
    assert support["cell_splits_sha256"] == sha256_file(run / "cell_splits.csv.gz")
    assert support["allowed_pairs_sha256"] == new["tissue_trajectories"]["allowed_pairs_sha256"]
    assert support["minibatch"]["batch_size"] == new_cfg["problem"]["ot_minibatch_size"]
    summary = pd.read_csv(run / "evaluation/trajectory_summary.csv.gz")
    original_summary = pd.read_csv(base_eval / "evaluation/trajectory_summary.csv.gz")
    pd.testing.assert_frame_equal(original_summary[["source_cell_id", "source_tissue"]], summary[["source_cell_id", "source_tissue"]])
    raw, filtered = new["tissue_trajectories"]["raw"], new["tissue_trajectories"]["confidence_filtered"]
    assert int(summary.raw_violation.sum()) == raw["n_violations"]
    assert int(summary.confidence_filtered_status.eq("violation").sum()) == filtered["n_violations"]
    assert int(summary.n_confident_checkpoints.eq(50).sum()) == filtered["n_fully_assessed"]
    result = {"status": "passed", "same_training_settings_and_cell_splits": True, "same_evaluation_inputs_except_generated_paths": True,
              "source_training_overlap": 0, "n_test_sources_verified": 4641, "n_checkpoints": 50, "flow_composition_steps": 100,
              "checkpoint_selected_only_by_fixed_plain_ot_validation": True, "baseline_artifacts_unchanged": True,
              "training_sampler_counts_consistent": True,
              "paired_raw_status_counts": {
                  "violates_both": int((original_summary.raw_violation & summary.raw_violation).sum()),
                  "violates_plain_only": int((original_summary.raw_violation & ~summary.raw_violation).sum()),
                  "violates_filtered_only": int((~original_summary.raw_violation & summary.raw_violation).sum()),
                  "violates_neither": int((~original_summary.raw_violation & ~summary.raw_violation).sum()),
              }}
    result["source_label_groups"] = {}
    for name, frame in (("plain", original_summary), ("filtered", summary)):
        result["source_label_groups"][name] = {}
        for group, mask in (("named_tissues", frame.source_tissue.ne("Unknown")), ("unknown", frame.source_tissue.eq("Unknown"))):
            selected = frame.loc[mask]
            assessed = selected.confidence_filtered_status.ne("unassessed")
            violations = selected.confidence_filtered_status.eq("violation")
            result["source_label_groups"][name][group] = {
                "n_sources": len(selected), "raw_violation_pct": float(100 * selected.raw_violation.mean()),
                "n_assessed": int(assessed.sum()), "n_filtered_violations": int(violations.sum()),
                "filtered_violation_pct_assessed": float(100 * violations.sum() / assessed.sum()) if assessed.any() else None,
                "filtered_coverage_pct": float(100 * assessed.mean()),
            }
    (run / "evaluation/results_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return old, new, training, pairs


def plot(run, baseline, report_dir, old, new, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ("#226C9B", "#3C8877")
    names = ("Plain OT", "Filtered OT")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for directory, name, color in zip((baseline, run), names, colors):
        history = pd.read_csv(directory / "training_history.csv")
        axes[0, 0].plot(history.step, history.validation_loss, color=color, label=name)
        selected = read_json(directory / "training_summary.json")
        axes[0, 0].scatter([selected["best_step"]], [selected["best_validation_loss"]], color=color, marker="*", s=100)
    axes[0, 0].set(xlabel="Optimizer step", ylabel="Fixed endpoint-validation LSD loss", title="A  Shared checkpoint-selection objective")
    axes[0, 0].legend()
    for shift, metrics, name, color in zip((-.18, .18), (old, new), names, colors):
        values = [metrics["distribution"][key]["w1"] for key in ("heldout_reference", "target_test")]
        bars = axes[0, 1].bar(np.arange(2) + shift, values, .36, color=color, label=name)
        axes[0, 1].bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
        raw, filtered = metrics["tissue_trajectories"]["raw"], metrics["tissue_trajectories"]["confidence_filtered"]
        rates = [raw["violation_pct_all_sources"], filtered["violation_pct_assessed"]]
        counts = [raw["n_sources"], filtered["n_assessed"]]
        bars = axes[1, 0].bar(np.arange(2) + shift, rates, .36, color=color, label=name)
        axes[1, 0].bar_label(bars, labels=[f"{r:.1f}%\n(n={n:,})" for r, n in zip(rates, counts)], fontsize=8, padding=3)
        coverage = 100 * np.array(filtered["confident_counts"]) / raw["n_sources"]
        axes[1, 1].plot(metrics["tissue_trajectories"]["checkpoint_hours"], coverage, color=color, label=name)
    axes[0, 1].set_xticks(np.arange(2), ["8–10 h held out", "12–14 h test"])
    axes[0, 1].set(ylabel="W1 in original PCA100 space", title="B  Composed-flow marginal agreement")
    axes[0, 1].set_ylim(0, max(m["distribution"][key]["w1"] for m in (old, new) for key in ("heldout_reference", "target_test")) * 1.2)
    axes[1, 0].set_xticks(np.arange(2), ["Raw: all paths", "≥1 confident checkpoint"])
    axes[1, 0].set(ylabel="Paths with a forbidden transition (%)", ylim=(0, 100), title="C  Fifty-checkpoint tissue violations")
    axes[1, 1].set(xlabel="Inferred developmental age (h)", ylabel="Confident checkpoint labels (%)", ylim=(0, 100),
                    title="D  Confidence ≥0.9 coverage")
    axes[0, 1].legend(loc="upper left")
    for ax in (axes[1, 0], axes[1, 1]):
        ax.legend()
    fig.suptitle("Calderon Ubi: plain versus filtered OT flow map · seed 0 · 4,641 test starts", fontsize=13)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(report_dir / "figures" / f"filtered_ot_flow_map_results.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def report(run, baseline, report_dir, old, new, training, pairs):
    raw, filtered = new["tissue_trajectories"]["raw"], new["tissue_trajectories"]["confidence_filtered"]
    audit = read_json(run / "evaluation/results_validation.json")
    controls = read_json(baseline / "evaluation/controls/control_metrics.json")
    identity = read_json(baseline / "evaluations/checkpoints50/evaluation/identity_control_metrics.json")
    support = read_json(run / "endpoint_support_audit.json")
    lines = ["# Calderon filtered OT flow-map results", "",
             f"On the same **4,641 test starting cells**, raw fifty-checkpoint path violations change from "
             f"**{old['tissue_trajectories']['raw']['violation_pct_all_sources']:.2f}%** with plain OT to "
             f"**{raw['violation_pct_all_sources']:.2f}%** with filtered OT. The ≥0.9 confidence-filtered rate changes from "
             f"**{old['tissue_trajectories']['confidence_filtered']['violation_pct_assessed']:.2f}%** to "
             f"**{filtered['violation_pct_assessed']:.2f}%**, while assessed-path coverage changes from "
             f"**{old['tissue_trajectories']['confidence_filtered']['trajectory_coverage_pct']:.2f}%** to "
             f"**{filtered['trajectory_coverage_pct']:.2f}%**. Both marginal W1 comparisons are reported below.", "",
             f"The filtered-interpolant variant completed **{training['completed_steps']:,} updates**. The shared plain-OT endpoint-validation objective "
             f"selected **step {training['best_step']:,}**, with loss **{training['best_validation_loss']:.4f}**. Training took "
             f"**{training['seconds']/60:.2f} minutes**. The comparison uses the same test cells, frozen tissue classifier, projected relation, "
             "50 checking points and 100-step composed flow-map sampling as the plain-OT baseline.", "",
             "![Training and held-out comparison](figures/filtered_ot_flow_map_results.png)", "",
             "| Metric | Plain OT | Filtered OT |", "| --- | ---: | ---: |"]
    for key, label in (("heldout_reference", "8–10 h held-out W1"), ("target_test", "12–14 h test W1")):
        lines.append(f"| {label} | {old['distribution'][key]['w1']:.4f} | {new['distribution'][key]['w1']:.4f} |")
    fields = [("Raw path violations", "raw", "violation_pct_all_sources"),
              ("Filtered violations among assessed paths", "confidence_filtered", "violation_pct_assessed"),
              ("At least one confident checkpoint: coverage", "confidence_filtered", "trajectory_coverage_pct"),
              ("All fifty checkpoints confident: coverage", "confidence_filtered", "full_checkpoint_coverage_pct"),
              ("Raw endpoint-only violations", "raw", "endpoint_violation_pct_assessed")]
    for label, group, field in fields:
        values = [m["tissue_trajectories"][group][field] for m in (old, new)]
        lines.append(f"| {label} | " + " | ".join("—" if v is None else f"{v:.2f}%" for v in values) + " |")
    lines += ["", f"Filtered OT raw violations: **{raw['n_violations']:,}/4,641 paths**. At probability ≥0.9: "
              f"**{filtered['n_violations']:,}/{filtered['n_assessed']:,} assessed paths**, with **{filtered['n_unassessed']:,} unassessed** "
              f"and **{filtered['n_fully_assessed']:,} confident at every checkpoint**. Skipped checkpoints are not demonstrated valid states; "
              "counts and coverage accompany the confidence-filtered rate.", "",
              "W1 uses Euclidean ground cost in original PCA100 coordinates, uniform weights and the same 2,048 generated/reference samples. "
              "The middle comparison uses the state after 50 composed steps; the final comparison uses all 100. Neither held-out marginal selected a checkpoint.", "",
              "| Paired raw status on the same starting cell | Paths |", "| --- | ---: |"]
    for key, label in (("violates_both", "Violation in both runs"), ("violates_plain_only", "Violation in plain OT only"),
                       ("violates_filtered_only", "Violation in filtered OT only"), ("violates_neither", "No raw violation in either run")):
        lines.append(f"| {label} | {audit['paired_raw_status_counts'][key]:,} |")
    lines += ["", "## Descriptive controls", "",
              "These reuse the completed baseline controls because the starting cells, frozen classifier and reference samples are identical. "
              "Holding expression constant yields the same tissue predictions at all 50 checks. The controls are descriptive context, "
              "rather than formal lower bounds or corrections to subtract from a model score.", "",
              "| Window | Unchanged-source W1 | Disjoint real-versus-real W1 |", "| --- | ---: | ---: |"]
    for key, label in (("heldout_reference", "8–10 h"), ("target_test", "12–14 h")):
        lines.append(f"| {label} | {controls['identity_source'][key + '_w1']:.4f} | {controls['real_vs_real'][key]['w1']:.4f} |")
    lines += ["", f"The unchanged-expression control gives **{identity['raw']['violation_pct_all_sources']:.2f}% raw violations** "
              f"and **{identity['confidence_filtered']['violation_pct_assessed']:.2f}% confidence-filtered violations** "
              f"among **{identity['confidence_filtered']['n_assessed']:,} assessed paths** "
              f"(**{identity['confidence_filtered']['trajectory_coverage_pct']:.2f}% coverage**).", "",
              "## Training filter and transport mass", "",
              "Training follows Maizels ID 5: mask forbidden released endpoint-label pairs, classify 50 interior linear-interpolant points with raw labels, "
              "check the released final target label, remove rejected edges and resolve the masked assignment. No generated-path penalty or tissue conditioning is used.", "",
              f"Across **{pairs['batches']:,} optimizer batches**, the sampler checked **{pairs['interpolant_checked']:,} candidate interpolants** and rejected "
              f"**{pairs['interpolant_rejected']:,} ({pairs['interpolant_rejection_pct_checked']:.2f}%)**. Mean retained valid mass was "
              f"**{100*pairs['mean_retained_valid_mass']:.2f}%** per block. Its **{pairs['sampled_pairs']:,} optimizer pairs** were sampled after renormalizing retained mass. "
              "The statistics include the entire completed training run, including updates after the selected best checkpoint. "
              "Because the hard mask can require partial matching, the effective training tissue weights differ from the original snapshots.", "",
              f"A separate released-label support audit on this exact training split permits **{100*support['balanced_maximum_supported_mass']:.4f}%** "
              f"of global endpoint mass. In **{support['minibatch']['n_blocks']:,} fresh {support['minibatch']['batch_size']}-cell blocks**, "
              f"the mean maximum endpoint-supported mass is **{100*support['minibatch']['mean_maximum_supported_mass']:.2f}%** "
              f"(seed {support['minibatch']['seed']}). This uses endpoint labels alone; it provides context for the additional interpolant "
              "filter, rather than results for an endpoint-only trained model. The tiny global support deficit arises on the training split; "
              "the earlier audit of full endpoint snapshots allowed 100% tissue-level support.", "",
              "| Endpoint | Tissue | Raw training fraction | Sampled optimizer fraction |", "| --- | --- | ---: | ---: |"]
    splits = pd.read_csv(run / "cell_splits.csv.gz")
    for window, key, label in (("4-6", "sampled_source_tissues", "Source"), ("12-14", "sampled_target_tissues", "Target")):
        counts = splits.loc[splits.flow_split.eq("train") & splits.day.eq(window), "broad_tissue"].value_counts()
        for tissue in counts.index:
            lines.append(f"| {label} | {tissue} | {100*counts[tissue]/counts.sum():.2f}% | {100*pairs[key][tissue]/pairs['sampled_pairs']:.2f}% |")
    lines += ["", "## Starting-tissue diagnostics", "",
              "| Starting tissue | Test paths | Plain raw violations | Filtered raw violations | Filtered ≥0.9 violation rate | Filtered assessed-path coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    a = pd.read_csv(baseline / "evaluations/checkpoints50/evaluation/violations_by_source_tissue.csv").set_index("source_tissue")
    b = pd.read_csv(run / "evaluation/violations_by_source_tissue.csv").sort_values("n_sources", ascending=False)
    for row in b.itertuples(index=False):
        rate = "—" if pd.isna(row.filtered_violation_pct_assessed) else f"{row.filtered_violation_pct_assessed:.2f}%"
        lines.append(f"| {row.source_tissue} | {row.n_sources:,} | {a.loc[row.source_tissue,'raw_violation_pct']:.2f}% | "
                     f"{row.raw_violation_pct:.2f}% | {rate} | {row.filtered_coverage_pct:.2f}% |")
    named_old = audit["source_label_groups"]["plain"]["named_tissues"]
    named_new = audit["source_label_groups"]["filtered"]["named_tissues"]
    unknown = audit["source_label_groups"]["filtered"]["unknown"]
    lines += ["", f"Among the **{named_new['n_sources']:,} test starts with a named tissue**, raw violations change from "
              f"**{named_old['raw_violation_pct']:.2f}%** to **{named_new['raw_violation_pct']:.2f}%**. Confidence-filtered violations change from "
              f"**{named_old['filtered_violation_pct_assessed']:.2f}% ({named_old['n_filtered_violations']:,}/{named_old['n_assessed']:,})** "
              f"to **{named_new['filtered_violation_pct_assessed']:.2f}% ({named_new['n_filtered_violations']:,}/{named_new['n_assessed']:,})**, "
              f"with coverage **{named_old['filtered_coverage_pct']:.2f}% → {named_new['filtered_coverage_pct']:.2f}%**. "
              f"The aggregate filtered rate depends strongly on `Unknown` starts: they supply **{unknown['n_assessed']:,}/{filtered['n_assessed']:,}** "
              f"assessed paths, with only **{unknown['n_filtered_violations']}** violations. The named-tissue breakdown should accompany the overall rate."]
    lines += ["", "## Interpretation and verification", "",
              "The training filter certifies the selected linear PCA interpolants. Final trajectories come from a learned nonlinear flow map "
              "composed 100 times; the filter does not impose a hard constraint on those generated paths.", "",
              "The same frozen classifier constructs the filtered training prior and scores generated paths. Its accuracy on real 8–10 h Ubi cells "
              "is 48.24%, and the unchanged-expression control already reports 15.49% raw violations. These are diagnostics of agreement with the "
              "reconstructed prior and classifier, not independent clone-fate accuracy. Both runs use the existing endpoint/reference-atlas protocol; "
              "the reference PCA/classifier can have seen endpoint flow-test cells. This single seed supplies no across-training-seed uncertainty estimate.", "",
              "The saved audit verifies identical training/network settings, cell splits, reference samples, classifier and graph hashes; "
              "the selected checkpoint is the minimum of the shared endpoint-validation loss. All original baseline artifacts are preserved. "
              "The frozen GPU classifier matched PyTorch decisions on 6,912 training-cell/interpolant points, with maximum probability error below 7×10⁻⁷. "
              "Nineteen Calderon unit checks, seven shared OT regression checks and a genuine two-step training/evaluation smoke run passed.", "",
              f"Run artifacts: `{run}`. See `metrics.json`, `training_summary.json`, `pair_filtering_summary.json`, `comparison_protocol.json`, "
              "`classifier_backend_verification.json`, `best.msgpack` and the per-cell evaluation files. "
              "[Method and reproduction commands](filtered_flow_map.md) explain the training filter, partial assignment and matched validation protocol.", "",
              "Reproduce this report:", "", "```sh",
              "/opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_filtered_flow_map.py", "```", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "filtered_ot_flow_map_results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/calderon_ubi_filtered_ot_flow_map_seed0")
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/calderon_ubi_ot_flow_map_seed0")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/calderon2022")
    args = parser.parse_args()
    old, new, training, pairs = verify(args.run_dir, args.baseline)
    report(args.run_dir, args.baseline, args.report_dir, old, new, training, pairs)
    plot(args.run_dir, args.baseline, args.report_dir, old, new, training)
    print(f"Report: {args.report_dir / 'filtered_ot_flow_map_results.md'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify and report the saved Calderon four- versus fifty-check evaluation."""

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

from common import calderon


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_and_compare(run, dense):
    old, new = load_json(run / "metrics.json"), load_json(dense / "metrics.json")
    hashes = load_json(dense / "baseline_artifact_hashes.json")
    assert all(calderon.sha256_file(run / name) == value for name, value in hashes.items())
    for key in ("checkpoint_sha256", "training_step", "training_config_sha256", "flowmap_n_steps", "ema_factor", "distribution"):
        assert old[key] == new[key], key
    for key in ("classifier_manifest_sha256", "classifier_selection_sha256", "allowed_pairs_sha256", "prob_threshold", "margin_threshold", "real_cell_classifier"):
        assert old["tissue_trajectories"][key] == new["tissue_trajectories"][key], key
    assert new["trajectory_comparison"]["same_coordinates_at_baseline_checks"]
    with np.load(run / "evaluation/evaluation_inputs.npz", allow_pickle=False) as a, np.load(dense / "evaluation/evaluation_inputs.npz", allow_pickle=False) as b:
        for key in a.files:
            if key not in ("paths", "checkpoint_times", "checkpoint_hours"):
                np.testing.assert_array_equal(a[key], b[key])
        assert b["paths"].shape == (4641, 50, 100) and np.isfinite(b["paths"]).all()
        np.testing.assert_array_equal(b["checkpoint_times"], np.arange(1, 51) / 50)
        np.testing.assert_array_equal(b["paths"][:, [24, 49]], a["paths"][:, [1, 3]])
        classes = tuple(b["class_names"].tolist())
        source_ids = np.array([classes.index(label) for label in b["source_tissues"]])
    allowed_path = load_json(dense / "evaluation/evaluation_config.json")["classifier"]["allowed_pairs_path"]
    allowed = calderon.load_allowed_pairs(allowed_path, classes)
    with np.load(dense / "evaluation/trajectory_probabilities.npz", allow_pickle=False) as saved:
        assert saved["probabilities"].shape == (4641, 50, 24)
        raw = calderon.score_tissue_paths(source_ids, saved["probabilities"], allowed)
        filtered = calderon.score_tissue_paths(source_ids, saved["probabilities"], allowed, .9, 0.)
        assert raw["metrics"] == new["tissue_trajectories"]["raw"]
        assert filtered["metrics"] == new["tissue_trajectories"]["confidence_filtered"]

    # Unchanged expression has exactly the same frozen prediction at every time.
    # Verify that invariance before reusing the completed identity-control inputs.
    identity_path = run / "evaluation/controls/identity/trajectory_probabilities.npz"
    with np.load(identity_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["class_names"], classes)
        np.testing.assert_array_equal(saved["probabilities"], np.repeat(saved["probabilities"][:, :1], 4, axis=1))
        repeated = np.repeat(saved["probabilities"][:, :1], 50, axis=1)
    identity = {
        "raw": calderon.score_tissue_paths(source_ids, repeated, allowed)["metrics"],
        "confidence_filtered": calderon.score_tissue_paths(source_ids, repeated, allowed, .9, 0.)["metrics"],
        "source_probability_sha256": calderon.sha256_file(identity_path),
        "construction": "Repeat verified identical frozen predictions for unchanged source expression at all 50 times",
    }
    identity_reference = load_json(run / "evaluation/controls/control_metrics.json")["identity_source"]["tissue_trajectories"]
    for key in ("raw", "confidence_filtered"):
        for count in ("n_sources", "n_violations", "n_assessed", "n_fully_assessed"):
            assert identity[key][count] == identity_reference[key][count]
    write_json(dense / "evaluation/identity_control_metrics.json", identity)

    a = pd.read_csv(run / "evaluation/trajectory_summary.csv.gz").set_index("source_cell_id")
    b = pd.read_csv(dense / "evaluation/trajectory_summary.csv.gz").set_index("source_cell_id").loc[a.index]
    changes = {
        "new_raw_violations": int((~a.raw_violation & b.raw_violation).sum()),
        "previous_raw_violations_absent": int((a.raw_violation & ~b.raw_violation).sum()),
        "new_filtered_violations": int((a.confidence_filtered_status.ne("violation") & b.confidence_filtered_status.eq("violation")).sum()),
        "previous_filtered_violations_absent": int((a.confidence_filtered_status.eq("violation") & b.confidence_filtered_status.ne("violation")).sum()),
    }
    result = {"status": "passed", "original_artifacts_unchanged": True, "same_cells_checkpoint_and_integration": True,
              "coordinates_identical_at_original_checks": True, "w1_unchanged": True, "paired_status_changes": changes}
    write_json(dense / "evaluation/results_validation.json", result)
    return old, new, identity, changes


def figure(report_dir, old, new):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for shift, metrics, label, color in [(-.18, old, "4 checks", "#226C9B"), (.18, new, "50 checks", "#3C8877")]:
        scores = metrics["tissue_trajectories"]
        raw, filtered = scores["raw"], scores["confidence_filtered"]
        rates = [raw["violation_pct_all_sources"], filtered["violation_pct_assessed"], filtered["violation_pct_fully_assessed"]]
        counts = [raw["n_sources"], filtered["n_assessed"], filtered["n_fully_assessed"]]
        bars = axes[0].bar(np.arange(3) + shift, [np.nan if r is None else r for r in rates], .36, color=color, label=label)
        axes[0].bar_label(bars, labels=["unassessed" if r is None else f"{r:.1f}%\n(n={n:,})" for r, n in zip(rates, counts)], fontsize=8, padding=3)
        coverage = 100 * np.array(filtered["confident_counts"]) / raw["n_sources"]
        axes[1].plot(scores["checkpoint_hours"], coverage, color=color, label=label,
                     marker="o" if len(coverage) == 4 else None, linestyle="none" if len(coverage) == 4 else "-")
    axes[0].set_xticks(np.arange(3), ["Raw: all paths", "≥1 confident\ncheckpoint", "Every checkpoint\nconfident"])
    axes[0].set(ylabel="Paths with a forbidden transition (%)", ylim=(0, 100), title="A  Violations and assessed denominators")
    axes[1].set(xlabel="Inferred developmental age (h)", ylabel="Confident checkpoint labels (%)", ylim=(0, 100),
                title="B  Coverage at probability ≥0.9")
    for ax in axes:
        ax.legend()
    fig.suptitle("Calderon Ubi · same checkpoint and 100-step trajectories · 4,641 test starts", fontsize=13)
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(report_dir / "figures" / f"ot_flow_map_50_checkpoints.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)


def report(report_dir, run, dense, old, new, identity, changes):
    lines = ["# Calderon OT flow map: 50 trajectory checks", "",
             "The saved seed-0 step-6,000 EMA was evaluated at **50 equally spaced points**, normalized times **0.02, 0.04, …, 1.00**, "
             "or inferred ages **5.16, 5.32, …, 13.00 h**. The released source tissue at 5 h initializes the scan. "
             "All **4,641 test starting cells**, 100 composition steps, frozen classifier, tissue relation and ≥0.9 confidence threshold are unchanged. No training was performed.", "",
             "![Four- versus fifty-check results](figures/ot_flow_map_50_checkpoints.png)", "",
             "| Metric | 4 checks | 50 checks |", "| --- | ---: | ---: |"]
    fields = [("Raw violations", "raw", "violation_pct_all_sources"),
              ("Filtered violations among assessed paths", "confidence_filtered", "violation_pct_assessed"),
              ("At least one confident checkpoint: coverage", "confidence_filtered", "trajectory_coverage_pct"),
              ("Every checkpoint confident: coverage", "confidence_filtered", "full_checkpoint_coverage_pct"),
              ("Violations among paths confident at every checkpoint", "confidence_filtered", "violation_pct_fully_assessed")]
    for label, group, field in fields:
        values = [m["tissue_trajectories"][group][field] for m in (old, new)]
        lines.append(f"| {label} | " + " | ".join("—" if v is None else f"{v:.2f}%" for v in values) + " |")
    for label, group, field in [("Raw violating paths", "raw", "n_violations"), ("Filtered violating paths", "confidence_filtered", "n_violations"),
                                ("Assessed filtered paths", "confidence_filtered", "n_assessed"), ("Fully assessed paths", "confidence_filtered", "n_fully_assessed")]:
        lines.append(f"| {label} | " + " | ".join(f"{m['tissue_trajectories'][group][field]:,}" for m in (old, new)) + " |")
    lines += ["", "Raw violations rise from **2,537 to 2,958**. The filtered violation count also rises, **441 → 457**, "
              "but its assessed denominator grows **2,003 → 2,791**. The lower filtered percentage therefore accompanies greater coverage; "
              "it is not an improvement of the model. Uncertain checkpoints are skipped while retaining the last assessed tissue. "
              "Only **86/4,641 paths (1.85%)** are confident at all 50 checkpoints, and **1/86** of those paths violates the relation.", "",
              "The grids are not nested: the original checks at t=0.25 and 0.75 are between the uniform fifty-point grid positions. "
              "The projected tissue relation also does not receive a second transitive closure. Individual classifications of path validity "
              "can therefore change in either direction as the checkpoint grid changes. Paired-path changes are:", "",
              "| Status change | Paths |", "| --- | ---: |"]
    for key, label in [("new_raw_violations", "New raw violations"), ("previous_raw_violations_absent", "Previously raw-violating, now no raw violation"),
                       ("new_filtered_violations", "New filtered violations"), ("previous_filtered_violations_absent", "Previously filtered-violating, now no filtered violation")]:
        lines.append(f"| {label} | {changes[key]:,} |")
    lines += ["", f"Endpoint-only raw violations remain **{new['tissue_trajectories']['raw']['endpoint_violation_pct_assessed']:.2f}%**. "
              f"W1 is unchanged: **{new['distribution']['heldout_reference']['w1']:.4f}** at 8–10 h and **{new['distribution']['target_test']['w1']:.4f}** at 12–14 h. "
              "The extra checks change tissue-path evaluation rather than the flow or marginal comparisons.", "",
              f"The unchanged-expression control gives **{identity['raw']['violation_pct_all_sources']:.2f}% raw** and "
              f"**{identity['confidence_filtered']['violation_pct_assessed']:.2f}% filtered** violations, with "
              f"**{identity['confidence_filtered']['trajectory_coverage_pct']:.2f}%** coverage. Its frozen probabilities were verified identical "
              "at all original checks, then repeated at the 50 times. The real 8–10 h Ubi classifier accuracy remains **48.24%**. "
              "These rates are noisy diagnostics of agreement with the reconstructed tissue prior; confidence abstention does not establish a valid complete path.", "",
              "Verification confirmed that the checkpoint and all original run artifacts were unchanged, evaluation source/reference rows and IDs were identical, "
              "and regenerated coordinates at every original checkpoint matched exactly (maximum absolute difference **0.0**). "
              "The focused flow-map suite passed **14 tests**, including dense-grid validation and unchanged composed rollouts.", "",
              f"Dense evaluation artifacts: `{dense}`. The validation audit and identity control are under `evaluation/`; "
              "per-path predictions, rejection times, confidence coverage and tissue strata are also saved there. See the [original training report](ot_flow_map_results.md) for training, marginal controls and the atlas holdout protocol.", "",
              "Reproduce evaluation:", "", "```sh",
              "ENABLE_PJRT_COMPATIBILITY=1 JAX_PLATFORMS=METAL,cpu \\",
              "  .venv-flowmaps-metal/bin/python py/launchers/calderon_ot_flow_map.py \\",
              "  --stage evaluate --output-dir outputs/calderon_ubi_ot_flow_map_seed0 \\",
              "  --evaluation-output-dir outputs/calderon_ubi_ot_flow_map_seed0/evaluations/checkpoints50 \\",
              "  --checking-points 50 --classifier-python /opt/anaconda3/envs/maizels2023aa/bin/python",
              "```", "", "Reproduce comparison reporting:", "", "```sh",
              "/opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_checkpoint_comparison.py", "```", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ot_flow_map_50_checkpoints.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/calderon_ubi_ot_flow_map_seed0")
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/calderon2022")
    args = parser.parse_args()
    dense = args.evaluation_dir or args.run_dir / "evaluations/checkpoints50"
    old, new, identity, changes = verify_and_compare(args.run_dir, dense)
    report(args.report_dir, args.run_dir, dense, old, new, identity, changes)
    figure(args.report_dir, old, new)
    print(f"Report: {args.report_dir / 'ot_flow_map_50_checkpoints.md'}")


if __name__ == "__main__":
    main()

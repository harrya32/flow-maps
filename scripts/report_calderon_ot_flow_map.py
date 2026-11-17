#!/usr/bin/env python3
"""Report the completed Calderon run with fixed-source and sampling controls.

Controls are evaluated after checkpoint selection; they never affect training.
The identity trajectory control measures reference-label/classifier disagreement
when the starting expression is held fixed. It is not a correction to subtract
from the learned trajectory violation percentage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd
from ml_collections import ConfigDict

from common import calderon, wasserstein


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def controls(run_dir, cfg, metrics):
    directory = run_dir / "evaluation/controls"
    directory.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "evaluation/evaluation_inputs.npz"
    input_hash = calderon.sha256_file(input_path)
    cache = directory / "control_metrics.json"
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored["evaluation_inputs_sha256"] == input_hash and stored["checkpoint_sha256"] == metrics["checkpoint_sha256"]:
            return stored
    with np.load(input_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    generated_indices = arrays["generated_w1_indices"]
    identity_x = arrays["source_x"][generated_indices]
    identity = {
        "heldout_reference_w1": wasserstein.exact_emd(identity_x, arrays["reference_x"]),
        "target_test_w1": wasserstein.exact_emd(identity_x, arrays["target_test_x"]),
    }
    identity_dir = directory / "identity"
    identity_dir.mkdir(exist_ok=True)
    identity_input = dict(arrays)
    identity_input["paths"] = np.repeat(arrays["source_x"][:, None, :], arrays["paths"].shape[1], axis=1)
    identity_path = identity_dir / "evaluation_inputs.npz"
    np.savez_compressed(identity_path, **identity_input)
    subprocess.run([
        cfg.classifier.python, str(ROOT / "scripts/evaluate_calderon_tissue_paths.py"),
        "--input", str(identity_path), "--manifest", calderon.evaluation_manifest_path(cfg),
        "--allowed-pairs", cfg.classifier.allowed_pairs_path, "--output-dir", str(identity_dir),
        "--prob-threshold", str(cfg.classifier.prob_threshold), "--margin-threshold", str(cfg.classifier.margin_threshold),
        "--batch-size", str(cfg.classifier.batch_size), "--threads", str(cfg.classifier.threads),
    ], check=True, cwd=ROOT)
    identity["tissue_trajectories"] = json.loads((identity_dir / "tissue_trajectory_metrics.json").read_text())

    data = calderon.load_endpoint_experiment(cfg)
    real_vs_real = {}
    reference_rows = []
    settings = [
        ("heldout_reference", data.reference_time, arrays["reference_x"], arrays["reference_cell_ids"], None),
        ("target_test", data.target_time, arrays["target_test_x"], arrays["target_test_cell_ids"], "test"),
    ]
    for offset, (name, window, first, excluded_ids, split) in enumerate(settings):
        marginal = data.marginals[window]
        eligible = np.arange(len(marginal.x)) if split is None else data.splits[window][split]
        excluded = marginal.metadata.cell_id.isin(set(excluded_ids.tolist())).to_numpy()
        eligible = eligible[~excluded[eligible]]
        if len(eligible) == 0:
            real_vs_real[name] = {"w1": None, "n_first": len(first), "n_second": 0}
            continue
        chosen = eligible[calderon.select_population_indices(len(eligible), len(first), cfg.evaluation.seed + 101 + offset)]
        real_vs_real[name] = {"w1": wasserstein.exact_emd(first, marginal.x[chosen]),
                              "n_first": len(first), "n_second": len(chosen)}
        records = marginal.metadata.iloc[chosen][["cell_id", "day", "pca_row"]].copy()
        records["control"] = name + "_disjoint_real_reference"
        reference_rows.append(records)
    if reference_rows:
        pd.concat(reference_rows, ignore_index=True).to_csv(directory / "real_reference_cells.csv.gz", index=False)
    result = {"checkpoint_sha256": metrics["checkpoint_sha256"], "evaluation_inputs_sha256": input_hash,
              "identity_source": identity, "real_vs_real": real_vs_real,
              "interpretation": "Descriptive controls after model selection; neither is a formal lower bound or subtractive correction"}
    write_json(cache, result)
    return result


def figure(report_dir, run_dir, metrics, control, training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = pd.read_csv(run_dir / "training_history.csv")
    learned = metrics["tissue_trajectories"]
    identity = control["identity_source"]["tissue_trajectories"]
    colors = {"model": "#226C9B", "identity": "#9A6A45", "real": "#3C8877"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(history.step, history.training_loss.rolling(5, min_periods=1).mean(), color="#9BA5AD", label="Training (5-log mean)", linewidth=1.2)
    ax.plot(history.step, history.validation_loss, color=colors["model"], label="Endpoint validation", linewidth=1.4)
    ax.axvline(training["best_step"], color="#58656F", linestyle="--", linewidth=1, label="Selected checkpoint")
    ax.set(xlabel="Optimizer step", ylabel="LSD loss", title="A  Training and checkpoint selection")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    positions = np.arange(2)
    width = .24
    keys = ["heldout_reference", "target_test"]
    model_w1 = [metrics["distribution"][key]["w1"] for key in keys]
    identity_w1 = [control["identity_source"][key + "_w1"] for key in keys]
    real_w1 = [control["real_vs_real"][key]["w1"] for key in keys]
    for shift, values, label, color in [(-width, identity_w1, "Unchanged source", colors["identity"]),
                                       (0, model_w1, "OT flow map", colors["model"]),
                                       (width, real_w1, "Disjoint real samples", colors["real"])]:
        bars = ax.bar(positions + shift, [np.nan if value is None else value for value in values], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=3)
    ax.set_xticks(positions, ["8–10 h held out", "12–14 h test"])
    ax.set(ylabel="W1 in original PCA100 space", title="B  Marginal agreement")
    ax.set_ylim(0, max(model_w1 + identity_w1 + [v for v in real_w1 if v is not None]) * 1.22)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    categories = np.arange(3)
    for shift, item, label, color in [(-.18, identity, "Unchanged source", colors["identity"]),
                                     (.18, learned, "OT flow map", colors["model"])]:
        values = [item["raw"]["violation_pct_all_sources"],
                  item["confidence_filtered"]["violation_pct_assessed"],
                  item["confidence_filtered"]["violation_pct_fully_assessed"]]
        denominators = [item["raw"]["n_sources"], item["confidence_filtered"]["n_assessed"],
                        item["confidence_filtered"]["n_fully_assessed"]]
        bars = ax.bar(categories + shift, [np.nan if value is None else value for value in values], .36, label=label, color=color)
        labels = ["—" if value is None else f"{value:.1f}%\n(n={denominator:,})"
                  for value, denominator in zip(values, denominators)]
        ax.bar_label(bars, labels=labels, fontsize=7, padding=3)
    ax.set_xticks(categories, ["Raw: all paths", "≥1 confident\ncheckpoint", "All 4 confident\ncheckpoints"])
    ax.set(ylabel="Trajectories with a forbidden transition (%)", title="C  Tissue classifier path violations", ylim=(0, 105))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    n_sources = metrics["n_test_sources_evaluated"]
    hours = learned["checkpoint_hours"]
    for shift, item, label, color in [(-.18, identity, "Unchanged source", colors["identity"]),
                                     (.18, learned, "OT flow map", colors["model"])]:
        coverage = 100 * np.asarray(item["confidence_filtered"]["confident_counts"]) / n_sources
        bars = ax.bar(np.arange(len(hours)) + shift, coverage, .36, label=label, color=color)
        ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=3)
    ax.set_xticks(np.arange(len(hours)), [f"{hour:g}" for hour in hours])
    ax.set(xlabel="Inferred-age midpoint (h)", ylabel="Confident cells (%)",
           title="D  Coverage at probability ≥0.9", ylim=(0, 110))
    ax.legend(fontsize=8)
    fig.suptitle(f"Calderon Ubi: 4–6 → 12–14 h OT flow map · seed 0 · {n_sources:,} test starts", fontsize=14)
    folder = report_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(folder / f"ot_flow_map_results.{suffix}", dpi=220)
    plt.close(fig)


def percent(value):
    return "—" if value is None else f"{value:.2f}%"


def report(report_dir, run_dir, cfg, metrics, control, training):
    learned = metrics["tissue_trajectories"]
    identity = control["identity_source"]["tissue_trajectories"]
    n_sources = metrics["n_test_sources_evaluated"]
    lines = ["# Calderon OT flow-map results", "",
             f"The declared **4–6 → 12–14 h Ubi** experiment completed **{training['completed_steps']:,} optimizer steps**. "
             f"Endpoint validation selected step **{training['best_step']:,}**, with validation loss **{training['best_validation_loss']:.4f}**. "
             f"Training took **{training['seconds']/60:.2f} minutes**. The 0.999 EMA was evaluated on **{n_sources:,} flow-test starting cells**; "
             "8–10 h remained reserved for interpolation evaluation and did not select the checkpoint.", "",
             "![Training, marginal agreement, violations and coverage](figures/ot_flow_map_results.png)", "",
             "## Marginal agreement", "", "| Evaluation marginal | OT flow map W1 | Unchanged-source W1 | Disjoint real-versus-real W1 |",
             "| --- | ---: | ---: | ---: |"]
    for key, name in [("heldout_reference", "8–10 h held out"), ("target_test", "12–14 h test")]:
        real = control["real_vs_real"][key]["w1"]
        lines.append(f"| {name} | {metrics['distribution'][key]['w1']:.4f} | {control['identity_source'][key+'_w1']:.4f} | "
                     + (f"{real:.4f}" if real is not None else "—") + " |")
    n_generated = metrics["distribution"]["heldout_reference"]["n_generated"]
    lines += ["", f"W1 uses Euclidean ground cost in the original PCA100 coordinates and uniform cell weights, with **{n_generated:,} generated points** "
              "and up to 2,048 reference points per comparison. The unchanged-source control uses exactly the same source/reference samples. "
              "Real-versus-real compares disjoint samples from the same reserved marginal; it supplies sampling context rather than a formal lower bound.", "",
              f"The model reduces endpoint W1 by **{100*(1-metrics['distribution']['target_test']['w1']/control['identity_source']['target_test_w1']):.2f}%** "
              "relative to unchanged sources. Its held-out 8–10 h W1 is slightly worse than that control, so this run shows endpoint improvement "
              "without evidence of improved interpolation at the reserved middle window.", "",
              "## Tissue trajectory violations", "",
              "| Path | Raw violation rate | Filtered rate among assessed paths | Assessed-path coverage | All-four-checkpoint coverage | Rate among all-four-assessed paths |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in [("OT flow map", learned), ("Unchanged expression control", identity)]:
        raw = item["raw"]
        filtered = item["confidence_filtered"]
        lines.append(f"| {name} | {percent(raw['violation_pct_all_sources'])} | {percent(filtered['violation_pct_assessed'])} | "
                     f"{percent(filtered['trajectory_coverage_pct'])} | {percent(filtered['full_checkpoint_coverage_pct'])} | "
                     f"{percent(filtered['violation_pct_fully_assessed'])} |")
    lines += ["", "The scorer starts from each cell's released tissue label and checks generated labels at **7, 9, 11 and 13 h**. "
              "The confidence filter is probability ≥0.9 and probability margin ≥0.0. A path with no confident checkpoint is unassessed. "
              "The supplied projected tissue relation is used without another closure; Unknown retains its specified transitions.", "",
              f"The raw flow result is **{learned['raw']['n_violations']:,}/{n_sources:,} trajectories**. "
              f"The filtered result is **{learned['confidence_filtered']['n_violations']:,}/{learned['confidence_filtered']['n_assessed']:,} assessed paths**, "
              f"with **{learned['confidence_filtered']['n_unassessed']:,} unassessed** and **{learned['confidence_filtered']['n_fully_assessed']:,} fully assessed paths**. "
              "The filtered rate permits skipped checkpoints and is not a complete-path assessment. Only "
              f"**{percent(learned['confidence_filtered']['full_checkpoint_coverage_pct'])}** of paths have all four checkpoints assessed.", "",
              f"Checking only the source-to-final tissue relation gives **{percent(learned['raw']['endpoint_violation_pct_assessed'])}** raw violations, "
              "compared with the whole-path rate above.", "",
              f"The unchanged-expression control reports {percent(identity['raw']['violation_pct_all_sources'])} raw violations even though expression is held fixed. "
              "Its positive violations therefore reflect reference-label/classifier disagreement. This is a noise diagnostic; it should not be subtracted "
              "from the learned rate, because classification behavior changes along generated trajectories.", "",
              "## Classifier checks on real evaluation cells", "",
              "| Population | Cells | Accuracy | Confidence coverage | Accuracy among confident cells |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for key, label in [("source_test", "4–6 h test starts"), ("heldout_reference", "8–10 h reference"), ("target_test", "12–14 h test reference")]:
        score = learned["real_cell_classifier"][key]
        lines.append(f"| {label} | {score['n_cells']:,} | {percent(100*score['accuracy'])} | {percent(score['confidence_coverage_pct'])} | "
                     f"{percent(None if score['confident_accuracy'] is None else 100*score['confident_accuracy'])} |")
    checkpoints = pd.read_csv(run_dir / "evaluation/checkpoint_rejections.csv")
    lines += ["", "## Where paths first fail", "",
              "| Midpoint (h) | First raw rejections | Fraction of all test paths | First filtered rejections | Confident coverage |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in checkpoints.itertuples(index=False):
        lines.append(f"| {row.hours:g} | {row.raw_first_rejections:,} | {percent(100*row.raw_first_rejections/n_sources)} | "
                     f"{row.filtered_first_rejections:,} | {percent(row.confidence_coverage_pct)} |")
    strata = pd.read_csv(run_dir / "evaluation/violations_by_source_tissue.csv").sort_values("n_sources", ascending=False)
    lines += ["", "## Starting-tissue strata", "",
              "| Starting tissue | Test paths | Raw violations | Filtered violations among assessed paths | Filtered path coverage |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in strata.itertuples(index=False):
        lines.append(f"| {row.source_tissue} | {row.n_sources:,} | {percent(row.raw_violation_pct)} | "
                     f"{percent(None if pd.isna(row.filtered_violation_pct_assessed) else row.filtered_violation_pct_assessed)} | "
                     f"{percent(row.filtered_coverage_pct)} |")
    lines += ["", "## Scope and reproduction", "",
              "This is one seed-0 unconstrained run of the declared native Maizels OT flow-map recipe, with three 1,024-unit hidden layers, "
              "batch size 128 and RAdam at 0.0003. The checkpoint was selected using fixed endpoint-validation batches and the sampling EMA. "
              "The saved model and all input hashes accompany the metrics. The Apple GPU backend matched CPU losses in a five-step full-model check.", "",
              "The holdouts are for flow training. The frozen reference-atlas PCA and classifier can have seen endpoint flow-test cells. "
              "Ages and tissue labels are atlas-derived; these percentages measure agreement with the reconstructed prior and classifier, "
              "rather than traced clone-fate violations. A single seed supplies no across-training-seed uncertainty estimate.", "",
              f"Artifacts: `{run_dir}`. See `metrics.json`, `training_summary.json`, `evaluation/controls/control_metrics.json`, "
              "the per-cell trajectory summaries and probability arrays, and `best.msgpack`.", "",
              "Reproduce post-training reporting:", "", "```sh",
              "/opt/anaconda3/envs/flowmaps/bin/python scripts/report_calderon_ot_flow_map.py",
              "```", ""]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "ot_flow_map_results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/calderon_ubi_ot_flow_map_seed0")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "docs/calderon2022")
    parser.add_argument("--classifier-python")
    args = parser.parse_args()
    cfg = ConfigDict(json.loads((args.run_dir / "config.json").read_text()))
    if args.classifier_python:
        cfg.classifier.python = args.classifier_python
    metrics = json.loads((args.run_dir / "metrics.json").read_text())
    training = json.loads((args.run_dir / "training_summary.json").read_text())
    control = controls(args.run_dir, cfg, metrics)
    report(args.report_dir, args.run_dir, cfg, metrics, control, training)
    figure(args.report_dir, args.run_dir, metrics, control, training)
    print(f"Report: {args.report_dir / 'ot_flow_map_results.md'}", flush=True)


if __name__ == "__main__":
    main()

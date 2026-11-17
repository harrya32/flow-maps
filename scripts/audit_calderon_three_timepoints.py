#!/usr/bin/env python3
"""Audit tissue coverage and graph support for three-time Calderon designs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import numpy as np
import pandas as pd

from docs.calderon2022.audit_timepoint_support import coupling_support

CANDIDATES = [
    ("Proposed", ("4-6", "10-12", "16-18")),
    ("Recommended compromise", ("4-6", "8-10", "14-16")),
    ("Symmetric shorter interval", ("4-6", "8-10", "12-14")),
    ("Longer interval with more middle cells", ("4-6", "8-10", "16-18")),
    ("Later middle, earlier final", ("4-6", "10-12", "14-16")),
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage(counts, times, pair_support):
    windows = counts.index.tolist()
    indices = [windows.index(time) for time in times]
    interval = windows[indices[0]:indices[-1] + 1]
    heldout = [time for time in interval if time not in times]
    training = counts.loc[list(times)].sum()
    required = counts.loc[interval].sum()
    required = required[required.gt(0)]
    missing = required.index[training.reindex(required.index).eq(0)].tolist()
    present = training[training.gt(0)]
    pairs = [pair_support[(a, b)] for a, b in zip(times[:-1], times[1:])]
    return {"training_times": "; ".join(times), "heldout_times": "; ".join(heldout),
            "first": times[0], "last": times[-1], "training_time_cells": int(training.sum()),
            "heldout_time_cells": int(counts.loc[heldout].to_numpy().sum()) if heldout else 0,
            "tissues_in_interval": len(required), "tissues_at_training_times": len(present),
            "missing_interval_tissues": "; ".join(missing),
            "global_tissues_absent_from_training": "; ".join(training.index[training.eq(0)]),
            "all_interval_tissues_covered": not missing,
            "heldout_window_in_each_gap": all(b - a >= 2 for a, b in zip(indices[:-1], indices[1:])),
            "rarest_present_training_tissue": present.idxmin(),
            "rarest_present_training_tissue_cells": int(present.min()),
            "unknown_training_pct": 100 * training.get("Unknown", 0) / training.sum(),
            "first_interval_forced_forbidden_pct": 100 * (1 - pairs[0][0]),
            "second_interval_forced_forbidden_pct": 100 * (1 - pairs[1][0]),
            "first_interval_orphan_source_pct": 100 * pairs[0][1],
            "second_interval_orphan_source_pct": 100 * pairs[1][1]}


def figure(counts, candidates, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [5, 1.45]},
                             constrained_layout=True)
    values = counts.T.to_numpy()
    log_counts = np.ma.masked_where(values == 0, np.log10(np.maximum(values, 1)))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#F0F2F4")
    image = axes[0].imshow(log_counts, cmap=cmap, vmin=1.5, vmax=4.5, aspect="auto")
    axes[0].set(xticks=range(len(counts)), xticklabels=counts.index,
                yticks=range(len(counts.columns)), yticklabels=counts.columns,
                title="Released broad tissue support at each window")
    axes[0].tick_params(axis="y", labelsize=9)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if value:
                axes[0].text(j, i, f"{value:,}", ha="center", va="center", fontsize=7.5,
                             color="white" if value >= 4000 else "#24292F")
    fig.colorbar(image, ax=axes[0], shrink=.75, label="log10 cell count")
    schedule = []
    for _, times in CANDIDATES:
        indices = [counts.index.get_loc(time) for time in times]
        schedule.append([2 if time in times else 1 if indices[0] <= i <= indices[-1] else 0
                         for i, time in enumerate(counts.index)])
    axes[1].imshow(schedule, cmap=ListedColormap(["#E8EBEE", "#E6BB6B", "#266A9C"]),
                   vmin=0, vmax=2, aspect="auto")
    axes[1].set(xticks=range(len(counts)), xticklabels=counts.index,
                yticks=range(len(CANDIDATES)), yticklabels=[name for name, _ in CANDIDATES],
                xlabel="Inferred-age window (h)", title="T = training time · H = held-out W1 time · — = outside interval")
    axes[1].tick_params(axis="y", labelsize=8.5)
    for i, row in enumerate(schedule):
        for j, value in enumerate(row):
            axes[1].text(j, i, ["—", "H", "T"][value], ha="center", va="center", fontsize=10,
                         color="white" if value == 2 else "#32383E")
    fig.suptitle("Calderon three-time designs · all 314,387 lineage-assigned cells\nCounts before cell-level training/validation/test splits", fontsize=13)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination.with_suffix(f".{extension}"), dpi=220, facecolor="white")
    plt.close(fig)


def report(counts, candidates, global_minimum, destination):
    lines = ["# Three-time Calderon experiment design", "",
        "The selected follow-up uses **4–6 / 8–10 / 12–14 h**, retaining the **4–14 h** interval and all 23 tissues present within it. "
        "The [completed tissue-classifier experiment](three_time_classifier.md) achieves **75.48% accuracy on held-out cells from the selected windows** "
        "and **62.50% across the two entire intermediate marginals**. The schedules below preserve the original coverage and compatibility comparisons.", "",
        "The proposed **4–6 / 10–12 / 16–18 h** training times cover all **23 tissue labels present within 4–18 h**. "
        "Only `Blastoderm` is absent from the global 24-label vocabulary; it occurs only before the selected interval.", "",
        "No three windows cover all 24 atlas-wide labels. Four disjoint support requirements establish this: "
        "`Blastoderm` occurs at 0–2/2–4 h, `Mesoderm` only at 4–6 h, `Crystal cells` at 8–10/10–12 h, "
        "and `Ventral nerve cord` at 12–14/14–16/16–18 h. Exhaustive enumeration confirms a minimum of four windows. "
        f"One four-window witness is {' / '.join(global_minimum['witness'])} h.", "",
        "## Candidate schedules", "",
        "Counts below include `Unknown` and precede cell-level splits. All five candidates cover the 23 labels "
        "present between their first and last selected windows, with a held-out window in both gaps.", "",
        "| Design | Training windows (h) | Held-out windows (h) | Cells at training times | Held-out-time cells |",
        "| --- | --- | --- | ---: | ---: |"]
    for candidate in candidates:
        row = candidate["coverage"]
        lines.append(f"| {candidate['name']} | {row['training_times']} | {row['heldout_times']} | "
                     f"{row['training_time_cells']:,} | {row['heldout_time_cells']:,} |")
    lines += ["", "## Support for rare classifier targets", "",
        "Moving the middle training point to 8–10 h supplies more examples of the transient classes. "
        "The following labels are absent at 4–6, 14–16 and 16–18 h. For schedules ending at 14–16 or 16–18 h, "
        "their support comes entirely from the middle training window.", "",
        "| Tissue | 8–10 h cells | 10–12 h cells |", "| --- | ---: | ---: |"]
    for tissue in ("Muscle", "Crystal cells", "Midline/glia", "Visceral muscle"):
        lines.append(f"| {tissue} | {counts.loc['8-10', tissue]:,} | {counts.loc['10-12', tissue]:,} |")
    lines += ["", "The 12–14 h final-point alternative adds further `Midline/glia` and `Visceral muscle` examples. "
        "`Muscle` remains rare even with the earlier middle point: 147 cells before classifier validation/test splitting. "
        "Coverage is necessary for the classifier, but does not establish held-out-stage accuracy; "
        "the [completed selected-times classifier evaluation](three_time_classifier.md) measures that stage transfer separately.", "",
        "## Compatibility with the existing tissue prior", "",
        "Each interval below normalizes its full observed snapshots independently. We maximize coupling mass allowed "
        "by the existing projected tissue relation, subject to source and target histogram capacities. The remaining "
        "percentage is a lower bound on forbidden mass for an exactly balanced coupling. It is a graph/snapshot "
        "compatibility diagnostic, not a measured classifier or flow violation rate.", "",
        "| Design | First consecutive interval: minimum forbidden mass | Second consecutive interval: minimum forbidden mass |",
        "| --- | ---: | ---: |"]
    for candidate in candidates:
        row = candidate["coverage"]
        lines.append(f"| {candidate['name']} | {row['first_interval_forced_forbidden_pct']:.2f}% | "
                     f"{row['second_interval_forced_forbidden_pct']:.2f}% |")
    lines += ["", "The 16–18 h final marginal has only 11 tissue labels. Its approximately 40% compatibility floor "
        "in both long designs makes the lineage/marginal tradeoff substantial under the present prior. "
        "The earlier final windows reduce that floor to about 20–22%; it remains material. "
        "For soft constraints, report the tradeoff with W1. For filtered assignments, record supported mass and the changed sampling proportions.", "",
        "## Selected design and information split", "",
        "Use **4–6 / 8–10 / 12–14 h** for the selected symmetric pilot: training at **5 / 9 / 13 h**, evaluation at **7 / 11 h**, "
        "with 162,206 training-time cells before splitting and two interpolation checks. On a normalized 5–13 h clock, "
        "training is at **0 / 0.5 / 1** and evaluation at **0.25 / 0.75**.", "",
        "The earlier compromise proposal, **4–6 / 8–10 / 14–16 h**, also covers all 23 in-range tissues, provides 151,669 cells before splitting, "
        "and leaves **6–8 / 10–12 / 12–14 h** for W1 evaluation. Its midpoint training clock is 5/9/15 h and evaluation clock 7/11/13 h. "
        "The longer **4–6 / 8–10 / 16–18 h** schedule remains an alternative for a 4–18 h benchmark, with the larger compatibility floor described above.", "",
        "Use separate classifier weights for the two roles. The **constraint classifier** learns only from the selected "
        "training windows, with its own validation/test cell splits inside those windows and frozen weights during flow training. "
        "The existing **all-times reference ('oracle') classifier** supplies final violation tracking. Both use the same "
        "already fitted all-times PCA100 space. Outside-range windows are excluded from the flow benchmark. "
        "The intermediate-window marginals and labels are excluded from flow/constraint-classifier fitting and checkpoint selection; "
        "the reference classifier is intentionally allowed atlas information across all times.", "",
        "![Tissue counts and candidate schedules](figures/three_timepoint_coverage.png)", "",
        "## Reproduction", "",
        "[three_timepoint_coverage.csv](three_timepoint_coverage.csv) contains all 84 ordered three-window choices. "
        "[tissue_counts_by_window.csv](tissue_counts_by_window.csv) contains every tissue/window count. "
        "[three_timepoint_audit.json](three_timepoint_audit.json) records source hashes, class support, the minimum-window proof, "
        "selected candidate counts and independently recomputed graph support. Existing support-table results were checked against the recomputation.", "",
        "```sh", "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /opt/anaconda3/envs/maizels2023aa/bin/python \\",
        "  scripts/audit_calderon_three_timepoints.py", "```", ""]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/calderon2022")
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    source = ROOT / "datasets/calderon2022/lineage_assigned_metadata.csv.gz"
    graph = ROOT / "docs/calderon2022/graph_data/broad_tissue_allowed_pairs.csv"
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (source, graph)}
    cells = pd.read_csv(source, usecols=["cell_id", "day", "broad_tissue"])
    assert cells.cell_id.is_unique and len(cells) == 314387
    windows = sorted(cells.day.unique(), key=lambda time: int(time.split("-")[0]))
    counts = pd.crosstab(cells.day, cells.broad_tissue).reindex(windows)
    assert counts.shape == (9, 24) and int(counts.to_numpy().sum()) == len(cells)
    edges = pd.read_csv(graph)
    assert edges.shape[1] == 2
    allowed = set(map(tuple, edges.to_numpy()))
    pairs = {}
    existing = pd.read_csv(ROOT / "docs/calderon2022/timepoint_support.csv")
    for a, b in itertools.combinations(windows, 2):
        pairs[(a, b)] = coupling_support(cells.loc[cells.day.eq(a), "broad_tissue"],
                                         cells.loc[cells.day.eq(b), "broad_tissue"], allowed)
        row = existing.loc[existing.cohort.eq("all_assigned") & existing.source.eq(a) & existing.target.eq(b)]
        assert len(row) == 1
        np.testing.assert_allclose(pairs[(a, b)][0], row.broad_tissue_max_supported_mass.iloc[0], atol=1e-10, rtol=0)
    rows = [coverage(counts, times, pairs) for times in itertools.combinations(windows, 3)]
    audit = pd.DataFrame(rows)
    assert len(audit) == 84 and not audit.global_tissues_absent_from_training.eq("").any()
    minimum = None
    for size in range(1, 5):
        for times in itertools.combinations(windows, size):
            if counts.loc[list(times)].sum().gt(0).all():
                minimum = {"n_windows": size, "witness": list(times)}
                break
        if minimum:
            break
    assert minimum["n_windows"] == 4
    candidates = []
    for name, times in CANDIDATES:
        row = coverage(counts, times, pairs)
        assert row["all_interval_tissues_covered"] and row["heldout_window_in_each_gap"]
        assert row["tissues_in_interval"] == row["tissues_at_training_times"] == 23
        candidates.append({"name": name, "coverage": row,
                           "tissue_counts_at_training_times": {label: int(n) for label, n in counts.loc[list(times)].sum().items()}})
    for name, digest in hashes.items():
        assert sha256(ROOT / name) == digest
    destination.mkdir(parents=True, exist_ok=True)
    audit.to_csv(destination / "three_timepoint_coverage.csv", index=False)
    counts.to_csv(destination / "tissue_counts_by_window.csv")
    result = {"status": "passed", "cohort": "all_lineage_assigned_cells", "n_cells": len(cells),
              "source_hashes": hashes, "sources_unchanged": True, "n_triplets_audited": len(audit),
              "minimum_windows_for_all_24_tissues": minimum,
              "per_tissue_support_windows": {label: counts.index[counts[label].gt(0)].tolist() for label in counts.columns},
              "existing_pair_support_recomputed_and_matched": True, "candidates": candidates}
    (destination / "three_timepoint_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    report(counts, candidates, minimum, destination / "three_timepoint_design.md")
    figure(counts, candidates, destination / "figures/three_timepoint_coverage")
    print(json.dumps({"status": "passed", "n_triplets": len(audit), "all_global_labels_minimum": minimum}, indent=2), flush=True)
    print(pd.DataFrame([c["coverage"] for c in candidates]).to_string(index=False), flush=True)
    print(f"Report: {destination / 'three_timepoint_design.md'}", flush=True)


if __name__ == "__main__":
    main()

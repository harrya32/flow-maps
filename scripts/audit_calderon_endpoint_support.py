#!/usr/bin/env python3
"""Measure endpoint-label support on the saved Calderon training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, linprog

from common.calderon import load_allowed_pairs, sha256_file


def audit_interval(cfg, names, allowed, splits, source_window, target_window, n_blocks, seed):
    counts = []
    for window in (source_window, target_window):
        selected = splits.loc[splits.flow_split.eq("train") & splits.day.eq(window), "broad_tissue"]
        counts.append(selected.value_counts().reindex(names, fill_value=0).to_numpy())
    source, target = [x / x.sum() for x in counts]
    k = len(names)
    constraints = np.zeros((2 * k, k * k))
    for i in range(k):
        constraints[i, i*k:(i+1)*k] = 1
        constraints[k+i, i::k] = 1
    solution = linprog(-allowed.ravel().astype(float), A_ub=constraints,
                       b_ub=np.concatenate([source, target]), bounds=(0, None), method="highs")
    if not solution.success:
        raise RuntimeError(solution.message)
    batch = int(cfg["problem"]["ot_minibatch_size"])
    rng = np.random.default_rng(seed)
    retained = np.empty(n_blocks)
    for i in range(n_blocks):
        a, b = rng.choice(k, batch, p=source), rng.choice(k, batch, p=target)
        mask = allowed[a[:, None], b[None, :]]
        rows, columns = linear_sum_assignment((~mask).astype(np.float64))
        retained[i] = mask[rows, columns].mean()
    result = {
        "protocol": "Released endpoint labels only; no interpolant classifier or generated paths",
        "source_window": source_window, "target_window": target_window,
        "source_training_cells": int(counts[0].sum()), "target_training_cells": int(counts[1].sum()),
        "balanced_maximum_supported_mass": float(-solution.fun),
        "independent_pair_allowed_fraction": float(source @ allowed @ target),
        "minibatch": {"n_blocks": n_blocks, "batch_size": batch, "seed": seed,
                      "sampling": "Independent labels with replacement from the saved training fractions",
                      "mean_maximum_supported_mass": float(retained.mean()),
                      "minimum": float(retained.min()), "maximum": float(retained.max()),
                      "percentile_2_5": float(np.percentile(retained, 2.5)),
                      "percentile_97_5": float(np.percentile(retained, 97.5))},
        "interpretation": "Label-support audit, not a trained model; minibatch percentiles describe block variability",
    }
    return result


def audit(run, n_blocks=2000, seed=3029):
    cfg = json.loads((run / "config.json").read_text())
    manifest = Path(cfg["classifier"]["manifest_path"])
    names = json.loads(manifest.read_text())["class_names"]
    relation = Path(cfg["classifier"]["allowed_pairs_path"])
    allowed = load_allowed_pairs(relation, names, allow_extra_classes=True)
    splits = pd.read_csv(run / "cell_splits.csv.gz")
    windows = cfg["problem"].get("retained_timepoints", (cfg["problem"]["source_time"], cfg["problem"]["target_time"]))
    intervals = [audit_interval(cfg, names, allowed, splits, source, target, n_blocks, seed + i)
                 for i, (source, target) in enumerate(zip(windows[:-1], windows[1:]))]
    result = intervals[0] if len(intervals) == 1 else {
        "protocol": "Released training labels on each adjacent retained interval; no interpolant classifier or generated paths",
        "training_windows": list(windows), "intervals": intervals,
    }
    result.update({"cell_splits_sha256": sha256_file(run / "cell_splits.csv.gz"),
                   "allowed_pairs_sha256": sha256_file(relation), "classifier_manifest_sha256": sha256_file(manifest)})
    (run / "endpoint_support_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/calderon_symmetric_4_8_12_filtered_ot_flow_map_seed0")
    parser.add_argument("--blocks", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3029)
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    print(json.dumps(audit(args.run_dir, args.blocks, args.seed), indent=2))


if __name__ == "__main__":
    main()

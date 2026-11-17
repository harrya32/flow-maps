#!/usr/bin/env python3
"""Reuse all-times PCA100 with classifier labels split at three selected times."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.prepare_calderon_all_times_data import split_diagnostics
from scripts.prepare_calderon_lineage_data import file_hash, h5ad_safe_obs

VARIANT = "symmetric_4_8_12"
TRAINING_WINDOWS = ("4-6", "8-10", "12-14")
INTERMEDIATE_WINDOWS = ("6-8", "10-12")
SPLITS = ("train", "validation", "test")


def split_selected_times(obs, seed=0):
    """Stratify selected cells by window/branch, leaving other times untouched."""
    if not obs.index.is_unique:
        raise ValueError("Expected unique cell identifiers")
    selected = np.flatnonzero(obs.day.astype(str).isin(TRAINING_WINDOWS).to_numpy())
    if not len(selected):
        raise ValueError("No cells at the requested training windows")
    subset = obs.iloc[selected]
    if set(subset.day.astype(str)) != set(TRAINING_WINDOWS):
        raise ValueError("A requested training window is missing")
    if (subset.groupby("branch_id", observed=True).broad_tissue.nunique() != 1).any():
        raise ValueError("Each branch must map to exactly one tissue")
    strata = (subset.day.astype(str) + "|" + subset.branch_id.astype(str)).to_numpy()
    training, remainder = train_test_split(np.arange(len(selected)), test_size=.2,
                                           stratify=strata, random_state=seed)
    validation, test = train_test_split(remainder, test_size=.5,
                                        stratify=strata[remainder], random_state=seed + 1)
    assignment = np.full(len(obs), "outside_interval", dtype="<U24")
    assignment[obs.day.astype(str).isin(INTERMEDIATE_WINDOWS).to_numpy()] = "intermediate_holdout"
    for name, indices in zip(SPLITS, (training, validation, test)):
        assignment[selected[indices]] = name
    if set(assignment[selected]) != set(SPLITS):
        raise ValueError("Incomplete cell-level training split")
    required = set(obs.loc[obs.day.astype(str).isin(TRAINING_WINDOWS + INTERMEDIATE_WINDOWS), "broad_tissue"].astype(str))
    available = set(obs.iloc[selected].broad_tissue.astype(str))
    if not required <= available:
        raise ValueError(f"Selected windows do not cover intermediate tissues: {sorted(required - available)}")
    for name in SPLITS:
        portion = obs.loc[assignment == name]
        if set(portion.broad_tissue.astype(str)) != available:
            raise ValueError(f"Missing tissue classes in {name}")
        if set(portion.day.astype(str)) != set(TRAINING_WINDOWS):
            raise ValueError(f"Missing selected windows in {name}")
        if set(portion.branch_id.astype(str)) != set(subset.branch_id.astype(str)):
            raise ValueError(f"Missing auxiliary branch classes in {name}")
    return assignment, selected


def prepare(source_dir, output_dir, seed=0):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Use a separate directory for the selected-time label split")
    source = source_dir / "lineage_assigned_hvg3000_pca100_all_times.h5ad"
    original_manifest_path = source_dir / "preprocessing_all_times.json"
    original = json.loads(original_manifest_path.read_text())
    if original["hvg_fit"] != "all_cells" or original["pca_fit"] != "all_cells":
        raise ValueError("Expected the all-times fitted PCA representation")
    source_hash = file_hash(source)
    if source_hash != original["h5ad_sha256"]:
        raise ValueError("The source all-times representation changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"lineage_assigned_hvg3000_pca100_{VARIANT}.h5ad"
    manifest_path = output_dir / f"preprocessing_{VARIANT}.json"
    if destination.exists():
        if not manifest_path.exists():
            raise ValueError("Incomplete selected-time dataset; use a separate output directory")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["source_h5ad_sha256"], manifest["split_seed"], manifest["training_windows"],
                manifest["intermediate_windows"]) != (source_hash, seed, list(TRAINING_WINDOWS), list(INTERMEDIATE_WINDOWS)):
            raise ValueError("Existing selected-time split has different settings")
        if file_hash(destination) != manifest["h5ad_sha256"]:
            raise ValueError("Cached selected-time projection changed")
        print(f"Reusing verified selected-time projection: {destination}", flush=True)
        return destination
    backed = ad.read_h5ad(source, backed="r")
    try:
        all_obs = backed.obs.copy()
        x = np.asarray(backed.obsm["X_pca"], dtype=np.float32)
        var = backed.var.copy()
        uns = copy.deepcopy(dict(backed.uns))
        components = np.asarray(backed.varm["PCs"]).copy()
    finally:
        backed.file.close()
    if x.shape != (len(all_obs), 100) or not np.isfinite(x).all():
        raise ValueError("Expected finite, aligned PCA100 coordinates")
    assignment, selected = split_selected_times(all_obs, seed)
    selected_obs = all_obs.iloc[selected].copy()
    selected_obs["reference_classifier_split"] = selected_obs.classifier_split.copy()
    selected_obs["classifier_split"] = assignment[selected]
    selected_obs["source_pca_row"] = selected
    proportions, audit = split_diagnostics(selected_obs, assignment[selected])
    audit["stratification"] = "inferred-age window + branch_id; deterministic branch-to-tissue mapping preserves tissue proportions"
    for window in TRAINING_WINDOWS:
        for name in SPLITS:
            part = selected_obs.loc[selected_obs.day.astype(str).eq(window) & selected_obs.classifier_split.eq(name)]
            whole = selected_obs.loc[selected_obs.day.astype(str).eq(window)]
            assert set(part.broad_tissue.astype(str)) == set(whole.broad_tissue.astype(str))
    interval = all_obs.day.astype(str).isin(TRAINING_WINDOWS + INTERMEDIATE_WINDOWS).to_numpy()
    table = all_obs.loc[interval, ["branch_id", "broad_tissue", "day", "classifier_split"]].copy()
    table = table.rename(columns={"classifier_split": "reference_classifier_split"})
    table["classifier_split"] = assignment[interval]
    table["source_pca_row"] = np.flatnonzero(interval)
    table.to_csv(output_dir / "cell_splits.csv.gz")
    proportions.to_csv(output_dir / "split_proportions.csv", index=False)
    (output_dir / "split_validation.json").write_text(json.dumps(audit, indent=2) + "\n")
    uns["preprocessing"].update({"variant": VARIANT, "split_seed": seed,
        "classifier_training_windows": list(TRAINING_WINDOWS),
        "classifier_intermediate_holdout_windows": list(INTERMEDIATE_WINDOWS),
        "classifier_holdout_scope": "Selected-time validation/test labels and all intermediate labels excluded from fitting; original global HVG/PCA retained",
        "source_h5ad": str(source), "source_h5ad_sha256": source_hash,
        "projection_only": True})
    # X is omitted because this file supplies existing PCs, rather than a new count matrix.
    projection = ad.AnnData(obs=h5ad_safe_obs(selected_obs), var=var)
    projection.obsm["X_pca"] = x[selected].copy()
    projection.varm["PCs"] = components
    projection.uns = uns
    temporary = destination.with_name(destination.name + ".part")
    projection.write_h5ad(temporary, compression="gzip", compression_opts=1)
    with h5py.File(temporary, "r") as stored:
        assert "X" not in stored
    check = ad.read_h5ad(temporary, backed="r")
    try:
        np.testing.assert_array_equal(check.obs_names.astype(str), selected_obs.index.astype(str))
        np.testing.assert_array_equal(check.obsm["X_pca"], x[selected])
    finally:
        check.file.close()
    assert file_hash(source) == source_hash
    temporary.replace(destination)
    manifest = {"variant": VARIANT, "source_h5ad": str(source), "source_h5ad_sha256": source_hash,
        "source_preprocessing_manifest_sha256": file_hash(original_manifest_path),
        "source_preprocessing_npz_sha256": original["preprocessing_npz_sha256"],
        "h5ad_sha256": file_hash(destination), "n_hvgs": 3000, "n_pcs": 100,
        "hvg_fit": "all_cells", "pca_fit": "all_cells", "n_fit_cells": original["n_fit_cells"],
        "n_cells": len(selected_obs), "split_seed": seed, "seed": 0,
        "training_windows": list(TRAINING_WINDOWS), "intermediate_windows": list(INTERMEDIATE_WINDOWS),
        "split_sizes": audit["split_sizes"], "n_intermediate_cells": int((assignment == "intermediate_holdout").sum()),
        "intermediate_cells_per_window": {window: int(all_obs.day.astype(str).eq(window).sum()) for window in INTERMEDIATE_WINDOWS},
        "training_cells_per_window": {window: int(selected_obs.day.astype(str).eq(window).sum()) for window in TRAINING_WINDOWS},
        "tissue_classes": sorted(selected_obs.broad_tissue.astype(str).unique().tolist()),
        "n_tissue_classes": selected_obs.broad_tissue.nunique(), "n_auxiliary_branch_classes": selected_obs.branch_id.nunique(),
        "source_unchanged": True, "source_pca_projection_identical": True, "expression_matrix_omitted": True}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected_obs):,} selected-time cells; split {audit['split_sizes']}; "
          f"{manifest['n_intermediate_cells']:,} complete intermediate holdouts", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "datasets/calderon2022/all_times")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"datasets/calderon2022/{VARIANT}")
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.split_seed)


if __name__ == "__main__":
    main()

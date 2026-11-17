#!/usr/bin/env python3
"""Fit Calderon HVG/PCA on every labelled cell, with proportional label splits."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.utils.sparsefuncs import inplace_row_scale
from threadpoolctl import threadpool_limits

from scripts import prepare_calderon_lineage_data as preparation

VARIANT = "all_times"


def split_diagnostics(obs, split):
    """Audit the common branch-stratified split and both target proportions."""
    if not obs.index.is_unique or len(split) != len(obs):
        raise ValueError("Cell identities and split assignments must align uniquely")
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Expected disjoint training, validation and test assignments")
    if (obs.groupby("branch_id", observed=True).broad_tissue.nunique() != 1).any():
        raise ValueError("Each branch must map to exactly one tissue")
    rows, deviations = [], {}
    sizes = {name: int((split == name).sum()) for name in ("train", "validation", "test")}
    for target in ("branch_id", "broad_tissue", "day"):
        labels = obs[target].astype(str).to_numpy()
        for label in sorted(set(labels)):
            selected = labels == label
            n = int(selected.sum())
            row = {"target": target, "class": label, "dataset_n": n,
                   "dataset_pct": 100*n/len(obs)}
            for name, size in sizes.items():
                count = int((selected & (split == name)).sum())
                if count == 0:
                    raise ValueError(f"Missing {target}/{label} in {name}")
                row[f"{name}_n"] = count
                row[f"{name}_pct"] = 100*count/size
                row[f"{name}_difference_pp"] = row[f"{name}_pct"]-row["dataset_pct"]
            rows.append(row)
        subset = [r for r in rows if r["target"] == target]
        deviations[target] = {name: max(abs(r[f"{name}_difference_pp"]) for r in subset)
                              for name in ("validation", "test")}
    audit = {"status": "passed", "n_cells": len(obs), "split_sizes": sizes,
             "unique_cells_and_disjoint_complete_split": True,
             "stratification": "branch_id; deterministic branch-to-tissue mapping also preserves tissue proportions",
             "all_classes_and_time_windows_present_in_every_split": True,
             "max_absolute_proportion_difference_pp": deviations}
    return pd.DataFrame(rows), audit


def prepare(source_dir, output_dir, split_seed=0, n_hvgs=3000, n_pcs=100):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir:
        raise ValueError("Use a separate output directory for the all-cell representation")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir/f"lineage_assigned_hvg{n_hvgs}_pca{n_pcs}_{VARIANT}.h5ad"
    manifest_path = output_dir/f"preprocessing_{VARIANT}.json"
    source_files = ("pruning.json", "lineage_assigned_metadata.csv.gz", "feature_names.txt",
                    "csr/data.npy", "csr/indices.npy", "csr/indptr.npy",
                    "lineage_assigned_hvg3000_pca100_random.h5ad",
                    "lineage_assigned_hvg3000_pca100_heldout_8_10.h5ad")
    source_hashes = {name: preparation.file_hash(source_dir/name) for name in source_files}
    if destination.exists():
        if not manifest_path.exists():
            raise ValueError("Incomplete all-cell representation; choose another output directory")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["split_seed"], manifest["n_hvgs"], manifest["n_pcs"], manifest["source_hashes"]) != (
                split_seed, n_hvgs, n_pcs, source_hashes):
            raise ValueError("Existing all-cell representation settings or sources differ")
        if preparation.file_hash(destination) != manifest["h5ad_sha256"]:
            raise ValueError("Existing all-cell representation changed")
        preparation.log(f"Reusing verified all-cell representation: {destination}")
        return destination
    counts = preparation.load_counts(source_dir)
    obs = pd.read_csv(source_dir/"lineage_assigned_metadata.csv.gz", index_col=0)
    genes = (source_dir/"feature_names.txt").read_text().splitlines()
    if counts.shape != (len(obs), len(genes)) or not obs.index.is_unique:
        raise ValueError("Raw count dimensions and labelled metadata differ")
    if not np.array_equal(np.diff(counts.indptr), obs.nFeature_RNA.to_numpy()):
        raise ValueError("Raw detected-feature totals differ from metadata")
    for start in range(0, len(obs), 10000):
        stop = min(start+10000, len(obs))
        left, right = int(counts.indptr[start]), int(counts.indptr[stop])
        totals = np.add.reduceat(counts.data[left:right], counts.indptr[start:stop]-left, dtype=np.int64)
        np.testing.assert_array_equal(totals, obs.nCount_RNA.to_numpy(dtype=np.int64)[start:stop])
    split = preparation.split_cells(obs, "random", split_seed)
    obs["classifier_split"] = split
    proportions, split_audit = split_diagnostics(obs, split)
    proportions.to_csv(output_dir/"split_proportions.csv", index=False)
    (output_dir/"split_validation.json").write_text(json.dumps(split_audit, indent=2)+"\n")
    obs[["branch_id", "broad_tissue", "day", "classifier_split"]].to_csv(output_dir/"cell_splits.csv.gz")
    fit_mask = np.ones(len(obs), dtype=bool)
    preparation.log(f"All times: {len(obs):,} cells, split {dict(Counter(split))}; fitting HVG/PCA on ALL cells")
    selected, stats = preparation.seurat_v3_hvgs(counts, fit_mask, n_hvgs)
    stats = stats.rename(columns={"training_mean_counts": "all_cells_mean_counts",
                                  "training_variance_counts": "all_cells_variance_counts",
                                  "training_detected_cells": "all_cells_detected_cells"})
    preparation.log("Normalising selected genes using complete original cell-library totals")
    expression = counts[:, selected].astype(np.float32)
    del counts
    gc.collect()
    inplace_row_scale(expression, (10000/obs.nCount_RNA.to_numpy(dtype=np.float64)).astype(np.float32))
    np.log1p(expression.data, out=expression.data)
    scores, mean, scale, components, eigenvalues = preparation.fit_scaled_pca(expression, fit_mask, n_pcs)
    stats.index = pd.Index(genes, name="feature")
    var = stats.iloc[selected].copy()
    var["pca_all_cells_mean_log1p"] = mean
    var["pca_all_cells_scale_log1p"] = scale
    processed = ad.AnnData(X=expression, obs=preparation.h5ad_safe_obs(obs), var=var, dtype=expression.dtype)
    processed.obsm["X_pca"] = scores
    processed.varm["PCs"] = components.T.astype(np.float32)
    processed.uns["pca"] = {"variance": eigenvalues, "variance_ratio": eigenvalues/n_hvgs,
                            "fit_split": "all_cells", "n_fit_cells": len(obs), "zero_center": True,
                            "scale_genes_to_unit_variance": True}
    processed.uns["preprocessing"] = {"variant": VARIANT, "seed": 0, "split_seed": split_seed,
        "n_hvgs": n_hvgs, "n_pcs": n_pcs, "hvg_fit": "all_cells", "pca_fit": "all_cells",
        "n_fit_cells": len(obs), "heldout_window": "none", "normalization_target": 10000,
        "normalization_library": "All 23,932 original feature counts per cell",
        "hvg_method": "Seurat v3 VST, LOESS span 0.3, min 3 cells; all labelled raw counts",
        "pca_solver": "Exact all-cell covariance eigh; implicit centering and unit variance scaling",
        "classifier_holdout_scope": "Labels held out; all expressions enter unsupervised HVG/PCA fitting"}
    stats.to_csv(output_dir/f"hvg_statistics_{VARIANT}.csv.gz")
    transform_path = output_dir/f"preprocessing_hvg{n_hvgs}_pca{n_pcs}_{VARIANT}.npz"
    np.savez_compressed(transform_path, selected_feature_indices=selected,
                        feature_names=np.asarray(genes)[selected], gene_mean=mean, gene_scale=scale,
                        components=components, variance=eigenvalues, fit_indices=np.arange(len(obs)),
                        classifier_train_indices=np.flatnonzero(split == "train"),
                        classifier_validation_indices=np.flatnonzero(split == "validation"),
                        classifier_test_indices=np.flatnonzero(split == "test"), split_seed=np.asarray(split_seed))
    preparation.log(f"Writing all-cell latent dataset; PCA{n_pcs} explains {eigenvalues.sum()/n_hvgs:.2%}")
    processed.write_h5ad(destination, compression="gzip", compression_opts=1)
    rng = np.random.default_rng(981)
    check_indices = np.sort(rng.choice(len(obs), min(256, len(obs)), replace=False))
    check = ((expression[check_indices].toarray().astype(np.float64)-mean)/scale)@components.T
    error = float(np.max(abs(check-scores[check_indices])))
    np.testing.assert_allclose(check, scores[check_indices], rtol=1e-4, atol=2e-5)
    if any(preparation.file_hash(source_dir/name) != digest for name, digest in source_hashes.items()):
        raise ValueError("A source or previous representation changed during preprocessing")
    manifest = {"variant": VARIANT, "seed": 0, "split_seed": split_seed, "n_cells": len(obs),
        "n_hvgs": n_hvgs, "n_pcs": n_pcs, "n_fit_cells": len(obs), "hvg_fit": "all_cells",
        "pca_fit": "all_cells", "heldout_window": None, "split_sizes": split_audit["split_sizes"],
        "explained_variance_ratio": float(eigenvalues.sum()/n_hvgs),
        "max_projection_error_256_cells": error, "source_directory": str(source_dir),
        "source_hashes": source_hashes, "source_artifacts_unchanged": True,
        "h5ad_sha256": preparation.file_hash(destination),
        "preprocessing_npz_sha256": preparation.file_hash(transform_path),
        "heldout_accuracy_scope": "Transductive label holdout: HVG/PCA use every expression profile; classifier labels are split"}
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    preparation.log(f"COMPLETE: {destination}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT/"datasets/calderon2022")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"datasets/calderon2022/all_times")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        prepare(args.source_dir, args.output_dir, args.split_seed)


if __name__ == "__main__":
    main()

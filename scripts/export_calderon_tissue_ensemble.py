#!/usr/bin/env python3
"""Export the frozen tissue ensemble and a training-only inference check set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "py"))

import h5py
import numpy as np
import pandas as pd
import torch

from common.calderon import (
    load_tissue_manifest, sha256_file, DEFAULT_TRAINING_MANIFEST,
    DEFAULT_TRAINING_EXPORT, DEFAULT_SPLIT_MANIFEST,
)
from scripts.train_calderon_expressive_classifiers import predict_selected


def save_npz_if_changed(path, arrays, *, immutable=False):
    """Keep identical exports byte-for-byte, preserving saved run hashes."""
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            same = set(stored.files) == set(arrays) and all(np.array_equal(stored[key], value) for key, value in arrays.items())
        if same:
            return False
        if immutable:
            raise FileExistsError(f"Existing frozen export differs; choose another --output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRAINING_EXPORT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--dataset-location", type=Path)
    parser.add_argument("--weights-only", action="store_true", help="Export frozen weights without constructing a training-window probe")
    parser.add_argument("--baseline-run", type=Path,
                        help="Optional saved flow run supplying its training folds (for an older classifier export)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = load_tissue_manifest(args.manifest)
    arrays = {}
    members = []
    for i, member in enumerate(spec["members"]):
        if member["kind"] != "expressive":
            raise ValueError("This export expects the already selected expressive ensemble.")
        path = (args.manifest.parent / member["checkpoint"]).resolve()
        stored = torch.load(path, map_location="cpu", weights_only=False)
        if stored["metadata"]["class_names"] != spec["class_names"]:
            raise ValueError("Ensemble member class order differs.")
        family = stored["metadata"]["recipe_config"]["family"]
        if family not in ("wide", "residual"):
            raise ValueError(f"Unsupported frozen family: {family}")
        members.append({"family": family, "checkpoint_sha256": sha256_file(path)})
        for name, value in stored["model_state_dict"].items():
            if (name.startswith("trunk.") or name.startswith("heads.broad_tissue.")) and not name.endswith("num_batches_tracked"):
                arrays[f"member{i}__{name}"] = value.numpy().astype(np.float32)
    metadata = {"manifest": spec, "manifest_sha256": sha256_file(args.manifest), "members": members,
                "construction": "Frozen checkpoint export, no fitting; exact GELU and eval-mode normalization"}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    if args.weights_only:
        updated = save_npz_if_changed(args.output, arrays, immutable=True)
        print(f"{'Exported' if updated else 'Reused'} frozen weights: {args.output}")
        return
    if args.baseline_run:
        table_path = args.baseline_run / "cell_splits.csv.gz"
        table = pd.read_csv(table_path)
        cfg = json.loads((args.baseline_run / "config.json").read_text())
        dataset = Path(cfg["problem"]["dataset_location"])
        windows = cfg["problem"].get("retained_timepoints", (cfg["problem"]["source_time"], cfg["problem"]["target_time"]))
        split_column, row_column = "flow_split", "pca_row"
    else:
        table_path = args.split_manifest
        table = pd.read_csv(table_path)
        dataset = args.dataset_location or Path(spec.get("pca_representation_source_h5ad", spec["source_h5ad"]))
        windows = spec["training_windows"]
        split_column, row_column = "classifier_split", "source_pca_row"
    expected_dataset_hash = spec.get("pca_representation_source_sha256", spec.get("source_h5ad_sha256"))
    if expected_dataset_hash is not None and sha256_file(dataset) != expected_dataset_hash:
        raise ValueError("Verification points must use the classifier's original PCA representation.")
    rng = np.random.default_rng(9401)
    rows = []
    for window in windows:
        eligible = table.loc[table[split_column].eq("train") & table.day.eq(window), row_column].to_numpy()
        rows.append(np.sort(rng.choice(eligible, size=256, replace=False)))
    with h5py.File(dataset, "r") as handle:
        endpoints = [handle["obsm/X_pca"][indices] for indices in rows]
    taus = np.linspace(0., 1., 52, dtype=np.float32)[1:-1]
    paths = [source[:128, None] * (1 - taus[None, :, None]) + target[:128, None] * taus[None, :, None]
             for source, target in zip(endpoints[:-1], endpoints[1:])]
    probe = np.concatenate([*endpoints, *[path.reshape(-1, 100) for path in paths]])
    probabilities = predict_selected(args.manifest, probe, batch_size=4096)
    updated = save_npz_if_changed(args.output, arrays, immutable=True)
    check_path = args.output.with_name(args.output.stem + "_verification.npz")
    save_npz_if_changed(check_path, dict(x_pca=probe, torch_probabilities=probabilities,
        class_names=np.asarray(spec["class_names"]), export_sha256=np.asarray(sha256_file(args.output)),
        endpoint_pca_rows=np.stack(rows), endpoint_windows=np.asarray(windows),
        split_manifest_sha256=np.asarray(sha256_file(table_path)), dataset_sha256=np.asarray(sha256_file(dataset))))
    print(f"{'Exported' if updated else 'Reused'}: {args.output}; verification: {check_path}; "
          f"{len(probe)} endpoint/interpolant points at {windows}, training cells only.")


if __name__ == "__main__":
    main()

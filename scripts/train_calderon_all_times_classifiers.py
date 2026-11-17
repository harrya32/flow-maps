#!/usr/bin/env python3
"""Validation-selected branch/tissue classifiers in the all-cell PCA100 atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts import train_calderon_expressive_classifiers as expressive
from scripts.prepare_calderon_all_times_data import VARIANT, split_diagnostics
from scripts.prepare_calderon_lineage_data import file_hash


def dataset(args):
    path = args.data_dir/f"lineage_assigned_hvg3000_pca100_{VARIANT}.h5ad"
    manifest = json.loads((args.data_dir/f"preprocessing_{VARIANT}.json").read_text())
    if manifest["hvg_fit"] != "all_cells" or manifest["pca_fit"] != "all_cells":
        raise ValueError("Expected HVG/PCA fitted on every labelled cell")
    if file_hash(path) != manifest["h5ad_sha256"]:
        raise ValueError("Latent dataset differs from its preprocessing manifest")
    data = expressive.load_data(path, expected_seed=0, expected_variant=VARIANT)
    _, obs, _, indices = data
    _, audit = split_diagnostics(obs, obs.classifier_split.astype(str).to_numpy())
    if audit["split_sizes"] != manifest["split_sizes"]:
        raise ValueError("Classifier split differs from the preprocessing manifest")
    return path, manifest, data


def cell_split_hash(obs):
    digest = hashlib.sha256()
    digest.update(obs.index.to_numpy(dtype=str).tobytes())
    digest.update(obs.classifier_split.astype(str).to_numpy(dtype=str).tobytes())
    return digest.hexdigest()


def freeze(args):
    path, preprocessing, (x, obs, features, indices) = dataset(args)
    plan = json.loads((args.output_dir/"search_plan.json").read_text())
    if plan["recipes"] != expressive.RECIPES or plan["seed"] != args.seed or plan["variants"] != [VARIANT]:
        raise ValueError("Search settings differ from the requested selection")
    scaler = StandardScaler().fit(x[indices["train"]])
    groups = []
    split_hash = cell_split_hash(obs)
    for target in expressive.TARGETS:
        classes = sorted(obs[target].astype(str).unique().tolist())
        ids = {label: i for i, label in enumerate(classes)}
        encoded = obs[target].astype(str).map(ids).to_numpy(dtype=np.int64)
        truth = encoded[indices["validation"]]
        counts = np.bincount(encoded[indices["train"]], minlength=len(classes))
        descriptions, probabilities = [], {}
        for recipe in args.recipes:
            targets = expressive.TARGETS if recipe == "joint" else (target,)
            run_id = f"{VARIANT}_{'+'.join(targets)}_{recipe}_seed{args.seed}"
            directory = args.output_dir/"search"/run_id
            if not (directory/"complete.json").exists():
                raise ValueError(f"Incomplete model search: {run_id}")
            metadata = json.loads((directory/f"{target}_validation.json").read_text())
            if (metadata["class_names"] != classes or metadata["cell_and_split_sha256"] != split_hash
                    or metadata["selected_genes"] != features or metadata["source_h5ad"] != str(path)):
                raise ValueError("Candidate data, labels or representation differ")
            np.testing.assert_allclose(scaler.mean_, metadata["scaler_mean"], rtol=1e-8, atol=1e-8)
            with np.load(directory/f"{target}_validation_logits.npz", allow_pickle=False) as saved:
                logits = saved["logits"].copy()
                np.testing.assert_array_equal(saved["class_names"], classes)
                assert saved["cell_and_split_sha256"].item() == split_hash
            temperature = expressive.fit_temperature(logits, truth)
            probabilities[recipe] = expressive.scaled_probabilities(logits, temperature)
            checkpoint = directory/f"{target}.pt"
            descriptions.append({"id": recipe, "kind": "expressive", "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_hash(checkpoint), "temperature": temperature,
                "parameter_count": metadata["parameter_count"],
                "best_epoch": metadata["best_epoch"],
                "validation": expressive.scores(truth, probabilities[recipe], counts)})
        members, average, _, individual = expressive.greedy_ensemble(probabilities, truth, counts)
        temperature = expressive.fit_temperature(np.log(np.maximum(average, 1e-30)), truth)
        ensemble_prob = expressive.scaled_probabilities(np.log(np.maximum(average, 1e-30)), temperature)
        winner = max(individual, key=lambda name: (individual[name]["macro_f1"], -individual[name]["log_loss"]))
        group = {"variant": VARIANT, "target": target, "class_names": classes,
            "cell_and_split_sha256": split_hash, "split_sizes": preprocessing["split_sizes"],
            "candidates": descriptions, "best_expressive": winner, "single_winner": winner,
            "ensemble_members": members, "ensemble_temperature": temperature,
            "ensemble_validation": expressive.scores(truth, ensemble_prob, counts),
            "selected": "ensemble" if len(members) > 1 else members[0],
            "selection_metric": "Validation macro-F1; ties use validation cross entropy",
            "test_used_for_selection": False}
        groups.append(group)
        expressive.log(f"FROZEN {target}: selected={group['selected']}, members={members}, validation macro-F1={group['ensemble_validation']['macro_f1']:.4f}")
    result = {"schema_version": 2, "groups": groups, "recipes": expressive.RECIPES, "seed": args.seed,
        "selection_frozen_before_new_test_evaluation": True, "source_h5ad": str(path),
        "source_h5ad_sha256": preprocessing["h5ad_sha256"],
        "preprocessing_npz_sha256": preprocessing["preprocessing_npz_sha256"],
        "representation_fit": "All labelled cells, including classifier validation/test expressions",
        "classifier_supervision": "Training labels only; selection/calibration use validation labels only",
        "calibration": "Scalar temperatures fitted using validation negative log likelihood"}
    expressive.write_json(args.output_dir/"selection.json", result)


def evaluate(args):
    path, preprocessing, (x, obs, _, indices) = dataset(args)
    selection_path = args.output_dir/"selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["selection_frozen_before_new_test_evaluation"] or selection["source_h5ad_sha256"] != preprocessing["h5ad_sha256"]:
        raise ValueError("Selection must be frozen for this exact representation before testing")
    selection_hash = file_hash(selection_path)
    scaler = StandardScaler().fit(x[indices["train"]])
    test_x = torch.from_numpy(scaler.transform(x[indices["test"]]).astype(np.float32))
    records = []
    verification = []
    for group in selection["groups"]:
        target, classes = group["target"], group["class_names"]
        ids = {label: i for i, label in enumerate(classes)}
        encoded = obs[target].astype(str).map(ids).to_numpy(dtype=np.int64)
        truth = encoded[indices["test"]]
        counts = np.bincount(encoded[indices["train"]], minlength=len(classes))
        probabilities = {}
        for candidate in group["candidates"]:
            checkpoint = Path(candidate["checkpoint"])
            if file_hash(checkpoint) != candidate["checkpoint_sha256"]:
                raise ValueError("A selected candidate checkpoint changed after selection")
            model, metadata = expressive.load_checkpoint_model(checkpoint)
            np.testing.assert_allclose(scaler.mean_, metadata["scaler_mean"], rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(scaler.scale_, metadata["scaler_scale"], rtol=1e-8, atol=1e-8)
            logits = expressive.predict_logits(model, test_x, torch.device("cpu"))[target]
            probabilities[candidate["id"]] = expressive.scaled_probabilities(logits, candidate["temperature"])
            records.append(expressive.save_test_details(args, group, candidate["id"], probabilities[candidate["id"]],
                truth, obs, indices, counts, candidate["validation"], selection_hash))
        if group["selected"] == "ensemble":
            average = np.mean([probabilities[name] for name in group["ensemble_members"]], axis=0)
            probabilities["ensemble"] = expressive.scaled_probabilities(np.log(np.maximum(average, 1e-30)), group["ensemble_temperature"])
            records.append(expressive.save_test_details(args, group, "ensemble", probabilities["ensemble"],
                truth, obs, indices, counts, group["ensemble_validation"], selection_hash))
        manifest_path = args.output_dir/"models"/f"{VARIANT}_{target}_selected.json"
        chosen = group["ensemble_members"] if group["selected"] == "ensemble" else [group["selected"]]
        members = []
        for name in chosen:
            candidate = next(c for c in group["candidates"] if c["id"] == name)
            members.append({"checkpoint": os.path.relpath(candidate["checkpoint"], manifest_path.parent),
                "kind": "expressive", "temperature": candidate["temperature"], "method": name})
        manifest = {"target": target, "variant": VARIANT, "class_names": classes, "members": members,
            "temperature": group["ensemble_temperature"] if group["selected"] == "ensemble" else 1.,
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "n_pcs": 100,
            "selection_sha256": selection_hash, "selected": group["selected"], "source_h5ad": str(path),
            "source_h5ad_sha256": preprocessing["h5ad_sha256"], "representation_fit": "all_cells"}
        expressive.write_json(manifest_path, manifest)
        reloaded = expressive.predict_selected(manifest_path, x[indices["test"][:256]])
        np.testing.assert_allclose(reloaded, probabilities[group["selected"]][:256], rtol=1e-5, atol=1e-5)
        np.savez_compressed(args.output_dir/"reports"/f"{target}_selected_test_probabilities.npz",
            probabilities=probabilities[group["selected"]], class_names=np.asarray(classes),
            cell_ids=obs.index.to_numpy(dtype=str)[indices["test"]], truth=truth,
            selection_sha256=np.asarray(selection_hash))
        verification.append({"target": target, "selected_manifest_sha256": file_hash(manifest_path),
            "probabilities_reproduced_after_manifest_reload": True,
            "max_absolute_probability_difference_256_cells": float(np.max(abs(reloaded-probabilities[group["selected"]][:256]))),
            "n_test_cells": len(truth), "n_test_classes": len(np.unique(truth))})
    frame = pd.DataFrame([{ "target": record["target"], "method": record["method"], "selected": record["selected"],
        "validation_accuracy": record["validation"]["accuracy"],
        "validation_macro_f1": record["validation"]["macro_f1"], "accuracy": record["test"]["accuracy"],
        "balanced_accuracy": record["test"]["balanced_accuracy"], "macro_f1": record["test"]["macro_f1"],
        "test_cells": record["test"]["n_cells"], "test_classes": record["test"]["n_present_classes"]} for record in records])
    frame.to_csv(args.output_dir/"reports/results.csv", index=False)
    expressive.write_json(args.output_dir/"evaluation_complete.json", {"selection_sha256": selection_hash,
        "methods_evaluated": len(records), "selection_used_validation_only": True,
        "selected_manifests_verified": verification, "test_labels_excluded_from_supervised_training": True,
        "representation_fit_includes_validation_test_expressions": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"datasets/calderon2022/all_times")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"calderon-classifiers/all_times")
    parser.add_argument("--stage", choices=("search", "freeze", "evaluate", "report", "all"), default="all")
    parser.add_argument("--recipes", nargs="+", choices=tuple(expressive.RECIPES), default=list(expressive.RECIPES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.variants = [VARIANT]
    args.classifier_inputs = "PCA100 only; HVG/PCA fitted on all labelled cells; classifier PC scaler fitted on training cells"
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) <= 0:
        parser.error("Epoch, patience, batch and thread settings must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("search", "all"):
            dataset(args)
            expressive.search(args)
        if args.stage in ("freeze", "all"):
            freeze(args)
        if args.stage in ("evaluate", "all"):
            evaluate(args)
        if args.stage in ("report", "all"):
            from scripts.report_calderon_all_times_classifiers import report
            report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a tissue constraint classifier at 4–6, 8–10 and 12–14 h only."""

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
from scripts.prepare_calderon_lineage_data import file_hash
from scripts.prepare_calderon_three_time_data import (
    VARIANT, TRAINING_WINDOWS, INTERMEDIATE_WINDOWS, SPLITS, prepare,
)

TARGET = "broad_tissue"


def split_hash(obs):
    digest = hashlib.sha256()
    digest.update(obs.index.to_numpy(dtype=str).tobytes())
    digest.update(obs.classifier_split.astype(str).to_numpy(dtype=str).tobytes())
    return digest.hexdigest()


def evaluation_indices(table):
    if not table.cell_id.is_unique or not table.source_pca_row.is_unique:
        raise ValueError("Evaluation cells and source rows must be unique")
    for name in SPLITS:
        if not table.loc[table.classifier_split.eq(name), "day"].isin(TRAINING_WINDOWS).all():
            raise ValueError("Intermediate cells entered a selected-time training split")
    intermediate = table.classifier_split.eq("intermediate_holdout")
    if not table.loc[intermediate, "day"].isin(INTERMEDIATE_WINDOWS).all():
        raise ValueError("Invalid intermediate holdout assignment")
    if set(table.classifier_split) != set(SPLITS) | {"intermediate_holdout"}:
        raise ValueError("Unexpected interval split categories")
    result = {"selected_time_test": table.loc[table.classifier_split.eq("test"), "source_pca_row"].to_numpy(dtype=np.int64),
              "intermediate_pooled": table.loc[intermediate, "source_pca_row"].to_numpy(dtype=np.int64)}
    for window in INTERMEDIATE_WINDOWS:
        rows = table.loc[table.day.eq(window)]
        if not rows.classifier_split.eq("intermediate_holdout").all():
            raise ValueError("A held-out intermediate marginal was split for fitting")
        result[f"intermediate_{window.replace('-', '_')}"] = rows.source_pca_row.to_numpy(dtype=np.int64)
    fitting = table.loc[table.classifier_split.isin(("train", "validation")), "source_pca_row"].to_numpy()
    for rows in result.values():
        if len(np.intersect1d(rows, fitting)):
            raise ValueError("Evaluation overlaps classifier fitting or selection cells")
    if len(np.intersect1d(result["selected_time_test"], result["intermediate_pooled"])):
        raise ValueError("Cell holdouts and temporal holdouts overlap")
    return result


def dataset(args):
    path = args.data_dir / f"lineage_assigned_hvg3000_pca100_{VARIANT}.h5ad"
    manifest = json.loads((args.data_dir / f"preprocessing_{VARIANT}.json").read_text())
    if (manifest["training_windows"], manifest["intermediate_windows"]) != (list(TRAINING_WINDOWS), list(INTERMEDIATE_WINDOWS)):
        raise ValueError("Selected training or intermediate windows differ")
    if file_hash(path) != manifest["h5ad_sha256"] or file_hash(manifest["source_h5ad"]) != manifest["source_h5ad_sha256"]:
        raise ValueError("Selected or source PCA representation changed")
    data = expressive.load_data(path, expected_seed=0, expected_variant=VARIANT)
    _, obs, _, indices = data
    if set(obs.day.astype(str)) != set(TRAINING_WINDOWS):
        raise ValueError("Training dataset contains unselected times")
    if {name: len(rows) for name, rows in indices.items()} != manifest["split_sizes"]:
        raise ValueError("Classifier split differs from its manifest")
    table = pd.read_csv(args.data_dir / "cell_splits.csv.gz")
    selected = table.loc[table.classifier_split.isin(SPLITS)]
    np.testing.assert_array_equal(selected.cell_id.to_numpy(dtype=str), obs.index.to_numpy(dtype=str))
    np.testing.assert_array_equal(selected.classifier_split, obs.classifier_split.astype(str))
    np.testing.assert_array_equal(selected.source_pca_row, obs.source_pca_row)
    populations = evaluation_indices(table)
    if len(populations["intermediate_pooled"]) != manifest["n_intermediate_cells"]:
        raise ValueError("An intermediate marginal is incomplete")
    return path, manifest, data, table


def search(args):
    path, manifest, data, _ = dataset(args)
    plan = {"variant": VARIANT, "target": TARGET, "requested_recipes": args.recipes,
            "recipes": expressive.RECIPES, "seed": args.seed, "max_epochs": args.max_epochs,
            "patience": args.patience, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "source_h5ad": str(path), "source_h5ad_sha256": manifest["h5ad_sha256"],
            "source_global_pca_sha256": manifest["source_h5ad_sha256"],
            "training_windows": list(TRAINING_WINDOWS), "intermediate_windows": list(INTERMEDIATE_WINDOWS),
            "selection": "Selected-time validation macro-F1 only; intermediate/test labels excluded",
            "joint_recipe": "Branch prediction is an auxiliary task trained only at the same selected times"}
    plan_path = args.output_dir / "search_plan.json"
    if plan_path.exists() and not args.overwrite and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing search protocol differs; use a separate output directory")
    expressive.write_json(plan_path, plan)
    implementation = ("scripts/prepare_calderon_three_time_data.py", "scripts/train_calderon_three_time_classifier.py",
                      "scripts/train_calderon_expressive_classifiers.py", "scripts/train_calderon_lineage_classifiers.py")
    expressive.write_json(args.output_dir / "implementation_hashes.json", {name: file_hash(ROOT / name) for name in implementation})
    oracle = ROOT / "calderon-classifiers/all_times/models/all_times_broad_tissue_selected.json"
    if oracle.exists():
        spec = json.loads(oracle.read_text())
        paths = [oracle] + [(oracle.parent / member["checkpoint"]).resolve() for member in spec["members"]]
        expressive.write_json(args.output_dir / "reference_oracle_hashes.json", {str(p): file_hash(p) for p in paths})
    for recipe in args.recipes:
        targets = expressive.TARGETS if recipe == "joint" else (TARGET,)
        expressive.fit_recipe(args, data, VARIANT, recipe, targets)
    expressive.log("Selected-time search complete; no intermediate or test scores were used.")


def freeze(args):
    path, preprocessing, (x, obs, features, indices), _ = dataset(args)
    plan = json.loads((args.output_dir / "search_plan.json").read_text())
    if plan["requested_recipes"] != args.recipes or plan["seed"] != args.seed or plan["source_h5ad_sha256"] != preprocessing["h5ad_sha256"]:
        raise ValueError("Search plan differs from this selection")
    scaler = StandardScaler().fit(x[indices["train"]])
    classes = sorted(obs[TARGET].astype(str).unique().tolist())
    ids = {label: i for i, label in enumerate(classes)}
    labels = obs[TARGET].astype(str).map(ids).to_numpy(dtype=np.int64)
    truth = labels[indices["validation"]]
    counts = np.bincount(labels[indices["train"]], minlength=len(classes))
    digest = split_hash(obs)
    probabilities, descriptions = {}, []
    for recipe in args.recipes:
        targets = expressive.TARGETS if recipe == "joint" else (TARGET,)
        directory = args.output_dir / "search" / f"{VARIANT}_{'+'.join(targets)}_{recipe}_seed{args.seed}"
        if not (directory / "complete.json").exists():
            raise ValueError(f"Incomplete search recipe: {recipe}")
        metadata = json.loads((directory / f"{TARGET}_validation.json").read_text())
        if (metadata["class_names"] != classes or metadata["cell_and_split_sha256"] != digest
                or metadata["selected_genes"] != features or metadata["source_h5ad"] != str(path)):
            raise ValueError("Candidate labels, source representation or split differ")
        np.testing.assert_allclose(scaler.mean_, metadata["scaler_mean"], atol=1e-8, rtol=1e-8)
        with np.load(directory / f"{TARGET}_validation_logits.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored["class_names"], classes)
            assert stored["cell_and_split_sha256"].item() == digest
            logits = stored["logits"].copy()
        temperature = expressive.fit_temperature(logits, truth)
        probabilities[recipe] = expressive.scaled_probabilities(logits, temperature)
        checkpoint = directory / f"{TARGET}.pt"
        descriptions.append({"id": recipe, "kind": "expressive", "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_hash(checkpoint), "temperature": temperature,
            "best_epoch": metadata["best_epoch"], "parameter_count": metadata["parameter_count"],
            "validation": expressive.scores(truth, probabilities[recipe], counts)})
    members, average, _, individual = expressive.greedy_ensemble(probabilities, truth, counts)
    ensemble_temperature = expressive.fit_temperature(np.log(np.maximum(average, 1e-30)), truth)
    ensemble_probabilities = expressive.scaled_probabilities(np.log(np.maximum(average, 1e-30)), ensemble_temperature)
    winner = max(individual, key=lambda name: (individual[name]["macro_f1"], -individual[name]["log_loss"]))
    group = {"variant": VARIANT, "target": TARGET, "class_names": classes, "cell_and_split_sha256": digest,
        "split_sizes": preprocessing["split_sizes"], "candidates": descriptions, "best_expressive": winner,
        "ensemble_members": members, "ensemble_temperature": ensemble_temperature,
        "ensemble_validation": expressive.scores(truth, ensemble_probabilities, counts),
        "selected": "ensemble" if len(members) > 1 else members[0],
        "selection_metric": "Selected-time validation macro-F1; ties use validation cross entropy",
        "test_used_for_selection": False, "intermediate_used_for_selection": False}
    expressive.write_json(args.output_dir / "selection.json", {"schema_version": 3, "group": group,
        "training_windows": list(TRAINING_WINDOWS), "intermediate_windows": list(INTERMEDIATE_WINDOWS),
        "selection_frozen_before_test_and_intermediate_evaluation": True,
        "source_h5ad_sha256": preprocessing["h5ad_sha256"], "source_global_pca_sha256": preprocessing["source_h5ad_sha256"],
        "split_seed": preprocessing["split_seed"], "seed": args.seed,
        "supervision": "Training cells at selected windows only; validation at those same windows only",
        "representation_fit": "Existing HVG3000/PCA100 fitted on all 314,387 lineage-assigned cells"})
    expressive.log(f"FROZEN {TARGET}: selected={group['selected']}, members={members}, "
                   f"validation accuracy={group['ensemble_validation']['accuracy']:.4f}, "
                   f"macro-F1={group['ensemble_validation']['macro_f1']:.4f}")


def evaluate(args):
    selection_path = args.output_dir / "selection.json"
    if not selection_path.exists():
        raise ValueError("Freeze selected-time validation choices before test/intermediate evaluation")
    selection = json.loads(selection_path.read_text())
    path, preprocessing, (x, obs, _, indices), table = dataset(args)
    if (not selection["selection_frozen_before_test_and_intermediate_evaluation"]
            or selection["source_h5ad_sha256"] != preprocessing["h5ad_sha256"]):
        raise ValueError("Selection must be frozen for this exact representation")
    selection_digest = file_hash(selection_path)
    group = selection["group"]
    classes = group["class_names"]
    ids = {label: i for i, label in enumerate(classes)}
    counts = np.bincount(obs[TARGET].astype(str).map(ids).to_numpy(dtype=np.int64)[indices["train"]], minlength=len(classes))
    populations = evaluation_indices(table)
    all_rows = np.unique(np.concatenate([populations["selected_time_test"], populations["intermediate_pooled"]]))
    global_x, global_obs, _, _ = expressive.load_data(Path(preprocessing["source_h5ad"]), expected_seed=0, expected_variant="all_times")
    np.testing.assert_array_equal(global_obs.index.to_numpy(dtype=str)[table.source_pca_row], table.cell_id.to_numpy(dtype=str))
    np.testing.assert_array_equal(global_x[obs.source_pca_row.to_numpy()], x)
    scaler = StandardScaler().fit(x[indices["train"]])
    eval_tensor = torch.from_numpy(scaler.transform(global_x[all_rows]).astype(np.float32))
    probabilities = {}
    for candidate in group["candidates"]:
        checkpoint = Path(candidate["checkpoint"])
        if file_hash(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("A candidate checkpoint changed after frozen selection")
        model, metadata = expressive.load_checkpoint_model(checkpoint)
        np.testing.assert_allclose(scaler.mean_, metadata["scaler_mean"], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(scaler.scale_, metadata["scaler_scale"], atol=1e-8, rtol=1e-8)
        logits = expressive.predict_logits(model, eval_tensor, torch.device("cpu"))[TARGET]
        probabilities[candidate["id"]] = expressive.scaled_probabilities(logits, candidate["temperature"])
    if group["selected"] == "ensemble":
        average = np.mean([probabilities[name] for name in group["ensemble_members"]], axis=0)
        probabilities["ensemble"] = expressive.scaled_probabilities(np.log(np.maximum(average, 1e-30)), group["ensemble_temperature"])
    manifest_path = args.output_dir / "models" / f"{VARIANT}_{TARGET}_selected.json"
    chosen = group["ensemble_members"] if group["selected"] == "ensemble" else [group["selected"]]
    members = []
    for name in chosen:
        candidate = next(c for c in group["candidates"] if c["id"] == name)
        members.append({"checkpoint": os.path.relpath(candidate["checkpoint"], manifest_path.parent),
                        "kind": "expressive", "temperature": candidate["temperature"], "method": name})
    expressive.write_json(manifest_path, {"target": TARGET, "variant": VARIANT, "class_names": classes,
        "members": members, "temperature": group["ensemble_temperature"] if group["selected"] == "ensemble" else 1.,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "n_pcs": 100,
        "selection_sha256": selection_digest, "selected": group["selected"],
        "source_h5ad": str(path), "source_h5ad_sha256": preprocessing["h5ad_sha256"],
        "pca_representation_source_h5ad": preprocessing["source_h5ad"],
        "pca_representation_source_sha256": preprocessing["source_h5ad_sha256"],
        "training_windows": list(TRAINING_WINDOWS), "intermediate_holdout_windows": list(INTERMEDIATE_WINDOWS),
        "representation_fit": "all_cells", "classifier_supervision": "Selected-time training cells only"})
    records, verification = [], []
    for population, rows in populations.items():
        positions = np.searchsorted(all_rows, rows)
        truth_names = global_obs.iloc[rows][TARGET].astype(str)
        if not set(truth_names) <= set(classes):
            raise ValueError("An evaluation marginal has unseen tissue labels")
        truth = truth_names.map(ids).to_numpy(dtype=np.int64)
        population_group = dict(group, variant=f"{VARIANT}_{population}")
        for method, all_probabilities in probabilities.items():
            validation = group["ensemble_validation"] if method == "ensemble" else next(c["validation"] for c in group["candidates"] if c["id"] == method)
            record = expressive.save_test_details(args, population_group, method, all_probabilities[positions],
                truth, global_obs, {"test": rows}, counts, validation, selection_digest)
            record["population"] = population
            records.append(record)
        selected_probabilities = probabilities[group["selected"]][positions]
        reloaded = expressive.predict_selected(manifest_path, global_x[rows[:256]])
        np.testing.assert_allclose(reloaded, selected_probabilities[:256], atol=1e-5, rtol=1e-5)
        np.savez_compressed(args.output_dir / "reports" / f"{population}_selected_probabilities.npz",
            probabilities=selected_probabilities, truth=truth, class_names=np.asarray(classes),
            cell_ids=global_obs.index.to_numpy(dtype=str)[rows], source_pca_rows=rows,
            selection_sha256=np.asarray(selection_digest))
        verification.append({"population": population, "n_cells": len(rows), "n_present_classes": len(np.unique(truth)),
            "probabilities_reproduced_after_manifest_reload": True,
            "max_absolute_probability_difference_256_cells": float(np.max(abs(reloaded - selected_probabilities[:256])))})
    expressive.write_json(args.output_dir / "reports/all_evaluation_metrics.json", records)
    frame = pd.DataFrame([{"population": r["population"], "method": r["method"], "selected": r["selected"],
        "validation_accuracy": r["validation"]["accuracy"], "validation_macro_f1": r["validation"]["macro_f1"],
        **{key: r["test"][key] for key in ("n_cells", "n_present_classes", "accuracy", "balanced_accuracy", "macro_f1", "log_loss")}}
        for r in records])
    frame.to_csv(args.output_dir / "reports/results.csv", index=False)
    references = args.output_dir / "reference_oracle_hashes.json"
    if references.exists():
        for name, digest in json.loads(references.read_text()).items():
            assert file_hash(name) == digest
    expressive.write_json(args.output_dir / "evaluation_complete.json", {"status": "passed",
        "selection_sha256": selection_digest, "selected_manifest_sha256": file_hash(manifest_path),
        "selection_used_selected_time_validation_only": True, "intermediate_labels_excluded_from_training_and_selection": True,
        "test_cells_excluded_from_training_and_selection": True, "selected_time_supervised_split_sizes": preprocessing["split_sizes"],
        "verified_populations": verification, "reference_oracle_unchanged": True,
        "same_original_global_pca_coordinates": True})
    expressive.log(frame.loc[frame.selected].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / f"datasets/calderon2022/{VARIANT}")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "datasets/calderon2022/all_times")
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"calderon-classifiers/{VARIANT}")
    parser.add_argument("--stage", choices=("prepare", "search", "freeze", "evaluate", "report", "all"), default="all")
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
    args.data_dir, args.source_dir, args.output_dir = args.data_dir.resolve(), args.source_dir.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.classifier_inputs = "PCA100 only; original all-cell HVG/PCA retained; supervision only at 4–6, 8–10 and 12–14 h; PC scaler fitted on selected-time training cells only"
    if min(args.max_epochs, args.patience, args.batch_size, args.threads) <= 0:
        parser.error("Epoch, patience, batch and thread settings must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with threadpool_limits(limits=args.threads):
        if args.stage in ("prepare", "all"):
            prepare(args.source_dir, args.data_dir, seed=0)
        if args.stage in ("search", "all"):
            search(args)
        if args.stage in ("freeze", "all"):
            freeze(args)
        if args.stage in ("evaluate", "all"):
            evaluate(args)
        if args.stage in ("report", "all"):
            from scripts.report_calderon_three_time_classifier import report
            report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

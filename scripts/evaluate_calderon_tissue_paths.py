#!/usr/bin/env python3
"""Score saved Calderon flow trajectories with the frozen tissue ensemble.

Run with the classifier's PyTorch environment. The JAX experiment launcher
invokes this worker automatically, so JAX and PyTorch can use separate envs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "py"))

import numpy as np
import pandas as pd

from common import calderon


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def reference_accuracy(labels, probabilities, classes, threshold):
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    confident = probabilities.max(axis=1) >= threshold
    correct = predicted == labels
    return {
        "n_cells": len(labels), "accuracy": float(correct.mean()),
        "confidence_coverage_pct": 100.0 * float(confident.mean()),
        "confident_accuracy": float(correct[confident].mean()) if confident.any() else None,
    }


def validate_input_vocabulary(data, classes):
    """Observed names align with the oracle even if output IDs differ."""
    saved_classes = tuple(data["class_names"].astype(str))
    if len(saved_classes) != len(set(saved_classes)) or not set(saved_classes).issubset(classes):
        raise ValueError("The evaluation classifier must cover all saved training tissue names.")
    label_fields = ["source_tissues"]
    if "marginal_windows" in data:
        label_fields += [f"marginal_{i}_tissues" for i in range(len(data["marginal_windows"]))]
    else:
        label_fields += ["reference_tissues", "target_test_tissues"]
    if any(not np.isin(data[name], classes).all() for name in label_fields):
        raise ValueError("Observed tissue names are missing from the evaluation classifier vocabulary.")


def evaluate(args):
    import torch
    from scripts.train_calderon_expressive_classifiers import predict_selected

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = calderon.load_tissue_manifest(args.manifest)
    classes = tuple(spec["class_names"])
    allowed = calderon.load_allowed_pairs(args.allowed_pairs, classes)
    with np.load(args.input, allow_pickle=False) as stored:
        data = {name: stored[name] for name in stored.files}
    validate_input_vocabulary(data, classes)
    paths = data["paths"]
    n_sources, n_times, dimension = paths.shape
    if dimension != 100 or data["source_x"].shape != (n_sources, 100):
        raise ValueError("Expected source and path coordinates in original PCA100.")
    if "marginal_windows" in data:
        real_populations = [
            (str(key), str(window), data[f"marginal_{i}_x"], data[f"marginal_{i}_tissues"])
            for i, (key, window) in enumerate(zip(data["marginal_metric_keys"], data["marginal_windows"]))
        ]
    else:
        real_populations = [("heldout_reference", None, data["reference_x"], data["reference_tissues"]),
                            ("target_test", None, data["target_test_x"], data["target_test_tissues"])]
    segments = [paths.reshape(-1, 100), data["source_x"], *[x for _, _, x, _ in real_populations]]
    print(f"Scoring {sum(len(segment) for segment in segments)} PCA100 points with the frozen tissue ensemble.", flush=True)
    probabilities = predict_selected(args.manifest, np.concatenate(segments), batch_size=args.batch_size)
    boundaries = np.cumsum([len(segment) for segment in segments])[:-1]
    path_flat, source_prob, *real_probs = np.split(probabilities, boundaries)
    path_prob = path_flat.reshape(n_sources, n_times, len(classes))
    lookup = {name: index for index, name in enumerate(classes)}
    source_ids = np.array([lookup[label] for label in data["source_tissues"]])
    raw = calderon.score_tissue_paths(source_ids, path_prob, allowed)
    filtered = calderon.score_tissue_paths(source_ids, path_prob, allowed, args.prob_threshold, args.margin_threshold)
    metrics = {
        "classifier_manifest": str(args.manifest.resolve()),
        "classifier_manifest_sha256": calderon.sha256_file(args.manifest),
        "classifier_selection_sha256": spec["selection_sha256"],
        "classifier_variant": spec["variant"],
        "classifier_role": "all_times_reference_oracle" if spec["variant"] == "all_times" else "saved_run_reference_classifier",
        "training_class_names": data["class_names"].tolist(), "evaluation_class_names": list(classes),
        "label_alignment": "Observed tissue names mapped to the evaluation classifier IDs",
        "allowed_pairs_sha256": calderon.sha256_file(args.allowed_pairs),
        "allowed_relation": "Projected branch self/descendant pairs, with no second tissue-graph closure",
        "start_labels": "Released tissue annotations of flow-test source cells",
        "checkpoint_times": data["checkpoint_times"].tolist(),
        "checkpoint_hours": data["checkpoint_hours"].tolist(),
        "raw": raw["metrics"], "confidence_filtered": filtered["metrics"],
        "prob_threshold": args.prob_threshold, "margin_threshold": args.margin_threshold,
        "real_cell_classifier": {
            "source_test": reference_accuracy(data["source_tissues"], source_prob, classes, args.prob_threshold),
            **{key: {"window": window, **reference_accuracy(labels, probs, classes, args.prob_threshold)}
               for (key, window, _, labels), probs in zip(real_populations, real_probs)},
        },
    }
    write_json(args.output_dir / "tissue_trajectory_metrics.json", metrics)
    summary = pd.DataFrame({
        "source_cell_id": data["source_cell_ids"], "source_tissue": data["source_tissues"],
        "raw_violation": raw["first_rejection"] >= 0,
        "raw_first_rejection_index": raw["first_rejection"],
        "confidence_filtered_status": filtered["status"],
        "confidence_filtered_first_rejection_index": filtered["first_rejection"],
        "n_confident_checkpoints": filtered["confident"].sum(axis=1),
    })
    for name, result in (("raw", raw), ("confidence_filtered", filtered)):
        first = result["first_rejection"]
        rejection_hours = np.full(n_sources, np.nan)
        rejected = first >= 0
        rejection_hours[rejected] = data["checkpoint_hours"][first[rejected]]
        summary[f"{name}_first_rejection_hours"] = rejection_hours
    summary.to_csv(args.output_dir / "trajectory_summary.csv.gz", index=False)
    predicted_names = np.asarray(classes)[raw["predicted_ids"]]
    point_table = pd.DataFrame({
        "source_cell_id": np.repeat(data["source_cell_ids"], n_times),
        "checkpoint_time": np.tile(data["checkpoint_times"], n_sources),
        "checkpoint_hours": np.tile(data["checkpoint_hours"], n_sources),
        "predicted_tissue": predicted_names.reshape(-1),
        "confidence": raw["confidence"].reshape(-1), "margin": raw["margin"].reshape(-1),
        "confident": filtered["confident"].reshape(-1),
    })
    point_table.to_csv(args.output_dir / "trajectory_predictions.csv.gz", index=False)
    checkpoint_table = pd.DataFrame({
        "time": data["checkpoint_times"], "hours": data["checkpoint_hours"],
        "raw_first_rejections": raw["metrics"]["first_rejection_counts"],
        "filtered_first_rejections": filtered["metrics"]["first_rejection_counts"],
        "confident_cells": filtered["metrics"]["confident_counts"],
        "confidence_coverage_pct": 100.0 * filtered["confident"].mean(axis=0),
    })
    checkpoint_table.to_csv(args.output_dir / "checkpoint_rejections.csv", index=False)
    by_tissue = []
    for tissue in sorted(set(data["source_tissues"])):
        selected = data["source_tissues"] == tissue
        raw_group = calderon.score_tissue_paths(source_ids[selected], path_prob[selected], allowed)["metrics"]
        filtered_group = calderon.score_tissue_paths(source_ids[selected], path_prob[selected], allowed,
                                                   args.prob_threshold, args.margin_threshold)["metrics"]
        by_tissue.append({"source_tissue": tissue, "n_sources": int(selected.sum()),
                          "raw_violation_pct": raw_group["violation_pct_all_sources"],
                          "filtered_violation_pct_assessed": filtered_group["violation_pct_assessed"],
                          "filtered_coverage_pct": filtered_group["trajectory_coverage_pct"],
                          "filtered_full_checkpoint_coverage_pct": filtered_group["full_checkpoint_coverage_pct"]})
    pd.DataFrame(by_tissue).to_csv(args.output_dir / "violations_by_source_tissue.csv", index=False)
    np.savez_compressed(args.output_dir / "trajectory_probabilities.npz", probabilities=path_prob,
                        class_names=np.asarray(classes), source_cell_ids=data["source_cell_ids"],
                        checkpoint_times=data["checkpoint_times"], raw_first_rejection=raw["first_rejection"],
                        filtered_first_rejection=filtered["first_rejection"], confident=filtered["confident"])
    print(f"Raw trajectory violations: {raw['metrics']['violation_pct_all_sources']:.2f}%", flush=True)
    print("Confidence-filtered violations/coverage: "
          f"{filtered['metrics']['violation_pct_assessed']}% / {filtered['metrics']['trajectory_coverage_pct']:.2f}%", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allowed-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Verify the frozen ensemble before training")
    parser.add_argument("--prob-threshold", type=float, default=0.9)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error("batch-size and threads must be positive")
    if args.check_only:
        import torch
        from scripts.train_calderon_expressive_classifiers import predict_selected
        torch.set_num_threads(args.threads)
        spec = calderon.load_tissue_manifest(args.manifest)
        calderon.load_allowed_pairs(args.allowed_pairs, spec["class_names"])
        probabilities = predict_selected(args.manifest, np.zeros((1, 100), dtype=np.float32))
        if probabilities.shape != (1, len(spec["class_names"])) or not np.isfinite(probabilities).all():
            raise ValueError("Frozen classifier failed inference preflight.")
        print("Frozen tissue ensemble and allowed-pair relation verified.")
        return
    if args.input is None or args.output_dir is None:
        parser.error("--input and --output-dir are required for trajectory evaluation")
    evaluate(args)


if __name__ == "__main__":
    main()

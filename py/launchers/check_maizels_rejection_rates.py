"""
Estimate Maizels D3 -> D8 biological-prior rejection rates.

This script uses the saved PCA50 CSV used for cell-type classifier training and
the trained PCA50 classifier checkpoint. It reports:

1. The exact endpoint/coupling rejection rate under independent D3/D8 coupling.
2. A Monte Carlo estimate of the additional rejection rate from checking
   straight-line interpolants with a confidence-gated classifier.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

import common.maizels as maizels_common

DEFAULT_DATASET = (
    "/Users/harryamad/Desktop/Maizels2023aa/data/"
    "celltype_classification_pca50_dataset.csv.gz"
)
DEFAULT_CLASSIFIER = (
    "/Users/harryamad/Desktop/Maizels2023aa/models/"
    "celltype_classifier_pca50.pt"
)

TRANSITION_EDGES = maizels_common.TRANSITION_EDGES


def build_reachable(edges: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Return reflexive transitive closure for the transition graph."""
    nodes = set()
    children = defaultdict(set)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
        children[src].add(dst)

    reachable: Dict[str, Set[str]] = {}
    for node in nodes:
        seen = {node}
        stack = list(children[node])
        while stack:
            curr = stack.pop()
            if curr in seen:
                continue
            seen.add(curr)
            stack.extend(children[curr])
        reachable[node] = seen
    return reachable


def load_endpoint_arrays(
    dataset_path: Path,
    source_time: str,
    target_time: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load PCA rows and annotations for the two endpoint timepoints."""
    source_x: List[List[float]] = []
    target_x: List[List[float]] = []
    source_types: List[str] = []
    target_types: List[str] = []
    pc_cols = [f"PC{ii}" for ii in range(1, 51)]

    with gzip.open(dataset_path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in ["timepoint", "cell_annotation", *pc_cols] if col not in reader.fieldnames]
        if missing:
            raise KeyError(f"Missing expected columns in {dataset_path}: {missing}")

        for row in reader:
            timepoint = row["timepoint"]
            if timepoint not in (source_time, target_time):
                continue
            cell_type = row["cell_annotation"]
            pcs = [float(row[col]) for col in pc_cols]
            if timepoint == source_time:
                source_x.append(pcs)
                source_types.append(cell_type)
            else:
                target_x.append(pcs)
                target_types.append(cell_type)

    if not source_x:
        raise RuntimeError(f"No rows found for source_time={source_time!r}.")
    if not target_x:
        raise RuntimeError(f"No rows found for target_time={target_time!r}.")

    return (
        np.asarray(source_x, dtype=np.float32),
        np.asarray(source_types, dtype=object),
        np.asarray(target_x, dtype=np.float32),
        np.asarray(target_types, dtype=object),
    )


def load_classifier(
    classifier_path: Path,
) -> Tuple[Any, List[str], np.ndarray, np.ndarray]:
    """Load the PCA50 classifier and scaler metadata."""
    return maizels_common.load_classifier(classifier_path)


def endpoint_valid(
    src_type: str,
    dst_type: str,
    reachable: Dict[str, Set[str]],
) -> bool:
    return dst_type in reachable.get(src_type, {src_type})


def exact_endpoint_counts(
    source_types: np.ndarray,
    target_types: np.ndarray,
    reachable: Dict[str, Set[str]],
) -> Tuple[int, int]:
    """Compute exact accepted endpoint pairs from type counts."""
    source_counts = Counter(source_types.tolist())
    target_counts = Counter(target_types.tolist())

    valid = 0
    total = int(len(source_types) * len(target_types))
    for src_type, src_count in source_counts.items():
        for dst_type, dst_count in target_counts.items():
            if endpoint_valid(src_type, dst_type, reachable):
                valid += int(src_count * dst_count)
    return valid, total


def classifier_predictions(
    model: Any,
    class_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    x: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top class names, top probabilities, and logit margins."""
    return maizels_common.classifier_predictions(
        model,
        class_names,
        scaler_mean,
        scaler_scale,
        x,
        batch_size=batch_size,
    )


def sample_pair_indices(
    n_source: int,
    n_target: int,
    n_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, n_source, size=n_pairs, endpoint=False),
        rng.integers(0, n_target, size=n_pairs, endpoint=False),
    )


def check_interpolants(
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    reachable: Dict[str, Set[str]],
    model: Any,
    class_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    n_check_times: int,
    prob_threshold: float,
    margin_threshold: float,
    classifier_batch_size: int,
) -> Dict[str, float]:
    """Check interpolants for sampled pairs and return counters/rates."""
    src_types = source_types[source_idx]
    dst_types = target_types[target_idx]
    endpoint_ok = np.asarray(
        [
            endpoint_valid(str(src), str(dst), reachable)
            for src, dst in zip(src_types, dst_types)
        ],
        dtype=bool,
    )
    valid_pair_positions = np.flatnonzero(endpoint_ok)

    stats = {
        "sampled_pairs": int(source_idx.shape[0]),
        "sampled_endpoint_valid": int(endpoint_ok.sum()),
        "sampled_endpoint_rejected": int((~endpoint_ok).sum()),
        "interpolant_checked": int(valid_pair_positions.shape[0]),
        "interpolant_rejected": 0,
        "interpolant_rejected_at_checkpoint": 0,
        "interpolant_rejected_at_final_target": 0,
        "accepted_after_both_filters": 0,
        "classifier_points_total": 0,
        "classifier_points_confident": 0,
        "classifier_points_low_confidence": 0,
    }

    if valid_pair_positions.shape[0] == 0:
        return stats

    taus = np.linspace(0.0, 1.0, n_check_times + 2, dtype=np.float32)[1:-1]
    src_x = source_x[source_idx[valid_pair_positions]]
    dst_x = target_x[target_idx[valid_pair_positions]]

    line_points = []
    for tau in taus:
        line_points.append((1.0 - tau) * src_x + tau * dst_x)
    line_points_arr = np.concatenate(line_points, axis=0)

    pred_types_flat, top_probs_flat, margins_flat = classifier_predictions(
        model,
        class_names,
        scaler_mean,
        scaler_scale,
        line_points_arr,
        batch_size=classifier_batch_size,
    )

    n_valid = valid_pair_positions.shape[0]
    pred_types = pred_types_flat.reshape(n_check_times, n_valid).T
    top_probs = top_probs_flat.reshape(n_check_times, n_valid).T
    margins = margins_flat.reshape(n_check_times, n_valid).T
    confident = (top_probs >= prob_threshold) & (margins >= margin_threshold)

    stats["classifier_points_total"] = int(confident.size)
    stats["classifier_points_confident"] = int(confident.sum())
    stats["classifier_points_low_confidence"] = int((~confident).sum())

    rejected = np.zeros(n_valid, dtype=bool)
    rejected_at_checkpoint = np.zeros(n_valid, dtype=bool)
    rejected_at_final = np.zeros(n_valid, dtype=bool)

    current_types = src_types[valid_pair_positions].astype(object).copy()
    endpoint_target_types = dst_types[valid_pair_positions]

    for ii in range(n_valid):
        curr = str(current_types[ii])
        for jj in range(n_check_times):
            if not confident[ii, jj]:
                continue
            pred = str(pred_types[ii, jj])
            if not endpoint_valid(curr, pred, reachable):
                rejected[ii] = True
                rejected_at_checkpoint[ii] = True
                break
            curr = pred

        if rejected[ii]:
            continue
        final_target = str(endpoint_target_types[ii])
        if not endpoint_valid(curr, final_target, reachable):
            rejected[ii] = True
            rejected_at_final[ii] = True

    stats["interpolant_rejected"] = int(rejected.sum())
    stats["interpolant_rejected_at_checkpoint"] = int(rejected_at_checkpoint.sum())
    stats["interpolant_rejected_at_final_target"] = int(rejected_at_final.sum())
    stats["accepted_after_both_filters"] = int((~rejected).sum())
    return stats


def print_reachable(reachable: Dict[str, Set[str]]) -> None:
    print("Reachable cell types:")
    for src in sorted(reachable):
        print(f"  {src}: {', '.join(sorted(reachable[src]))}")


def pct(num: float, den: float) -> float:
    if den == 0:
        return float("nan")
    return 100.0 * float(num) / float(den)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Maizels D3 -> D8 coupling/interpolant rejection rates."
    )
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--classifier", type=Path, default=Path(DEFAULT_CLASSIFIER))
    parser.add_argument("--source_time", default="D3")
    parser.add_argument("--target_time", default="D8")
    parser.add_argument("--num_pairs", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_check_times", type=int, default=5)
    parser.add_argument("--prob_threshold", type=float, default=0.85)
    parser.add_argument("--margin_threshold", type=float, default=1.0)
    parser.add_argument("--classifier_batch_size", type=int, default=8192)
    args = parser.parse_args()

    reachable = build_reachable(TRANSITION_EDGES)
    source_x, source_types, target_x, target_types = load_endpoint_arrays(
        args.dataset,
        args.source_time,
        args.target_time,
    )
    model, class_names, scaler_mean, scaler_scale = load_classifier(args.classifier)

    missing_types = (
        set(source_types.tolist()) | set(target_types.tolist())
    ) - set(reachable.keys())
    missing_classes = set(class_names) - set(reachable.keys())
    if missing_types:
        raise KeyError(f"Endpoint cell types missing from transition graph: {sorted(missing_types)}")
    if missing_classes:
        raise KeyError(f"Classifier classes missing from transition graph: {sorted(missing_classes)}")

    source_counts = Counter(source_types.tolist())
    target_counts = Counter(target_types.tolist())
    exact_endpoint_valid, exact_endpoint_total = exact_endpoint_counts(
        source_types,
        target_types,
        reachable,
    )
    exact_endpoint_rejected = exact_endpoint_total - exact_endpoint_valid

    source_idx, target_idx = sample_pair_indices(
        source_x.shape[0],
        target_x.shape[0],
        args.num_pairs,
        args.seed,
    )
    stats = check_interpolants(
        source_x=source_x,
        source_types=source_types,
        target_x=target_x,
        target_types=target_types,
        source_idx=source_idx,
        target_idx=target_idx,
        reachable=reachable,
        model=model,
        class_names=class_names,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        n_check_times=args.n_check_times,
        prob_threshold=args.prob_threshold,
        margin_threshold=args.margin_threshold,
        classifier_batch_size=args.classifier_batch_size,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Classifier: {args.classifier}")
    print(f"Source time: {args.source_time} (n={source_x.shape[0]})")
    print(f"Target time: {args.target_time} (n={target_x.shape[0]})")
    print(f"Source counts: {dict(sorted(source_counts.items()))}")
    print(f"Target counts: {dict(sorted(target_counts.items()))}")
    print_reachable(reachable)
    print()
    print("Endpoint/coupling filter, exact over all independent source-target pairs:")
    print(f"  total pairs: {exact_endpoint_total}")
    print(
        f"  accepted: {exact_endpoint_valid} "
        f"({pct(exact_endpoint_valid, exact_endpoint_total):.2f}%)"
    )
    print(
        f"  rejected: {exact_endpoint_rejected} "
        f"({pct(exact_endpoint_rejected, exact_endpoint_total):.2f}%)"
    )
    print()
    print("Interpolant filter, Monte Carlo over sampled independent pairs:")
    print(f"  sampled pairs: {stats['sampled_pairs']}")
    print(
        f"  endpoint rejected in sample: {stats['sampled_endpoint_rejected']} "
        f"({pct(stats['sampled_endpoint_rejected'], stats['sampled_pairs']):.2f}%)"
    )
    print(f"  endpoint-valid interpolants checked: {stats['interpolant_checked']}")
    print(
        f"  interpolant rejected among endpoint-valid: {stats['interpolant_rejected']} "
        f"({pct(stats['interpolant_rejected'], stats['interpolant_checked']):.2f}%)"
    )
    print(
        "    at classifier checkpoint: "
        f"{stats['interpolant_rejected_at_checkpoint']} "
        f"({pct(stats['interpolant_rejected_at_checkpoint'], stats['interpolant_checked']):.2f}%)"
    )
    print(
        "    at final target after confident intermediate state: "
        f"{stats['interpolant_rejected_at_final_target']} "
        f"({pct(stats['interpolant_rejected_at_final_target'], stats['interpolant_checked']):.2f}%)"
    )
    print(
        f"  accepted after both filters in sample: {stats['accepted_after_both_filters']} "
        f"({pct(stats['accepted_after_both_filters'], stats['sampled_pairs']):.2f}% of sampled pairs)"
    )
    total_rejected = stats["sampled_endpoint_rejected"] + stats["interpolant_rejected"]
    print(
        f"  total rejected by endpoint or interpolant filters: {total_rejected} "
        f"({pct(total_rejected, stats['sampled_pairs']):.2f}% of sampled pairs)"
    )
    print()
    print("Classifier confidence gate:")
    print(f"  check times: {args.n_check_times}")
    print(f"  prob_threshold: {args.prob_threshold}")
    print(f"  margin_threshold: {args.margin_threshold}")
    print(f"  classifier points: {stats['classifier_points_total']}")
    print(
        f"  confident points: {stats['classifier_points_confident']} "
        f"({pct(stats['classifier_points_confident'], stats['classifier_points_total']):.2f}%)"
    )
    print(
        f"  low-confidence ignored points: {stats['classifier_points_low_confidence']} "
        f"({pct(stats['classifier_points_low_confidence'], stats['classifier_points_total']):.2f}%)"
    )


if __name__ == "__main__":
    main()

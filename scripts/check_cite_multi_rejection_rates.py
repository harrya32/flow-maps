#!/usr/bin/env python3
"""Measure CITE/Multi coupling and interpolant rejection rates.

This applies the same mechanics and defaults used by the Maizels experiment:

* source and target are the first/last observed days (2 and 7 here),
* annotated endpoint pairs are checked with descendant reachability,
* 50 equally spaced interior points of each straight-line interpolant are
  classified, and
* probability and logit-margin thresholds are both zero, so every prediction
  participates in the path check.

The hematopoietic lineage is deliberately conservative: HSC may remain HSC or
differentiate into any observed committed progenitor; each committed lineage
may only remain itself.  No cross-lineage conversion or dedifferentiation is
allowed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set, Tuple

import anndata as ad
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import maizels as maizels_common  # noqa: E402


DATASET_FILES = {
    "cite": "op_cite_inputs_0.h5ad",
    "multi": "op_train_multi_targets_0.h5ad",
}
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "CITE_MULTI_DATA_DIR",
        str(Path.home() / "Desktop" / "flow-maps-data"),
    )
).expanduser()
CELL_TYPES = ("HSC", "EryP", "NeuP", "MasP", "MkP", "BP", "MoP")
TRANSITION_EDGES = tuple(("HSC", cell_type) for cell_type in CELL_TYPES if cell_type != "HSC")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_FILES),
        default=list(DATASET_FILES),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument("--cite-path", type=Path, default=None)
    parser.add_argument("--multi-path", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-day", default="2")
    parser.add_argument("--target-day", default="7")
    parser.add_argument("--pca-key", default="X_pca")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--day-key", default="day")
    parser.add_argument("--num-pairs", type=int, default=100_000)
    parser.add_argument("--target-accepted-pairs", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-check-times", type=int, default=50)
    parser.add_argument("--prob-threshold", type=float, default=0.0)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--classifier-batch-size", type=int, default=8192)
    parser.add_argument("--pair-chunk-size", type=int, default=2048)
    parser.add_argument(
        "--transition-mode",
        choices=("descendant", "direct"),
        default="descendant",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT
        / "outputs"
        / "celltype_classification"
        / "cite_multi_rejection_rates.json",
    )
    args = parser.parse_args(argv)
    for name in (
        "num_pairs",
        "target_accepted_pairs",
        "n_check_times",
        "classifier_batch_size",
        "pair_chunk_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def build_reachable(mode: str) -> Dict[str, Set[str]]:
    reachable = {cell_type: {cell_type} for cell_type in CELL_TYPES}
    for source, target in TRANSITION_EDGES:
        reachable[source].add(target)
    if mode == "direct":
        return reachable

    for source in CELL_TYPES:
        stack = list(reachable[source])
        while stack:
            current = stack.pop()
            for target in reachable[current]:
                if target not in reachable[source]:
                    reachable[source].add(target)
                    stack.append(target)
    return reachable


def endpoint_valid(source: str, target: str, reachable: Mapping[str, Set[str]]) -> bool:
    return target in reachable.get(source, {source})


def dataset_path(args: argparse.Namespace, dataset: str) -> Path:
    explicit = getattr(args, f"{dataset}_path")
    path = explicit if explicit is not None else args.data_dir / DATASET_FILES[dataset]
    return path.expanduser().resolve()


def classifier_path(args: argparse.Namespace, dataset: str, n_pcs: int = 100) -> Path:
    return (args.model_dir / f"celltype_classifier_{dataset}_pca{n_pcs}.pt").expanduser().resolve()


def load_endpoints(
    path: Path,
    source_day: str,
    target_day: str,
    pca_key: str,
    label_key: str,
    day_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path, backed="r")
    try:
        if pca_key not in adata.obsm:
            raise KeyError(f"{path.name}: missing obsm[{pca_key!r}]")
        for key in (label_key, day_key):
            if key not in adata.obs:
                raise KeyError(f"{path.name}: missing obs[{key!r}]")
        days = adata.obs[day_key].astype("string").astype(str).to_numpy()
        labels = adata.obs[label_key].astype("string").astype(str).to_numpy()
        source_mask = days == str(source_day)
        target_mask = days == str(target_day)
        if not source_mask.any() or not target_mask.any():
            raise ValueError(
                f"{path.name}: could not find both day {source_day} and day {target_day}"
            )
        x = np.asarray(adata.obsm[pca_key], dtype=np.float32)
        return (
            x[source_mask],
            labels[source_mask],
            x[target_mask],
            labels[target_mask],
            int(x.shape[1]),
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def exact_endpoint_stats(
    source_types: np.ndarray,
    target_types: np.ndarray,
    reachable: Mapping[str, Set[str]],
) -> Dict[str, Any]:
    source_counts = Counter(str(value) for value in source_types)
    target_counts = Counter(str(value) for value in target_types)
    valid = 0
    rejected_by_pair: Dict[str, int] = {}
    for source, source_count in source_counts.items():
        for target, target_count in target_counts.items():
            count = int(source_count * target_count)
            if endpoint_valid(source, target, reachable):
                valid += count
            else:
                rejected_by_pair[f"{source}->{target}"] = count
    total = int(len(source_types) * len(target_types))
    return {
        "source_counts": dict(sorted(source_counts.items())),
        "target_counts": dict(sorted(target_counts.items())),
        "total": total,
        "accepted": int(valid),
        "rejected": int(total - valid),
        "accepted_rate": float(valid / total),
        "rejected_rate": float(1.0 - valid / total),
        "rejected_by_type_pair": dict(
            sorted(rejected_by_pair.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def sampled_interpolant_stats(
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    model: Any,
    class_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    reachable: Mapping[str, Set[str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    source_idx = rng.integers(0, len(source_x), size=args.num_pairs, endpoint=False)
    target_idx = rng.integers(0, len(target_x), size=args.num_pairs, endpoint=False)
    sampled_source_types = source_types[source_idx]
    sampled_target_types = target_types[target_idx]
    endpoint_ok = np.fromiter(
        (
            endpoint_valid(str(source), str(target), reachable)
            for source, target in zip(sampled_source_types, sampled_target_types)
        ),
        dtype=bool,
        count=args.num_pairs,
    )
    valid_positions = np.flatnonzero(endpoint_ok)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    missing = (set(source_types) | set(target_types)) - set(class_to_id)
    if missing:
        raise KeyError(f"Classifier is missing annotated cell types: {sorted(missing)}")

    stats: Dict[str, Any] = {
        "sampled_pairs": int(args.num_pairs),
        "sampled_endpoint_accepted": int(endpoint_ok.sum()),
        "sampled_endpoint_rejected": int((~endpoint_ok).sum()),
        "interpolants_checked": int(len(valid_positions)),
        "interpolants_rejected": 0,
        "rejected_at_checkpoint": 0,
        "rejected_at_final_target": 0,
        "accepted_after_both_filters": 0,
        "classifier_points_total": 0,
        "classifier_points_confident": 0,
    }
    pair_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    taus = np.linspace(
        0.0, 1.0, args.n_check_times + 2, dtype=np.float32
    )[1:-1]

    for start in range(0, len(valid_positions), args.pair_chunk_size):
        positions = valid_positions[start : start + args.pair_chunk_size]
        source_rows = source_idx[positions]
        target_rows = target_idx[positions]
        paths = (
            (1.0 - taus[None, :, None]) * source_x[source_rows, None, :]
            + taus[None, :, None] * target_x[target_rows, None, :]
        ).astype(np.float32, copy=False)
        flat_types, flat_probs, flat_margins = maizels_common.classifier_predictions(
            model,
            class_names,
            scaler_mean,
            scaler_scale,
            paths.reshape(-1, paths.shape[-1]),
            batch_size=args.classifier_batch_size,
        )
        shape = (len(positions), args.n_check_times)
        start_type_ids = np.asarray(
            [class_to_id[str(value)] for value in sampled_source_types[positions]],
            dtype=np.int32,
        )
        target_type_ids = np.asarray(
            [class_to_id[str(value)] for value in sampled_target_types[positions]],
            dtype=np.int32,
        )
        validity = maizels_common.path_validity_from_predictions(
            start_type_ids=start_type_ids,
            pred_types=flat_types.reshape(shape),
            top_probs=flat_probs.reshape(shape),
            margins=flat_margins.reshape(shape),
            class_names=class_names,
            reachable=dict(reachable),
            prob_threshold=args.prob_threshold,
            margin_threshold=args.margin_threshold,
            final_type_ids=target_type_ids,
        )
        valid = np.asarray(validity["valid"], dtype=bool)
        rejected_checkpoint = np.asarray(
            validity["rejected_at_checkpoint"], dtype=bool
        )
        rejected_final = np.asarray(validity["rejected_at_final"], dtype=bool)
        stats["interpolants_rejected"] += int((~valid).sum())
        stats["rejected_at_checkpoint"] += int(rejected_checkpoint.sum())
        stats["rejected_at_final_target"] += int(rejected_final.sum())
        stats["accepted_after_both_filters"] += int(valid.sum())
        stats["classifier_points_total"] += int(validity["n_points"])
        stats["classifier_points_confident"] += int(validity["n_confident"])

        for source, target, is_valid in zip(
            sampled_source_types[positions], sampled_target_types[positions], valid
        ):
            key = f"{source}->{target}"
            pair_counts[key]["checked"] += 1
            pair_counts[key]["rejected"] += int(not is_valid)

    stats["sampled_endpoint_rejected_rate"] = float((~endpoint_ok).mean())
    stats["interpolant_rejected_rate_given_endpoint_valid"] = float(
        stats["interpolants_rejected"] / max(stats["interpolants_checked"], 1)
    )
    stats["total_rejected"] = int(
        stats["sampled_endpoint_rejected"] + stats["interpolants_rejected"]
    )
    stats["total_rejected_rate"] = float(stats["total_rejected"] / args.num_pairs)
    stats["accepted_rate"] = float(
        stats["accepted_after_both_filters"] / args.num_pairs
    )
    stats["classifier_points_confident_rate"] = float(
        stats["classifier_points_confident"]
        / max(stats["classifier_points_total"], 1)
    )
    stats["by_endpoint_type_pair"] = {
        key: {
            "checked": int(counts["checked"]),
            "rejected": int(counts["rejected"]),
            "rejected_rate": float(counts["rejected"] / counts["checked"]),
        }
        for key, counts in sorted(pair_counts.items())
    }
    return stats


def projected_rejection_stats(
    sampled: Mapping[str, Any], target_accepted_pairs: int
) -> Dict[str, Any]:
    acceptance_rate = float(sampled["accepted_rate"])
    if acceptance_rate <= 0.0:
        return {"target_accepted_pairs": target_accepted_pairs, "feasible": False}
    candidates = int(math.ceil(target_accepted_pairs / acceptance_rate))
    endpoint_rate = float(sampled["sampled_endpoint_rejected_rate"])
    interpolant_rate = float(sampled["interpolants_rejected"] / sampled["sampled_pairs"])
    endpoint_rejected = int(round(candidates * endpoint_rate))
    interpolant_rejected = int(round(candidates * interpolant_rate))
    return {
        "target_accepted_pairs": int(target_accepted_pairs),
        "estimated_candidates_needed": candidates,
        "estimated_endpoint_rejected": endpoint_rejected,
        "estimated_interpolant_rejected": interpolant_rejected,
        "estimated_total_rejected": int(candidates - target_accepted_pairs),
        "feasible": True,
    }


def percent(value: int | float, total: int | float) -> float:
    return 100.0 * float(value) / float(total) if total else float("nan")


def inspect_dataset(args: argparse.Namespace, dataset: str) -> Dict[str, Any]:
    path = dataset_path(args, dataset)
    source_x, source_types, target_x, target_types, n_pcs = load_endpoints(
        path,
        args.source_day,
        args.target_day,
        args.pca_key,
        args.label_key,
        args.day_key,
    )
    checkpoint = classifier_path(args, dataset, n_pcs=n_pcs)
    model, class_names, scaler_mean, scaler_scale = maizels_common.load_classifier(
        checkpoint
    )
    if len(scaler_mean) != n_pcs:
        raise ValueError(
            f"{checkpoint.name} expects {len(scaler_mean)} PCs, but {path.name} "
            f"contains {n_pcs}"
        )
    reachable = build_reachable(args.transition_mode)
    unknown = (set(source_types) | set(target_types) | set(class_names)) - set(
        reachable
    )
    if unknown:
        raise KeyError(f"Cell types missing from transition graph: {sorted(unknown)}")

    exact = exact_endpoint_stats(source_types, target_types, reachable)
    sampled = sampled_interpolant_stats(
        source_x,
        source_types,
        target_x,
        target_types,
        model,
        class_names,
        scaler_mean,
        scaler_scale,
        reachable,
        args,
    )
    projected = projected_rejection_stats(sampled, args.target_accepted_pairs)
    result = {
        "dataset": dataset,
        "dataset_path": str(path),
        "classifier_path": str(checkpoint),
        "source_day": str(args.source_day),
        "target_day": str(args.target_day),
        "source_n": int(len(source_x)),
        "target_n": int(len(target_x)),
        "n_pcs": n_pcs,
        "transition_mode": args.transition_mode,
        "transition_edges": [list(edge) for edge in TRANSITION_EDGES],
        "n_check_times": int(args.n_check_times),
        "prob_threshold": float(args.prob_threshold),
        "margin_threshold": float(args.margin_threshold),
        "seed": int(args.seed),
        "exact_endpoint_filter": exact,
        "sampled_interpolant_filter": sampled,
        "projected_training_pool": projected,
    }

    print(f"\n{dataset.upper()}: day {args.source_day} -> day {args.target_day}")
    print(f"  endpoints: {len(source_x):,} source x {len(target_x):,} target")
    print(
        f"  exact endpoint rejection: {exact['rejected']:,}/{exact['total']:,} "
        f"({percent(exact['rejected'], exact['total']):.2f}%)"
    )
    print(
        f"  sampled endpoint rejection: {sampled['sampled_endpoint_rejected']:,}/"
        f"{sampled['sampled_pairs']:,} "
        f"({percent(sampled['sampled_endpoint_rejected'], sampled['sampled_pairs']):.2f}%)"
    )
    print(
        f"  interpolant rejection among endpoint-valid pairs: "
        f"{sampled['interpolants_rejected']:,}/{sampled['interpolants_checked']:,} "
        f"({percent(sampled['interpolants_rejected'], sampled['interpolants_checked']):.2f}%)"
    )
    print(
        f"    checkpoint violation: {sampled['rejected_at_checkpoint']:,}; "
        f"final-target violation: {sampled['rejected_at_final_target']:,}"
    )
    print(
        f"  total sampled rejection: {sampled['total_rejected']:,}/"
        f"{sampled['sampled_pairs']:,} "
        f"({percent(sampled['total_rejected'], sampled['sampled_pairs']):.2f}%)"
    )
    print(
        f"  accepted after both filters: {sampled['accepted_after_both_filters']:,}/"
        f"{sampled['sampled_pairs']:,} ({100.0 * sampled['accepted_rate']:.2f}%)"
    )
    if projected["feasible"]:
        print(
            f"  projected for {args.target_accepted_pairs:,} accepted training pairs: "
            f"draw ~{projected['estimated_candidates_needed']:,}, reject "
            f"~{projected['estimated_total_rejected']:,}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = [inspect_dataset(args, dataset) for dataset in args.datasets]
        if args.output_json is not None:
            output = args.output_json.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(results, indent=2) + "\n")
            os.replace(temporary, output)
            print(f"\nSaved {output}")
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

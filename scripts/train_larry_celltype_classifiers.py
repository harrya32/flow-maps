#!/usr/bin/env python3
"""Train the two frozen classifiers used by each LARRY flow representation.

``all_days`` uses D2, D4, and D6 and is reserved for evaluation. The
``train_days_d2_d6`` model excludes held-out D4 and is used for candidate-pair
filtering and differentiable lineage constraints during flow training.

Both ``.pt`` and runtime-ready ``.npz`` checkpoints, validation reports, and
loss curves are stored under ``larry-classifiers`` by default. PCA50 remains
the default; ``--representation spring2d`` trains separate two-dimensional
checkpoints on the authors' supplied layout.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
for path in (REPO_ROOT, PY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import larry  # noqa: E402
from scripts import train_cite_multi_celltype_classifiers as shared  # noqa: E402


CLASSIFIER_VARIANTS: Mapping[str, shared.ClassifierVariant] = {
    "all_days": shared.ClassifierVariant(
        key="all_days",
        excluded_day=None,
        usage="flow_evaluation_only",
    ),
    "train_days_d2_d6": shared.ClassifierVariant(
        key="train_days_d2_d6",
        excluded_day=None,
        usage="flow_training_for_d2_d6",
        included_days=("D2", "D6"),
    ),
}
DEFAULT_VARIANTS = tuple(CLASSIFIER_VARIANTS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train all-days and D2/D6-only cell-type classifiers on the "
            "selected cached LARRY representation."
        )
    )
    parser.add_argument(
        "--dataset-location",
        type=Path,
        default=larry.DEFAULT_DATA_DIR,
        help=(
            "Directory containing the canonical LARRY representation cache, "
            "or an explicit precomputed .h5ad path."
        ),
    )
    parser.add_argument(
        "--representation",
        choices=larry.REPRESENTATIONS,
        default=larry.PCA50_REPRESENTATION,
        help="State-space representation used by the flow experiment.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(CLASSIFIER_VARIANTS),
        default=list(DEFAULT_VARIANTS),
        help="Classifier variants to train (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=larry.DEFAULT_CLASSIFIER_DIR,
        help="Checkpoint directory (default: repository/larry-classifiers).",
    )
    parser.add_argument("--n-hvgs", type=int, default=larry.DEFAULT_N_HVGS)
    parser.add_argument("--n-pcs", type=int, default=larry.DEFAULT_N_PCS)
    parser.add_argument(
        "--clone-labelled-only",
        "--clone-labeled-only",
        action="store_true",
        help=(
            "Use the separately cached PCA representation fitted only on "
            "clone-labelled cells and write separately named checkpoints."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=0)
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        default=(128, 64),
        metavar=("H1", "H2"),
    )
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="CPU is the safe default; choose CUDA explicitly when available.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested checkpoints, including incomplete pairs.",
    )
    args = parser.parse_args(argv)
    args.representation = larry.canonical_representation(args.representation)
    if args.clone_labelled_only and args.representation != larry.PCA50_REPRESENTATION:
        parser.error("--clone-labelled-only is supported only for PCA data")
    if args.representation == larry.SPRING2D_REPRESENTATION:
        # The shared trainer uses n_pcs as its generic input-dimension field.
        args.n_pcs = larry.DEFAULT_SPRING_DIM

    for name in (
        "n_hvgs",
        "n_pcs",
        "batch_size",
        "max_epochs",
        "patience",
        "log_every",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.min_delta < 0:
        parser.error("--min-delta must be non-negative")
    if args.lr_patience < 0:
        parser.error("--lr-patience must be non-negative")
    if not 0.0 < args.lr_factor < 1.0:
        parser.error("--lr-factor must be in (0, 1)")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")
    if not 0.0 <= args.class_weight_power <= 1.0:
        parser.error("--class-weight-power must be in [0, 1]")
    if tuple(args.hidden_dims) != (128, 64):
        parser.error("LARRY flow checkpoints currently require --hidden-dims 128 64")
    return args


def checkpoint_path(
    output_dir: Path,
    variant: str,
    *,
    n_pcs: int = larry.DEFAULT_N_PCS,
    n_hvgs: int = larry.DEFAULT_N_HVGS,
    representation: str = larry.PCA50_REPRESENTATION,
    clone_labelled_only: bool = False,
) -> Path:
    if variant not in CLASSIFIER_VARIANTS:
        raise KeyError(f"Unknown LARRY classifier variant {variant!r}.")
    if variant == "all_days":
        return larry.classifier_checkpoint_path(
            all_days=True,
            classifier_dir=output_dir,
            n_pcs=n_pcs,
            n_hvgs=n_hvgs,
            representation=representation,
            clone_labelled_only=clone_labelled_only,
        )
    return larry.classifier_checkpoint_path(
        training_timepoints=("D2", "D6"),
        classifier_dir=output_dir,
        n_pcs=n_pcs,
        n_hvgs=n_hvgs,
        representation=representation,
        clone_labelled_only=clone_labelled_only,
    )


def requested_runs(
    args: argparse.Namespace,
) -> List[Tuple[shared.ClassifierVariant, Path]]:
    output_dir = args.output_dir.expanduser().resolve()
    return [
        (
            CLASSIFIER_VARIANTS[name],
            checkpoint_path(
                output_dir,
                name,
                n_pcs=args.n_pcs,
                n_hvgs=args.n_hvgs,
                representation=args.representation,
                clone_labelled_only=bool(args.clone_labelled_only),
            ),
        )
        for name in args.variants
    ]


def checkpoint_pair_state(pt_path: Path) -> str:
    present = (pt_path.is_file(), pt_path.with_suffix(".npz").is_file())
    if all(present):
        return "complete"
    if any(present):
        return "partial"
    return "missing"


def load_training_data(
    args: argparse.Namespace,
    input_path: Path,
) -> Dict[str, Any]:
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "LARRY classifier training requires working anndata, NumPy, and "
            "pandas installations."
        ) from exc

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        representation_name = larry.canonical_representation(args.representation)
        representation_key = larry.representation_key(representation_name)
        expected_dim = (
            int(args.n_pcs)
            if representation_name == larry.PCA50_REPRESENTATION
            else larry.representation_dim(representation_name)
        )
        if representation_key not in adata.obsm:
            raise KeyError(
                f"{input_path.name} is missing adata.obsm[{representation_key!r}]."
            )
        representation = adata.obsm[representation_key]
        stored_n_pcs = int(representation.shape[1])
        if representation_name == larry.PCA50_REPRESENTATION and int(
            adata.n_vars
        ) != int(args.n_hvgs):
            raise ValueError(
                f"{input_path.name} stores {adata.n_vars} HVGs; {args.n_hvgs} "
                "were requested."
            )
        if stored_n_pcs != expected_dim:
            raise ValueError(
                f"{input_path.name} stores {stored_n_pcs} {representation_name} "
                f"dimensions; expected {expected_dim}."
            )
        label_key = "cell_type" if "cell_type" in adata.obs else "Cell type annotation"
        day_key = "day_label" if "day_label" in adata.obs else "Time point"
        if label_key not in adata.obs or day_key not in adata.obs:
            raise KeyError(
                f"{input_path.name} does not contain LARRY cell-type/day metadata."
            )
        if args.clone_labelled_only:
            if "clone_id" not in adata.obs:
                raise KeyError(
                    f"{input_path.name} is missing obs['clone_id']; it is not a "
                    "valid clone-labelled-only cache."
                )
            clone_values = np.asarray(adata.obs["clone_id"])
            missing_clone = np.asarray(
                [
                    value is None
                    or str(value).strip().lower()
                    in {"", "nan", "none", "<na>", "-1", "-1.0"}
                    for value in clone_values
                ],
                dtype=bool,
            )
            if bool(missing_clone.any()):
                raise ValueError(
                    f"{input_path.name} contains {int(missing_clone.sum())} cells "
                    "without clone assignments; rebuild it with "
                    "--clone-labelled-only."
                )

        labels = adata.obs[label_key].astype("string").str.strip()
        if bool(labels.isna().any()) or bool(labels.eq("").fillna(True).any()):
            raise ValueError(f"{input_path.name} contains missing cell-type labels.")
        labels = labels.astype(str).to_numpy()
        unknown_types = sorted(set(labels) - set(larry.CLASS_NAMES))
        if unknown_types:
            raise ValueError(f"Unexpected LARRY cell types: {unknown_types}.")
        missing_types = sorted(set(larry.CLASS_NAMES) - set(labels))
        if missing_types:
            raise ValueError(
                f"LARRY classifier data is missing classes: {missing_types}."
            )

        raw_days = adata.obs[day_key]
        if bool(raw_days.isna().any()):
            raise ValueError(f"{input_path.name} contains missing observation days.")
        days = np.asarray(
            [larry.format_timepoint(value) for value in raw_days.astype(str)],
            dtype=object,
        )
        unknown_days = sorted(set(days) - set(larry.TIMEPOINTS))
        if unknown_days:
            raise ValueError(f"Unexpected LARRY observation days: {unknown_days}.")

        x = np.asarray(representation[:, :expected_dim], dtype=np.float32).copy()
        if not bool(np.isfinite(x).all()):
            raise ValueError(
                f"{input_path.name} contains non-finite {representation_name} "
                "coordinates."
            )
        return {
            "x": x,
            "labels": labels,
            "cell_ids": np.asarray(adata.obs_names.astype(str), dtype=object),
            "days": days,
            "feature_names": (
                ["SPRING-x", "SPRING-y"]
                if representation_name == larry.SPRING2D_REPRESENTATION
                else [f"PC{index + 1}" for index in range(expected_dim)]
            ),
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "stored_n_pcs": stored_n_pcs,
        }
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = Path(tempfile.gettempdir()) / "flow_maps_numba_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))

    try:
        runs = requested_runs(args)
        pending: List[Tuple[shared.ClassifierVariant, Path]] = []
        for variant, pt_path in runs:
            state = checkpoint_pair_state(pt_path)
            if state == "complete" and not args.overwrite:
                print(f"[{variant.key}] Reusing cached classifier {pt_path}")
                continue
            if state == "partial" and not args.overwrite:
                raise RuntimeError(
                    f"Incomplete classifier checkpoint pair at {pt_path}; rerun "
                    "with --overwrite to rebuild both .pt and .npz files."
                )
            pending.append((variant, pt_path))
        if not pending:
            print("All requested LARRY classifier checkpoints already exist.")
            return 0

        dependencies = shared.import_dependencies()
        device = shared.choose_device(dependencies["torch"], args.device)
        input_path = larry.resolve_dataset_path(
            args.dataset_location,
            n_pcs=args.n_pcs,
            n_hvgs=args.n_hvgs,
            representation=args.representation,
            clone_labelled_only=bool(args.clone_labelled_only),
        )
        print(f"[larry] Reading {input_path}", flush=True)
        data = load_training_data(args, input_path)

        args.output_root = REPO_ROOT
        args.pca_key = larry.representation_key(args.representation)
        args.label_key = "cell_type"
        args.day_key = "day_label"
        metrics_by_variant: Dict[str, Dict[str, Any]] = {}
        for variant, pt_path in pending:
            run_args = copy.copy(args)
            run_args.classifier_output_dir = pt_path.parent
            run_args.checkpoint_stem_override = pt_path.stem
            metrics_by_variant[variant.key] = shared.train_one_variant(
                run_args,
                "larry",
                variant,
                input_path,
                data,
                dependencies,
                device,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nSummary")
    for variant, metrics in metrics_by_variant.items():
        print(
            f"  larry/{variant}: val_loss={metrics['validation_loss']:.4f}, "
            f"accuracy={metrics['validation_accuracy']:.4f}, "
            f"balanced_accuracy={metrics['validation_balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['validation_macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

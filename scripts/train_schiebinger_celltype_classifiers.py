#!/usr/bin/env python3
"""Train cached Schiebinger classifiers for flow training and evaluation.

The all-days classifier is reserved for evaluation.  A selected-days model is
trained only on the exact observation days supplied with ``--train-times`` and
is used by endpoint-interpolant filtering and lineage constraint losses.
Existing complete ``.pt``/``.npz`` checkpoint pairs are skipped by default.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
for path in (REPO_ROOT, PY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import schiebinger  # noqa: E402
from scripts import train_cite_multi_celltype_classifiers as shared  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the all-days evaluation classifier and/or one classifier "
            "for an exact Schiebinger flow-training schedule."
        )
    )
    parser.add_argument(
        "--dataset-location",
        type=Path,
        default=schiebinger.DEFAULT_DATA_DIR,
        help=(
            "Directory containing the canonical HVG-PCA H5AD, or an explicit "
            "precomputed .h5ad path."
        ),
    )
    parser.add_argument(
        "--all-days",
        action="store_true",
        help="Train the all-days classifier reserved for evaluation.",
    )
    parser.add_argument(
        "--train-times",
        type=str,
        default=None,
        help="Comma-separated observation days for a flow-training classifier.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=schiebinger.DEFAULT_CLASSIFIER_DIR,
        help="Checkpoint directory (default: repository/schiebinger_classifiers).",
    )
    parser.add_argument(
        "--all-days-checkpoint",
        type=Path,
        default=None,
        help="Optional exact .pt destination for the all-days model.",
    )
    parser.add_argument(
        "--training-checkpoint",
        type=Path,
        default=None,
        help="Optional exact .pt destination for the selected-days model.",
    )
    parser.add_argument("--n-pcs", type=int, default=5)
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
        "--hidden-dims", type=int, nargs=2, default=(128, 64), metavar=("H1", "H2")
    )
    parser.add_argument("--class-weight-power", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="CPU is the safe default when launched immediately before JAX training.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace requested checkpoints, including incomplete checkpoint pairs.",
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.all_days and args.train_times is None:
        args.all_days = True
    for name in ("n_pcs", "batch_size", "max_epochs", "patience", "log_every"):
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
    if any(width <= 0 for width in args.hidden_dims):
        parser.error("--hidden-dims values must be positive")
    if tuple(args.hidden_dims) != (128, 64):
        parser.error(
            "Schiebinger flow checkpoints currently require --hidden-dims 128 64"
        )
    return args


def training_variant(training_times: Sequence[str]) -> shared.ClassifierVariant:
    canonical = schiebinger.parse_training_timepoints(training_times)
    return shared.ClassifierVariant(
        key=f"train_days_{schiebinger.classifier_schedule_slug(canonical)}",
        excluded_day=None,
        usage="flow_training_for_selected_days",
        included_days=canonical,
    )


def requested_runs(
    args: argparse.Namespace,
) -> List[Tuple[shared.ClassifierVariant, Path]]:
    output_dir = args.output_dir.expanduser().resolve()
    runs: List[Tuple[shared.ClassifierVariant, Path]] = []
    if args.all_days:
        variant = shared.ClassifierVariant(
            key="all_days",
            excluded_day=None,
            usage="flow_evaluation_only",
        )
        path = args.all_days_checkpoint or schiebinger.classifier_checkpoint_path(
            all_days=True,
            classifier_dir=output_dir,
            n_pcs=args.n_pcs,
        )
        runs.append((variant, path.expanduser().resolve()))
    if args.train_times is not None:
        times = schiebinger.parse_training_timepoints(args.train_times)
        variant = training_variant(times)
        path = args.training_checkpoint or schiebinger.classifier_checkpoint_path(
            training_timepoints=times,
            classifier_dir=output_dir,
            n_pcs=args.n_pcs,
        )
        runs.append((variant, path.expanduser().resolve()))
    return runs


def checkpoint_pair_state(pt_path: Path) -> str:
    """Return ``complete``, ``missing``, or ``partial`` for a checkpoint pair."""
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
            "Schiebinger classifier training requires working anndata, NumPy, "
            "and pandas installations."
        ) from exc

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        for key in ("day", "cell_sets"):
            if key not in adata.obs:
                raise KeyError(f"{input_path.name} is missing adata.obs[{key!r}].")
        if "highly_variable" not in adata.var:
            raise KeyError(
                f"{input_path.name} is not an HVG dataset: missing "
                "adata.var['highly_variable']."
            )
        hvg_mask = adata.var["highly_variable"].astype(bool).to_numpy()
        if int(hvg_mask.sum()) != int(adata.n_vars):
            raise ValueError(
                f"{input_path.name} still contains non-HVG genes. Run "
                "scripts/create_schiebinger_hvg_pca.py first."
            )
        representation = adata.obsm.get("X_pca")
        stored_n_pcs = 0 if representation is None else int(representation.shape[1])
        if representation is None or stored_n_pcs < args.n_pcs:
            raise ValueError(
                f"{input_path.name} stores {stored_n_pcs} PCs; {args.n_pcs} were "
                "requested. Run scripts/create_schiebinger_hvg_pca.py with the "
                "matching --n-pcs value."
            )

        numeric_days = pd.to_numeric(adata.obs["day"], errors="coerce").to_numpy()
        valid = np.isfinite(numeric_days)
        if "serum" in adata.obs:
            raw_serum = adata.obs["serum"]
            if str(raw_serum.dtype) == "bool":
                serum_mask = raw_serum.to_numpy(dtype=bool)
            else:
                serum_mask = (
                    raw_serum.astype("string")
                    .str.lower()
                    .isin(("true", "1", "yes"))
                    .to_numpy()
                )
            valid &= serum_mask
        labels = (
            adata.obs["cell_sets"]
            .astype("string")
            .str.strip()
            .astype(str)
            .to_numpy()
        )
        if bool(pd.isna(adata.obs["cell_sets"]).any()) or bool(
            (labels == "").any()
        ):
            raise ValueError(f"{input_path.name} contains missing cell_sets labels.")
        x = np.asarray(representation[:, : args.n_pcs], dtype=np.float32)[valid]
        if not bool(np.isfinite(x).all()):
            raise ValueError(f"{input_path.name} contains non-finite PCA coordinates.")
        days = np.asarray(
            [schiebinger.format_timepoint(value) for value in numeric_days[valid]],
            dtype=object,
        )
        labels = labels[valid]
        unknown_types = sorted(set(labels) - set(schiebinger.CLASS_NAMES))
        if unknown_types:
            raise ValueError(f"Unexpected Schiebinger cell types: {unknown_types}.")
        return {
            "x": x,
            "labels": labels,
            "cell_ids": np.asarray(adata.obs_names.astype(str), dtype=object)[valid],
            "days": days,
            "feature_names": [str(value) for value in adata.var_names],
            "n_cells": int(x.shape[0]),
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
            print("All requested Schiebinger classifier checkpoints already exist.")
            return 0

        dependencies = shared.import_dependencies()
        device = shared.choose_device(dependencies["torch"], args.device)
        input_path = schiebinger.resolve_dataset_path(
            args.dataset_location,
            n_pcs=args.n_pcs,
        )
        print(f"[schiebinger] Reading {input_path}", flush=True)
        data = load_training_data(args, input_path)

        # Fields consumed by the shared CITE/Multi training engine.
        args.output_root = REPO_ROOT
        args.pca_key = "X_pca"
        args.label_key = "cell_sets"
        args.day_key = "day"
        metrics_by_variant: Dict[str, Dict[str, Any]] = {}
        for variant, pt_path in pending:
            run_args = copy.copy(args)
            run_args.classifier_output_dir = pt_path.parent
            run_args.checkpoint_stem_override = pt_path.stem
            metrics_by_variant[variant.key] = shared.train_one_variant(
                run_args,
                "schiebinger",
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
            f"  schiebinger/{variant}: "
            f"val_loss={metrics['validation_loss']:.4f}, "
            f"accuracy={metrics['validation_accuracy']:.4f}, "
            f"balanced_accuracy={metrics['validation_balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['validation_macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

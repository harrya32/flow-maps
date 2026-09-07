#!/usr/bin/env python3
"""Create a compact Schiebinger H5AD containing HVGs and stored PCs."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import schiebinger  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subset the Schiebinger expression matrix to its supplied highly "
            "variable genes and store a PCA representation in a new H5AD."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=schiebinger.DEFAULT_DATA_DIR / schiebinger.SOURCE_FILENAME,
        help="Original Schiebinger serum H5AD.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output H5AD. By default it is written beside the source using "
            "the canonical hvg<N>_pca<N> filename."
        ),
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=schiebinger.DEFAULT_N_PCS,
        help="Number of principal components to store (default: 5).",
    )
    parser.add_argument("--hvg-key", default="highly_variable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    args = parser.parse_args(argv)
    if args.n_pcs <= 0:
        parser.error("--n-pcs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cache_root = Path(tempfile.gettempdir()) / "flow_maps_numba_cache"
    matplotlib_root = Path(tempfile.gettempdir()) / "flow_maps_matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    matplotlib_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_root))

    try:
        import scanpy as sc
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(
            f"error: a working scanpy installation is required: {exc}",
            file=sys.stderr,
        )
        return 1

    source = args.source.expanduser().resolve()
    if not source.is_file():
        print(f"error: source H5AD does not exist: {source}", file=sys.stderr)
        return 1

    print(f"Reading {source}", flush=True)
    adata = sc.read_h5ad(source)
    if args.hvg_key not in adata.var:
        print(
            f"error: missing adata.var[{args.hvg_key!r}] in {source.name}",
            file=sys.stderr,
        )
        return 1

    hvg_mask = adata.var[args.hvg_key].astype(bool).to_numpy()
    n_hvgs = int(hvg_mask.sum())
    if n_hvgs == 0:
        print(f"error: adata.var[{args.hvg_key!r}] selects no genes", file=sys.stderr)
        return 1
    if args.n_pcs >= min(int(adata.n_obs), n_hvgs):
        print(
            "error: --n-pcs must be smaller than both the number of cells and "
            f"the number of selected HVGs ({n_hvgs})",
            file=sys.stderr,
        )
        return 1

    target = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(
            schiebinger.pca_dataset_filename(args.n_pcs, n_hvgs=n_hvgs)
        )
    )
    if source == target:
        print("error: output must differ from the source H5AD", file=sys.stderr)
        return 1
    if target.exists() and not args.overwrite:
        print(f"Reusing existing PCA dataset: {target}")
        return 0

    print(
        f"Selecting {n_hvgs:,}/{adata.n_vars:,} genes and computing "
        f"{args.n_pcs} PCs.",
        flush=True,
    )
    reduced = adata[:, hvg_mask].copy()
    del adata
    gc.collect()
    reduced.obsm.pop("X_pca", None)
    reduced.uns.pop("pca", None)
    reduced.varm.pop("PCs", None)
    sc.pp.pca(reduced, n_comps=args.n_pcs, random_state=args.seed)
    reduced.uns["hvg_pca_provenance"] = {
        "source_file": source.name,
        "highly_variable_key": str(args.hvg_key),
        "n_highly_variable_genes": n_hvgs,
        "n_pcs": int(args.n_pcs),
        "pca_random_state": int(args.seed),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.stem}.partial-{os.getpid()}{target.suffix}"
    )
    try:
        reduced.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {target}")
    print(
        f"AnnData shape: {reduced.shape}; "
        f"X_pca shape: {reduced.obsm['X_pca'].shape}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

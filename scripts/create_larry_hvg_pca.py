#!/usr/bin/env python3
"""Create the compact HVG/PCA H5AD consumed by the LARRY flow config."""

from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import larry  # noqa: E402


RAW_FILENAMES = {
    "counts": "stateFate_inVitro_normed_counts.mtx.gz",
    "genes": "stateFate_inVitro_gene_names.txt.gz",
    "metadata": "stateFate_inVitro_metadata.txt.gz",
    "clones": "stateFate_inVitro_clone_matrix.mtx.gz",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select highly variable genes and fit PCA using D2, D4, and D6, then "
            "retain cell and clone barcodes in a compact H5AD."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=larry.DEFAULT_DATA_DIR,
        help="Directory containing the four downloaded stateFate_inVitro files.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-hvgs", type=int, default=larry.DEFAULT_N_HVGS)
    parser.add_argument("--n-pcs", type=int, default=larry.DEFAULT_N_PCS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clone-labelled-only",
        "--clone-labeled-only",
        action="store_true",
        help=(
            "Fit HVGs and PCA using only cells with a LARRY clone assignment. "
            "D2, D4, and D6 remain included in the fit."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.n_hvgs <= 0 or args.n_pcs <= 0:
        parser.error("--n-hvgs and --n-pcs must be positive")
    return args


def _read_sparse(path: Path):
    from scipy import sparse
    from scipy.io import mmread

    with gzip.open(path, "rb") as handle:
        matrix = mmread(handle)
    return matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    target = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_dir
        / larry.pca_dataset_filename(
            args.n_pcs,
            args.n_hvgs,
            clone_labelled_only=bool(args.clone_labelled_only),
        )
    )
    if target.exists() and not args.overwrite:
        print(f"Reusing existing PCA dataset: {target}")
        return 0

    # Some managed environments make the installed Scanpy package directory
    # read-only.  Giving Numba an explicit writable cache keeps Scanpy imports
    # and repeated preprocessing runs reliable in those environments.
    numba_cache = Path(tempfile.gettempdir()) / f"flow-maps-numba-{os.getuid()}"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scanpy as sc
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(
            f"error: working scanpy/anndata packages are required: {exc}",
            file=sys.stderr,
        )
        return 1

    paths = {name: data_dir / filename for name, filename in RAW_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        print(
            "error: missing LARRY source files:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
        return 1

    print("Reading LARRY expression and clone matrices.", flush=True)
    expression = _read_sparse(paths["counts"]).astype(np.float32)
    clone_matrix = _read_sparse(paths["clones"])
    clone_matrix.eliminate_zeros()
    metadata = pd.read_csv(paths["metadata"], sep="\t", compression="gzip")
    with gzip.open(paths["genes"], "rt") as handle:
        gene_names = [line.rstrip("\r\n") for line in handle]

    n_cells = len(metadata)
    source_n_cells = n_cells
    if expression.shape == (len(gene_names), n_cells):
        expression = expression.T.tocsr()
    if clone_matrix.shape[0] != n_cells and clone_matrix.shape[1] == n_cells:
        clone_matrix = clone_matrix.T.tocsr()
    if expression.shape != (n_cells, len(gene_names)):
        raise ValueError(
            f"Expression shape {expression.shape} does not match {n_cells} cells "
            f"and {len(gene_names)} genes."
        )
    if clone_matrix.shape[0] != n_cells:
        raise ValueError(
            f"Clone matrix shape {clone_matrix.shape} does not match {n_cells} cells."
        )

    required = ["Library", "Cell barcode", "Time point", "Cell type annotation"]
    missing_columns = [column for column in required if column not in metadata]
    if missing_columns:
        raise KeyError(f"Metadata is missing columns: {missing_columns}")

    day = pd.to_numeric(metadata["Time point"], errors="raise")
    day_label = day.map(lambda value: f"D{float(value):g}")
    unexpected_days = sorted(set(day_label) - set(larry.TIMEPOINTS))
    if unexpected_days:
        raise ValueError(f"Unexpected LARRY days: {unexpected_days}")

    clone_assignments = np.asarray(clone_matrix.getnnz(axis=1)).ravel()
    if np.any(clone_assignments > 1):
        raise ValueError(
            "Some cells have multiple clone assignments; clone-conditioned "
            "evaluation requires one clone id per cell."
        )
    clone_ids = np.full(n_cells, -1, dtype=np.int64)
    clone_coo = clone_matrix.tocoo()
    clone_ids[clone_coo.row] = clone_coo.col.astype(np.int64)
    del clone_assignments, clone_coo, clone_matrix

    if args.clone_labelled_only:
        selected = np.flatnonzero(clone_ids >= 0)
        if selected.size == 0:
            raise ValueError("The LARRY clone matrix contains no labelled cells.")
        expression = expression[selected].tocsr()
        metadata = metadata.iloc[selected].copy()
        day = day.iloc[selected]
        day_label = day_label.iloc[selected]
        clone_ids = clone_ids[selected]
        n_cells = int(selected.size)
        print(
            f"Restricting PCA fit to {n_cells:,}/{source_n_cells:,} "
            "clone-labelled cells across D2, D4, and D6.",
            flush=True,
        )

    cell_barcodes = metadata["Cell barcode"].astype(str)
    libraries = metadata["Library"].astype(str)
    obs_names = libraries + ":" + cell_barcodes
    if obs_names.duplicated().any():
        raise ValueError("Library-qualified cell barcodes are not unique.")
    obs = metadata.copy()
    obs.index = pd.Index(obs_names, name="cell_id")
    obs["library"] = libraries.to_numpy()
    obs["cell_barcode"] = cell_barcodes.to_numpy()
    obs["day"] = day.to_numpy(dtype=np.float32)
    obs["day_label"] = day_label.to_numpy()
    obs["cell_type"] = metadata["Cell type annotation"].astype(str).to_numpy()
    obs["clone_id"] = clone_ids
    obs["has_clone"] = clone_ids >= 0

    processed = ad.AnnData(
        X=expression,
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_names, name="gene")),
    )
    processed.var_names_make_unique()
    print(
        f"Fitting {min(int(args.n_hvgs), processed.n_vars):,} HVGs and "
        f"{int(args.n_pcs)} PCs jointly on D2, D4, and D6.",
        flush=True,
    )
    # Work in place until the HVG subset is taken: the source matrix is large,
    # so keeping a second complete sparse copy needlessly doubles peak memory.
    sc.pp.log1p(processed)
    sc.pp.highly_variable_genes(
        processed,
        n_top_genes=min(int(args.n_hvgs), processed.n_vars),
        flavor="seurat",
    )
    hvg_mask = processed.var["highly_variable"].to_numpy(dtype=bool)
    requested_hvgs = min(int(args.n_hvgs), int(processed.n_vars))
    if int(hvg_mask.sum()) != requested_hvgs:
        # Scanpy selects every gene tied at the n_top_genes cutoff. Break those
        # ties stably so that the cache filename and stored dimension agree.
        scores = np.asarray(processed.var["dispersions_norm"], dtype=np.float64)
        order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
        hvg_mask = np.zeros(processed.n_vars, dtype=bool)
        hvg_mask[order[:requested_hvgs]] = True
        processed.var["highly_variable"] = hvg_mask
    n_hvgs = int(hvg_mask.sum())
    if args.n_pcs >= min(int(processed.n_obs), n_hvgs):
        raise ValueError(
            "--n-pcs must be smaller than both the cell count and "
            f"selected HVG count ({n_hvgs})."
        )

    full_hvg = processed[:, hvg_mask].copy()
    del processed, expression
    sc.pp.pca(
        full_hvg,
        n_comps=int(args.n_pcs),
        zero_center=True,
        svd_solver="arpack",
        random_state=int(args.seed),
    )
    full_hvg.obsm["X_pca"] = np.asarray(full_hvg.obsm["X_pca"], dtype=np.float32)
    full_hvg.uns["hvg_pca_provenance"] = {
        "source_counts": paths["counts"].name,
        "source_metadata": paths["metadata"].name,
        "source_clone_matrix": paths["clones"].name,
        "fit_days": ["D2", "D4", "D6"],
        "heldout_day": "D4",
        "cell_subset": (
            "clone_labelled_only" if args.clone_labelled_only else "all_cells"
        ),
        "source_cell_count": int(source_n_cells),
        "retained_cell_count": int(n_cells),
        "n_highly_variable_genes": n_hvgs,
        "n_pcs": int(args.n_pcs),
        "random_seed": int(args.seed),
        "expression_transform": "log1p of supplied total-count-normalized values",
        "clone_id_convention": "zero-based clone-matrix column; -1 means unassigned",
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial-{os.getpid()}{target.suffix}")
    try:
        full_hvg.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {target}")
    print(
        f"Cells={full_hvg.n_obs:,}, HVGs={full_hvg.n_vars:,}, "
        f"PCA shape={full_hvg.obsm['X_pca'].shape}, "
        f"clone-labelled cells={int((clone_ids >= 0).sum()):,}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

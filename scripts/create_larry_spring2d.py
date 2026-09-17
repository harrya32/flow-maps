#!/usr/bin/env python3
"""Create the compact LARRY dataset in the authors' supplied SPRING space."""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import larry  # noqa: E402


METADATA_FILENAME = "stateFate_inVitro_metadata.txt.gz"
CLONE_FILENAME = "stateFate_inVitro_clone_matrix.mtx.gz"
SPRING_COLUMNS = ("SPRING-x", "SPRING-y")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, isotropically scale, and cache the published two-dimensional "
            "LARRY SPRING coordinates with cell and clone metadata."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=larry.DEFAULT_DATA_DIR,
        help="Directory containing the downloaded LARRY metadata and clone matrix.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_clone_matrix(path: Path):
    from scipy import sparse
    from scipy.io import mmread

    with gzip.open(path, "rb") as handle:
        matrix = mmread(handle)
    matrix = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(matrix)
    matrix.eliminate_zeros()
    return matrix


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    metadata_path = data_dir / METADATA_FILENAME
    clone_path = data_dir / CLONE_FILENAME
    target = (
        args.output.expanduser().resolve()
        if args.output is not None
        else data_dir / larry.spring_dataset_filename()
    )
    if target.exists() and not args.overwrite:
        print(f"Reusing existing SPRING dataset: {target}")
        return 0
    missing = [str(path) for path in (metadata_path, clone_path) if not path.is_file()]
    if missing:
        print(
            "error: missing LARRY source files:\n  " + "\n  ".join(missing),
            file=sys.stderr,
        )
        return 1

    try:
        import anndata as ad
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: working anndata is required: {exc}", file=sys.stderr)
        return 1

    metadata = pd.read_csv(metadata_path, sep="\t", compression="gzip")
    required = [
        "Library",
        "Cell barcode",
        "Time point",
        "Cell type annotation",
        *SPRING_COLUMNS,
    ]
    missing_columns = [column for column in required if column not in metadata]
    if missing_columns:
        raise KeyError(f"LARRY metadata is missing columns: {missing_columns}")

    clone_matrix = _read_clone_matrix(clone_path)
    if clone_matrix.shape[0] != len(metadata) and clone_matrix.shape[1] == len(
        metadata
    ):
        clone_matrix = clone_matrix.T.tocsr()
    if clone_matrix.shape[0] != len(metadata):
        raise ValueError(
            f"Clone matrix shape {clone_matrix.shape} does not match "
            f"{len(metadata)} metadata rows."
        )
    clone_assignments = np.asarray(clone_matrix.getnnz(axis=1)).ravel()
    if np.any(clone_assignments > 1):
        raise ValueError("Some cells have multiple LARRY clone assignments.")
    clone_ids = np.full(len(metadata), -1, dtype=np.int64)
    clone_coo = clone_matrix.tocoo()
    clone_ids[clone_coo.row] = clone_coo.col.astype(np.int64)

    raw = metadata.loc[:, SPRING_COLUMNS].apply(pd.to_numeric, errors="raise")
    raw = raw.to_numpy(dtype=np.float32)
    if raw.shape != (len(metadata), larry.DEFAULT_SPRING_DIM):
        raise ValueError(f"Unexpected SPRING coordinate shape {raw.shape}.")
    if not bool(np.isfinite(raw).all()):
        raise ValueError("Published SPRING coordinates contain non-finite values.")
    center = np.mean(raw, axis=0, dtype=np.float64)
    centered = raw.astype(np.float64) - center
    # One shared scale preserves the geometry and aspect ratio of the published
    # force-directed layout. Per-axis standardization would distort distances.
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Invalid SPRING coordinate scale {scale}.")
    spring = ((raw.astype(np.float64) - center) / scale).astype(np.float32)

    day = pd.to_numeric(metadata["Time point"], errors="raise")
    day_label = day.map(lambda value: f"D{float(value):g}")
    unexpected_days = sorted(set(day_label) - set(larry.TIMEPOINTS))
    if unexpected_days:
        raise ValueError(f"Unexpected LARRY days: {unexpected_days}")
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

    result = ad.AnnData(
        X=spring.copy(),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(["SPRING-1", "SPRING-2"], name="coordinate")),
    )
    result.obsm["X_spring"] = spring
    result.obsm["X_spring_raw"] = raw
    result.uns["spring2d_provenance"] = {
        "source_metadata": metadata_path.name,
        "source_clone_matrix": clone_path.name,
        "source_coordinate_columns": list(SPRING_COLUMNS),
        "fit_days": ["D2", "D4", "D6"],
        "heldout_flow_training_day": "D4",
        "normalization": "subtract coordinate mean and divide both axes by one global RMS",
        "coordinate_center": center.tolist(),
        "coordinate_scale": scale,
        "published_layout": True,
        "clone_id_convention": "zero-based clone-matrix column; -1 means unassigned",
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial-{os.getpid()}{target.suffix}")
    try:
        result.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"Saved {target}: cells={result.n_obs:,}, dim={spring.shape[1]}, "
        f"center={center.tolist()}, shared_scale={scale:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

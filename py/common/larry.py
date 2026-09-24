"""Weinreb et al. LARRY in-vitro differentiation data utilities.

The flow experiment trains only on the day-2 and day-6 marginals.  Day 4 is
kept out of pair construction and is used for population- and clone-level
evaluation.  The supported state spaces are the 50-dimensional HVG-PCA
representation and the authors' supplied two-dimensional SPRING layout.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

from . import cite_multi, maizels


DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "LARRY_INVITRO_DATA_DIR",
        str(Path.home() / "Desktop" / "flow-maps-data"),
    )
).expanduser()
DEFAULT_N_HVGS = 2_000
DEFAULT_N_PCS = 50
DEFAULT_SPRING_DIM = 2
PCA50_REPRESENTATION = "pca50"
SPRING2D_REPRESENTATION = "spring2d"
REPRESENTATIONS = (PCA50_REPRESENTATION, SPRING2D_REPRESENTATION)
REPRESENTATION_KEYS = {
    PCA50_REPRESENTATION: "X_pca",
    SPRING2D_REPRESENTATION: "X_spring",
}
REPRESENTATION_DIMS = {
    PCA50_REPRESENTATION: DEFAULT_N_PCS,
    SPRING2D_REPRESENTATION: DEFAULT_SPRING_DIM,
}
TARGETS = ("larry_pca50", "larry_spring2d")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_DIR = Path(
    os.environ.get(
        "LARRY_CLASSIFIER_DIR",
        str(REPO_ROOT / "larry-classifiers"),
    )
).expanduser()


def pca_dataset_filename(
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
    *,
    clone_labelled_only: bool = False,
) -> str:
    """Return the canonical processed-data filename."""
    if int(n_pcs) <= 0 or int(n_hvgs) <= 0:
        raise ValueError("PCA dimensions and HVG counts must be positive.")
    subset = "_clone_labelled" if clone_labelled_only else ""
    return (
        f"stateFate_inVitro{subset}_hvg{int(n_hvgs)}_pca{int(n_pcs)}.h5ad"
    )


def spring_dataset_filename() -> str:
    return "stateFate_inVitro_spring2d.h5ad"


def canonical_representation(value: str | None) -> str:
    aliases = {
        "pca": PCA50_REPRESENTATION,
        "pca50": PCA50_REPRESENTATION,
        "spring": SPRING2D_REPRESENTATION,
        "spring2d": SPRING2D_REPRESENTATION,
        "spring_2d": SPRING2D_REPRESENTATION,
    }
    key = aliases.get(str(value or PCA50_REPRESENTATION).lower())
    if key is None:
        raise ValueError(
            f"Unknown LARRY representation {value!r}; choose from "
            f"{list(REPRESENTATIONS)}."
        )
    return key


def representation_key(value: str | None) -> str:
    return REPRESENTATION_KEYS[canonical_representation(value)]


def representation_dim(value: str | None) -> int:
    return int(REPRESENTATION_DIMS[canonical_representation(value)])


def representation_dataset_filename(
    value: str | None,
    *,
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
    clone_labelled_only: bool = False,
) -> str:
    representation = canonical_representation(value)
    if representation == SPRING2D_REPRESENTATION:
        if clone_labelled_only:
            raise ValueError(
                "clone_labelled_only requires the PCA representation; the "
                "SPRING layout is not refitted."
            )
        return spring_dataset_filename()
    return pca_dataset_filename(
        n_pcs,
        n_hvgs,
        clone_labelled_only=clone_labelled_only,
    )


def is_larry_target(target: str | None) -> bool:
    return str(target) in TARGETS


DEFAULT_FILENAME = pca_dataset_filename()
DEFAULT_SPRING_FILENAME = spring_dataset_filename()
TIMEPOINTS = ("D2", "D4", "D6")
NORMALIZED_TIMES = {"D2": 0.0, "D4": 0.5, "D6": 1.0}

# Keep the source annotation spelling and the alphabetic order emitted by the
# shared classifier trainer's LabelEncoder.  Pair labels and classifier output
# indices must use this same order for hard interpolant checks.
CLASS_NAMES = (
    "Baso",
    "Ccr7_DC",
    "Eos",
    "Erythroid",
    "Lymphoid",
    "Mast",
    "Meg",
    "Monocyte",
    "Neutrophil",
    "Undifferentiated",
    "pDC",
)
TRANSITION_EDGES = tuple(
    ("Undifferentiated", cell_type)
    for cell_type in CLASS_NAMES
    if cell_type != "Undifferentiated"
)
UNCONSTRAINED_CLASS_NAMES: Tuple[str, ...] = ()
CLASSIFIER_PAIR_MODES = ("endpoint_interpolant", "ot_endpoint_interpolant")

_DATA_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def classifier_schedule_slug(
    training_timepoints: Sequence[str | float | int],
) -> str:
    canonical = tuple(format_timepoint(value) for value in training_timepoints)
    if not canonical:
        raise ValueError("At least one classifier training day is required.")
    return "_".join(value.lower().replace(".", "p") for value in canonical)


def classifier_checkpoint_path(
    training_timepoints: Sequence[str | float | int] | None = None,
    *,
    all_days: bool = False,
    classifier_dir: str | Path | None = None,
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
    representation: str = PCA50_REPRESENTATION,
    clone_labelled_only: bool = False,
) -> Path:
    """Return a canonical LARRY classifier checkpoint path."""
    if all_days == (training_timepoints is not None):
        raise ValueError("Specify exactly one of all_days=True or training_timepoints.")
    root = Path(classifier_dir or DEFAULT_CLASSIFIER_DIR).expanduser().resolve()
    variant = (
        "all_days"
        if all_days
        else f"train_days_{classifier_schedule_slug(training_timepoints or ())}"
    )
    representation = canonical_representation(representation)
    if representation == SPRING2D_REPRESENTATION:
        if clone_labelled_only:
            raise ValueError(
                "clone_labelled_only classifiers require the PCA representation."
            )
        stem = f"celltype_classifier_larry_spring2d_{variant}"
    else:
        subset = "_clone_labelled" if clone_labelled_only else ""
        stem = (
            f"celltype_classifier_larry{subset}_hvg{int(n_hvgs)}_"
            f"pca{int(n_pcs)}_{variant}"
        )
    return root / f"{stem}.pt"


DEFAULT_CLASSIFIER = str(classifier_checkpoint_path(all_days=True))


def resolve_dataset_path(
    dataset_location: str | Path | None,
    *,
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
    representation: str = PCA50_REPRESENTATION,
    clone_labelled_only: bool = False,
) -> Path:
    """Resolve a processed H5AD directory or explicit H5AD path."""
    representation = canonical_representation(representation)
    if clone_labelled_only and representation != PCA50_REPRESENTATION:
        raise ValueError(
            "clone_labelled_only requires the PCA representation; the SPRING "
            "layout is not refitted."
        )
    location = (
        DEFAULT_DATA_DIR
        if dataset_location in (None, "")
        else Path(str(dataset_location)).expanduser()
    )
    if location.suffix.lower() == ".h5ad":
        return location.resolve()
    return (
        location
        / representation_dataset_filename(
            representation,
            n_pcs=n_pcs,
            n_hvgs=n_hvgs,
            clone_labelled_only=clone_labelled_only,
        )
    ).resolve()


def parse_timepoint(value: str | float | int) -> float:
    text = str(value).strip()
    if text.upper().startswith("D"):
        text = text[1:]
    return float(text)


def format_timepoint(value: str | float | int) -> str:
    return f"D{parse_timepoint(value):g}"


def normalized_time(timepoint: str | float | int, cfg=None) -> float:
    key = format_timepoint(timepoint)
    if cfg is not None:
        order = getattr(cfg.problem, "timepoint_order", None)
        values = getattr(cfg.problem, "timepoint_values", None)
        if order is not None and values is not None:
            mapping = {
                format_timepoint(name): float(value)
                for name, value in zip(list(order), list(values))
            }
            if key in mapping:
                return mapping[key]
    if key not in NORMALIZED_TIMES:
        raise KeyError(f"Unknown LARRY timepoint {timepoint!r}.")
    return float(NORMALIZED_TIMES[key])


def retained_timepoints(cfg) -> Tuple[str, str]:
    configured = tuple(
        format_timepoint(value)
        for value in getattr(cfg.problem, "retained_timepoints", ("D2", "D6"))
    )
    if configured != ("D2", "D6"):
        raise ValueError(
            "The current LARRY experiment expects retained_timepoints=['D2', 'D6']; "
            f"got {configured}."
        )
    return configured


def flow_training_requires_classifier(cfg) -> bool:
    """Return whether the selected future variant consumes a classifier."""
    pair_mode = str(
        getattr(
            cfg.problem,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        )
    )
    constraints = getattr(cfg, "constraints", None)
    lineage_loss = (
        bool(getattr(constraints, "enabled", False))
        and str(getattr(constraints, "type", "")) == "maizels_lineage_path"
    )
    return pair_mode in CLASSIFIER_PAIR_MODES or lineage_loss


def _obs_values(adata, preferred: str, fallback: str | None = None) -> np.ndarray:
    if preferred in adata.obs:
        return adata.obs[preferred].astype("string").astype(str).to_numpy()
    if fallback is not None and fallback in adata.obs:
        return adata.obs[fallback].astype("string").astype(str).to_numpy()
    options = [preferred] + ([] if fallback is None else [fallback])
    raise KeyError(f"Missing required H5AD obs column; expected one of {options}.")


def load_dataset(
    dataset_path: str | Path,
    *,
    representation: str | None = None,
    n_pcs: int | None = None,
) -> Dict[str, np.ndarray]:
    """Load one LARRY representation, annotations, barcodes, and clone ids."""
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed LARRY dataset not found: {path}. Run the matching "
            "scripts/create_larry_hvg_pca.py or scripts/create_larry_spring2d.py "
            "preparation command first."
        )

    try:
        import anndata as ad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading the processed LARRY dataset requires the 'anndata' package."
        ) from exc

    adata = ad.read_h5ad(path, backed="r")
    try:
        if representation is None:
            available = [
                name for name, key in REPRESENTATION_KEYS.items() if key in adata.obsm
            ]
            if len(available) != 1:
                raise ValueError(
                    f"{path.name}: representation is ambiguous; available LARRY "
                    f"representations are {available}."
                )
            selected_representation = available[0]
        else:
            selected_representation = canonical_representation(representation)
        selected_key = representation_key(selected_representation)
        if selected_representation == PCA50_REPRESENTATION:
            requested_dim = None if n_pcs is None else int(n_pcs)
            if requested_dim is not None and requested_dim <= 0:
                raise ValueError("n_pcs must be positive.")
        else:
            requested_dim = representation_dim(selected_representation)
        dimension_key = "auto" if requested_dim is None else str(requested_dim)
        cache_key = f"{path}::{selected_representation}::{dimension_key}"
        if cache_key in _DATA_CACHE:
            return _DATA_CACHE[cache_key]
        if selected_key not in adata.obsm:
            raise KeyError(f"{path.name}: missing obsm[{selected_key!r}]")
        x = np.asarray(adata.obsm[selected_key], dtype=np.float32)
        actual_dim = x.shape[1] if x.ndim == 2 else -1
        selected_dim = actual_dim if requested_dim is None else requested_dim
        raw_times = _obs_values(adata, "day_label", "Time point")
        timepoints = np.asarray(
            [format_timepoint(value) for value in raw_times], dtype=object
        )
        cell_types = _obs_values(adata, "cell_type", "Cell type annotation").astype(
            object
        )
        cell_barcodes = _obs_values(adata, "cell_barcode", "Cell barcode").astype(
            object
        )
        if "library" in adata.obs:
            libraries = _obs_values(adata, "library").astype(object)
        elif "Library" in adata.obs:
            libraries = _obs_values(adata, "Library").astype(object)
        else:
            libraries = np.full(adata.n_obs, "", dtype=object)
        if "clone_id" not in adata.obs:
            raise KeyError(
                f"{path.name}: missing obs['clone_id']; rebuild it with the "
                "LARRY clone matrix."
            )
        clone_raw = np.asarray(adata.obs["clone_id"])
        clone_ids = np.full(adata.n_obs, -1, dtype=np.int64)
        for idx, value in enumerate(clone_raw):
            if value is None or str(value).strip().lower() in {
                "",
                "nan",
                "none",
                "<na>",
                "-1",
            }:
                continue
            clone_ids[idx] = int(float(value))
        obs_names = np.asarray(adata.obs_names.astype(str), dtype=object)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    unknown_times = sorted(set(timepoints) - set(TIMEPOINTS))
    if unknown_times:
        raise ValueError(f"Unexpected LARRY timepoints in {path.name}: {unknown_times}")
    unknown_types = sorted(set(cell_types) - set(CLASS_NAMES))
    if unknown_types:
        raise ValueError(f"Unexpected LARRY cell types in {path.name}: {unknown_types}")
    if x.ndim != 2 or x.shape[1] != selected_dim:
        raise ValueError(
            f"{path.name}: expected {selected_dim} {selected_representation} "
            "dimensions, found "
            f"{x.shape[1] if x.ndim == 2 else x.shape}."
        )
    if not bool(np.isfinite(x).all()):
        raise ValueError(f"{path.name}: representation contains non-finite values.")
    if not (
        x.shape[0] == timepoints.shape[0] == cell_types.shape[0] == clone_ids.shape[0]
    ):
        raise RuntimeError("LARRY representation and metadata rows are misaligned.")

    data = {
        "obs_names": obs_names,
        "x": x,
        "timepoints": timepoints,
        "time_values": np.asarray(
            [parse_timepoint(value) for value in timepoints], dtype=np.float32
        ),
        "normalized_times": np.asarray(
            [normalized_time(value) for value in timepoints], dtype=np.float32
        ),
        "cell_types": cell_types,
        "cell_barcodes": cell_barcodes,
        "libraries": libraries,
        "clone_ids": clone_ids,
        "representation": selected_representation,
        "representation_key": selected_key,
    }
    _DATA_CACHE[cache_key] = data
    return data


def all_timepoint_data(
    dataset_location: str | Path | None = None,
    *,
    representation: str | None = None,
    n_pcs: int | None = None,
) -> Dict[str, np.ndarray]:
    selected = (
        PCA50_REPRESENTATION
        if representation is None and dataset_location in (None, "")
        else representation
    )
    return load_dataset(
        resolve_dataset_path(
            dataset_location,
            n_pcs=DEFAULT_N_PCS if n_pcs is None else int(n_pcs),
            representation=selected or PCA50_REPRESENTATION,
        ),
        representation=representation,
        n_pcs=n_pcs,
    )


def subset_time(
    data: Dict[str, np.ndarray], timepoint: str | float | int
) -> Tuple[np.ndarray, np.ndarray]:
    mask = data["timepoints"] == format_timepoint(timepoint)
    return data["x"][mask], data["cell_types"][mask]


def timepoint_pool_splits(cfg, dataset_location=None):
    """Split D2/D6 for validation while leaving all D4 cells evaluation-only."""
    representation = canonical_representation(
        getattr(cfg.problem, "larry_representation", PCA50_REPRESENTATION)
    )
    clone_labelled_only = bool(
        getattr(cfg.problem, "larry_clone_labelled_only", False)
    )
    resolved_path = resolve_dataset_path(
        (
            dataset_location
            if dataset_location not in (None, "")
            else getattr(cfg.problem, "dataset_location", None)
        ),
        n_pcs=int(getattr(cfg.problem, "n_pcs", DEFAULT_N_PCS)),
        n_hvgs=int(getattr(cfg.problem, "n_hvgs", DEFAULT_N_HVGS)),
        representation=representation,
        clone_labelled_only=clone_labelled_only,
    )
    data = load_dataset(
        resolved_path,
        representation=representation,
        n_pcs=int(getattr(cfg.problem, "n_pcs", DEFAULT_N_PCS)),
    )
    if clone_labelled_only and bool(np.any(data["clone_ids"] < 0)):
        n_unlabelled = int(np.count_nonzero(data["clone_ids"] < 0))
        raise ValueError(
            f"{resolved_path.name} contains {n_unlabelled:,} cells without clone "
            "assignments. The clone-labelled-only setting requires the PCA cache "
            "created with scripts/create_larry_hvg_pca.py "
            "--clone-labelled-only."
        )
    retained = set(retained_timepoints(cfg))
    training_seed = int(getattr(getattr(cfg, "training", None), "seed", 0))
    split_seed = int(getattr(cfg.problem, "maizels_holdout_seed", training_seed + 701))
    holdout_fraction = float(getattr(cfg.problem, "maizels_holdout_fraction", 0.0))
    holdout_n = int(getattr(cfg.problem, "maizels_holdout_n", 0))

    result = {}
    for index, timepoint in enumerate(TIMEPOINTS):
        x, cell_types = subset_time(data, timepoint)
        if x.shape[0] == 0:
            raise RuntimeError(f"LARRY {timepoint} has no cells.")
        if timepoint in retained:
            train_idx, holdout_idx = maizels._split_train_holdout_indices(
                x.shape[0],
                holdout_fraction=holdout_fraction,
                holdout_n=holdout_n,
                seed=split_seed + 101 * (index + 1),
            )
        else:
            # D4 is never sampled by training or its endpoint validation loss.
            train_idx = np.arange(x.shape[0], dtype=np.int64)
            holdout_idx = np.empty((0,), dtype=np.int64)
        result[timepoint] = {
            "x": x,
            "types": cell_types,
            "train_x": x[train_idx],
            "train_types": cell_types[train_idx],
            "holdout_x": x[holdout_idx],
            "holdout_types": cell_types[holdout_idx],
            "train_idx": train_idx,
            "holdout_idx": holdout_idx,
        }
    return result


def endpoint_pool_splits(cfg, dataset_location=None) -> Dict[str, np.ndarray]:
    pools = timepoint_pool_splits(cfg, dataset_location)
    source = pools[format_timepoint(getattr(cfg.problem, "source_time", "D2"))]
    target = pools[format_timepoint(getattr(cfg.problem, "target_time", "D6"))]
    return maizels._endpoint_split_dict(source, target)


def _training_pair_mode(cfg) -> str:
    return str(
        getattr(
            cfg.problem,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        )
    )


def uses_minibatch_ot(cfg, pair_mode: str | None = None) -> bool:
    return maizels.uses_minibatch_ot(cfg, pair_mode or _training_pair_mode(cfg))


def _pair_from_arrays(
    cfg,
    source_x,
    source_types,
    target_x,
    target_types,
    *,
    n_pairs: int,
    pair_mode: str,
    seed: int,
    minibatch_ot: bool = False,
):
    builder = (
        cite_multi._make_minibatch_ot_pairs_from_arrays
        if minibatch_ot
        else maizels._make_pair_pool_from_endpoint_arrays
    )
    paired, stats = builder(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=np.random.default_rng(int(seed)),
        pair_mode=str(pair_mode),
        class_names=CLASS_NAMES,
        transition_edges=TRANSITION_EDGES,
    )
    return paired, stats


def make_pair_pool(cfg, dataset_location=None):
    """Create D2--D6 training pairs; D4 cannot enter this function."""
    splits = endpoint_pool_splits(cfg, dataset_location)
    pair_mode = _training_pair_mode(cfg)
    dynamic_ot = uses_minibatch_ot(cfg, pair_mode)
    paired, stats = _pair_from_arrays(
        cfg,
        splits["source_train_x"],
        splits["source_train_types"],
        splits["target_train_x"],
        splits["target_train_types"],
        n_pairs=int(getattr(cfg.problem, "n", 500_000)),
        pair_mode="none" if dynamic_ot else pair_mode,
        seed=int(getattr(cfg.training, "seed", 0)) + 301,
    )
    if dynamic_ot:
        stats.update(
            {
                "pair_mode": maizels._canonical_maizels_pair_mode(pair_mode),
                "pair_pool_mode": "independent_candidates",
                "coupling": "dynamic_minibatch_ot",
                "ot_minibatch_size": int(
                    getattr(cfg.problem, "ot_minibatch_size", 128)
                ),
                "ot_cost": "raw_sqeuclidean",
            }
        )
    stats.update(
        {
            "source_time": "D2",
            "target_time": "D6",
            "heldout_timepoint": "D4",
            "source_total_n": splits["source_n"],
            "source_train_n": splits["source_train_n"],
            "source_holdout_n": splits["source_holdout_n"],
            "target_total_n": splits["target_n"],
            "target_train_n": splits["target_train_n"],
            "target_holdout_n": splits["target_holdout_n"],
        }
    )
    return paired, stats


def make_minibatch_ot_training_pools(cfg, dataset_location=None):
    """Return compact D2/D6 endpoint pools for minibatch-OT variants."""
    pair_mode = maizels._canonical_maizels_pair_mode(_training_pair_mode(cfg))
    if not uses_minibatch_ot(cfg, pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")
    splits = endpoint_pool_splits(cfg, dataset_location)
    source_types = _cell_type_ids(splits["source_train_types"])
    target_types = _cell_type_ids(splits["target_train_types"])
    source_x = np.asarray(splits["source_train_x"], dtype=np.float32)
    target_x = np.asarray(splits["target_train_x"], dtype=np.float32)
    interval = {
        "source_time": "D2",
        "target_time": "D6",
        "t_start": 0.0,
        "t_end": 1.0,
        "nominal_pairs": int(getattr(cfg.problem, "n", 500_000)),
    }
    compact = {
        "timepoints": {
            "D2": {"x": source_x, "type_ids": source_types},
            "D6": {"x": target_x, "type_ids": target_types},
        },
        "intervals": (interval,),
        "nominal_n": int(getattr(cfg.problem, "n", 500_000)),
        "dimension": int(source_x.shape[1]),
        "include_time_bounds": False,
    }
    stored_cells = source_x.shape[0] + target_x.shape[0]
    stored_bytes = sum(
        value.nbytes for value in (source_x, source_types, target_x, target_types)
    )
    stats = {
        "pair_mode": pair_mode,
        "pair_pool_mode": "direct_timepoint_pools",
        "coupling": "dynamic_minibatch_ot",
        "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
        "ot_cost": "raw_sqeuclidean",
        "sampled_pairs": compact["nominal_n"],
        "candidate_pairs": compact["nominal_n"],
        "accepted_pairs": compact["nominal_n"],
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "candidate_acceptance_rate": 1.0,
        "stored_endpoint_cells": int(stored_cells),
        "stored_endpoint_bytes": int(stored_bytes),
        "intervals": {"D2_to_D6": dict(interval, sampled_pairs=compact["nominal_n"])},
        "heldout_timepoint": "D4",
    }
    return compact, stats


def couple_minibatch_ot_timepoint_pools(
    cfg,
    pools,
    n_pairs: int,
    *,
    seed: int,
    pair_mode: str | None = None,
):
    pair_mode = pair_mode or _training_pair_mode(cfg)
    intervals = tuple(pools["intervals"])
    if len(intervals) != 1:
        raise ValueError("The LARRY endpoint experiment expects exactly one interval.")
    interval = intervals[0]
    source = pools["timepoints"][interval["source_time"]]
    target = pools["timepoints"][interval["target_time"]]
    class_names = np.asarray(CLASS_NAMES, dtype=object)
    return _pair_from_arrays(
        cfg,
        source["x"],
        class_names[source["type_ids"]],
        target["x"],
        class_names[target["type_ids"]],
        n_pairs=int(n_pairs),
        pair_mode=pair_mode,
        seed=int(seed),
        minibatch_ot=True,
    )


def _cell_type_ids(cell_types: np.ndarray) -> np.ndarray:
    lookup = maizels.class_to_id_map(CLASS_NAMES)
    values = np.asarray(cell_types)
    if values.dtype.kind in "iu":
        result = values.astype(np.int32, copy=False)
    else:
        result = np.asarray([lookup[str(value)] for value in values], dtype=np.int32)
    if np.any(result < 0) or np.any(result >= len(CLASS_NAMES)):
        raise ValueError("LARRY pool contains an invalid cell-type id.")
    return result


def make_endpoint_split_pair_pool(
    cfg,
    n_pairs: int,
    *,
    split: str,
    dataset_location=None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    splits = endpoint_pool_splits(cfg, dataset_location)
    split = str(split).lower()
    prefix = "holdout" if split in ("holdout", "heldout") else split
    if prefix not in ("holdout", "train", "all"):
        raise ValueError("split must be one of 'heldout', 'train', or 'all'.")
    if prefix == "all":
        source_x, source_types = splits["source_x"], splits["source_types"]
        target_x, target_types = splits["target_x"], splits["target_types"]
    else:
        source_x = splits[f"source_{prefix}_x"]
        source_types = splits[f"source_{prefix}_types"]
        target_x = splits[f"target_{prefix}_x"]
        target_types = splits[f"target_{prefix}_types"]
    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError(f"LARRY endpoint split {split!r} is empty.")
    pair_mode = pair_mode or _training_pair_mode(cfg)
    seed = int(seed if seed is not None else int(cfg.training.seed) + 997)
    paired, stats = _pair_from_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        pair_mode=pair_mode,
        seed=seed,
        minibatch_ot=uses_minibatch_ot(cfg, pair_mode),
    )
    stats.update(
        {
            "split": "heldout" if prefix == "holdout" else prefix,
            "source_total_n": splits["source_n"],
            "source_train_n": splits["source_train_n"],
            "source_holdout_n": splits["source_holdout_n"],
            "target_total_n": splits["target_n"],
            "target_train_n": splits["target_train_n"],
            "target_holdout_n": splits["target_holdout_n"],
        }
    )
    return paired, stats


def make_validation_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location=None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    return make_endpoint_split_pair_pool(
        cfg,
        n_pairs,
        split="heldout",
        dataset_location=dataset_location,
        pair_mode=pair_mode,
        seed=seed,
    )


def make_heldout_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location=None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    return make_validation_pair_pool(
        cfg,
        n_pairs,
        dataset_location=dataset_location,
        pair_mode=pair_mode,
        seed=seed,
    )


def endpoint_pools(dataset_location, source_time: str, target_time: str):
    data = all_timepoint_data(dataset_location)
    source_x, source_types = subset_time(data, source_time)
    target_x, target_types = subset_time(data, target_time)
    return source_x, source_types, target_x, target_types


# Shared classifier machinery. The LARRY trainer exports the same MLP schema
# used by the Maizels and CITE/Multi experiments, so both hard path filtering
# and future differentiable constraints can reuse these implementations.
load_classifier = maizels.load_classifier
load_jax_classifier_params = maizels.load_jax_classifier_params
jax_classifier_logits = maizels.jax_classifier_logits
classifier_predictions = maizels.classifier_predictions
lineage_soft_terms_from_probs = maizels.lineage_soft_terms_from_probs
resolve_lineage_transition_mode = maizels.resolve_lineage_transition_mode
lineage_transition_mode_from_config = maizels.lineage_transition_mode_from_config


def classifier_index_lookup(class_names: Sequence[str]) -> np.ndarray:
    return maizels.classifier_index_lookup(
        class_names,
        canonical_class_names=CLASS_NAMES,
    )


def lineage_invalid_transition_matrix(
    class_names: Sequence[str], transition_mode: str | None = "descendant"
) -> np.ndarray:
    return maizels.lineage_invalid_transition_matrix(
        class_names,
        transition_mode=transition_mode,
        transition_edges=TRANSITION_EDGES,
        unconstrained_class_names=UNCONSTRAINED_CLASS_NAMES,
    )


def check_paths_with_classifier(*args, **kwargs):
    kwargs["transition_edges"] = TRANSITION_EDGES
    kwargs["unconstrained_class_names"] = UNCONSTRAINED_CLASS_NAMES
    return maizels.check_paths_with_classifier(*args, **kwargs)

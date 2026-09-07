"""Schiebinger serum reprogramming trajectories and lineage-aware pairing.

The observed training days are configurable.  Pairs are balanced over adjacent
retained days, while every omitted internal day is reserved for evaluation.
Only the explicitly supported MEF/MET/iPS and stromal transitions are
constrained; transitions involving the remaining cell types are intentionally
left unconstrained.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import cite_multi, maizels


DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "SCHIEBINGER_DATA_DIR",
        str(Path.home() / "Desktop" / "schiebinger"),
    )
).expanduser()
SOURCE_FILENAME = "schiebinger_serum_serum.h5ad"
DEFAULT_N_HVGS = 1479
DEFAULT_N_PCS = 5


def pca_dataset_filename(
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
) -> str:
    """Return the canonical filename for a precomputed HVG-PCA dataset."""
    n_pcs = int(n_pcs)
    n_hvgs = int(n_hvgs)
    if n_pcs <= 0 or n_hvgs <= 0:
        raise ValueError("PCA dimensions and HVG counts must be positive.")
    return f"schiebinger_serum_serum_hvg{n_hvgs}_pca{n_pcs}.h5ad"


DEFAULT_FILENAME = pca_dataset_filename()
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLASSIFIER_DIR = Path(
    os.environ.get(
        "SCHIEBINGER_CLASSIFIER_DIR",
        str(REPO_ROOT / "schiebinger_classifiers"),
    )
).expanduser()

TIME_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    5.5,
    6.0,
    6.5,
    7.0,
    7.5,
    8.0,
    8.25,
    8.5,
    8.75,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
    15.5,
    16.0,
    16.5,
    17.0,
    17.5,
    18.0,
)


def format_timepoint(value: str | float | int) -> str:
    """Return the stable string used for a Schiebinger observation day."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Schiebinger timepoints must be finite, got {value!r}.")
    return f"{numeric:g}"


TIMEPOINTS = tuple(format_timepoint(value) for value in TIME_VALUES)
NORMALIZED_TIMES = {
    timepoint: (value - TIME_VALUES[0]) / (TIME_VALUES[-1] - TIME_VALUES[0])
    for timepoint, value in zip(TIMEPOINTS, TIME_VALUES)
}

# Canonical ids stored in pair labels.  A classifier may use another order;
# classifier_index_lookup maps these ids by class name at loss/evaluation time.
CLASS_NAMES = (
    "MEF/other",
    "MET",
    "Stromal",
    "Epithelial",
    "Trophoblast",
    "Neural",
    "IPS",
)
TRANSITION_EDGES = (
    ("MEF/other", "MET"),
    ("MET", "IPS"),
    ("MEF/other", "Stromal"),
)
UNCONSTRAINED_CLASS_NAMES = ("Epithelial", "Trophoblast", "Neural")
CLASSIFIER_PAIR_MODES = ("endpoint_interpolant", "ot_endpoint_interpolant")

_DATA_CACHE: Dict[Tuple[Any, ...], Dict[str, np.ndarray]] = {}


def classifier_schedule_slug(training_timepoints: Sequence[str | float]) -> str:
    canonical = parse_training_timepoints(training_timepoints)
    return "_".join(value.replace(".", "p") for value in canonical)


def classifier_checkpoint_path(
    training_timepoints: Sequence[str | float] | None = None,
    *,
    all_days: bool = False,
    classifier_dir: str | Path | None = None,
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
) -> Path:
    """Return the canonical Schiebinger classifier checkpoint path."""
    if all_days == (training_timepoints is not None):
        raise ValueError("Specify exactly one of all_days=True or training_timepoints.")
    root = Path(classifier_dir or DEFAULT_CLASSIFIER_DIR).expanduser().resolve()
    variant = (
        "all_days"
        if all_days
        else f"train_days_{classifier_schedule_slug(training_timepoints or ())}"
    )
    return root / (
        f"celltype_classifier_schiebinger_hvg{int(n_hvgs)}_"
        f"pca{int(n_pcs)}_{variant}.pt"
    )


def flow_training_requires_classifier(cfg) -> bool:
    """Whether the selected pair mode or constraint consumes a classifier."""
    pair_mode = str(
        getattr(
            cfg.problem,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        )
    )
    constraints = getattr(cfg, "constraints", None)
    lineage_loss = bool(getattr(constraints, "enabled", False)) and str(
        getattr(constraints, "type", "")
    ) == "maizels_lineage_path"
    return pair_mode in CLASSIFIER_PAIR_MODES or lineage_loss


def resolve_dataset_path(
    dataset_location: str | Path | None,
    filename: str | None = None,
    *,
    n_pcs: int = DEFAULT_N_PCS,
    n_hvgs: int = DEFAULT_N_HVGS,
) -> Path:
    """Resolve a precomputed PCA H5AD directory or explicit file path."""
    if dataset_location in (None, ""):
        location = DEFAULT_DATA_DIR
    else:
        location = Path(str(dataset_location)).expanduser()
    if location.suffix.lower() == ".h5ad":
        return location.resolve()
    selected_filename = filename or pca_dataset_filename(n_pcs, n_hvgs)
    return (location / selected_filename).resolve()


def parse_training_timepoints(
    value: str | Sequence[str | float | int] | None,
) -> Tuple[str, ...]:
    """Parse and validate selected training days in chronological order."""
    if value in (None, ""):
        raw_values: Sequence[str | float | int] = (TIMEPOINTS[0], TIMEPOINTS[-1])
    elif isinstance(value, str):
        raw_values = tuple(token for token in re.split(r"[\s,]+", value) if token)
    else:
        raw_values = value

    selected = tuple(format_timepoint(item) for item in raw_values)
    if len(selected) < 2:
        raise ValueError("Schiebinger training requires at least two timepoints.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate Schiebinger training timepoints: {selected}.")
    unknown = [item for item in selected if item not in NORMALIZED_TIMES]
    if unknown:
        raise ValueError(
            f"Unknown Schiebinger training timepoints {unknown}; choose from "
            f"{list(TIMEPOINTS)}."
        )

    selected_set = set(selected)
    ordered = tuple(item for item in TIMEPOINTS if item in selected_set)
    return ordered


def retained_timepoints(cfg) -> Tuple[str, ...]:
    configured = getattr(cfg.problem, "retained_timepoints", None)
    return parse_training_timepoints(configured)


def experiment_timepoints(cfg) -> Tuple[str, ...]:
    """Observed days inside the window defined by the selected endpoints."""
    retained = retained_timepoints(cfg)
    start, end = float(retained[0]), float(retained[-1])
    return tuple(item for item in TIMEPOINTS if start <= float(item) <= end)


def evaluation_timepoints(cfg) -> Tuple[str, ...]:
    """Unselected observation days strictly inside the experiment window."""
    retained = set(retained_timepoints(cfg))
    return tuple(item for item in experiment_timepoints(cfg) if item not in retained)


def normalized_time(timepoint: str | float, cfg=None) -> float:
    key = format_timepoint(timepoint)
    if cfg is not None:
        order = getattr(cfg.problem, "timepoint_order", None)
        values = getattr(cfg.problem, "timepoint_values", None)
        if order is not None and values is not None:
            mapping = {
                format_timepoint(name): float(value)
                for name, value in zip(list(order), list(values))
            }
            if key not in mapping:
                raise KeyError(f"Unknown configured Schiebinger timepoint {key!r}.")
            return mapping[key]
    if key not in NORMALIZED_TIMES:
        raise KeyError(f"Unknown Schiebinger timepoint {key!r}.")
    return float(NORMALIZED_TIMES[key])


def retained_interval_for_timepoint(cfg, timepoint: str) -> Tuple[str, str]:
    value = float(timepoint)
    retained = retained_timepoints(cfg)
    for source_time, target_time in zip(retained[:-1], retained[1:]):
        if float(source_time) < value < float(target_time):
            return source_time, target_time
    raise ValueError(
        f"Timepoint {timepoint!r} is not inside a retained interval from {retained}."
    )


def _cfg_with_location(cfg, dataset_location: str | Path | None):
    if dataset_location in (None, ""):
        return cfg
    if str(dataset_location) == str(getattr(cfg.problem, "dataset_location", "")):
        return cfg
    resolved = copy.deepcopy(cfg)
    resolved.problem.dataset_location = str(dataset_location)
    return resolved


def _numeric_obs(values) -> np.ndarray:
    try:
        import pandas as pd

        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    except ModuleNotFoundError:
        raw = np.asarray(values)
        result = np.full(raw.shape[0], np.nan, dtype=float)
        for index, value in enumerate(raw):
            try:
                result[index] = float(value)
            except (TypeError, ValueError):
                pass
        return result


def load_dataset(cfg, dataset_location: str | Path | None = None) -> Dict[str, np.ndarray]:
    """Load the configured PCA representation, days, and ``cell_sets`` labels."""
    cfg = _cfg_with_location(cfg, dataset_location)
    path = resolve_dataset_path(
        getattr(cfg.problem, "dataset_location", None),
        str(getattr(cfg.problem, "schiebinger_filename", DEFAULT_FILENAME)),
    )
    cache_key = (
        str(path),
        str(getattr(cfg.problem, "embedding_key", "X_pca")),
        int(getattr(cfg.problem, "n_pcs", 5)),
        int(getattr(cfg.problem, "pca_random_state", 0)),
        bool(getattr(cfg.problem, "whiten_pca", False)),
        bool(getattr(cfg.problem, "subset_to_serum", True)),
        str(getattr(cfg.problem, "time_key", "day")),
        str(getattr(cfg.problem, "cell_type_key", "cell_sets")),
        experiment_timepoints(cfg),
        tuple(getattr(cfg.problem, "timepoint_order", ())),
        tuple(float(value) for value in getattr(cfg.problem, "timepoint_values", ())),
    )
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    # Import lazily to avoid a module cycle: datasets dispatches to this module.
    from . import datasets

    embedding, numeric_times = datasets.load_schiebinger_embedding(cfg)
    adata = datasets._load_schiebinger_anndata(cfg)
    time_key = str(getattr(cfg.problem, "time_key", "day"))
    cell_type_key = str(getattr(cfg.problem, "cell_type_key", "cell_sets"))
    if cell_type_key not in adata.obs:
        raise KeyError(f"Cell-type key {cell_type_key!r} not found in adata.obs.")

    raw_times = _numeric_obs(adata.obs[time_key])
    valid = np.isfinite(raw_times)
    cell_types = (
        adata.obs[cell_type_key].astype("string").astype(str).to_numpy()[valid]
    )
    obs_names = np.asarray(adata.obs_names.astype(str), dtype=object)[valid]
    if embedding.shape[0] != cell_types.shape[0]:
        raise RuntimeError(
            "Schiebinger embedding and cell-type rows became misaligned after "
            "filtering invalid timepoints."
        )

    timepoints = np.asarray(
        [format_timepoint(value) for value in numeric_times], dtype=object
    )
    unknown_times = sorted(set(timepoints) - set(TIMEPOINTS), key=float)
    if unknown_times:
        raise ValueError(f"Unexpected Schiebinger timepoints: {unknown_times}.")
    unknown_types = sorted(set(cell_types) - set(CLASS_NAMES))
    if unknown_types:
        raise ValueError(f"Unexpected Schiebinger cell types: {unknown_types}.")

    # The selected endpoints define the complete experiment window.  Cells
    # before the first selected day or after the last selected day must not
    # enter training, validation, or held-out evaluation.
    included = np.isin(timepoints, np.asarray(experiment_timepoints(cfg)))
    embedding = embedding[included]
    numeric_times = numeric_times[included]
    timepoints = timepoints[included]
    cell_types = cell_types[included]
    obs_names = obs_names[included]

    data = {
        "obs_names": obs_names,
        "x": np.asarray(embedding, dtype=np.float32),
        "timepoints": timepoints,
        "time_values": np.asarray(numeric_times, dtype=np.float32),
        "normalized_times": np.asarray(
            [normalized_time(value, cfg) for value in timepoints], dtype=np.float32
        ),
        "cell_types": np.asarray(cell_types, dtype=object),
    }
    _DATA_CACHE[cache_key] = data
    return data


def all_timepoint_data(dataset_location=None, cfg=None) -> Dict[str, np.ndarray]:
    """Compatibility entry point used by shared lineage evaluation code."""
    if cfg is None:
        from ml_collections import ConfigDict

        cfg = ConfigDict()
        cfg.training = ConfigDict({"seed": 0})
        cfg.problem = ConfigDict(
            {
                "dataset_location": str(dataset_location or DEFAULT_DATA_DIR),
                "schiebinger_filename": DEFAULT_FILENAME,
                "subset_to_serum": False,
                "embedding_key": "X_pca",
                "n_pcs": DEFAULT_N_PCS,
                "pca_random_state": 0,
                "whiten_pca": False,
                "time_key": "day",
                "cell_type_key": "cell_sets",
                "timepoint_order": list(TIMEPOINTS),
                "timepoint_values": [NORMALIZED_TIMES[item] for item in TIMEPOINTS],
            }
        )
    return load_dataset(cfg, dataset_location)


def subset_time(
    data: Dict[str, np.ndarray], timepoint: str | float
) -> Tuple[np.ndarray, np.ndarray]:
    mask = data["timepoints"] == format_timepoint(timepoint)
    return data["x"][mask], data["cell_types"][mask]


def _cell_type_ids(cell_types: np.ndarray) -> np.ndarray:
    values = np.asarray(cell_types)
    if values.dtype.kind in "iu":
        ids = values.astype(np.int32, copy=False)
    else:
        lookup = maizels.class_to_id_map(CLASS_NAMES)
        ids = np.asarray([lookup[str(value)] for value in values], dtype=np.int32)
    if np.any(ids < 0) or np.any(ids >= len(CLASS_NAMES)):
        raise ValueError("Schiebinger pool contains an invalid cell-type id.")
    return ids


def timepoint_pool_splits(cfg, dataset_location=None):
    """Split retained days for training/validation; omitted days remain eval-only."""
    data = load_dataset(cfg, dataset_location)
    retained = set(retained_timepoints(cfg))
    training_seed = int(getattr(getattr(cfg, "training", None), "seed", 0))
    split_seed = int(
        getattr(
            cfg.problem,
            "schiebinger_holdout_seed",
            getattr(cfg.problem, "maizels_holdout_seed", training_seed + 701),
        )
    )
    holdout_fraction = float(
        getattr(
            cfg.problem,
            "schiebinger_holdout_fraction",
            getattr(cfg.problem, "maizels_holdout_fraction", 0.0),
        )
    )
    holdout_n = int(
        getattr(
            cfg.problem,
            "schiebinger_holdout_n",
            getattr(cfg.problem, "maizels_holdout_n", 0),
        )
    )

    result = {}
    for index, timepoint in enumerate(experiment_timepoints(cfg)):
        x, cell_types = subset_time(data, timepoint)
        if x.shape[0] == 0:
            raise RuntimeError(f"Schiebinger day {timepoint} has no cells.")
        if timepoint in retained:
            train_idx, holdout_idx = maizels._split_train_holdout_indices(
                x.shape[0],
                holdout_fraction=holdout_fraction,
                holdout_n=holdout_n,
                seed=split_seed + 101 * (index + 1),
            )
        else:
            # Omitted populations are never used by training or validation.
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
    source_time = format_timepoint(getattr(cfg.problem, "source_time", TIMEPOINTS[0]))
    target_time = format_timepoint(getattr(cfg.problem, "target_time", TIMEPOINTS[-1]))
    return maizels._endpoint_split_dict(pools[source_time], pools[target_time])


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


def _allocate_pairs(total: int, n_intervals: int) -> List[int]:
    return maizels._allocate_pairs(total, n_intervals)


def _add_time_bounds(cfg, paired, source_time: str, target_time: str):
    n_pairs = paired["x0"].shape[0]
    bounds = np.tile(
        np.asarray(
            [normalized_time(source_time, cfg), normalized_time(target_time, cfg)],
            dtype=np.float32,
        ),
        (n_pairs, 1),
    )
    result = dict(paired)
    result["label"] = np.concatenate(
        [np.asarray(paired["label"], dtype=np.float32), bounds], axis=1
    )
    return result


def _make_interval_pairs(
    cfg,
    source_x,
    source_types,
    target_x,
    target_types,
    *,
    source_time: str,
    target_time: str,
    n_pairs: int,
    rng,
    pair_mode: str,
):
    paired, stats = maizels._make_pair_pool_from_endpoint_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=pair_mode,
        class_names=CLASS_NAMES,
        transition_edges=TRANSITION_EDGES,
    )
    paired = _add_time_bounds(cfg, paired, source_time, target_time)
    stats.update(
        {
            "source_time": source_time,
            "target_time": target_time,
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "sampled_pairs": int(paired["x0"].shape[0]),
        }
    )
    return paired, stats


def _make_minibatch_ot_interval_pairs(
    cfg,
    source_x,
    source_types,
    target_x,
    target_types,
    *,
    source_time: str,
    target_time: str,
    n_pairs: int,
    rng,
    pair_mode: str,
    sample_with_replacement: bool = False,
):
    paired, stats = cite_multi._make_minibatch_ot_pairs_from_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=pair_mode,
        sample_with_replacement=sample_with_replacement,
        class_names=CLASS_NAMES,
        transition_edges=TRANSITION_EDGES,
    )
    paired = _add_time_bounds(cfg, paired, source_time, target_time)
    stats.update(
        {
            "source_time": source_time,
            "target_time": target_time,
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "sampled_pairs": int(paired["x0"].shape[0]),
            "ot_cost": "raw_sqeuclidean",
        }
    )
    return paired, stats


def _make_retained_interval_pair_pool(
    cfg,
    n_pairs: int,
    *,
    split: str,
    dataset_location,
    pair_mode: str,
    seed: int,
    minibatch_ot: bool = False,
):
    if split == "train":
        x_key, types_key = "train_x", "train_types"
    elif split in ("holdout", "heldout"):
        split = "heldout"
        x_key, types_key = "holdout_x", "holdout_types"
    else:
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}.")

    pools = timepoint_pool_splits(cfg, dataset_location)
    retained = retained_timepoints(cfg)
    intervals = list(zip(retained[:-1], retained[1:]))
    counts = _allocate_pairs(int(n_pairs), len(intervals))
    rng = np.random.default_rng(int(seed))
    paired_parts = []
    interval_stats = []

    for (source_time, target_time), count in zip(intervals, counts):
        source, target = pools[source_time], pools[target_time]
        builder = (
            _make_minibatch_ot_interval_pairs if minibatch_ot else _make_interval_pairs
        )
        paired, stats = builder(
            cfg,
            source[x_key],
            source[types_key],
            target[x_key],
            target[types_key],
            source_time=source_time,
            target_time=target_time,
            n_pairs=count,
            rng=rng,
            pair_mode=pair_mode,
        )
        stats.update(
            {
                "source_total_n": int(source["x"].shape[0]),
                "source_train_n": int(source["train_x"].shape[0]),
                "source_holdout_n": int(source["holdout_x"].shape[0]),
                "target_total_n": int(target["x"].shape[0]),
                "target_train_n": int(target["train_x"].shape[0]),
                "target_holdout_n": int(target["holdout_x"].shape[0]),
            }
        )
        paired_parts.append(paired)
        interval_stats.append(stats)

    paired = {
        key: np.concatenate([part[key] for part in paired_parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    permutation = rng.permutation(paired["x0"].shape[0])
    paired = {key: value[permutation] for key, value in paired.items()}
    candidate_pairs = sum(int(item["candidate_pairs"]) for item in interval_stats)
    accepted = sum(
        int(item.get("collected_accepted_pairs", item["accepted_pairs"]))
        for item in interval_stats
    )
    stats = {
        "retained_timepoints": list(retained),
        "evaluation_timepoints": list(evaluation_timepoints(cfg)),
        "split": split,
        "pair_mode": pair_mode,
        "sampled_pairs": int(paired["x0"].shape[0]),
        "candidate_pairs": candidate_pairs,
        "endpoint_rejected": sum(int(item["endpoint_rejected"]) for item in interval_stats),
        "interpolant_rejected": sum(
            int(item["interpolant_rejected"]) for item in interval_stats
        ),
        "candidate_acceptance_rate": accepted / candidate_pairs,
        "intervals": {
            f"{item['source_time']}_to_{item['target_time']}".replace(".", "p"): item
            for item in interval_stats
        },
    }
    return paired, stats


def make_pair_pool(cfg, dataset_location=None):
    pair_mode = _training_pair_mode(cfg)
    dynamic_ot = uses_minibatch_ot(cfg, pair_mode)
    paired, stats = _make_retained_interval_pair_pool(
        cfg,
        int(getattr(cfg.problem, "n", 500_000)),
        split="train",
        dataset_location=dataset_location,
        pair_mode="none" if dynamic_ot else pair_mode,
        seed=int(getattr(cfg.training, "seed", 0)) + 301,
    )
    if dynamic_ot:
        canonical = maizels._canonical_maizels_pair_mode(pair_mode)
        stats.update(
            {
                "pair_mode": canonical,
                "pair_pool_mode": "independent_candidates",
                "coupling": "dynamic_minibatch_ot",
                "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
            }
        )
    return paired, stats


def make_minibatch_ot_training_pools(cfg, dataset_location=None):
    pair_mode = maizels._canonical_maizels_pair_mode(_training_pair_mode(cfg))
    if not uses_minibatch_ot(cfg, pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")
    pools = timepoint_pool_splits(cfg, dataset_location)
    retained = retained_timepoints(cfg)
    interval_names = list(zip(retained[:-1], retained[1:]))
    nominal_n = int(getattr(cfg.problem, "n", 500_000))
    counts = _allocate_pairs(nominal_n, len(interval_names))

    timepoints = {}
    for timepoint in retained:
        x = np.asarray(pools[timepoint]["train_x"], dtype=np.float32)
        if x.shape[0] == 0:
            raise RuntimeError(f"Schiebinger training pool for day {timepoint} is empty.")
        timepoints[timepoint] = {
            "x": x,
            "type_ids": _cell_type_ids(pools[timepoint]["train_types"]),
        }

    intervals = []
    interval_stats = {}
    for (source_time, target_time), count in zip(interval_names, counts):
        source, target = pools[source_time], pools[target_time]
        interval = {
            "source_time": source_time,
            "target_time": target_time,
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "nominal_pairs": int(count),
        }
        intervals.append(interval)
        interval_stats[f"{source_time}_to_{target_time}".replace(".", "p")] = {
            **interval,
            "sampled_pairs": int(count),
            "candidate_pairs": int(count),
            "accepted_pairs": int(count),
            "endpoint_rejected": 0,
            "interpolant_rejected": 0,
            "candidate_acceptance_rate": 1.0,
            "source_total_n": int(source["x"].shape[0]),
            "source_train_n": int(source["train_x"].shape[0]),
            "source_holdout_n": int(source["holdout_x"].shape[0]),
            "target_total_n": int(target["x"].shape[0]),
            "target_train_n": int(target["train_x"].shape[0]),
            "target_holdout_n": int(target["holdout_x"].shape[0]),
            "pair_mode": pair_mode,
            "pair_pool_mode": "direct_timepoint_pools",
            "coupling": "dynamic_minibatch_ot",
            "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
            "ot_cost": "raw_sqeuclidean",
        }

    compact = {
        "timepoints": timepoints,
        "intervals": tuple(intervals),
        "nominal_n": nominal_n,
        "dimension": int(next(iter(timepoints.values()))["x"].shape[1]),
        "include_time_bounds": True,
    }
    stored_cells = sum(pool["x"].shape[0] for pool in timepoints.values())
    stored_bytes = sum(
        pool["x"].nbytes + pool["type_ids"].nbytes for pool in timepoints.values()
    )
    stats = {
        "retained_timepoints": list(retained),
        "evaluation_timepoints": list(evaluation_timepoints(cfg)),
        "split": "train",
        "pair_mode": pair_mode,
        "sampled_pairs": nominal_n,
        "candidate_pairs": nominal_n,
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "candidate_acceptance_rate": 1.0,
        "intervals": interval_stats,
        "pair_pool_mode": "direct_timepoint_pools",
        "coupling": "dynamic_minibatch_ot",
        "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
        "ot_cost": "raw_sqeuclidean",
        "stored_endpoint_cells": int(stored_cells),
        "stored_endpoint_bytes": int(stored_bytes),
        "expanded_pair_rows_avoided": nominal_n,
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
    pair_mode = maizels._canonical_maizels_pair_mode(
        pair_mode or _training_pair_mode(cfg)
    )
    intervals = tuple(pools["intervals"])
    counts = _allocate_pairs(int(n_pairs), len(intervals))
    rng = np.random.default_rng(int(seed))
    parts, stats = [], []
    for interval, count in zip(intervals, counts):
        source_time, target_time = interval["source_time"], interval["target_time"]
        source, target = pools["timepoints"][source_time], pools["timepoints"][target_time]
        paired, interval_stats = _make_minibatch_ot_interval_pairs(
            cfg,
            source["x"],
            source["type_ids"],
            target["x"],
            target["type_ids"],
            source_time=source_time,
            target_time=target_time,
            n_pairs=count,
            rng=rng,
            pair_mode=pair_mode,
            sample_with_replacement=True,
        )
        parts.append(paired)
        stats.append(interval_stats)
    result = {
        key: np.concatenate([part[key] for part in parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    permutation = rng.permutation(result["x0"].shape[0])
    result = {key: value[permutation] for key, value in result.items()}
    return result, {
        "pair_mode": pair_mode,
        "coupling": "dynamic_minibatch_ot",
        "pair_pool_mode": "direct_timepoint_pools",
        "sampled_pairs": int(result["x0"].shape[0]),
        "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
        "ot_cost": "raw_sqeuclidean",
        "intervals": stats,
    }


def make_validation_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location=None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    pair_mode = pair_mode or _training_pair_mode(cfg)
    seed = int(seed if seed is not None else int(cfg.training.seed) + 2701)
    return _make_retained_interval_pair_pool(
        cfg,
        int(n_pairs),
        split="heldout",
        dataset_location=dataset_location,
        pair_mode=str(pair_mode),
        seed=seed,
        minibatch_ot=uses_minibatch_ot(cfg, pair_mode),
    )


def make_heldout_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location=None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    splits = endpoint_pool_splits(cfg, dataset_location)
    if splits["source_holdout_n"] == 0 or splits["target_holdout_n"] == 0:
        raise RuntimeError("Schiebinger endpoint holdout pools are empty.")
    pair_mode = pair_mode or _training_pair_mode(cfg)
    seed = int(seed if seed is not None else int(cfg.training.seed) + 997)
    builder = (
        _make_minibatch_ot_interval_pairs
        if uses_minibatch_ot(cfg, pair_mode)
        else _make_interval_pairs
    )
    paired, stats = builder(
        cfg,
        splits["source_holdout_x"],
        splits["source_holdout_types"],
        splits["target_holdout_x"],
        splits["target_holdout_types"],
        source_time=format_timepoint(cfg.problem.source_time),
        target_time=format_timepoint(cfg.problem.target_time),
        n_pairs=int(n_pairs),
        rng=np.random.default_rng(seed),
        pair_mode=str(pair_mode),
    )
    stats.update(
        {
            "source_total_n": splits["source_n"],
            "source_train_n": splits["source_train_n"],
            "source_holdout_n": splits["source_holdout_n"],
            "target_total_n": splits["target_n"],
            "target_train_n": splits["target_train_n"],
            "target_holdout_n": splits["target_holdout_n"],
        }
    )
    return paired, stats


# Shared classifier implementation, with Schiebinger graph semantics supplied
# explicitly where hard filtering or differentiable constraints need them.
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

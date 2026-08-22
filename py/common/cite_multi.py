"""CITE/Multi four-day trajectory data and lineage-aware pair construction.

Training retains three of days 2, 3, 4, and 7 and constructs an equal number
of pairs for each adjacent retained interval.  Each pair label has columns
``[source_type_id, target_type_id, t_start, t_end]`` so the shared flow-map
network is trained on the correct sub-interval of the global time axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import maizels


DATASET_FILES = {
    "cite": "op_cite_inputs_0.h5ad",
    "multi": "op_train_multi_targets_0.h5ad",
}
TIMEPOINTS = ("2", "3", "4", "7")
NORMALIZED_TIMES = {
    timepoint: index / float(len(TIMEPOINTS) - 1)
    for index, timepoint in enumerate(TIMEPOINTS)
}
# Match the exported classifiers' class order so hard path checks and the
# differentiable canonical-to-classifier lookup use identical ids.
CLASS_NAMES = ("BP", "EryP", "HSC", "MasP", "MkP", "MoP", "NeuP")
TRANSITION_EDGES = tuple(
    ("HSC", cell_type) for cell_type in CLASS_NAMES if cell_type != "HSC"
)

_DATA_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def canonical_dataset_name(dataset_name: str | None) -> str:
    name = str(dataset_name or "cite").lower()
    aliases = {
        "cite_seq": "cite",
        "citeseq": "cite",
        "multiome": "multi",
        "multi_seq": "multi",
    }
    name = aliases.get(name, name)
    if name not in DATASET_FILES:
        raise ValueError(
            f"dataset_name must be one of {sorted(DATASET_FILES)}, got {dataset_name!r}."
        )
    return name


def infer_dataset_name(path: str | Path) -> str:
    filename = Path(path).name
    for name, expected in DATASET_FILES.items():
        if filename == expected:
            return name
    lowered = filename.lower()
    if "cite" in lowered:
        return "cite"
    if "multi" in lowered:
        return "multi"
    raise ValueError(
        f"Could not infer CITE/Multi dataset from {filename!r}; set problem.dataset_name."
    )


def resolve_dataset_path(
    dataset_location: str | None,
    dataset_name: str | None = None,
) -> Path:
    name = canonical_dataset_name(dataset_name)
    if dataset_location not in (None, ""):
        location = Path(str(dataset_location)).expanduser()
        if location.suffix == ".h5ad":
            return location.resolve()
        return (location / DATASET_FILES[name]).resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "metric-flow-matching" / "data" / DATASET_FILES[name]).resolve()


def parse_timepoint(value: str) -> float:
    return maizels.parse_timepoint(value)


def normalized_time(timepoint: str) -> float:
    key = str(timepoint)
    if key not in NORMALIZED_TIMES:
        raise KeyError(f"Unknown CITE/Multi timepoint {timepoint!r}.")
    return float(NORMALIZED_TIMES[key])


def load_dataset(dataset_path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(dataset_path).expanduser().resolve()
    cache_key = str(path)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import anndata as ad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reading the CITE/Multi benchmark requires the 'anndata' package."
        ) from exc

    adata = ad.read_h5ad(path, backed="r")
    try:
        if "X_pca" not in adata.obsm:
            raise KeyError(f"{path.name}: missing obsm['X_pca']")
        for key in ("day", "cell_type"):
            if key not in adata.obs:
                raise KeyError(f"{path.name}: missing obs[{key!r}]")
        timepoints = adata.obs["day"].astype("string").astype(str).to_numpy()
        cell_types = adata.obs["cell_type"].astype("string").astype(str).to_numpy()
        x = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        obs_names = np.asarray(adata.obs_names.astype(str), dtype=object)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    unknown_times = sorted(set(timepoints) - set(TIMEPOINTS))
    if unknown_times:
        raise ValueError(f"Unexpected timepoints in {path.name}: {unknown_times}")
    unknown_types = sorted(set(cell_types) - set(CLASS_NAMES))
    if unknown_types:
        raise ValueError(f"Unexpected cell types in {path.name}: {unknown_types}")
    if x.shape[1] != 100:
        raise ValueError(f"{path.name}: expected 100 PCA dimensions, found {x.shape[1]}.")

    data = {
        "obs_names": obs_names,
        "x": x,
        "timepoints": timepoints.astype(object),
        "time_values": np.asarray(
            [parse_timepoint(value) for value in timepoints], dtype=np.float32
        ),
        "normalized_times": np.asarray(
            [normalized_time(value) for value in timepoints], dtype=np.float32
        ),
        "cell_types": cell_types.astype(object),
    }
    _DATA_CACHE[cache_key] = data
    return data


def all_timepoint_data(
    dataset_location: str | None = None,
    dataset_name: str | None = None,
) -> Dict[str, np.ndarray]:
    if dataset_name is None and dataset_location not in (None, ""):
        location = Path(str(dataset_location)).expanduser()
        if location.suffix == ".h5ad":
            dataset_name = infer_dataset_name(location)
    path = resolve_dataset_path(dataset_location, dataset_name)
    return load_dataset(path)


def subset_time(
    data: Dict[str, np.ndarray], timepoint: str
) -> Tuple[np.ndarray, np.ndarray]:
    mask = data["timepoints"] == str(timepoint)
    return data["x"][mask], data["cell_types"][mask]


def retained_timepoints(cfg) -> Tuple[str, str, str]:
    heldout = str(getattr(cfg.problem, "heldout_timepoint", "4"))
    if heldout not in ("3", "4"):
        raise ValueError("problem.heldout_timepoint must be '3' or '4'.")
    return tuple(timepoint for timepoint in TIMEPOINTS if timepoint != heldout)


def _dataset_name_from_cfg(cfg) -> str:
    return canonical_dataset_name(getattr(cfg.problem, "dataset_name", "cite"))


def _dataset_path_from_cfg(cfg, dataset_location: str | None = None) -> Path:
    return resolve_dataset_path(
        dataset_location
        if dataset_location not in (None, "")
        else getattr(cfg.problem, "dataset_location", None),
        _dataset_name_from_cfg(cfg),
    )


def _timepoint_splits(cfg, dataset_location: str | None = None):
    path = _dataset_path_from_cfg(cfg, dataset_location)
    data = load_dataset(path)
    training_seed = int(getattr(getattr(cfg, "training", None), "seed", 0))
    split_seed = int(getattr(cfg.problem, "maizels_holdout_seed", training_seed + 701))
    holdout_fraction = float(getattr(cfg.problem, "maizels_holdout_fraction", 0.0))
    holdout_n = int(getattr(cfg.problem, "maizels_holdout_n", 0))

    result = {}
    for index, timepoint in enumerate(TIMEPOINTS):
        x, cell_types = subset_time(data, timepoint)
        train_idx, holdout_idx = maizels._split_train_holdout_indices(
            x.shape[0],
            holdout_fraction=holdout_fraction,
            holdout_n=holdout_n,
            seed=split_seed + 101 * (index + 1),
        )
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


def endpoint_pool_splits(
    cfg, dataset_location: str | None = None
) -> Dict[str, np.ndarray]:
    pools = _timepoint_splits(cfg, dataset_location)
    source_time = str(getattr(cfg.problem, "source_time", "2"))
    target_time = str(getattr(cfg.problem, "target_time", "7"))
    source = pools[source_time]
    target = pools[target_time]
    return {
        "source_x": source["x"],
        "source_types": source["types"],
        "target_x": target["x"],
        "target_types": target["types"],
        "source_train_x": source["train_x"],
        "source_train_types": source["train_types"],
        "target_train_x": target["train_x"],
        "target_train_types": target["train_types"],
        "source_holdout_x": source["holdout_x"],
        "source_holdout_types": source["holdout_types"],
        "target_holdout_x": target["holdout_x"],
        "target_holdout_types": target["holdout_types"],
        "source_train_idx": source["train_idx"],
        "source_holdout_idx": source["holdout_idx"],
        "target_train_idx": target["train_idx"],
        "target_holdout_idx": target["holdout_idx"],
        "source_n": int(source["x"].shape[0]),
        "target_n": int(target["x"].shape[0]),
        "source_train_n": int(source["train_x"].shape[0]),
        "source_holdout_n": int(source["holdout_x"].shape[0]),
        "target_train_n": int(target["train_x"].shape[0]),
        "target_holdout_n": int(target["holdout_x"].shape[0]),
    }


def _training_pair_mode(cfg) -> str:
    return str(
        getattr(
            cfg.problem,
            "pair_mode",
            getattr(cfg.problem, "maizels_pair_mode", "none"),
        )
    )


def _add_time_bounds(
    paired: Dict[str, np.ndarray], source_time: str, target_time: str
) -> Dict[str, np.ndarray]:
    n_pairs = paired["x0"].shape[0]
    bounds = np.tile(
        np.asarray(
            [normalized_time(source_time), normalized_time(target_time)],
            dtype=np.float32,
        ),
        (n_pairs, 1),
    )
    paired = dict(paired)
    paired["label"] = np.concatenate(
        [paired["label"].astype(np.float32), bounds], axis=1
    )
    return paired


def _make_interval_pairs(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    *,
    source_time: str,
    target_time: str,
    n_pairs: int,
    rng: np.random.Generator,
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
    paired = _add_time_bounds(paired, source_time, target_time)
    stats.update(
        {
            "source_time": str(source_time),
            "target_time": str(target_time),
            "t_start": normalized_time(source_time),
            "t_end": normalized_time(target_time),
            "sampled_pairs": int(paired["x0"].shape[0]),
        }
    )
    return paired, stats


def _allocate_pairs(total: int, n_intervals: int) -> List[int]:
    if total < n_intervals:
        raise ValueError(
            f"problem.n={total} must be at least the number of intervals ({n_intervals})."
        )
    counts = [total // n_intervals] * n_intervals
    for index in range(total % n_intervals):
        counts[index] += 1
    return counts


def make_pair_pool(
    cfg, dataset_location: str | None = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build a balanced pair pool over adjacent retained timepoints."""
    pools = _timepoint_splits(cfg, dataset_location)
    retained = retained_timepoints(cfg)
    intervals = list(zip(retained[:-1], retained[1:]))
    n_total = int(getattr(cfg.problem, "n", 500_000))
    counts = _allocate_pairs(n_total, len(intervals))
    seed = int(getattr(cfg.training, "seed", 0))
    rng = np.random.default_rng(seed + 301)
    pair_mode = _training_pair_mode(cfg)

    paired_parts = []
    interval_stats = []
    for (source_time, target_time), n_pairs in zip(intervals, counts):
        source = pools[source_time]
        target = pools[target_time]
        paired, stats = _make_interval_pairs(
            cfg,
            source["train_x"],
            source["train_types"],
            target["train_x"],
            target["train_types"],
            source_time=source_time,
            target_time=target_time,
            n_pairs=n_pairs,
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

    candidate_pairs = sum(int(stats["candidate_pairs"]) for stats in interval_stats)
    collected = sum(
        int(stats.get("collected_accepted_pairs", stats["accepted_pairs"]))
        for stats in interval_stats
    )
    aggregate = {
        "dataset_name": _dataset_name_from_cfg(cfg),
        "heldout_timepoint": str(getattr(cfg.problem, "heldout_timepoint", "4")),
        "retained_timepoints": list(retained),
        "pair_mode": pair_mode,
        "sampled_pairs": int(paired["x0"].shape[0]),
        "candidate_pairs": candidate_pairs,
        "endpoint_rejected": sum(
            int(stats["endpoint_rejected"]) for stats in interval_stats
        ),
        "interpolant_rejected": sum(
            int(stats["interpolant_rejected"]) for stats in interval_stats
        ),
        "candidate_acceptance_rate": (
            collected / candidate_pairs if candidate_pairs > 0 else 0.0
        ),
        "intervals": {
            f"{stats['source_time']}_to_{stats['target_time']}": stats
            for stats in interval_stats
        },
    }
    return paired, aggregate


def _endpoint_split_arrays(splits, split: str):
    split = str(split).lower()
    if split in ("holdout", "heldout"):
        return (
            splits["source_holdout_x"],
            splits["source_holdout_types"],
            splits["target_holdout_x"],
            splits["target_holdout_types"],
            "heldout",
        )
    if split == "train":
        return (
            splits["source_train_x"],
            splits["source_train_types"],
            splits["target_train_x"],
            splits["target_train_types"],
            "train",
        )
    if split == "all":
        return (
            splits["source_x"],
            splits["source_types"],
            splits["target_x"],
            splits["target_types"],
            "all",
        )
    raise ValueError(f"split must be 'heldout', 'train', or 'all', got {split!r}.")


def make_endpoint_split_pair_pool(
    cfg,
    n_pairs: int,
    *,
    split: str,
    dataset_location: str | None = None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    """Pair global day-2/day-7 endpoints for diagnostics and evaluation."""
    splits = endpoint_pool_splits(cfg, dataset_location)
    source_x, source_types, target_x, target_types, split_name = _endpoint_split_arrays(
        splits, split
    )
    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError(f"CITE/Multi endpoint split {split_name!r} is empty.")
    if pair_mode is None:
        pair_mode = _training_pair_mode(cfg)
    if seed is None:
        seed = int(getattr(cfg.training, "seed", 0)) + 997
    source_time = str(getattr(cfg.problem, "source_time", "2"))
    target_time = str(getattr(cfg.problem, "target_time", "7"))
    paired, stats = _make_interval_pairs(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        source_time=source_time,
        target_time=target_time,
        n_pairs=int(n_pairs),
        rng=np.random.default_rng(int(seed)),
        pair_mode=str(pair_mode),
    )
    stats.update(
        {
            "split": split_name,
            "source_total_n": int(splits["source_n"]),
            "source_train_n": int(splits["source_train_n"]),
            "source_holdout_n": int(splits["source_holdout_n"]),
            "target_total_n": int(splits["target_n"]),
            "target_train_n": int(splits["target_train_n"]),
            "target_holdout_n": int(splits["target_holdout_n"]),
        }
    )
    return paired, stats


def make_heldout_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location: str | None = None,
    pair_mode: str | None = None,
    seed: int | None = None,
):
    """Create global day-2/day-7 diagnostic pairs from held-out endpoint cells."""
    return make_endpoint_split_pair_pool(
        cfg,
        n_pairs,
        split="heldout",
        dataset_location=dataset_location,
        pair_mode=pair_mode,
        seed=seed,
    )


def endpoint_pools(dataset_location: str | None, source_time: str, target_time: str):
    data = all_timepoint_data(dataset_location)
    source_x, source_types = subset_time(data, source_time)
    target_x, target_types = subset_time(data, target_time)
    return source_x, source_types, target_x, target_types


# Shared classifier implementation, with the CITE/Multi lineage graph supplied
# explicitly where graph semantics matter.
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
    )


def check_paths_with_classifier(*args, **kwargs):
    kwargs["transition_edges"] = TRANSITION_EDGES
    return maizels.check_paths_with_classifier(*args, **kwargs)

"""CITE/Multi four-day trajectory data and lineage-aware pair construction.

Training retains three of days 2, 3, 4, and 7 and constructs an equal number
of pairs for each adjacent retained interval.  Each pair label has columns
``[source_type_id, target_type_id, t_start, t_end]`` so the shared flow-map
network is trained on the correct sub-interval of the global time axis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import maizels


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

    return (DEFAULT_DATA_DIR / DATASET_FILES[name]).resolve()


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
    train_fraction = getattr(cfg.problem, "cite_multi_train_fraction", None)
    if train_fraction is not None:
        train_fraction = float(train_fraction)
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(
                "problem.cite_multi_train_fraction must be between 0 and 1."
            )

    result = {}
    for index, timepoint in enumerate(TIMEPOINTS):
        x, cell_types = subset_time(data, timepoint)
        timepoint_holdout_n = holdout_n
        timepoint_holdout_fraction = holdout_fraction
        if train_fraction is not None:
            # Match MFM's ``split_index = int(n_cells * train_fraction)``.
            timepoint_holdout_n = x.shape[0] - int(x.shape[0] * train_fraction)
            timepoint_holdout_fraction = 0.0
        train_idx, holdout_idx = maizels._split_train_holdout_indices(
            x.shape[0],
            holdout_fraction=timepoint_holdout_fraction,
            holdout_n=timepoint_holdout_n,
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


def _canonical_pair_mode(pair_mode: str) -> str:
    return maizels._canonical_maizels_pair_mode(str(pair_mode))


def uses_minibatch_ot(pair_mode: str) -> bool:
    """Return whether CITE/Multi should couple samples with minibatch OT."""
    return _canonical_pair_mode(pair_mode) in {
        "ot_plain",
        "ot_endpoint",
        "ot_endpoint_interpolant",
    }


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


def _squared_euclidean_cost(source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """Match torchcfm's unnormalised squared-Euclidean minibatch cost."""
    source = np.asarray(source_x, dtype=np.float64)
    target = np.asarray(target_x, dtype=np.float64)
    cost = (
        np.sum(source * source, axis=1)[:, None]
        + np.sum(target * target, axis=1)[None, :]
        - 2.0 * source @ target.T
    )
    return np.maximum(cost, 0.0)


def _sample_dense_ot_plan(
    cost: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve and sample the same exact uniform OT plan used by torchcfm."""
    try:
        import ot as pot
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CITE/Multi minibatch OT requires POT (pip install POT==0.9.3)."
        ) from exc

    n_source, n_target = cost.shape
    plan = np.asarray(
        pot.emd(
            pot.unif(n_source),
            pot.unif(n_target),
            cost,
            numItermax=1_000_000,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(plan)) or float(plan.sum()) <= 1e-12:
        raise RuntimeError("CITE/Multi minibatch OT returned an invalid plan.")
    probability = np.maximum(plan.reshape(-1), 0.0)
    probability /= probability.sum()
    chosen = rng.choice(
        probability.shape[0],
        size=int(n_samples),
        replace=True,
        p=probability,
    )
    source_idx, target_idx = np.divmod(chosen, n_target)
    stats = {
        "ot_solver_mode": "exact_minibatch",
        "ot_objective": float(np.sum(plan * cost)),
        "ot_positive_edges": int(np.count_nonzero(plan > 1e-12)),
        "ot_retained_valid_mass": 1.0,
    }
    return source_idx.astype(np.int64), target_idx.astype(np.int64), stats


def _solve_masked_assignment(
    cost: np.ndarray,
    allowed: np.ndarray,
    *,
    infeasible_fallback: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve equal-mass masked OT as a minimum-cost bipartite assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Masked CITE/Multi minibatch OT requires scipy."
        ) from exc

    if cost.shape[0] != cost.shape[1]:
        raise ValueError("Masked minibatch OT expects equal source and target sizes.")
    if not allowed.any():
        raise RuntimeError("Masked minibatch OT has no allowed edges.")
    if infeasible_fallback not in ("error", "partial"):
        raise ValueError(
            "ot_infeasible_fallback must be 'error' or 'partial', "
            f"got {infeasible_fallback!r}."
        )

    n = cost.shape[0]
    max_valid_cost = float(np.max(cost[allowed]))
    # This makes minimizing the number of forbidden assignments dominate every
    # possible difference in transport cost, yielding maximum-cardinality
    # partial OT when a balanced hard-masked plan does not exist.
    forbidden_cost = (max_valid_cost + 1.0) * (n + 1.0)
    assignment_cost = np.where(allowed, cost, forbidden_cost)
    source_idx, target_idx = linear_sum_assignment(assignment_cost)
    selected_valid = allowed[source_idx, target_idx]
    retained = int(np.count_nonzero(selected_valid))
    if retained < n and infeasible_fallback == "error":
        raise RuntimeError(
            "Exact masked CITE/Multi minibatch OT is infeasible: "
            f"only {retained}/{n} assignments satisfy the hard mask."
        )
    if retained == 0:
        raise RuntimeError("Partial masked minibatch OT retained zero valid mass.")

    source_idx = source_idx[selected_valid].astype(np.int64)
    target_idx = target_idx[selected_valid].astype(np.int64)
    mass = np.full(retained, 1.0 / float(retained), dtype=np.float64)
    stats = {
        "ot_solver_mode": (
            "exact_masked_assignment" if retained == n else "max_valid_partial_assignment"
        ),
        "ot_positive_edges": retained,
        "ot_objective": float(np.mean(cost[source_idx, target_idx])),
        "ot_total_mass": 1.0,
        "ot_retained_valid_mass": retained / float(n),
        "ot_source_cells_with_mass": retained,
        "ot_target_cells_with_mass": retained,
    }
    return source_idx, target_idx, mass, stats


def _make_batched_masked_minibatch_ot_pairs(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    *,
    n_pairs: int,
    rng: np.random.Generator,
    pair_mode: str,
    ot_batch_size: int,
    max_attempts: int,
    class_names: Sequence[str] = CLASS_NAMES,
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Solve many masked OT blocks while batching classifier inference."""
    class_names = tuple(str(name) for name in class_names)
    transition_edges = tuple((str(src), str(dst)) for src, dst in transition_edges)
    class_to_id = maizels.class_to_id_map(class_names)
    lineage_transition_mode = maizels.lineage_transition_mode_from_config(cfg)
    reachable = maizels.build_transition_reachable(
        lineage_transition_mode,
        edges=transition_edges,
        class_names=class_names,
    )
    class_mask = np.asarray(
        [
            [maizels.endpoint_valid(src, dst, reachable) for dst in class_names]
            for src in class_names
        ],
        dtype=bool,
    )
    fallback = str(
        getattr(
            cfg.problem,
            "ot_minibatch_infeasible_fallback",
            getattr(cfg.problem, "ot_infeasible_fallback", "partial"),
        )
    )

    def draw_state(output_size: int) -> Dict[str, Any]:
        candidate_size = min(
            ot_batch_size,
            int(output_size),
            source_x.shape[0],
            target_x.shape[0],
        )
        last_error = None
        for _ in range(max_attempts):
            source_draw = rng.choice(
                source_x.shape[0], size=candidate_size, replace=False
            )
            target_draw = rng.choice(
                target_x.shape[0], size=candidate_size, replace=False
            )
            source_batch = source_x[source_draw]
            target_batch = target_x[target_draw]
            source_type_batch = source_types[source_draw]
            target_type_batch = target_types[target_draw]
            source_ids = np.asarray(
                [class_to_id[str(value)] for value in source_type_batch],
                dtype=np.int32,
            )
            target_ids = np.asarray(
                [class_to_id[str(value)] for value in target_type_batch],
                dtype=np.int32,
            )
            allowed = class_mask[source_ids[:, None], target_ids[None, :]]
            if allowed.any():
                return {
                    "source_x": source_batch,
                    "target_x": target_batch,
                    "source_ids": source_ids,
                    "target_ids": target_ids,
                    "cost": _squared_euclidean_cost(source_batch, target_batch),
                    "allowed": allowed,
                    "checked_valid": np.zeros_like(allowed, dtype=bool),
                    "endpoint_rejected": int(
                        allowed.size - np.count_nonzero(allowed)
                    ),
                    "interpolant_checked": 0,
                    "interpolant_rejected": 0,
                    "ot_refinements": 0,
                    "candidate_size": int(candidate_size),
                    "output_size": int(output_size),
                    "done": False,
                }
            last_error = RuntimeError(
                "CITE/Multi masked minibatch OT found no valid endpoints."
            )
        raise RuntimeError(
            "CITE/Multi masked minibatch OT could not draw a valid candidate block."
        ) from last_error

    states = []
    remaining = int(n_pairs)
    while remaining > 0:
        output_size = min(ot_batch_size, remaining)
        states.append(draw_state(output_size))
        remaining -= output_size

    max_refinements = max(
        int(state["candidate_size"]) ** 2 for state in states
    ) + 1
    for _ in range(max_refinements):
        pending = []
        all_check_source = []
        all_check_target = []
        all_start_ids = []
        all_target_ids = []

        for state_index, state in enumerate(states):
            if state["done"]:
                continue
            (
                positive_source,
                positive_target,
                positive_mass,
                solver_stats,
            ) = _solve_masked_assignment(
                state["cost"],
                state["allowed"],
                infeasible_fallback=fallback,
            )
            state["positive_source"] = positive_source
            state["positive_target"] = positive_target
            state["positive_mass"] = positive_mass
            state["solver_stats"] = solver_stats

            if pair_mode != "ot_endpoint_interpolant":
                state["done"] = True
                continue

            needs_check = ~state["checked_valid"][positive_source, positive_target]
            if not needs_check.any():
                state["done"] = True
                continue
            check_source = positive_source[needs_check]
            check_target = positive_target[needs_check]
            pending.append(
                (state_index, check_source, check_target, check_source.shape[0])
            )
            all_check_source.append(state["source_x"][check_source])
            all_check_target.append(state["target_x"][check_target])
            all_start_ids.append(state["source_ids"][check_source])
            all_target_ids.append(state["target_ids"][check_target])

        if all(state["done"] for state in states):
            break
        if not pending:
            raise RuntimeError("Masked minibatch OT refinement made no progress.")

        validity = maizels._check_candidate_interpolants(
            source_x=np.concatenate(all_check_source, axis=0),
            source_type_ids=np.concatenate(all_start_ids, axis=0),
            target_x=np.concatenate(all_check_target, axis=0),
            target_type_ids=np.concatenate(all_target_ids, axis=0),
            classifier_path=maizels.resolve_classifier_path(
                getattr(cfg.problem, "classifier_path", None)
            ),
            n_check_times=max(
                1, int(getattr(cfg.problem, "n_interpolant_check_times", 5))
            ),
            prob_threshold=float(
                getattr(cfg.problem, "classifier_prob_threshold", 0.85)
            ),
            margin_threshold=float(
                getattr(cfg.problem, "classifier_margin_threshold", 1.0)
            ),
            classifier_batch_size=int(
                getattr(cfg.problem, "classifier_batch_size", 8192)
            ),
            lineage_transition_mode=lineage_transition_mode,
            transition_edges=transition_edges,
            path_builder=getattr(cfg.problem, "interpolant_path_builder", None),
        )
        valid_all = np.asarray(validity["valid"], dtype=bool)
        offset = 0
        for state_index, check_source, check_target, count in pending:
            state = states[state_index]
            valid = valid_all[offset : offset + count]
            offset += count
            state["interpolant_checked"] += int(count)
            state["interpolant_rejected"] += int((~valid).sum())
            if valid.any():
                state["checked_valid"][check_source[valid], check_target[valid]] = True
            if (~valid).any():
                state["allowed"][check_source[~valid], check_target[~valid]] = False
            state["ot_refinements"] += 1
            if valid.all():
                state["done"] = True
    else:
        raise RuntimeError(
            "CITE/Multi masked minibatch OT exceeded its refinement bound."
        )

    paired_parts = []
    block_stats = []
    for state in states:
        chosen = rng.choice(
            state["positive_mass"].shape[0],
            size=state["output_size"],
            replace=True,
            p=state["positive_mass"],
        )
        source_idx = state["positive_source"][chosen]
        target_idx = state["positive_target"][chosen]
        paired_parts.append(
            {
                "x0": state["source_x"][source_idx].astype(np.float32, copy=False),
                "x1": state["target_x"][target_idx].astype(np.float32, copy=False),
                "label": np.stack(
                    [state["source_ids"][source_idx], state["target_ids"][target_idx]],
                    axis=1,
                ).astype(np.int32),
            }
        )
        block_stats.append(
            {
                "candidate_pairs": int(state["allowed"].size),
                "endpoint_rejected": int(state["endpoint_rejected"]),
                "interpolant_checked": int(state["interpolant_checked"]),
                "interpolant_rejected": int(state["interpolant_rejected"]),
                "ot_refinements": int(state["ot_refinements"]),
                "sampled_pairs": int(state["output_size"]),
                **state["solver_stats"],
            }
        )

    paired = {
        key: np.concatenate([part[key] for part in paired_parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    summary = {
        "pair_mode": pair_mode,
        "coupling": "minibatch_ot",
        "ot_minibatch_size": max(
            int(state["candidate_size"]) for state in states
        ),
        "ot_configured_minibatch_size": ot_batch_size,
        "ot_minibatches": len(block_stats),
        "sampled_pairs": int(paired["x0"].shape[0]),
        "candidate_pairs": sum(int(item["candidate_pairs"]) for item in block_stats),
        "endpoint_rejected": sum(
            int(item["endpoint_rejected"]) for item in block_stats
        ),
        "interpolant_checked": sum(
            int(item["interpolant_checked"]) for item in block_stats
        ),
        "interpolant_rejected": sum(
            int(item["interpolant_rejected"]) for item in block_stats
        ),
        "accepted_pairs": int(paired["x0"].shape[0]),
        "collected_accepted_pairs": int(paired["x0"].shape[0]),
        "ot_refinements": sum(int(item["ot_refinements"]) for item in block_stats),
        "ot_mean_retained_valid_mass": float(
            np.mean(
                [float(item.get("ot_retained_valid_mass", 1.0)) for item in block_stats]
            )
        ),
    }
    summary["candidate_acceptance_rate"] = (
        summary["accepted_pairs"] / summary["candidate_pairs"]
        if summary["candidate_pairs"] > 0
        else 0.0
    )
    return paired, summary


def _make_minibatch_ot_pairs_from_arrays(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    *,
    n_pairs: int,
    rng: np.random.Generator,
    pair_mode: str,
    class_names: Sequence[str] = CLASS_NAMES,
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Draw independent minibatches, solve OT, and sample coupled pairs."""
    pair_mode = _canonical_pair_mode(pair_mode)
    if not uses_minibatch_ot(pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} is not an OT pair mode.")
    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError("CITE/Multi minibatch OT received an empty population.")

    ot_batch_size = max(1, int(getattr(cfg.problem, "ot_minibatch_size", 128)))
    max_attempts = max(1, int(getattr(cfg.problem, "ot_minibatch_max_resamples", 20)))
    if pair_mode != "ot_plain":
        return _make_batched_masked_minibatch_ot_pairs(
            cfg,
            source_x,
            source_types,
            target_x,
            target_types,
            n_pairs=n_pairs,
            rng=rng,
            pair_mode=pair_mode,
            ot_batch_size=ot_batch_size,
            max_attempts=max_attempts,
            class_names=class_names,
            transition_edges=transition_edges,
        )

    class_to_id = maizels.class_to_id_map(class_names)
    paired_parts = []
    block_stats = []
    remaining = int(n_pairs)

    while remaining > 0:
        output_size = min(ot_batch_size, remaining)
        candidate_size = min(
            ot_batch_size,
            output_size,
            source_x.shape[0],
            target_x.shape[0],
        )
        last_error = None
        for _ in range(max_attempts):
            source_draw = rng.choice(
                source_x.shape[0], size=candidate_size, replace=False
            )
            target_draw = rng.choice(
                target_x.shape[0], size=candidate_size, replace=False
            )
            source_batch = source_x[source_draw]
            target_batch = target_x[target_draw]
            source_type_batch = source_types[source_draw]
            target_type_batch = target_types[target_draw]
            try:
                cost = _squared_euclidean_cost(source_batch, target_batch)
                source_idx, target_idx, stats = _sample_dense_ot_plan(
                    cost, output_size, rng
                )
                stats.update(
                    {
                        "candidate_pairs": int(cost.size),
                        "endpoint_rejected": 0,
                        "interpolant_checked": 0,
                        "interpolant_rejected": 0,
                        "ot_refinements": 0,
                        "candidate_size": int(candidate_size),
                    }
                )
                break
            except RuntimeError as exc:
                last_error = exc
        else:
            raise RuntimeError(
                "CITE/Multi minibatch OT could not construct a feasible block "
                f"after {max_attempts} attempts."
            ) from last_error

        source_ids = np.asarray(
            [class_to_id[str(value)] for value in source_type_batch[source_idx]],
            dtype=np.int32,
        )
        target_ids = np.asarray(
            [class_to_id[str(value)] for value in target_type_batch[target_idx]],
            dtype=np.int32,
        )
        paired_parts.append(
            {
                "x0": source_batch[source_idx].astype(np.float32, copy=False),
                "x1": target_batch[target_idx].astype(np.float32, copy=False),
                "label": np.stack([source_ids, target_ids], axis=1),
            }
        )
        stats["sampled_pairs"] = int(output_size)
        block_stats.append(stats)
        remaining -= output_size

    paired = {
        key: np.concatenate([part[key] for part in paired_parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    summary = {
        "pair_mode": pair_mode,
        "coupling": "minibatch_ot",
        "ot_minibatch_size": max(
            int(item["candidate_size"]) for item in block_stats
        ),
        "ot_configured_minibatch_size": ot_batch_size,
        "ot_minibatches": len(block_stats),
        "sampled_pairs": int(paired["x0"].shape[0]),
        "candidate_pairs": sum(int(item["candidate_pairs"]) for item in block_stats),
        "endpoint_rejected": sum(
            int(item["endpoint_rejected"]) for item in block_stats
        ),
        "interpolant_checked": sum(
            int(item["interpolant_checked"]) for item in block_stats
        ),
        "interpolant_rejected": sum(
            int(item["interpolant_rejected"]) for item in block_stats
        ),
        "accepted_pairs": int(paired["x0"].shape[0]),
        "collected_accepted_pairs": int(paired["x0"].shape[0]),
        "ot_refinements": sum(int(item["ot_refinements"]) for item in block_stats),
        "ot_mean_retained_valid_mass": float(
            np.mean(
                [float(item.get("ot_retained_valid_mass", 1.0)) for item in block_stats]
            )
        ),
    }
    summary["candidate_acceptance_rate"] = (
        summary["accepted_pairs"] / summary["candidate_pairs"]
        if summary["candidate_pairs"] > 0
        else 0.0
    )
    return paired, summary


def _make_minibatch_ot_interval_pairs(
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    paired, stats = _make_minibatch_ot_pairs_from_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=n_pairs,
        rng=rng,
        pair_mode=pair_mode,
    )
    paired = _add_time_bounds(paired, source_time, target_time)
    stats.update(
        {
            "source_time": str(source_time),
            "target_time": str(target_time),
            "t_start": normalized_time(source_time),
            "t_end": normalized_time(target_time),
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


def _make_retained_interval_pair_pool(
    cfg,
    n_pairs: int,
    *,
    split: str,
    dataset_location: str | None,
    pair_mode: str,
    seed: int,
    minibatch_ot: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build pairs over retained intervals from one timepoint split."""
    split = str(split).lower()
    if split == "train":
        x_key, types_key = "train_x", "train_types"
    elif split in ("holdout", "heldout"):
        split = "heldout"
        x_key, types_key = "holdout_x", "holdout_types"
    else:
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}.")

    pools = _timepoint_splits(cfg, dataset_location)
    retained = retained_timepoints(cfg)
    intervals = list(zip(retained[:-1], retained[1:]))
    counts = _allocate_pairs(int(n_pairs), len(intervals))
    rng = np.random.default_rng(int(seed))

    paired_parts = []
    interval_stats = []
    for (source_time, target_time), n_pairs in zip(intervals, counts):
        source = pools[source_time]
        target = pools[target_time]
        pair_builder = (
            _make_minibatch_ot_interval_pairs if minibatch_ot else _make_interval_pairs
        )
        paired, stats = pair_builder(
            cfg,
            source[x_key],
            source[types_key],
            target[x_key],
            target[types_key],
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
        "split": split,
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


def make_pair_pool(
    cfg, dataset_location: str | None = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build training pairs over adjacent retained timepoints."""
    pair_mode = _training_pair_mode(cfg)
    minibatch_ot = uses_minibatch_ot(pair_mode)
    paired, stats = _make_retained_interval_pair_pool(
        cfg,
        int(getattr(cfg.problem, "n", 500_000)),
        split="train",
        dataset_location=dataset_location,
        # Dynamic minibatch OT starts from independent source and target pools.
        pair_mode="none" if minibatch_ot else pair_mode,
        seed=int(getattr(cfg.training, "seed", 0)) + 301,
    )
    if minibatch_ot:
        canonical_mode = _canonical_pair_mode(pair_mode)
        for interval_stats in stats["intervals"].values():
            interval_stats.update(
                {
                    "pair_mode": canonical_mode,
                    "pair_pool_mode": "independent_candidates",
                    "coupling": "dynamic_minibatch_ot",
                    "ot_minibatch_size": int(
                        getattr(cfg.problem, "ot_minibatch_size", 128)
                    ),
                }
            )
        stats.update(
            {
                "pair_mode": canonical_mode,
                "pair_pool_mode": "independent_candidates",
                "coupling": "dynamic_minibatch_ot",
                "ot_minibatch_size": int(
                    getattr(cfg.problem, "ot_minibatch_size", 128)
                ),
            }
        )
    return paired, stats


def make_validation_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location: str | None = None,
    pair_mode: str | None = None,
    seed: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Build validation pairs from the held-out 10% of retained timepoints."""
    if pair_mode is None:
        pair_mode = _training_pair_mode(cfg)
    if seed is None:
        seed = int(getattr(cfg.training, "seed", 0)) + 2701
    minibatch_ot = uses_minibatch_ot(pair_mode)
    return _make_retained_interval_pair_pool(
        cfg,
        int(n_pairs),
        split="heldout",
        dataset_location=dataset_location,
        pair_mode=str(pair_mode),
        seed=int(seed),
        minibatch_ot=minibatch_ot,
    )


def couple_minibatch_ot_pair_pool(
    cfg,
    paired: Dict[str, np.ndarray],
    n_pairs: int,
    *,
    seed: int,
    pair_mode: str | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Construct a balanced optimizer batch from an independent pair pool."""
    if pair_mode is None:
        pair_mode = _training_pair_mode(cfg)
    pair_mode = _canonical_pair_mode(pair_mode)
    if not uses_minibatch_ot(pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")

    labels = np.asarray(paired["label"])
    if labels.ndim != 2 or labels.shape[1] < 4:
        raise ValueError("CITE/Multi minibatch OT requires four-column pair labels.")
    interval_bounds = np.unique(labels[:, 2:4], axis=0)
    interval_bounds = interval_bounds[np.argsort(interval_bounds[:, 0])]
    counts = _allocate_pairs(int(n_pairs), int(interval_bounds.shape[0]))
    rng = np.random.default_rng(int(seed))
    class_names = np.asarray(CLASS_NAMES, dtype=object)
    paired_parts = []
    interval_stats = []

    for bounds, interval_n in zip(interval_bounds, counts):
        in_interval = np.all(np.isclose(labels[:, 2:4], bounds[None, :]), axis=1)
        if not in_interval.any():
            raise RuntimeError(
                f"CITE/Multi candidate pool has no rows for interval {bounds.tolist()}."
            )
        source_x = np.asarray(paired["x0"])[in_interval]
        target_x = np.asarray(paired["x1"])[in_interval]
        source_types = class_names[labels[in_interval, 0].astype(np.int32)]
        target_types = class_names[labels[in_interval, 1].astype(np.int32)]
        interval_pairs, stats = _make_minibatch_ot_pairs_from_arrays(
            cfg,
            source_x,
            source_types,
            target_x,
            target_types,
            n_pairs=interval_n,
            rng=rng,
            pair_mode=pair_mode,
        )
        time_bounds = np.tile(
            np.asarray(bounds, dtype=np.float32),
            (interval_pairs["x0"].shape[0], 1),
        )
        interval_pairs["label"] = np.concatenate(
            [interval_pairs["label"].astype(np.float32), time_bounds], axis=1
        )
        stats.update(
            {
                "t_start": float(bounds[0]),
                "t_end": float(bounds[1]),
            }
        )
        paired_parts.append(interval_pairs)
        interval_stats.append(stats)

    result = {
        key: np.concatenate([part[key] for part in paired_parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    permutation = rng.permutation(result["x0"].shape[0])
    result = {key: value[permutation] for key, value in result.items()}
    return result, {
        "pair_mode": pair_mode,
        "coupling": "dynamic_minibatch_ot",
        "sampled_pairs": int(result["x0"].shape[0]),
        "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
        "intervals": interval_stats,
    }


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
    pair_builder = (
        _make_minibatch_ot_interval_pairs
        if uses_minibatch_ot(pair_mode)
        else _make_interval_pairs
    )
    paired, stats = pair_builder(
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

"""Calderon population/ensemble adapter for the shared Maizels SSFM trainer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import calderon, calderon_logging, cite_multi, maizels

# The existing adapter aligns the 23-class training ensemble and 24-class
# evaluation ensemble by tissue name, and uses the supplied projected relation.
class_names = calderon_logging.class_names
retained_timepoints = calderon_logging.retained_timepoints
normalized_time = calderon_logging.normalized_time
retained_interval_for_timepoint = calderon_logging.retained_interval_for_timepoint
timepoint_pool_splits = calderon_logging.timepoint_pool_splits
load_jax_classifier_params = calderon_logging.load_jax_classifier_params
jax_classifier_logits = calderon_logging.jax_classifier_logits
classifier_index_lookup = calderon_logging.classifier_index_lookup
lineage_transition_mode_from_config = calderon_logging.lineage_transition_mode_from_config
lineage_invalid_transition_matrix = calderon_logging.lineage_invalid_transition_matrix
lineage_soft_terms_from_probs = calderon_logging.lineage_soft_terms_from_probs
check_paths_with_classifier = calderon_logging.check_paths_with_classifier


def stochastic_classifier_path(path):
    """Calderon classifier paths identify manifests, not individual NPZ MLPs."""
    return Path(path).expanduser().resolve()


def uses_minibatch_ot_config(cfg):
    return (cfg.problem.maizels_pair_mode in ("ot_plain", "ot_endpoint")
            and cfg.problem.maizels_ot_coupling == "minibatch_ot")


def _compact_pools(cfg, split):
    data = calderon_logging._context(cfg)["data"]
    times = tuple(retained_timepoints(cfg))
    pools = {window: data.pool(window, split) for window in times}
    intervals = tuple(dict(source_time=a, target_time=b, t_start=normalized_time(a, cfg),
                           t_end=normalized_time(b, cfg)) for a, b in zip(times[:-1], times[1:]))
    stored_cells = sum(len(pool["x"]) for pool in pools.values())
    stored_bytes = sum(pool["x"].nbytes + pool["type_ids"].nbytes for pool in pools.values())
    compact = dict(timepoints=pools, intervals=intervals, nominal_n=int(cfg.problem.n),
                   dimension=cfg.problem.d, include_time_bounds=True, class_names=data.class_names)
    stats = dict(split=split, retained_timepoints=list(times), pair_mode=cfg.problem.maizels_pair_mode,
                 pair_pool_mode="direct_timepoint_pools", coupling="dynamic_minibatch_ot",
                 stored_endpoint_cells=stored_cells, stored_endpoint_bytes=stored_bytes,
                 expanded_pair_rows_avoided=int(cfg.problem.n), intervals=list(intervals),
                 ot_minibatch_size=int(cfg.problem.ot_minibatch_size), ot_cost="raw_sqeuclidean")
    return compact, stats


def make_minibatch_ot_training_pools(cfg, dataset_location=None):
    if not uses_minibatch_ot_config(cfg):
        raise ValueError("Calderon stochastic training requires plain or endpoint-filtered minibatch OT.")
    return _compact_pools(cfg, "train")


def couple_minibatch_ot_timepoint_pools(cfg, pools, n_pairs, *, seed, pair_mode=None):
    """Recompute exact OT on independent cell draws for every optimizer batch."""
    pair_mode = cfg.problem.maizels_pair_mode if pair_mode is None else pair_mode
    if pair_mode not in ("ot_plain", "ot_endpoint"):
        raise ValueError("Stochastic OT filters annotated endpoints only; use ot_plain or ot_endpoint.")
    names = tuple(pools["class_names"])
    allowed = (calderon.load_allowed_pairs(cfg.classifier.allowed_pairs_path, names, allow_extra_classes=True)
               if pair_mode == "ot_endpoint" else None)
    intervals = tuple(pools["intervals"])
    counts = maizels._allocate_pairs(int(n_pairs), len(intervals))
    rng = np.random.default_rng(int(seed))
    parts, interval_stats = [], []
    for interval, count in zip(intervals, counts):
        source = pools["timepoints"][interval["source_time"]]
        target = pools["timepoints"][interval["target_time"]]
        # Use the same exact uniform OT and masked-assignment/partial-mass
        # solver as the deterministic Calderon adapter. No straight-line
        # classifier checks are applied to stochastic training pairs.
        paired, stats = cite_multi._make_minibatch_ot_pairs_from_arrays(
            cfg, source["x"], source["type_ids"], target["x"], target["type_ids"],
            n_pairs=count, rng=rng, pair_mode=pair_mode, sample_with_replacement=True,
            class_names=names, transition_edges=(), allowed_class_mask=allowed,
        )
        bounds = np.tile(np.asarray([interval["t_start"], interval["t_end"]], dtype=np.float32), (count, 1))
        paired["label"] = np.concatenate([paired["label"].astype(np.float32), bounds], axis=1)
        parts.append(paired)
        interval_stats.append({**stats, **interval})
    batch = {key: np.concatenate([part[key] for part in parts]) for key in ("x0", "x1", "label")}
    permutation = rng.permutation(len(batch["x0"]))
    batch = {key: value[permutation] for key, value in batch.items()}
    return batch, dict(pair_mode=pair_mode, coupling="dynamic_minibatch_ot", pair_pool_mode="direct_timepoint_pools",
                       sampled_pairs=len(batch["x0"]), ot_minibatch_size=int(cfg.problem.ot_minibatch_size),
                       ot_cost="raw_sqeuclidean", intervals=interval_stats)


def make_validation_pair_pool(cfg, n_pairs, *, dataset_location=None, pair_mode=None, seed=None):
    pools, _ = _compact_pools(cfg, "validation")
    return couple_minibatch_ot_timepoint_pools(
        cfg, pools, n_pairs, pair_mode=pair_mode,
        seed=cfg.logging.maizels.validation_seed if seed is None else seed,
    )

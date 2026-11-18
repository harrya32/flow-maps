"""Calderon populations/classifiers for the existing Maizels diagnostics.

This adapter keeps the native validation batching, sampling, plotting and
logging functions. Tissue relations are always used exactly as supplied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import calderon

DEFAULT_CLASSIFIER = calderon.DEFAULT_TRAINING_MANIFEST
_CONTEXTS = {}
_MANIFEST_CONTEXTS = {}
_CLASS_CONTEXTS = {}
_MODELS = {}


def _key(cfg):
    return (str(Path(cfg.problem.dataset_location).resolve()), tuple(calderon.training_windows(cfg)),
            str(cfg.problem.cohort_branch_prefix), int(cfg.problem.maizels_holdout_seed),
            float(cfg.problem.maizels_holdout_fraction), int(cfg.problem.maizels_holdout_n),
            str(cfg.classifier.manifest_path), str(calderon.evaluation_manifest_path(cfg)))


def register(cfg, data):
    context = {"cfg": cfg, "data": data}
    _CONTEXTS[_key(cfg)] = context
    for path in (cfg.classifier.manifest_path, calderon.evaluation_manifest_path(cfg)):
        _MANIFEST_CONTEXTS[str(Path(path).resolve())] = context
        names = tuple(calderon.load_tissue_manifest(path)["class_names"])
        _CLASS_CONTEXTS[names] = context
    return context


def _context(cfg):
    key = _key(cfg)
    return _CONTEXTS[key] if key in _CONTEXTS else register(cfg, calderon.load_endpoint_experiment(cfg))


def class_names(cfg):
    return _context(cfg)["data"].class_names


def retained_timepoints(cfg):
    return calderon.training_windows(cfg)


def normalized_time(window, cfg):
    return calderon.normalized_window_time(window, cfg)


def retained_interval_for_timepoint(cfg, window):
    times = retained_timepoints(cfg)
    tau = normalized_time(window, cfg)
    for left, right in zip(times[:-1], times[1:]):
        if normalized_time(left, cfg) <= tau <= normalized_time(right, cfg):
            return left, right
    raise ValueError(f"No retained interval contains {window}.")


def timepoint_pool_splits(cfg, dataset_location=None):
    data = _context(cfg)["data"]
    pools = {}
    for window, marginal in data.marginals.items():
        labels = marginal.metadata.broad_tissue.to_numpy(dtype=str)
        train = data.splits[window]["train"] if window in data.splits else np.arange(len(labels))
        holdout = data.splits[window]["validation"] if window in data.splits else np.array([], dtype=int)
        pools[window] = {"x": marginal.x, "types": labels, "train_x": marginal.x[train],
                         "train_types": labels[train], "holdout_x": marginal.x[holdout],
                         "holdout_types": labels[holdout]}
    return pools


def endpoint_pool_splits(cfg, dataset_location=None):
    pools = timepoint_pool_splits(cfg, dataset_location)
    result = {}
    for role, window in (("source", cfg.problem.source_time), ("target", cfg.problem.target_time)):
        for name, value in pools[window].items():
            result[f"{role}_{name}"] = value
        result[f"{role}_train_n"] = len(pools[window]["train_x"])
        result[f"{role}_holdout_n"] = len(pools[window]["holdout_x"])
    return result


def all_timepoint_data(dataset_location, cfg=None):
    if cfg is None:
        dataset = str(Path(dataset_location).resolve())
        matches = [context for key, context in _CONTEXTS.items() if key[0] == dataset]
        if not matches:
            raise ValueError("Register the Calderon experiment before requesting populations.")
        data = matches[-1]["data"]
    else:
        data = _context(cfg)["data"]
    return {"x": np.concatenate([m.x for m in data.marginals.values()]),
            "timepoints": np.concatenate([np.repeat(window, len(m.x)) for window, m in data.marginals.items()]),
            "time_values": np.concatenate([np.repeat(calderon.window_midpoint(window), len(m.x))
                                           for window, m in data.marginals.items()])}


def _pair_config(cfg, pair_mode):
    from ml_collections import ConfigDict, FrozenConfigDict
    copy = ConfigDict(cfg.to_dict())
    copy.problem.maizels_pair_mode = cfg.problem.maizels_pair_mode if pair_mode == "same_as_training" else pair_mode
    return FrozenConfigDict(copy)


def make_validation_pair_pool(cfg, n_pairs, *, dataset_location=None, pair_mode=None, seed=None):
    import jax
    data = _context(cfg)["data"]
    copy = _pair_config(cfg, pair_mode or cfg.problem.maizels_pair_mode)
    ds = calderon.training_dataset(copy, data, "validation")
    pairs = ds.sample_device_batch(n_pairs, jax.random.PRNGKey(cfg.logging.maizels.validation_seed if seed is None else seed))
    return {key: np.asarray(value) for key, value in pairs.items()}, ds.last_pair_stats


def make_heldout_pair_pool(cfg, n_pairs, *, dataset_location=None, pair_mode=None, seed=None):
    import jax
    data = _context(cfg)["data"]
    copy = _pair_config(cfg, pair_mode or cfg.problem.maizels_pair_mode)
    ds = calderon.EndpointOTDataset(copy, data.pool(data.source_time, "validation"),
        data.pool(data.target_time, "validation"), data.class_names, time_bounds=(0., 1.))
    pairs = ds.sample_device_batch(n_pairs, jax.random.PRNGKey(cfg.logging.maizels.plot_seed if seed is None else seed))
    return {key: np.asarray(value) for key, value in pairs.items()}, ds.last_pair_stats


def _model(path, batch_size=8192):
    from .calderon_lineage import DifferentiableTissueEnsemble
    path = str(Path(path).resolve())
    context = _MANIFEST_CONTEXTS[path]
    cfg = context["cfg"]
    export = (cfg.classifier.jax_export_path if path == str(Path(cfg.classifier.manifest_path).resolve())
              else cfg.classifier.evaluation_jax_export_path)
    key = (path, str(export), int(batch_size))
    if key not in _MODELS:
        _MODELS[key] = DifferentiableTissueEnsemble(export, path, batch_size)
    return _MODELS[key]


def load_jax_classifier_params(classifier_path):
    model = _model(classifier_path)
    return model, model.class_names, None, None


def jax_classifier_logits(params, scaler_mean, scaler_scale, x_pca):
    return params.log_probabilities(x_pca)


def classifier_index_lookup(names):
    canonical = _CLASS_CONTEXTS[tuple(names)]["data"].class_names
    return np.array([tuple(names).index(name) for name in canonical], dtype=np.int32)


def lineage_transition_mode_from_config(cfg):
    return cfg.problem.lineage_transition_mode


def lineage_invalid_transition_matrix(names, transition_mode=None):
    context = _CLASS_CONTEXTS[tuple(names)]
    return ~calderon.load_allowed_pairs(context["cfg"].classifier.allowed_pairs_path, names, allow_extra_classes=True)


def lineage_soft_terms_from_probs(*args, **kwargs):
    from .maizels import lineage_soft_terms_from_probs as native
    return native(*args, **kwargs)


def check_paths_with_classifier(paths, start_type_ids, classifier_path, prob_threshold=0., margin_threshold=0.,
                                final_type_ids=None, classifier_batch_size=8192, lineage_transition_mode=None):
    model = _model(classifier_path, classifier_batch_size)
    probabilities = model.predict_proba(np.asarray(paths).reshape(-1, paths.shape[-1])).reshape(
        paths.shape[0], paths.shape[1], len(model.class_names))
    lookup = classifier_index_lookup(model.class_names)
    starts = lookup[np.asarray(start_type_ids, dtype=int)]
    allowed = ~lineage_invalid_transition_matrix(model.class_names)
    if final_type_ids is not None:
        return calderon.interpolant_validity(starts, lookup[np.asarray(final_type_ids, dtype=int)],
                                            probabilities, allowed, prob_threshold, margin_threshold)
    result = calderon.score_tissue_paths(starts, probabilities, allowed, prob_threshold, margin_threshold)
    rejected = result["first_rejection"] >= 0
    return {"valid": ~rejected, "confident": result["confident"], "rejected_at_checkpoint": rejected,
            "rejected_at_final": np.zeros_like(rejected), "n_confident": int(result["confident"].sum()),
            "n_points": int(result["confident"].size)}

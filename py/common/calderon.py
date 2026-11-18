"""Calderon marginal populations, OT pairs and tissue trajectory diagnostics.

This module reads only stored PCA100 scores and metadata, never the full RNA
count matrix. The filtered variant uses the frozen training classifier to select
pairs; cell splits and PCA coordinates are shared across the flow variants.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/calderon2022/all_times/lineage_assigned_hvg3000_pca100_all_times.h5ad"
DEFAULT_TRAINING_MANIFEST = ROOT / "calderon-classifiers/symmetric_4_8_12/models/symmetric_4_8_12_broad_tissue_selected.json"
DEFAULT_EVALUATION_MANIFEST = ROOT / "calderon-classifiers/all_times/models/all_times_broad_tissue_selected.json"
DEFAULT_SPLIT_MANIFEST = ROOT / "datasets/calderon2022/symmetric_4_8_12/cell_splits.csv.gz"
DEFAULT_TRAINING_EXPORT = ROOT / ".cache/calderon2022/symmetric_4_8_12_tissue_ensemble_jax.npz"
DEFAULT_EVALUATION_EXPORT = ROOT / ".cache/calderon2022/all_times_tissue_ensemble_jax.npz"


def uses_maizels_protocol(cfg):
    return cfg.evaluation.get("protocol") == "maizels"


def evaluation_manifest_path(cfg):
    # Saved endpoint runs predate the separation of classifier roles.
    return cfg.classifier.get("evaluation_manifest_path", cfg.classifier.manifest_path)


def window_midpoint(window):
    left, right = map(float, window.split("-"))
    if not left < right:
        raise ValueError(f"Invalid time window: {window}.")
    return (left + right) / 2


def training_windows(cfg):
    return tuple(cfg.problem.get("retained_timepoints", (cfg.problem.source_time, cfg.problem.target_time)))


def heldout_windows(cfg):
    if cfg.problem.get("retained_timepoints"):
        return tuple(cfg.problem.evaluation_timepoints)
    return (cfg.problem.reference_time,)


def normalized_window_time(window, cfg):
    return (window_midpoint(window) - cfg.problem.source_hours) / (cfg.problem.target_hours - cfg.problem.source_hours)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tissue_manifest(path, expected_variant=None):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get("target") != "broad_tissue":
        raise ValueError("Use a selected broad-tissue classifier.")
    if expected_variant is not None and spec.get("variant") != expected_variant:
        raise ValueError(f"Expected the {expected_variant} tissue classifier.")
    if spec.get("n_pcs") != 100 or len(spec["scaler_mean"]) != 100 or len(spec["scaler_scale"]) != 100:
        raise ValueError("The selected tissue classifier must use PCA100.")
    classes = tuple(spec["class_names"])
    if len(classes) != len(set(classes)):
        raise ValueError("Duplicate classifier class names.")
    if not np.isfinite(spec["scaler_mean"]).all() or not np.isfinite(spec["scaler_scale"]).all() or min(spec["scaler_scale"]) <= 0:
        raise ValueError("Invalid classifier scaler.")
    if not spec.get("members"):
        raise ValueError("The classifier manifest has no members.")
    temperatures = [spec.get("temperature", 1.), *[member.get("temperature", 1.) for member in spec["members"]]]
    if not np.isfinite(temperatures).all() or min(temperatures) <= 0:
        raise ValueError("Classifier temperatures must be positive and finite.")
    for member in spec["members"]:
        checkpoint = (path.parent / member["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    return spec


def load_allowed_pairs(path, class_names, *, allow_extra_classes=False):
    """Load the projected relation as supplied, without another closure."""
    frame = pd.read_csv(path)
    if list(frame.columns) != ["source", "target"]:
        raise ValueError("Allowed-pair CSV must have source,target columns.")
    lookup = {name: i for i, name in enumerate(class_names)}
    allowed = np.zeros((len(lookup), len(lookup)), dtype=bool)
    for source, target in frame.itertuples(index=False, name=None):
        if source not in lookup or target not in lookup:
            if allow_extra_classes:
                continue
            raise ValueError(f"Unknown tissue in allowed pair: {source!r}, {target!r}.")
        allowed[lookup[source], lookup[target]] = True
    if not allowed.diagonal().all():
        raise ValueError("Every tissue must permit a self transition.")
    return allowed


def split_endpoint_indices(n_cells, validation_fraction, test_fraction, seed):
    """Return disjoint uniform cell splits; validation and test are distinct."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Positive validation/test fractions must sum to less than one.")
    n_validation = int(np.ceil(n_cells * validation_fraction))
    n_test = int(np.ceil(n_cells * test_fraction))
    if n_validation + n_test >= n_cells:
        raise ValueError("Endpoint population is too small for all three splits.")
    order = np.random.default_rng(seed).permutation(n_cells)
    return {
        "test": np.sort(order[:n_test]),
        "validation": np.sort(order[n_test:n_test + n_validation]),
        "train": np.sort(order[n_test + n_validation:]),
    }


def _read_h5_column(group, name):
    import h5py
    item = group[name]
    if isinstance(item, h5py.Group):
        categories = item["categories"].asstr()[()]
        codes = item["codes"][()]
        if (codes < 0).any():
            raise ValueError(f"Missing values in {name}.")
        return categories[codes].astype(str)
    return item.asstr()[()].astype(str)


@dataclass
class Marginal:
    window: str
    x: np.ndarray
    metadata: pd.DataFrame
    type_ids: np.ndarray


@dataclass
class EndpointExperiment:
    marginals: dict
    splits: dict
    class_names: tuple
    source_time: str
    reference_time: str
    target_time: str
    provenance: dict

    @property
    def training_times(self):
        return tuple(self.provenance.get("training_windows", (self.source_time, self.target_time)))

    @property
    def heldout_times(self):
        return tuple(self.provenance.get("intermediate_windows", (self.reference_time,)))

    def normalized_time(self, window):
        return (window_midpoint(window) - window_midpoint(self.source_time)) / (
            window_midpoint(self.target_time) - window_midpoint(self.source_time))

    def evaluation_populations(self):
        for window in sorted(self.marginals, key=window_midpoint):
            if window == self.source_time:
                continue
            if window in self.heldout_times:
                key = "heldout_reference" if len(self.heldout_times) == 1 else f"heldout_{window.replace('-', '_')}"
                indices = np.arange(len(self.marginals[window].x))
                kind = "entire_heldout_marginal"
            else:
                if "test" in self.splits[window]:
                    key = "target_test" if window == self.target_time else f"training_time_test_{window.replace('-', '_')}"
                    indices = self.splits[window]["test"]
                    kind = "selected_time_test_cells"
                else:
                    key = "target_observed" if window == self.target_time else f"observed_time_{window.replace('-', '_')}"
                    indices = np.arange(len(self.marginals[window].x))
                    kind = "entire_observed_marginal"
            yield key, window, indices, kind

    def pool(self, window, split):
        marginal = self.marginals[window]
        if split == "all":
            indices = np.arange(len(marginal.x))
        else:
            split = "validation" if split in ("holdout", "heldout") else split
            indices = self.splits[window][split]
        return {"x": marginal.x[indices], "type_ids": marginal.type_ids[indices]}

    def split_table(self):
        parts = []
        for window, marginal in self.marginals.items():
            frame = marginal.metadata.copy()
            if window in self.heldout_times:
                frame["flow_split"] = "heldout_reference"
            else:
                assignments = np.empty(len(frame), dtype=object)
                for split, indices in self.splits[window].items():
                    assignments[indices] = split
                frame["flow_split"] = assignments
            parts.append(frame)
        return pd.concat(parts, ignore_index=True)

    def summary(self):
        return {
            window: {
                "n_cells": len(marginal.x),
                "n_branches": int(marginal.metadata.branch_id.nunique()),
                "n_tissues": int(marginal.metadata.broad_tissue.nunique()),
                "splits": {name: len(indices) for name, indices in self.splits.get(window, {}).items()},
            }
            for window, marginal in self.marginals.items()
        }


def load_endpoint_experiment(cfg):
    if cfg.problem.get("retained_timepoints"):
        return load_multitime_experiment(cfg)
    import h5py
    dataset = Path(cfg.problem.dataset_location)
    spec = load_tissue_manifest(cfg.classifier.manifest_path, expected_variant="heldout_8_10")
    class_names = tuple(spec["class_names"])
    load_allowed_pairs(cfg.classifier.allowed_pairs_path, class_names)
    windows = (cfg.problem.source_time, cfg.problem.reference_time, cfg.problem.target_time)
    if cfg.problem.reference_time != "8-10":
        raise ValueError("The current reference classifier/PCA protocol holds out 8-10 h.")
    if len(set(windows)) != 3:
        raise ValueError("Source, reference and target windows must differ.")
    hours = (cfg.problem.source_hours, cfg.problem.reference_hours, cfg.problem.target_hours)
    if not hours[0] < hours[1] < hours[2]:
        raise ValueError("Reference time must lie between the endpoints.")
    if not np.isclose((hours[1] - hours[0]) / (hours[2] - hours[0]), 0.5):
        raise ValueError("This endpoint example expects its held-out reference at t=0.5.")
    for window, midpoint in zip(windows, hours):
        left, right = map(float, window.split("-"))
        if not np.isclose((left + right) / 2, midpoint):
            raise ValueError(f"Window and midpoint disagree: {window}, {midpoint}.")
    with h5py.File(dataset, "r") as handle:
        preprocessing = handle["uns/preprocessing"]
        if preprocessing["variant"].asstr()[()] != "heldout_8_10" or preprocessing["heldout_window"].asstr()[()] != "8-10":
            raise ValueError("Use the PCA representation fitted without 8-10 h.")
        if int(preprocessing["n_pcs"][()]) != 100 or int(preprocessing["n_hvgs"][()]) != 3000:
            raise ValueError("This experiment requires the fitted HVG3000/PCA100 export.")
        group = handle["obs"]
        metadata = pd.DataFrame({name: _read_h5_column(group, name) for name in
                                 ("cell_id", "day", "branch_id", "broad_tissue", "classifier_split")})
        metadata["pca_row"] = np.arange(len(metadata))
        pca = np.asarray(handle["obsm/X_pca"][()], dtype=np.float32)
    if pca.shape != (len(metadata), 100) or metadata.cell_id.duplicated().any():
        raise ValueError("PCA rows must align with unique cell IDs and have 100 columns.")
    if not ((metadata.day == "8-10") == (metadata.classifier_split == "test")).all():
        raise ValueError("Classifier split is not the declared held-out-window protocol.")
    lookup = {name: i for i, name in enumerate(class_names)}
    marginals = {}
    for window in windows:
        selected = (metadata.day == window) & metadata.branch_id.str.startswith(cfg.problem.cohort_branch_prefix)
        frame = metadata.loc[selected].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Empty lineage-assigned population at {window}.")
        if not frame.broad_tissue.isin(class_names).all():
            raise ValueError("Dataset tissue vocabulary differs from the classifier.")
        x = pca[selected.to_numpy()]
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite PCA scores.")
        marginals[window] = Marginal(window, x, frame, frame.broad_tissue.map(lookup).to_numpy(dtype=np.int32))
    splits = {
        window: split_endpoint_indices(len(marginals[window].x), cfg.problem.validation_fraction,
                                       cfg.problem.test_fraction, cfg.problem.split_seed + index)
        for index, window in enumerate((cfg.problem.source_time, cfg.problem.target_time))
    }
    dataset_hash = sha256_file(dataset)
    classifier_dataset = Path(spec.get("source_h5ad", str(dataset))).expanduser()
    if classifier_dataset.is_file() and classifier_dataset.resolve() != dataset.resolve():
        if sha256_file(classifier_dataset) != dataset_hash:
            raise ValueError("The PCA export differs from the source file used by the frozen classifier.")
    provenance = {
        "dataset": str(dataset.resolve()), "dataset_sha256": dataset_hash,
        "classifier_source_h5ad": spec.get("source_h5ad"),
        "classifier_manifest": str(Path(cfg.classifier.manifest_path).resolve()),
        "classifier_manifest_sha256": sha256_file(cfg.classifier.manifest_path),
        "classifier_selection_sha256": spec["selection_sha256"],
        "allowed_pairs": str(Path(cfg.classifier.allowed_pairs_path).resolve()),
        "allowed_pairs_sha256": sha256_file(cfg.classifier.allowed_pairs_path),
        "representation": "Original stored HVG3000/PCA100 scores; no extra PC standardization for the flow",
        "information_regime": "Endpoint flow training with the existing reference-atlas PCA, graph and tissue classifier",
        "heldout_scope": "Flow cell splits; the reference classifier/PCA can have seen endpoint flow-test cells",
    }
    if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
        export = Path(cfg.classifier.jax_export_path)
        with np.load(export, allow_pickle=False) as stored:
            metadata = json.loads(stored["metadata"].item())
        if metadata["manifest_sha256"] != provenance["classifier_manifest_sha256"]:
            raise ValueError("Training classifier export differs from the frozen tissue manifest.")
        provenance.update({"training_classifier_export_sha256": sha256_file(export),
                           "training_classifier_forward_sha256": sha256_file(Path(__file__).with_name("calderon_classifier.py")),
                           "training_classifier_checkpoint_sha256": [m["checkpoint_sha256"] for m in metadata["members"]],
                           "training_pair_prior": "Projected tissue relation used directly, plus classified linear interpolants and observed target label"})
    if cfg.constraints.enabled:
        provenance.update({"lineage_loss_backend": "Native Maizels loss-point transition NLL with the frozen Calderon ensemble",
                           "lineage_loss_forward_sha256": sha256_file(Path(__file__).with_name("calderon_lineage.py"))})
    return EndpointExperiment(marginals, splits, class_names, *windows, provenance)


def load_multitime_experiment(cfg):
    """Load all-times PCs, classifier metadata, and the configured flow folds."""
    import h5py
    dataset = Path(cfg.problem.dataset_location)
    selected = training_windows(cfg)
    intermediate = heldout_windows(cfg)
    if selected != ("4-6", "8-10", "12-14") or intermediate != ("6-8", "10-12"):
        raise ValueError("The selected-times classifier requires the symmetric 4–6 / 8–10 / 12–14 h schedule.")
    if (cfg.problem.source_time, cfg.problem.target_time) != (selected[0], selected[-1]):
        raise ValueError("Source and target must be the first and last training windows.")
    for window, midpoint in ((selected[0], cfg.problem.source_hours), (selected[-1], cfg.problem.target_hours)):
        if not np.isclose(window_midpoint(window), midpoint):
            raise ValueError(f"Window and midpoint disagree: {window}, {midpoint}.")
    spec = load_tissue_manifest(cfg.classifier.manifest_path, expected_variant="symmetric_4_8_12")
    oracle_path = evaluation_manifest_path(cfg)
    oracle = load_tissue_manifest(oracle_path, expected_variant="all_times")
    if tuple(spec.get("training_windows", ())) != selected or tuple(spec.get("intermediate_holdout_windows", ())) != intermediate:
        raise ValueError("Classifier supervision windows differ from the flow schedule.")
    classes = tuple(spec["class_names"])
    if not set(classes).issubset(oracle["class_names"]):
        raise ValueError("The evaluation classifier must cover the training classifier's tissues.")
    # Restrict the original relation to the training vocabulary by name. Do
    # not add a closure after dropping the atlas-wide Blastoderm label.
    load_allowed_pairs(cfg.classifier.allowed_pairs_path, oracle["class_names"])
    load_allowed_pairs(cfg.classifier.allowed_pairs_path, classes, allow_extra_classes=True)
    dataset_hash = sha256_file(dataset)
    if spec.get("pca_representation_source_sha256") != dataset_hash or oracle.get("source_h5ad_sha256") != dataset_hash:
        raise ValueError("Training and evaluation classifiers require the same unchanged all-times PCA export.")
    selected_cache = Path(spec["source_h5ad"])
    if sha256_file(selected_cache) != spec["source_h5ad_sha256"]:
        raise ValueError("The selected-times classifier cache changed after fitting.")
    windows = tuple(sorted((*selected, *intermediate), key=window_midpoint))
    with h5py.File(dataset, "r") as handle:
        preprocessing = handle["uns/preprocessing"]
        if (preprocessing["variant"].asstr()[()] != "all_times"
                or preprocessing["hvg_fit"].asstr()[()] != "all_cells"
                or preprocessing["pca_fit"].asstr()[()] != "all_cells"):
            raise ValueError("Use the existing all-times HVG/PCA representation.")
        if int(preprocessing["n_pcs"][()]) != 100 or int(preprocessing["n_hvgs"][()]) != 3000:
            raise ValueError("This experiment requires HVG3000/PCA100.")
        group = handle["obs"]
        metadata = pd.DataFrame({name: _read_h5_column(group, name) for name in
                                 ("cell_id", "day", "branch_id", "broad_tissue", "classifier_split")})
        metadata.rename(columns={"classifier_split": "reference_classifier_split"}, inplace=True)
        metadata["pca_row"] = np.arange(len(metadata))
        pca = np.asarray(handle["obsm/X_pca"][()], dtype=np.float32)
    if pca.shape != (len(metadata), 100) or metadata.cell_id.duplicated().any() or not np.isfinite(pca).all():
        raise ValueError("PCA rows must be finite and align with unique cell IDs.")
    table_path = Path(cfg.problem.split_manifest_path)
    table = pd.read_csv(table_path)
    required = {"cell_id", "day", "branch_id", "broad_tissue", "classifier_split", "source_pca_row"}
    if not required.issubset(table.columns) or table.cell_id.duplicated().any() or table.source_pca_row.duplicated().any():
        raise ValueError("The shared split table must identify unique cells and source PCA rows.")
    interval = metadata.loc[metadata.day.isin(windows)].copy()
    if set(interval.cell_id) != set(table.cell_id):
        raise ValueError("The split table must contain exactly the five retained-interval marginals.")
    aligned = table.set_index("cell_id").loc[interval.cell_id]
    for name in ("day", "branch_id", "broad_tissue"):
        if not np.array_equal(interval[name].to_numpy(), aligned[name].to_numpy()):
            raise ValueError(f"Split metadata differs from the PCA source: {name}.")
    if not np.array_equal(interval.pca_row.to_numpy(), aligned.source_pca_row.to_numpy()):
        raise ValueError("Split cells do not align with their global PCA rows.")
    interval["classifier_split"] = aligned.classifier_split.to_numpy()
    selected_rows = interval.day.isin(selected)
    if not interval.loc[selected_rows, "classifier_split"].isin(("train", "validation", "test")).all():
        raise ValueError("Selected times require train, validation and test cell folds.")
    if not interval.loc[~selected_rows, "classifier_split"].eq("intermediate_holdout").all():
        raise ValueError("Intermediate cells must be excluded from training and selection.")
    with h5py.File(selected_cache, "r") as handle:
        cache = pd.Series(_read_h5_column(handle["obs"], "classifier_split"),
                          index=_read_h5_column(handle["obs"], "cell_id"))
    supervised = interval.loc[selected_rows]
    if cache.index.duplicated().any() or set(cache.index) != set(supervised.cell_id):
        raise ValueError("The split table and classifier cache contain different selected-time cells.")
    if not np.array_equal(cache.loc[supervised.cell_id].to_numpy(), supervised.classifier_split.to_numpy()):
        raise ValueError("Classifier metadata folds must match the fitted classifier cache.")
    lookup = {name: i for i, name in enumerate(classes)}
    marginals, splits = {}, {}
    for window in windows:
        frame = interval.loc[interval.day.eq(window) & interval.branch_id.str.startswith(cfg.problem.cohort_branch_prefix)].copy()
        frame.reset_index(drop=True, inplace=True)
        if frame.empty or not frame.broad_tissue.isin(classes).all():
            raise ValueError(f"Empty or unrecognized tissue population at {window}.")
        marginals[window] = Marginal(window, pca[frame.pca_row.to_numpy()], frame,
                                     frame.broad_tissue.map(lookup).to_numpy(dtype=np.int32))
        if window in selected:
            if cfg.problem.get("split_protocol") == "maizels_train_holdout":
                from .maizels import _split_train_holdout_indices
                offset = 11 if window == selected[0] else 29 if window == selected[-1] else 101 * (windows.index(window) + 1)
                train, heldout = _split_train_holdout_indices(
                    len(frame), holdout_fraction=cfg.problem.maizels_holdout_fraction,
                    holdout_n=cfg.problem.maizels_holdout_n, seed=cfg.problem.maizels_holdout_seed + offset)
                splits[window] = {"train": train, "validation": heldout}
            else:
                splits[window] = {fold: np.flatnonzero(frame.classifier_split.eq(fold).to_numpy())
                                  for fold in ("train", "validation", "test")}
            if any(len(rows) == 0 for rows in splits[window].values()):
                raise ValueError(f"Empty supervised cell fold at {window}.")
    provenance = {
        "dataset": str(dataset.resolve()), "dataset_sha256": dataset_hash,
        "classifier_source_h5ad": str(selected_cache.resolve()),
        "classifier_source_h5ad_sha256": spec["source_h5ad_sha256"],
        "classifier_manifest": str(Path(cfg.classifier.manifest_path).resolve()),
        "classifier_manifest_sha256": sha256_file(cfg.classifier.manifest_path),
        "classifier_selection_sha256": spec["selection_sha256"],
        "evaluation_classifier_manifest": str(Path(oracle_path).resolve()),
        "evaluation_classifier_manifest_sha256": sha256_file(oracle_path),
        "evaluation_classifier_selection_sha256": oracle["selection_sha256"],
        "shared_split_manifest": str(table_path.resolve()), "shared_split_manifest_sha256": sha256_file(table_path),
        "allowed_pairs": str(Path(cfg.classifier.allowed_pairs_path).resolve()),
        "allowed_pairs_sha256": sha256_file(cfg.classifier.allowed_pairs_path),
        "training_windows": list(selected), "intermediate_windows": list(intermediate),
        "training_intervals": [[a, b] for a, b in zip(selected[:-1], selected[1:])],
        "representation": "Original all-times HVG3000/PCA100 scores; no extra standardization for the flow",
        "information_regime": "Selected-window supervised classifier and flow training; all-times reference classifier for final violations",
        "heldout_scope": "Shared flow/classifier validation and test cells; full intermediate marginals excluded from fitting and selection; global PCA includes all expression profiles",
        "classifier_label_alignment": "Training and evaluation vocabularies aligned by tissue name",
    }
    if cfg.problem.get("split_protocol") == "maizels_train_holdout":
        provenance.update({
            "flow_split_protocol": "Independent Maizels 90/10 train/heldout split at each retained time; heldout used for validation and violations",
            "flow_split_seed": cfg.problem.maizels_holdout_seed,
            "flow_holdout_fraction": cfg.problem.maizels_holdout_fraction,
            "heldout_scope": "Independent flow validation cells; both full intermediate marginals excluded from flow fitting/selection; classifiers retain their original splits and global PCA",
        })
    if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
        export = Path(cfg.classifier.jax_export_path)
        with np.load(export, allow_pickle=False) as stored:
            export_metadata = json.loads(stored["metadata"].item())
        if export_metadata["manifest_sha256"] != provenance["classifier_manifest_sha256"]:
            raise ValueError("Training classifier export differs from the selected-times tissue manifest.")
        provenance.update({
            "training_classifier_export_sha256": sha256_file(export),
            "training_classifier_forward_sha256": sha256_file(Path(__file__).with_name("calderon_classifier.py")),
            "training_classifier_checkpoint_sha256": [m["checkpoint_sha256"] for m in export_metadata["members"]],
            "training_pair_prior": "Projected tissue relation restricted by name, plus selected-times classified interpolants and observed final labels",
            "training_interpolant_filter_backend": cfg.problem.get("interpolant_filter_backend", "reference"),
        })
        if cfg.problem.get("interpolant_filter_backend", "reference") == "jax":
            provenance["training_interpolant_filter_sha256"] = sha256_file(Path(__file__).with_name("calderon_filter.py"))
    if cfg.constraints.enabled:
        provenance.update({"lineage_loss_backend": "Native Maizels loss-point transition NLL with the frozen selected-times ensemble",
                           "lineage_loss_forward_sha256": sha256_file(Path(__file__).with_name("calderon_lineage.py"))})
    return EndpointExperiment(marginals, splits, classes, selected[0], intermediate[0], selected[-1], provenance)


class EndpointOTDataset:
    """Fresh plain or filtered OT with the Maizels solver/refinement loop."""

    def __init__(self, cfg, source_pool, target_pool, class_names, interpolant_checker=None,
                 time_bounds=None, classifier=None):
        self.cfg = cfg
        self.source = source_pool
        self.target = target_pool
        self.class_names = class_names
        self.time_bounds = time_bounds
        if time_bounds is not None and not 0 <= time_bounds[0] < time_bounds[1] <= 1:
            raise ValueError("Pair interval bounds must lie in [0,1] and increase.")
        self.last_pair_stats = None
        self.allowed = None
        self.interpolant_checker = None
        self.jax_interpolant_filter = None
        self.interpolant_reference_fallback_pairs = 0
        self.interpolant_reference_fallback_points = 0
        self.cumulative_stats = {name: 0 for name in ("batches", "candidate_pairs", "endpoint_rejected", "interpolant_checked",
                                                    "interpolant_rejected", "ot_refinements", "sampled_pairs", "resampled_empty_blocks")}
        self.retained_mass_sum = 0.
        self.interpolant_classifier_points = 0
        self.sampled_source_tissues = np.zeros(len(class_names), dtype=np.int64)
        self.sampled_target_tissues = np.zeros(len(class_names), dtype=np.int64)
        if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
            self.allowed = load_allowed_pairs(cfg.classifier.allowed_pairs_path, class_names, allow_extra_classes=True)
            if interpolant_checker is None:
                from .calderon_classifier import FrozenTissueEnsemble
                self.classifier = classifier if classifier is not None else FrozenTissueEnsemble(
                    cfg.classifier.jax_export_path, cfg.classifier.manifest_path, cfg.problem.classifier_batch_size)
                if tuple(self.classifier.class_names) != tuple(class_names):
                    raise ValueError("Training decoder vocabulary differs from the endpoint labels.")
                if cfg.problem.get("interpolant_filter_backend", "reference") == "jax":
                    from .calderon_filter import get_jax_interpolant_filter
                    n_times = int(cfg.problem.n_interpolant_check_times)
                    if cfg.problem.get("interpolant_check_chunk_size", n_times) != n_times:
                        raise ValueError("The JAX filter uses the full check grid; use the reference backend for early rejection.")
                    self.jax_interpolant_filter = get_jax_interpolant_filter(
                        self.classifier, self.allowed, n_times, cfg.problem.classifier_prob_threshold,
                        cfg.problem.classifier_margin_threshold, cfg.problem.get("gpu_filter_decision_tolerance", 1e-4),
                    )
                self.interpolant_checker = self.check_interpolants
            else:
                self.interpolant_checker = interpolant_checker

    def check_interpolants(self, source_x, source_type_ids, target_x, target_type_ids):
        if self.jax_interpolant_filter is not None:
            result = self.jax_interpolant_filter(source_x, source_type_ids, target_x, target_type_ids)
            self.interpolant_classifier_points += result["n_points"]
            self.interpolant_reference_fallback_pairs += result["reference_fallback_pairs"]
            self.interpolant_reference_fallback_points += result["reference_fallback_points"]
            return result
        n_times = int(self.cfg.problem.n_interpolant_check_times)
        taus = np.linspace(0., 1., n_times + 2, dtype=np.float32)[1:-1]
        chunk = int(self.cfg.problem.get("interpolant_check_chunk_size", n_times))
        if chunk < n_times:
            if self.cfg.problem.classifier_prob_threshold != 0 or self.cfg.problem.classifier_margin_threshold != 0:
                raise ValueError("Early interpolant rejection uses raw labels with zero probability/margin thresholds.")
            n_pairs = len(source_x)
            valid = np.ones(n_pairs, dtype=bool)
            current = np.asarray(source_type_ids).copy()
            confident = np.zeros((n_pairs, n_times), dtype=bool)
            inspected = 0
            for start in range(0, n_times, chunk):
                active = np.flatnonzero(valid)
                if not len(active):
                    break
                times = taus[start:start + chunk]
                points = source_x[active, None] * (1 - times[None, :, None]) + target_x[active, None] * times[None, :, None]
                probs = self.classifier.predict_proba(points.reshape(-1, 100)).reshape(len(active), len(times), len(self.class_names))
                predicted = probs.argmax(-1)
                bad = np.zeros(len(active), dtype=bool)
                for column in range(len(times)):
                    bad |= ~self.allowed[current[active], predicted[:, column]]
                    current[active] = predicted[:, column]
                valid[active[bad]] = False
                confident[active, start:start + len(times)] = True
                inspected += len(active) * len(times)
            rejected_at_checkpoint = ~valid.copy()
            rejected_at_final = valid & ~self.allowed[current, np.asarray(target_type_ids)]
            valid[rejected_at_final] = False
            self.interpolant_classifier_points += inspected
            return {"valid": valid, "confident": confident, "rejected_at_checkpoint": rejected_at_checkpoint,
                    "rejected_at_final": rejected_at_final, "n_confident": inspected, "n_points": inspected}
        paths = source_x[:, None] * (1 - taus[None, :, None]) + target_x[:, None] * taus[None, :, None]
        probabilities = self.classifier.predict_proba(paths.reshape(-1, 100)).reshape(len(paths), n_times, len(self.class_names))
        self.interpolant_classifier_points += len(paths) * n_times
        result = interpolant_validity(source_type_ids, target_type_ids, probabilities, self.allowed,
                                      self.cfg.problem.classifier_prob_threshold, self.cfg.problem.classifier_margin_threshold)
        return result

    def sample_device_batch(self, bs, key):
        import jax
        import jax.numpy as jnp
        from . import cite_multi
        words = np.asarray(jax.device_get(jax.random.key_data(key)), dtype=np.uint32)
        rng = np.random.default_rng(words)
        for attempt in range(int(self.cfg.problem.ot_minibatch_max_resamples)):
            try:
                paired, self.last_pair_stats = cite_multi._make_minibatch_ot_pairs_from_arrays(
                    self.cfg, self.source["x"], self.source["type_ids"], self.target["x"], self.target["type_ids"],
                    n_pairs=int(bs), rng=rng, pair_mode=self.cfg.problem.maizels_pair_mode,
                    sample_with_replacement=True, class_names=self.class_names, transition_edges=(),
                    allowed_class_mask=self.allowed, interpolant_checker=self.interpolant_checker,
                )
                break
            except RuntimeError as error:
                if self.allowed is None or "zero valid mass" not in str(error):
                    raise
        else:
            raise RuntimeError("Filtered OT repeatedly retained zero valid mass; the prior was not relaxed.")
        self.last_pair_stats["resampled_empty_blocks"] = attempt
        self.cumulative_stats["batches"] += 1
        for name in self.cumulative_stats:
            if name != "batches":
                self.cumulative_stats[name] += int(self.last_pair_stats.get(name, 0))
        self.retained_mass_sum += self.last_pair_stats["ot_mean_retained_valid_mass"]
        self.sampled_source_tissues += np.bincount(paired["label"][:, 0], minlength=len(self.class_names))
        self.sampled_target_tissues += np.bincount(paired["label"][:, 1], minlength=len(self.class_names))
        # Endpoint tissue IDs select prior-valid pairs, never network conditioning.
        result = {name: jnp.asarray(paired[name], dtype=jnp.float32) for name in ("x0", "x1")}
        if self.time_bounds is not None:
            bounds = np.tile(np.asarray(self.time_bounds, dtype=np.float32), (len(paired["x0"]), 1))
            result["label"] = jnp.asarray(np.concatenate([paired["label"].astype(np.float32), bounds], axis=1))
        elif self.cfg.problem.interp_uses_labels:
            result["label"] = jnp.asarray(paired["label"], dtype=jnp.int32)
        return result

    def pair_filter_summary(self):
        counts = dict(self.cumulative_stats)
        counts["mean_retained_valid_mass"] = self.retained_mass_sum / counts["batches"] if counts["batches"] else None
        counts["interpolant_rejection_pct_checked"] = (100 * counts["interpolant_rejected"] / counts["interpolant_checked"]
                                                       if counts["interpolant_checked"] else None)
        counts["sampled_source_tissues"] = dict(zip(self.class_names, self.sampled_source_tissues.tolist()))
        counts["sampled_target_tissues"] = dict(zip(self.class_names, self.sampled_target_tissues.tolist()))
        counts["interpolant_classifier_points"] = self.interpolant_classifier_points
        counts["interpolant_filter_backend"] = self.cfg.problem.get("interpolant_filter_backend", "reference")
        counts["interpolant_reference_fallback_pairs"] = self.interpolant_reference_fallback_pairs
        counts["interpolant_reference_fallback_points"] = self.interpolant_reference_fallback_points
        return counts


class MultiMarginalOTDataset:
    """Equal pair counts per adjacent training interval, then shuffled together."""

    def __init__(self, cfg, data, split, interpolant_checker=None):
        self.cfg = cfg
        self.intervals = tuple(zip(data.training_times[:-1], data.training_times[1:]))
        if len(self.intervals) < 2 or not cfg.problem.pair_time_bounds_in_label:
            raise ValueError("Multi-marginal OT requires multiple intervals and absolute-time pair labels.")
        self.datasets = []
        shared_classifier = None
        for source, target in self.intervals:
            ds = EndpointOTDataset(
                cfg, data.pool(source, split), data.pool(target, split), data.class_names,
                interpolant_checker=interpolant_checker, classifier=shared_classifier,
                time_bounds=(data.normalized_time(source), data.normalized_time(target)),
            )
            shared_classifier = getattr(ds, "classifier", shared_classifier)
            self.datasets.append(ds)
        self.last_pair_stats = None
        self.batches = 0
        self.last_filter_dispatch_stats = None
        self.filter_dispatches = 0

    @property
    def interpolant_classifier_points(self):
        return sum(ds.interpolant_classifier_points for ds in self.datasets)

    def sample_device_batch(self, bs, key):
        import jax
        import jax.numpy as jnp
        if bs < len(self.datasets):
            raise ValueError("A training batch must supply a pair for every retained interval.")
        keys = jax.random.split(key, len(self.datasets) + 1)
        counts = [bs // len(self.datasets) + (i < bs % len(self.datasets)) for i in range(len(self.datasets))]
        shared_filter = self.datasets[0].jax_interpolant_filter
        batch_checks = (self.cfg.problem.get("batch_interpolant_checks_across_intervals", False)
                        and shared_filter is not None
                        and all(ds.jax_interpolant_filter is shared_filter for ds in self.datasets))
        if batch_checks:
            from .calderon_filter import sample_intervals_with_batched_filter
            parts, self.last_filter_dispatch_stats = sample_intervals_with_batched_filter(self.datasets, counts, keys[:-1])
            self.filter_dispatches += self.last_filter_dispatch_stats["dispatches"]
        else:
            parts = [ds.sample_device_batch(count, interval_key)
                     for ds, count, interval_key in zip(self.datasets, counts, keys[:-1])]
            self.last_filter_dispatch_stats = None
        batch = {name: jnp.concatenate([part[name] for part in parts], axis=0) for name in ("x0", "x1", "label")}
        permutation = jax.random.permutation(keys[-1], bs)
        batch = {name: value[permutation] for name, value in batch.items()}
        fields = self.datasets[0].cumulative_stats
        self.last_pair_stats = {name: sum(ds.last_pair_stats.get(name, 0) for ds in self.datasets)
                                for name in fields if name != "batches"}
        self.last_pair_stats["ot_mean_retained_valid_mass"] = sum(
            count * ds.last_pair_stats["ot_mean_retained_valid_mass"] for count, ds in zip(counts, self.datasets)) / bs
        self.last_pair_stats["intervals"] = [
            {"source_window": source, "target_window": target, "time_bounds": list(ds.time_bounds), **ds.last_pair_stats}
            for (source, target), ds in zip(self.intervals, self.datasets)
        ]
        self.batches += 1
        return batch

    def pair_filter_summary(self):
        summaries = [ds.pair_filter_summary() for ds in self.datasets]
        combined = {name: sum(ds.cumulative_stats[name] for ds in self.datasets)
                    for name in self.datasets[0].cumulative_stats}
        combined["batches"] = self.batches
        interval_batches = sum(ds.cumulative_stats["batches"] for ds in self.datasets)
        combined["mean_retained_valid_mass"] = (
            sum(ds.retained_mass_sum for ds in self.datasets) / interval_batches if interval_batches else None)
        combined["interpolant_rejection_pct_checked"] = (
            100 * combined["interpolant_rejected"] / combined["interpolant_checked"]
            if combined["interpolant_checked"] else None)
        for name in ("sampled_source_tissues", "sampled_target_tissues"):
            combined[name] = {tissue: sum(row[name][tissue] for row in summaries) for tissue in self.datasets[0].class_names}
        combined["interpolant_classifier_points"] = self.interpolant_classifier_points
        combined["interpolant_filter_backend"] = summaries[0]["interpolant_filter_backend"]
        if self.filter_dispatches:
            combined["gpu_filter_dispatches"] = self.filter_dispatches
        for name in ("interpolant_reference_fallback_pairs", "interpolant_reference_fallback_points"):
            combined[name] = sum(row[name] for row in summaries)
        combined["intervals"] = [
            {"source_window": source, "target_window": target, "time_bounds": list(ds.time_bounds), **summary}
            for (source, target), ds, summary in zip(self.intervals, self.datasets, summaries)
        ]
        return combined


def training_dataset(cfg, data, split):
    if len(data.training_times) > 2:
        return MultiMarginalOTDataset(cfg, data, split)
    return EndpointOTDataset(cfg, data.pool(data.source_time, split), data.pool(data.target_time, split), data.class_names)


def interpolant_validity(source_ids, target_ids, probabilities, allowed, prob_threshold=0., margin_threshold=0.):
    """Match the Maizels interior scan, including its observed-final-label check."""
    n_times = probabilities.shape[1]
    final = np.eye(probabilities.shape[2], dtype=np.float32)[np.asarray(target_ids)]
    augmented = np.concatenate([probabilities, final[:, None]], axis=1)
    result = score_tissue_paths(source_ids, augmented, allowed, prob_threshold, margin_threshold)
    first = result["first_rejection"]
    return {"valid": first < 0, "confident": result["confident"][:, :n_times],
            "rejected_at_checkpoint": (first >= 0) & (first < n_times), "rejected_at_final": first == n_times,
            "n_confident": int(result["confident"][:, :n_times].sum()), "n_points": int(probabilities.shape[0] * n_times)}


def select_population_indices(n_cells, cap, seed):
    if cap < 0:
        raise ValueError("Population cap must be nonnegative; zero means all.")
    if cap == 0 or cap >= n_cells:
        return np.arange(n_cells)
    return np.sort(np.random.default_rng(seed).choice(n_cells, size=cap, replace=False))


def score_tissue_paths(source_ids, probabilities, allowed, prob_threshold=0.0, margin_threshold=0.0):
    """Check successive accepted labels; keep unassessed paths explicit."""
    probabilities = np.asarray(probabilities)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if probabilities.ndim != 3 or probabilities.shape[0] != len(source_ids) or probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("Expected nonempty probabilities shaped [source, checkpoint, tissue].")
    n_sources, n_times, n_classes = probabilities.shape
    if n_classes < 2 or allowed.shape != (n_classes, n_classes) or (source_ids < 0).any() or (source_ids >= n_classes).any():
        raise ValueError("Invalid tissue vocabularies or source IDs.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-5):
        raise ValueError("Invalid classifier probabilities.")
    if not 0 <= prob_threshold <= 1 or not 0 <= margin_threshold <= 1:
        raise ValueError("Probability/margin thresholds must lie in [0, 1].")
    predicted = probabilities.argmax(axis=-1)
    confidence = probabilities.max(axis=-1)
    margin = confidence - np.partition(probabilities, -2, axis=-1)[..., -2]
    confident = (confidence >= prob_threshold) & (margin >= margin_threshold)
    current = source_ids.copy()
    first_rejection = np.full(n_sources, -1, dtype=np.int32)
    for index in range(n_times):
        assessed = confident[:, index] & (first_rejection < 0)
        bad = assessed & ~allowed[current, predicted[:, index]]
        first_rejection[bad] = index
        good = assessed & ~bad
        current[good] = predicted[good, index]
    violated = first_rejection >= 0
    any_assessed = confident.any(axis=1)
    fully_assessed = confident.all(axis=1)
    endpoint_bad = ~allowed[source_ids, predicted[:, -1]] & confident[:, -1]
    status = np.where(violated, "violation", np.where(any_assessed, "no_detected_violation", "unassessed"))

    def fraction_pct(numerator, denominator):
        return 100.0 * int(numerator) / int(denominator) if denominator else None

    metrics = {
        "n_sources": n_sources, "n_checkpoints": n_times,
        "n_violations": int(violated.sum()), "n_assessed": int(any_assessed.sum()),
        "n_unassessed": int((~any_assessed).sum()), "n_fully_assessed": int(fully_assessed.sum()),
        "violation_pct_all_sources": fraction_pct(violated.sum(), n_sources),
        "violation_pct_assessed": fraction_pct(violated.sum(), any_assessed.sum()),
        "trajectory_coverage_pct": fraction_pct(any_assessed.sum(), n_sources),
        "full_checkpoint_coverage_pct": fraction_pct(fully_assessed.sum(), n_sources),
        "violation_pct_fully_assessed": fraction_pct((violated & fully_assessed).sum(), fully_assessed.sum()),
        "confident_checkpoint_pct": fraction_pct(confident.sum(), confident.size),
        "endpoint_violation_pct_assessed": fraction_pct(endpoint_bad.sum(), confident[:, -1].sum()),
        "first_rejection_counts": [int((first_rejection == index).sum()) for index in range(n_times)],
        "confident_counts": confident.sum(axis=0).astype(int).tolist(),
    }
    return {"metrics": metrics, "first_rejection": first_rejection, "status": status,
            "predicted_ids": predicted, "confident": confident, "confidence": confidence, "margin": margin}

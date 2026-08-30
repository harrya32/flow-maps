"""Utilities for the Maizels PCA50 trajectory experiment."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # NumPy pairing is also used by the PyTorch MFM code.
    jax = None
    jnp = None

DEFAULT_DATASET = (
    "/Users/harryamad/Desktop/Maizels2023aa/data/"
    "celltype_classification_pca50_dataset.csv.gz"
)
DEFAULT_CLASSIFIER = (
    "/Users/harryamad/Desktop/Maizels2023aa/models/" "celltype_classifier_pca50.pt"
)

TIMEPOINTS = (
    "D3",
    "D3.2",
    "D3.4",
    "D3.6",
    "D3.8",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
)

CLASS_NAMES = [
    "Early_Neural",
    "FP",
    "MN",
    "Mesoderm",
    "NMP",
    "Neural",
    "V3",
    "p3",
    "pMN",
]

TRANSITION_EDGES_OLD = [
    ("NMP", "Mesoderm"),
    ("NMP", "Early_Neural"),
    ("Early_Neural", "Neural"),
    ("Early_Neural", "pMN"),
    ("Early_Neural", "p3"),
    ("p3", "V3"),
    ("p3", "FP"),
    ("pMN", "MN"),
]

# Active lineage hierarchy. Self-transitions are added by the reachability
# builders below, so only differentiation edges are listed here.
TRANSITION_EDGES = [
    ("NMP", "Mesoderm"),
    ("NMP", "Early_Neural"),
    ("Early_Neural", "Neural"),
    ("Neural", "pMN"),
    ("pMN", "MN"),
    ("Neural", "p3"),
    ("p3", "V3"),
    ("p3", "FP"),
]

_DATA_CACHE: Dict[str, Dict[str, np.ndarray]] = {}
_CLASSIFIER_CACHE: Dict[str, Tuple[Any, List[str], np.ndarray, np.ndarray]] = {}
_JAX_CLASSIFIER_CACHE: Dict[
    str, Tuple[Dict[str, jnp.ndarray], List[str], jnp.ndarray, jnp.ndarray]
] = {}


def _canonical_maizels_pair_mode(pair_mode: str) -> str:
    pair_mode = str(pair_mode)
    if pair_mode in ("ot", "plain_ot"):
        return "ot_plain"
    return pair_mode


_OT_PAIR_MODES = {
    "ot_plain",
    "ot_endpoint",
    "ot_endpoint_interpolant",
}


def _canonical_maizels_ot_coupling(coupling: str) -> str:
    aliases = {
        "global": "global_ot",
        "exact": "global_ot",
        "minibatch": "minibatch_ot",
    }
    coupling = aliases.get(str(coupling).lower(), str(coupling).lower())
    if coupling not in ("global_ot", "minibatch_ot"):
        raise ValueError(
            "problem.maizels_ot_coupling must be 'global_ot' or "
            f"'minibatch_ot', got {coupling!r}."
        )
    return coupling


def maizels_ot_coupling_from_config(cfg) -> str:
    """Return the selected Maizels OT backend."""
    return _canonical_maizels_ot_coupling(
        getattr(cfg.problem, "maizels_ot_coupling", "global_ot")
    )


def uses_minibatch_ot(cfg, pair_mode: str | None = None) -> bool:
    """Return whether a Maizels OT mode should be coupled per training batch."""
    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    return (
        _canonical_maizels_pair_mode(str(pair_mode)) in _OT_PAIR_MODES
        and maizels_ot_coupling_from_config(cfg) == "minibatch_ot"
    )


class NumpyCellTypeMLP:
    """NumPy inference copy of the PyTorch PCA50 cell-type classifier."""

    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays

    @staticmethod
    def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return x @ weight.T + bias

    @staticmethod
    def _batch_norm(
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        return x_hat * weight + bias

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        a = self.arrays
        h = self._linear(x, a["net.0.weight"], a["net.0.bias"])
        h = self._batch_norm(
            h,
            a["net.1.weight"],
            a["net.1.bias"],
            a["net.1.running_mean"],
            a["net.1.running_var"],
        )
        h = np.maximum(h, 0.0)
        h = self._linear(h, a["net.4.weight"], a["net.4.bias"])
        h = self._batch_norm(
            h,
            a["net.5.weight"],
            a["net.5.bias"],
            a["net.5.running_mean"],
            a["net.5.running_var"],
        )
        h = np.maximum(h, 0.0)
        return self._linear(h, a["net.8.weight"], a["net.8.bias"])


def resolve_dataset_path(dataset_location: str | None) -> Path:
    if dataset_location in ("", None):
        return Path(DEFAULT_DATASET)
    location = Path(str(dataset_location))
    if location.suffixes[-2:] == [".csv", ".gz"] or location.suffix == ".csv":
        return location
    return location / "celltype_classification_pca50_dataset.csv.gz"


def resolve_classifier_path(classifier_path: str | None) -> Path:
    if classifier_path in ("", None):
        return Path(DEFAULT_CLASSIFIER)
    return Path(str(classifier_path))


def parse_timepoint(value: str) -> float:
    text = str(value)
    if text.startswith("D"):
        text = text[1:]
    return float(text)


def retained_timepoints(cfg) -> Tuple[str, ...]:
    """Return the observed timepoints used to construct training intervals."""
    configured = getattr(cfg.problem, "retained_timepoints", None)
    if configured is None:
        configured = (
            getattr(cfg.problem, "source_time", "D3"),
            getattr(cfg.problem, "target_time", "D8"),
        )
    retained = tuple(str(value) for value in configured)
    if len(retained) < 2:
        raise ValueError("Maizels training requires at least two retained timepoints.")
    if len(set(retained)) != len(retained):
        raise ValueError(f"Duplicate retained Maizels timepoints: {retained}.")
    ordered = tuple(sorted(retained, key=parse_timepoint))
    if retained != ordered:
        raise ValueError(
            "problem.retained_timepoints must be in increasing day order, "
            f"got {retained}."
        )
    return retained


def uses_retained_intervals(cfg) -> bool:
    """Whether Maizels training uses more than one adjacent interval."""
    return len(retained_timepoints(cfg)) > 2


def normalized_time(timepoint: str, cfg=None) -> float:
    """Map a Maizels observation day onto the configured global model clock."""
    key = str(timepoint)
    if cfg is not None:
        order = getattr(cfg.problem, "timepoint_order", None)
        values = getattr(cfg.problem, "timepoint_values", None)
        if order is not None and values is not None:
            mapping = {
                str(name): float(value) for name, value in zip(list(order), list(values))
            }
            if key not in mapping:
                raise KeyError(f"Unknown configured Maizels timepoint {key!r}.")
            return mapping[key]

    source = parse_timepoint(TIMEPOINTS[0])
    target = parse_timepoint(TIMEPOINTS[-1])
    return float((parse_timepoint(key) - source) / (target - source))


def retained_interval_for_timepoint(cfg, timepoint: str) -> Tuple[str, str]:
    """Return the adjacent retained interval containing an omitted timepoint."""
    value = parse_timepoint(timepoint)
    retained = retained_timepoints(cfg)
    for source_time, target_time in zip(retained[:-1], retained[1:]):
        if parse_timepoint(source_time) < value < parse_timepoint(target_time):
            return source_time, target_time
    raise ValueError(
        f"Timepoint {timepoint!r} is not strictly inside a retained interval "
        f"from {retained}."
    )


def build_reachable(
    edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Set[str]]:
    """Return the reflexive transitive closure of the cell-type transition graph."""
    nodes = set(class_names)
    children = defaultdict(set)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)
        children[src].add(dst)

    reachable: Dict[str, Set[str]] = {}
    for node in nodes:
        seen = {node}
        stack = list(children[node])
        while stack:
            curr = stack.pop()
            if curr in seen:
                continue
            seen.add(curr)
            stack.extend(children[curr])
        reachable[node] = seen
    return reachable


def build_direct_reachable(
    edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Set[str]]:
    """Return reflexive one-edge reachability for strict stepwise checks."""
    nodes = set(class_names)
    for src, dst in edges:
        nodes.add(src)
        nodes.add(dst)

    reachable: Dict[str, Set[str]] = {node: {node} for node in nodes}
    for src, dst in edges:
        reachable[src].add(dst)
    return reachable


def resolve_lineage_transition_mode(mode: str | None) -> str:
    """Normalize lineage validity mode names."""
    if mode in (None, "", "same_as_problem", "same_as_training"):
        return "descendant"
    mode = str(mode).lower()
    if mode in (
        "descendant",
        "descendants",
        "descendent",
        "descendents",
        "reachable",
        "transitive",
        "transitive_closure",
        "closure",
    ):
        return "descendant"
    if mode in (
        "direct",
        "direct_child",
        "direct_children",
        "edge",
        "immediate",
        "one_step",
        "strict",
    ):
        return "direct"
    raise ValueError(
        "lineage_transition_mode must be 'descendant' or 'direct', " f"got {mode!r}."
    )


def lineage_transition_mode_from_config(cfg, *, default: str = "descendant") -> str:
    """Resolve the lineage transition mode shared by pair filters and constraints."""
    problem_cfg = getattr(cfg, "problem", None)
    mode = getattr(problem_cfg, "lineage_transition_mode", default)
    constraints_cfg = getattr(cfg, "constraints", None)
    constraint_mode = getattr(constraints_cfg, "lineage_transition_mode", None)
    if constraint_mode not in (None, "", "same_as_problem", "same_as_training"):
        mode = constraint_mode
    return resolve_lineage_transition_mode(mode)


def build_transition_reachable(
    mode: str | None = "descendant",
    edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict[str, Set[str]]:
    """Return the reachability relation for the configured lineage mode."""
    mode = resolve_lineage_transition_mode(mode)
    if mode == "direct":
        return build_direct_reachable(edges, class_names=class_names)
    return build_reachable(edges, class_names=class_names)


def endpoint_valid(
    src_type: str, dst_type: str, reachable: Dict[str, Set[str]]
) -> bool:
    return dst_type in reachable.get(src_type, {src_type})


def load_pca50_dataset(dataset_path: str | Path) -> Dict[str, np.ndarray]:
    """Load the saved PCA50 CSV used by the cell-type classifier notebook."""
    path = Path(dataset_path).expanduser().resolve()
    cache_key = str(path)
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    pc_cols = [f"PC{ii}" for ii in range(1, 51)]
    obs_names: List[str] = []
    timepoints: List[str] = []
    cell_types: List[str] = []
    pcs: List[List[float]] = []

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        missing = [
            col
            for col in ["timepoint", "cell_annotation", *pc_cols]
            if col not in (reader.fieldnames or [])
        ]
        if missing:
            raise KeyError(f"Missing expected columns in {path}: {missing}")

        index_col = "" if "" in (reader.fieldnames or []) else None
        for row in reader:
            obs_names.append(
                row[index_col] if index_col is not None else str(len(obs_names))
            )
            timepoints.append(row["timepoint"])
            cell_types.append(row["cell_annotation"])
            pcs.append([float(row[col]) for col in pc_cols])

    data = {
        "obs_names": np.asarray(obs_names, dtype=object),
        "x": np.asarray(pcs, dtype=np.float32),
        "timepoints": np.asarray(timepoints, dtype=object),
        "time_values": np.asarray(
            [parse_timepoint(tp) for tp in timepoints], dtype=np.float32
        ),
        "cell_types": np.asarray(cell_types, dtype=object),
    }
    _DATA_CACHE[cache_key] = data
    return data


def subset_time(
    data: Dict[str, np.ndarray], timepoint: str
) -> Tuple[np.ndarray, np.ndarray]:
    mask = data["timepoints"] == timepoint
    return data["x"][mask], data["cell_types"][mask]


def class_to_id_map(class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def _cell_type_ids(
    cell_types: np.ndarray, class_names: Sequence[str] = CLASS_NAMES
) -> np.ndarray:
    """Return validated integer ids, accepting names or pre-encoded ids."""
    values = np.asarray(cell_types)
    if values.dtype.kind in "iu":
        ids = values.astype(np.int32, copy=False)
    else:
        mapping = class_to_id_map(class_names)
        ids = np.asarray([mapping[str(value)] for value in values], dtype=np.int32)
    if np.any(ids < 0) or np.any(ids >= len(class_names)):
        raise ValueError("Cell-type pool contains an invalid class id.")
    return ids


def _make_celltype_mlp(n_features: int, n_classes: int, dropout: float = 0.2):
    import torch
    from torch import nn

    class CellTypeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    return CellTypeMLP()


def load_classifier(
    classifier_path: str | Path,
) -> Tuple[Any, List[str], np.ndarray, np.ndarray]:
    """Load the PCA50 classifier lazily.

    PyTorch is intentionally imported inside this function so non-Maizels
    experiments do not need it at import time.
    """
    path = Path(classifier_path).expanduser().resolve()
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if npz_path.exists():
        cache_key = str(npz_path)
        if cache_key in _CLASSIFIER_CACHE:
            return _CLASSIFIER_CACHE[cache_key]
        with np.load(npz_path, allow_pickle=False) as raw:
            arrays = {
                key: raw[key].astype(np.float32)
                for key in raw.files
                if key.startswith("net.")
            }
            class_names = [str(x) for x in raw["class_names"].tolist()]
            scaler_mean = raw["scaler_mean"].astype(np.float32)
            scaler_scale = raw["scaler_scale"].astype(np.float32)
        result = (NumpyCellTypeMLP(arrays), class_names, scaler_mean, scaler_scale)
        _CLASSIFIER_CACHE[cache_key] = result
        return result

    cache_key = str(path)
    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Maizels prior/classifier path requires PyTorch to load " f"{path}."
        ) from exc

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    metadata = checkpoint["metadata"]
    class_names = list(metadata["class_names"])
    scaler_mean = np.asarray(metadata["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(metadata["scaler_scale"], dtype=np.float32)

    model = _make_celltype_mlp(50, len(class_names), dropout=0.2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    result = (model, class_names, scaler_mean, scaler_scale)
    _CLASSIFIER_CACHE[cache_key] = result
    return result


def load_jax_classifier_params(
    classifier_path: str | Path,
) -> Tuple[Dict[str, jnp.ndarray], List[str], jnp.ndarray, jnp.ndarray]:
    """Load frozen PCA50 classifier weights for differentiable JAX inference."""
    if jax is None or jnp is None:
        raise ModuleNotFoundError(
            "JAX is required for differentiable Maizels classifier constraints."
        )
    path = Path(classifier_path).expanduser().resolve()
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(
            "Maizels differentiable constraints require the exported NumPy "
            f"classifier checkpoint at {npz_path}."
        )

    cache_key = str(npz_path)
    if cache_key in _JAX_CLASSIFIER_CACHE:
        return _JAX_CLASSIFIER_CACHE[cache_key]

    with np.load(npz_path, allow_pickle=False) as raw:
        params = {
            key: jnp.asarray(raw[key].astype(np.float32))
            for key in raw.files
            if key.startswith("net.")
        }
        class_names = [str(x) for x in raw["class_names"].tolist()]
        scaler_mean = jnp.asarray(raw["scaler_mean"].astype(np.float32))
        scaler_scale = jnp.asarray(raw["scaler_scale"].astype(np.float32))

    result = (params, class_names, scaler_mean, scaler_scale)
    _JAX_CLASSIFIER_CACHE[cache_key] = result
    return result


def jax_classifier_logits(
    params: Dict[str, jnp.ndarray],
    scaler_mean: jnp.ndarray,
    scaler_scale: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable JAX copy of the PCA50 cell-type MLP classifier."""
    if jax is None or jnp is None:
        raise ModuleNotFoundError(
            "JAX is required for differentiable Maizels classifier constraints."
        )
    x = (x - scaler_mean) / scaler_scale

    def linear(h, prefix):
        return h @ params[f"{prefix}.weight"].T + params[f"{prefix}.bias"]

    def batch_norm(h, prefix):
        mean = params[f"{prefix}.running_mean"]
        var = params[f"{prefix}.running_var"]
        weight = params[f"{prefix}.weight"]
        bias = params[f"{prefix}.bias"]
        return (h - mean) / jnp.sqrt(var + 1e-5) * weight + bias

    h = linear(x, "net.0")
    h = batch_norm(h, "net.1")
    h = jax.nn.relu(h)
    h = linear(h, "net.4")
    h = batch_norm(h, "net.5")
    h = jax.nn.relu(h)
    return linear(h, "net.8")


def classifier_index_lookup(
    class_names: Sequence[str],
    canonical_class_names: Sequence[str] = CLASS_NAMES,
) -> np.ndarray:
    """Map canonical Maizels class ids to a classifier-specific class order."""
    by_name = {str(name): idx for idx, name in enumerate(class_names)}
    missing = [name for name in canonical_class_names if name not in by_name]
    if missing:
        raise KeyError(
            "Classifier is missing Maizels classes required for constraints: "
            f"{missing}"
        )
    return np.asarray([by_name[name] for name in canonical_class_names], dtype=np.int32)


def lineage_invalid_transition_matrix(
    class_names: Sequence[str],
    transition_mode: str | None = "descendant",
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> np.ndarray:
    """Return matrix M where M[i, j]=1 iff i -> j is biologically invalid."""
    reachable = build_transition_reachable(
        transition_mode,
        edges=transition_edges,
        class_names=class_names,
    )
    invalid = np.zeros((len(class_names), len(class_names)), dtype=np.float32)
    for ii, src in enumerate(class_names):
        for jj, dst in enumerate(class_names):
            invalid[ii, jj] = (
                0.0 if endpoint_valid(str(src), str(dst), reachable) else 1.0
            )
    return invalid


def lineage_soft_terms_from_probs(
    probs: jnp.ndarray,
    source_type_ids: jnp.ndarray,
    invalid_transition: jnp.ndarray,
    canonical_to_classifier: jnp.ndarray,
    target_type_ids: jnp.ndarray | None = None,
    transition_mask: jnp.ndarray | None = None,
) -> Dict[str, jnp.ndarray]:
    """Differentiable lineage-validity terms from path classifier probabilities."""
    if jax is None or jnp is None:
        raise ModuleNotFoundError(
            "JAX is required for differentiable Maizels classifier constraints."
        )
    source_cls = jnp.take(canonical_to_classifier, source_type_ids.astype(jnp.int32))
    source_probs = jax.nn.one_hot(source_cls, probs.shape[-1], dtype=probs.dtype)
    invalid_transition = invalid_transition.astype(probs.dtype)
    valid_transition = 1.0 - invalid_transition
    eps = jnp.asarray(1e-6, dtype=probs.dtype)
    if transition_mask is not None:
        transition_mask = transition_mask.astype(probs.dtype)
        mask_denom = jnp.maximum(jnp.sum(transition_mask, axis=1), eps)
        final_idx = jnp.sum(transition_mask.astype(jnp.int32), axis=1)
        final_probs = jnp.take_along_axis(
            probs,
            final_idx[:, None, None],
            axis=1,
        )[:, 0, :]
    else:
        final_probs = probs[:, -1, :]

    start_invalid = jnp.einsum(
        "bi,ij,bj->b",
        source_probs,
        invalid_transition,
        probs[:, 0, :],
    )
    start_valid = jnp.einsum(
        "bi,ij,bj->b",
        source_probs,
        valid_transition,
        probs[:, 0, :],
    )
    start_valid_nll = -jnp.log(jnp.clip(start_valid, eps, 1.0))
    if probs.shape[1] > 1:
        transition_invalid = jnp.einsum(
            "bti,ij,btj->bt",
            probs[:, :-1, :],
            invalid_transition,
            probs[:, 1:, :],
        )
        transition_valid = jnp.einsum(
            "bti,ij,btj->bt",
            probs[:, :-1, :],
            valid_transition,
            probs[:, 1:, :],
        )
        transition_invalid_per_path = jnp.mean(transition_invalid, axis=1)
        transition_valid_nll = -jnp.log(jnp.clip(transition_valid, eps, 1.0))
        transition_valid_nll_per_path = jnp.mean(transition_valid_nll, axis=1)
        transition_valid_per_path = jnp.mean(transition_valid, axis=1)
        if transition_mask is not None:
            transition_invalid_per_path = (
                jnp.sum(transition_invalid * transition_mask, axis=1) / mask_denom
            )
            transition_valid_nll_per_path = (
                jnp.sum(transition_valid_nll * transition_mask, axis=1) / mask_denom
            )
            transition_valid_per_path = (
                jnp.sum(transition_valid * transition_mask, axis=1) / mask_denom
            )
    else:
        transition_invalid = jnp.zeros((probs.shape[0], 0), dtype=probs.dtype)
        transition_valid = jnp.zeros((probs.shape[0], 0), dtype=probs.dtype)
        transition_invalid_per_path = jnp.zeros((probs.shape[0],), dtype=probs.dtype)
        transition_valid_nll = jnp.zeros((probs.shape[0], 0), dtype=probs.dtype)
        transition_valid_nll_per_path = jnp.zeros((probs.shape[0],), dtype=probs.dtype)
        transition_valid_per_path = jnp.zeros((probs.shape[0],), dtype=probs.dtype)

    if target_type_ids is None:
        final_invalid = jnp.zeros((probs.shape[0],), dtype=probs.dtype)
        final_valid = jnp.ones((probs.shape[0],), dtype=probs.dtype)
        final_valid_nll = jnp.zeros((probs.shape[0],), dtype=probs.dtype)
    else:
        target_cls = jnp.take(
            canonical_to_classifier, target_type_ids.astype(jnp.int32)
        )
        target_probs = jax.nn.one_hot(target_cls, probs.shape[-1], dtype=probs.dtype)
        final_invalid = jnp.einsum(
            "bi,ij,bj->b",
            final_probs,
            invalid_transition,
            target_probs,
        )
        final_valid = jnp.einsum(
            "bi,ij,bj->b",
            final_probs,
            valid_transition,
            target_probs,
        )
        final_valid_nll = -jnp.log(jnp.clip(final_valid, eps, 1.0))

    final_entropy = -jnp.sum(
        final_probs * jnp.log(jnp.clip(final_probs, eps, 1.0)),
        axis=-1,
    )
    if transition_mask is not None and probs.shape[1] > 1:
        active_transition_denom = jnp.maximum(jnp.sum(transition_mask), eps)
        transition_invalid_mass = (
            jnp.sum(transition_invalid * transition_mask) / active_transition_denom
        )
        transition_valid_mass = (
            jnp.sum(transition_valid * transition_mask) / active_transition_denom
        )
    else:
        transition_invalid_mass = (
            jnp.mean(transition_invalid)
            if transition_invalid.size > 0
            else jnp.asarray(0.0, dtype=probs.dtype)
        )
        transition_valid_mass = (
            jnp.mean(transition_valid)
            if transition_valid.size > 0
            else jnp.asarray(0.0, dtype=probs.dtype)
        )

    return {
        "start_invalid_loss": jnp.mean(start_invalid),
        "transition_invalid_loss": jnp.mean(transition_invalid_per_path),
        "final_invalid_loss": jnp.mean(final_invalid),
        "path_invalid_loss": jnp.mean(
            start_invalid + transition_invalid_per_path + final_invalid
        ),
        "start_invalid_mass": jnp.mean(start_invalid),
        "start_valid_mass": jnp.mean(start_valid),
        "start_valid_nll_loss": jnp.mean(start_valid_nll),
        "transition_invalid_mass": transition_invalid_mass,
        "transition_valid_mass": transition_valid_mass,
        "transition_valid_nll_loss": jnp.mean(transition_valid_nll_per_path),
        "final_invalid_mass": jnp.mean(final_invalid),
        "final_valid_mass": jnp.mean(final_valid),
        "final_valid_nll_loss": jnp.mean(final_valid_nll),
        "path_valid_nll_loss": jnp.mean(
            start_valid_nll + transition_valid_nll_per_path + final_valid_nll
        ),
        "final_entropy_loss": jnp.mean(final_entropy),
    }


def classifier_predictions(
    model: Any,
    class_names: Sequence[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    x: np.ndarray,
    batch_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top class names, top probabilities, and top-logit margins."""
    top_idx_batches = []
    top_prob_batches = []
    margin_batches = []

    if hasattr(model, "predict_logits"):
        for start in range(0, x.shape[0], batch_size):
            xb = (x[start : start + batch_size] - scaler_mean) / scaler_scale
            logits = model.predict_logits(xb.astype(np.float32, copy=False))
            shifted = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            top_idx = np.argmax(probs, axis=1)
            top_prob = probs[np.arange(probs.shape[0]), top_idx]
            top2_logits = np.sort(logits, axis=1)[:, -2:]
            margin = top2_logits[:, 1] - top2_logits[:, 0]
            top_idx_batches.append(top_idx)
            top_prob_batches.append(top_prob)
            margin_batches.append(margin)
    else:
        import torch

        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = (x[start : start + batch_size] - scaler_mean) / scaler_scale
                xb_t = torch.from_numpy(xb.astype(np.float32, copy=False))
                logits = model(xb_t)
                probs = torch.softmax(logits, dim=1)
                top_prob, top_idx = probs.max(dim=1)
                top2_logits = torch.topk(logits, k=2, dim=1).values
                margin = top2_logits[:, 0] - top2_logits[:, 1]
                top_idx_batches.append(top_idx.cpu().numpy())
                top_prob_batches.append(top_prob.cpu().numpy())
                margin_batches.append(margin.cpu().numpy())

    top_idx_all = np.concatenate(top_idx_batches)
    top_types = np.asarray([class_names[int(ii)] for ii in top_idx_all], dtype=object)
    top_probs = np.concatenate(top_prob_batches).astype(np.float32)
    margins = np.concatenate(margin_batches).astype(np.float32)
    return top_types, top_probs, margins


def _filter_endpoint_pairs(
    src_types: np.ndarray,
    dst_types: np.ndarray,
    reachable: Dict[str, Set[str]],
) -> np.ndarray:
    return np.asarray(
        [
            endpoint_valid(str(src), str(dst), reachable)
            for src, dst in zip(src_types, dst_types)
        ],
        dtype=bool,
    )


def path_validity_from_predictions(
    start_type_ids: np.ndarray,
    pred_types: np.ndarray,
    top_probs: np.ndarray,
    margins: np.ndarray,
    class_names: Sequence[str],
    reachable: Dict[str, Set[str]],
    prob_threshold: float,
    margin_threshold: float,
    final_type_ids: np.ndarray | None = None,
) -> Dict[str, np.ndarray | int]:
    """Check lineage monotonicity from classifier predictions along paths."""
    valid = np.ones(pred_types.shape[0], dtype=bool)
    rejected_at_checkpoint = np.zeros(pred_types.shape[0], dtype=bool)
    rejected_at_final = np.zeros(pred_types.shape[0], dtype=bool)
    confident = (top_probs >= prob_threshold) & (margins >= margin_threshold)

    for ii in range(pred_types.shape[0]):
        curr = class_names[int(start_type_ids[ii])]
        for jj in range(pred_types.shape[1]):
            if not confident[ii, jj]:
                continue
            pred = str(pred_types[ii, jj])
            if not endpoint_valid(curr, pred, reachable):
                valid[ii] = False
                rejected_at_checkpoint[ii] = True
                break
            curr = pred

        if not valid[ii] or final_type_ids is None:
            continue
        final_type = class_names[int(final_type_ids[ii])]
        if not endpoint_valid(curr, final_type, reachable):
            valid[ii] = False
            rejected_at_final[ii] = True

    return {
        "valid": valid,
        "confident": confident,
        "rejected_at_checkpoint": rejected_at_checkpoint,
        "rejected_at_final": rejected_at_final,
        "n_confident": int(confident.sum()),
        "n_points": int(confident.size),
    }


def check_paths_with_classifier(
    paths: np.ndarray,
    start_type_ids: np.ndarray,
    classifier_path: str | Path,
    prob_threshold: float,
    margin_threshold: float,
    final_type_ids: np.ndarray | None = None,
    classifier_batch_size: int = 8192,
    lineage_transition_mode: str | None = "descendant",
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> Dict[str, np.ndarray | int]:
    """Classify path points and apply the Maizels transition prior."""
    model, class_names, scaler_mean, scaler_scale = load_classifier(classifier_path)
    reachable = build_transition_reachable(
        lineage_transition_mode,
        edges=transition_edges,
        class_names=class_names,
    )
    flat = np.asarray(paths, dtype=np.float32).reshape((-1, paths.shape[-1]))
    pred_flat, prob_flat, margin_flat = classifier_predictions(
        model,
        class_names,
        scaler_mean,
        scaler_scale,
        flat,
        batch_size=classifier_batch_size,
    )
    pred_types = pred_flat.reshape(paths.shape[0], paths.shape[1])
    top_probs = prob_flat.reshape(paths.shape[0], paths.shape[1])
    margins = margin_flat.reshape(paths.shape[0], paths.shape[1])
    return path_validity_from_predictions(
        start_type_ids=np.asarray(start_type_ids, dtype=np.int32),
        pred_types=pred_types,
        top_probs=top_probs,
        margins=margins,
        class_names=class_names,
        reachable=reachable,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=final_type_ids,
    )


def _check_candidate_interpolants(
    source_x: np.ndarray,
    source_type_ids: np.ndarray,
    target_x: np.ndarray,
    target_type_ids: np.ndarray,
    classifier_path: str | Path,
    n_check_times: int,
    prob_threshold: float,
    margin_threshold: float,
    classifier_batch_size: int,
    lineage_transition_mode: str,
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
    path_builder=None,
) -> Dict[str, np.ndarray | int]:
    taus = np.linspace(0.0, 1.0, n_check_times + 2, dtype=np.float32)[1:-1]
    if path_builder is None:
        paths = np.stack(
            [(1.0 - tau) * source_x + tau * target_x for tau in taus],
            axis=1,
        )
    else:
        paths = path_builder(source_x, target_x, taus)
        expected_shape = (source_x.shape[0], taus.shape[0], source_x.shape[1])
        if paths.shape != expected_shape:
            raise ValueError(
                "Maizels interpolant path builder returned shape "
                f"{paths.shape}, expected {expected_shape}."
            )
    return check_paths_with_classifier(
        paths=paths,
        start_type_ids=source_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=target_type_ids,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
        transition_edges=transition_edges,
    )


def _sample_pair_indices(
    rng: np.random.Generator,
    n_source: int,
    n_target: int,
    n_pairs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        rng.integers(0, n_source, size=n_pairs, endpoint=False),
        rng.integers(0, n_target, size=n_pairs, endpoint=False),
    )


def _hash_array(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    array = np.asarray(array)
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    if array.dtype == object:
        for item in array.tolist():
            hasher.update(str(item).encode("utf-8"))
            hasher.update(b"\0")
    else:
        hasher.update(np.ascontiguousarray(array).view(np.uint8))
    return hasher.hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _ot_cache_metadata(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    *,
    pair_mode: str,
    lineage_transition_mode: str,
    class_names: Sequence[str] = CLASS_NAMES,
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> Dict[str, Any]:
    pair_mode = _canonical_maizels_pair_mode(pair_mode)
    metadata = {
        "cache_version": str(getattr(cfg.problem, "ot_cache_version", "v1")),
        "pair_mode": str(pair_mode),
        "source_time": str(getattr(cfg.problem, "source_time", "D3")),
        "target_time": str(getattr(cfg.problem, "target_time", "D8")),
        "dataset_location": str(getattr(cfg.problem, "dataset_location", "")),
        "endpoint_transition_mode": (
            "none" if pair_mode == "ot_plain" else "descendant"
        ),
        "lineage_transition_mode": (
            "none" if pair_mode == "ot_plain" else str(lineage_transition_mode)
        ),
        "cost": "whitened_sqeuclidean_global_std_v1",
        "ot_mass_tolerance": float(getattr(cfg.problem, "ot_mass_tolerance", 1e-12)),
        "ot_drop_orphan_cells": bool(
            getattr(cfg.problem, "ot_drop_orphan_cells", True)
        ),
        "source_x_hash": _hash_array(source_x.astype(np.float32, copy=False)),
        "target_x_hash": _hash_array(target_x.astype(np.float32, copy=False)),
        "source_types_hash": _hash_array(source_types.astype(object, copy=False)),
        "target_types_hash": _hash_array(target_types.astype(object, copy=False)),
        "source_n": int(source_x.shape[0]),
        "target_n": int(target_x.shape[0]),
        "dim": int(source_x.shape[1]),
        "class_names": [str(name) for name in class_names],
        "transition_edges": [list(edge) for edge in transition_edges],
    }
    if pair_mode == "ot_endpoint_interpolant":
        classifier_path = resolve_classifier_path(
            getattr(cfg.problem, "classifier_path", None)
        )
        metadata.update(
            {
                "classifier": _file_fingerprint(classifier_path),
                "n_interpolant_check_times": int(
                    getattr(cfg.problem, "n_interpolant_check_times", 5)
                ),
                "classifier_prob_threshold": float(
                    getattr(cfg.problem, "classifier_prob_threshold", 0.85)
                ),
                "classifier_margin_threshold": float(
                    getattr(cfg.problem, "classifier_margin_threshold", 1.0)
                ),
                "interpolant_path_kind": str(
                    getattr(cfg.problem, "interpolant_path_kind", "linear")
                ),
                "ot_infeasible_fallback": str(
                    getattr(cfg.problem, "ot_infeasible_fallback", "error")
                ),
            }
        )
    return metadata


def _ot_cache_path(cfg, metadata: Dict[str, Any]) -> Path | None:
    if not bool(getattr(cfg.problem, "ot_cache_enabled", True)):
        return None

    cache_dir = str(getattr(cfg.problem, "ot_cache_dir", "") or "")
    if cache_dir:
        root = Path(cache_dir).expanduser()
    else:
        output_folder = str(
            getattr(getattr(cfg, "logging", None), "output_folder", "") or ""
        )
        if output_folder:
            namespace = str(
                getattr(cfg.problem, "lineage_dataset_name", "maizels")
            ).replace(os.sep, "_")
            root = Path(output_folder).expanduser() / f"{namespace}_ot_cache"
        else:
            location = Path(
                str(getattr(cfg.problem, "dataset_location", "") or DEFAULT_DATASET)
            ).expanduser()
            root = (location.parent if location.suffix else location) / ".lineage_ot_cache"

    key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    version = str(metadata.get("cache_version", "v1")).replace(os.sep, "_")
    namespace = str(
        getattr(cfg.problem, "lineage_dataset_name", "maizels")
    ).replace(os.sep, "_")
    return root / f"{namespace}_exact_ot_{version}_{key}.npz"


def _load_cached_ot_plan(cache_path: Path):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as raw:
            source_idx = raw["source_idx"].astype(np.int64, copy=False)
            target_idx = raw["target_idx"].astype(np.int64, copy=False)
            mass = raw["mass"].astype(np.float64, copy=False)
            stats = json.loads(str(raw["stats_json"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"Ignoring unreadable Maizels OT cache {cache_path}: {exc}")
        return None

    mass_sum = float(mass.sum())
    if mass_sum <= 0.0:
        print(f"Ignoring empty Maizels OT cache {cache_path}.")
        return None
    mass = mass / mass_sum
    stats["ot_cache_hit"] = True
    stats["ot_cache_path"] = str(cache_path)
    return source_idx, target_idx, mass, stats


def _save_cached_ot_plan(
    cache_path: Path,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    mass: np.ndarray,
    stats: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    stats_to_save = dict(stats)
    stats_to_save["ot_cache_hit"] = False
    stats_to_save["ot_cache_path"] = str(cache_path)
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            source_idx=source_idx.astype(np.int64, copy=False),
            target_idx=target_idx.astype(np.int64, copy=False),
            mass=mass.astype(np.float64, copy=False),
            stats_json=json.dumps(stats_to_save, sort_keys=True),
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
    os.replace(tmp_path, cache_path)


def _progress_bar(total: int, desc: str, enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        return None
    return tqdm(total=total, desc=desc, unit="pair")


def _type_index_groups(types: np.ndarray) -> Dict[str, np.ndarray]:
    groups = {}
    for cell_type in np.unique(types):
        groups[str(cell_type)] = np.flatnonzero(types == cell_type).astype(np.int64)
    return groups


def _pairwise_whitened_sqeuclidean(
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    diff = (
        source_x[source_idx].astype(np.float64)
        - target_x[target_idx].astype(np.float64)
    ) / scale
    return np.sum(diff * diff, axis=1)


def _append_ot_edges(
    edge_source_idx: List[np.ndarray],
    edge_target_idx: List[np.ndarray],
    edge_costs: List[np.ndarray],
    source_x: np.ndarray,
    target_x: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    scale: np.ndarray,
) -> None:
    if source_idx.shape[0] == 0:
        return
    edge_source_idx.append(source_idx.astype(np.int64, copy=False))
    edge_target_idx.append(target_idx.astype(np.int64, copy=False))
    edge_costs.append(
        _pairwise_whitened_sqeuclidean(
            source_x,
            target_x,
            source_idx,
            target_idx,
            scale,
        ).astype(np.float64, copy=False)
    )


def _collect_exact_ot_edges(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    source_type_ids_all: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    target_type_ids_all: np.ndarray,
    *,
    pair_mode: str,
    endpoint_reachable: Dict[str, Set[str]],
    lineage_transition_mode: str,
    transition_edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Enumerate candidate edges for exact Maizels OT coupling."""
    pair_mode = _canonical_maizels_pair_mode(pair_mode)
    plain_ot = pair_mode == "ot_plain"
    classifier_path = None
    if pair_mode == "ot_endpoint_interpolant":
        classifier_path = resolve_classifier_path(
            getattr(cfg.problem, "classifier_path", None)
        )
    chunk_size = int(
        getattr(
            cfg.problem,
            "ot_candidate_chunk_size",
            getattr(cfg.problem, "rejection_chunk_size", 50_000),
        )
    )
    chunk_size = max(1, chunk_size)
    n_check_times = int(getattr(cfg.problem, "n_interpolant_check_times", 5))
    prob_threshold = float(getattr(cfg.problem, "classifier_prob_threshold", 0.85))
    margin_threshold = float(getattr(cfg.problem, "classifier_margin_threshold", 1.0))
    classifier_batch_size = int(getattr(cfg.problem, "classifier_batch_size", 8192))

    if plain_ot:
        source_groups = {
            "__all__": np.arange(source_x.shape[0], dtype=np.int64),
        }
        target_groups = {
            "__all__": np.arange(target_x.shape[0], dtype=np.int64),
        }
    else:
        source_groups = _type_index_groups(source_types)
        target_groups = _type_index_groups(target_types)
    all_x = np.concatenate([source_x, target_x], axis=0).astype(np.float64)
    cost_scale = np.std(all_x, axis=0)
    cost_scale = np.where(cost_scale > 1e-6, cost_scale, 1.0)
    edge_source_idx: List[np.ndarray] = []
    edge_target_idx: List[np.ndarray] = []
    edge_costs: List[np.ndarray] = []
    total_pairs = int(source_x.shape[0] * target_x.shape[0])
    stats = {
        "candidate_pairs": total_pairs,
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "accepted_pairs": 0,
        "ot_edge_mode": "full_support" if plain_ot else "hard_mask",
    }

    progress = _progress_bar(
        total_pairs,
        "Maizels OT full support" if plain_ot else "Maizels OT hard mask",
        bool(getattr(cfg.problem, "ot_progress_enabled", True)),
    )
    try:
        for src_type, sidx_group in source_groups.items():
            for dst_type, tidx_group in target_groups.items():
                n_block = int(sidx_group.shape[0] * tidx_group.shape[0])
                if n_block == 0:
                    continue
                if not plain_ot and not endpoint_valid(
                    src_type, dst_type, endpoint_reachable
                ):
                    stats["endpoint_rejected"] += n_block
                    if progress is not None:
                        progress.update(n_block)
                    continue

                n_target_group = tidx_group.shape[0]
                for start in range(0, n_block, chunk_size):
                    end = min(start + chunk_size, n_block)
                    flat = np.arange(start, end, dtype=np.int64)
                    source_idx = sidx_group[flat // n_target_group]
                    target_idx = tidx_group[flat % n_target_group]

                    keep = np.ones(source_idx.shape[0], dtype=bool)
                    if pair_mode == "ot_endpoint_interpolant":
                        validity = _check_candidate_interpolants(
                            source_x=source_x[source_idx],
                            source_type_ids=source_type_ids_all[source_idx],
                            target_x=target_x[target_idx],
                            target_type_ids=target_type_ids_all[target_idx],
                            classifier_path=classifier_path,
                            n_check_times=n_check_times,
                            prob_threshold=prob_threshold,
                            margin_threshold=margin_threshold,
                            classifier_batch_size=classifier_batch_size,
                            lineage_transition_mode=lineage_transition_mode,
                            transition_edges=transition_edges,
                            path_builder=getattr(
                                cfg.problem,
                                "interpolant_path_builder",
                                None,
                            ),
                        )
                        keep = np.asarray(validity["valid"], dtype=bool)
                        stats["interpolant_rejected"] += int((~keep).sum())

                    if keep.any():
                        _append_ot_edges(
                            edge_source_idx,
                            edge_target_idx,
                            edge_costs,
                            source_x,
                            target_x,
                            source_idx[keep],
                            target_idx[keep],
                            cost_scale,
                        )
                        stats["accepted_pairs"] += int(keep.sum())

                    if progress is not None:
                        progress.update(end - start)
    finally:
        if progress is not None:
            progress.close()

    if not edge_source_idx:
        raise RuntimeError("Maizels exact OT found no candidate edges.")

    return (
        np.concatenate(edge_source_idx),
        np.concatenate(edge_target_idx),
        np.concatenate(edge_costs),
        stats,
    )


def _solve_sparse_exact_ot(
    n_source: int,
    n_target: int,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    costs: np.ndarray,
    *,
    mass_tol: float,
    infeasible_fallback: str = "error",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve exact balanced OT on a sparse hard-valid edge set with HiGHS."""
    try:
        from scipy.optimize import linprog
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Exact Maizels OT coupling requires scipy.optimize.linprog."
        ) from exc

    if infeasible_fallback == "partial":
        return _solve_sparse_max_valid_partial_ot(
            n_source,
            n_target,
            source_idx,
            target_idx,
            costs,
            mass_tol=mass_tol,
        )
    if infeasible_fallback != "error":
        raise ValueError(
            "infeasible_fallback must be 'error' or 'partial', "
            f"got {infeasible_fallback!r}."
        )

    n_edges = int(source_idx.shape[0])
    edge_ids = np.arange(n_edges, dtype=np.int64)
    rows = np.concatenate([source_idx, n_source + target_idx])
    cols = np.concatenate([edge_ids, edge_ids])
    data = np.ones(2 * n_edges, dtype=np.float64)
    a_eq = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(n_source + n_target, n_edges),
    ).tocsr()
    b_eq = np.concatenate(
        [
            np.full(n_source, 1.0 / float(n_source), dtype=np.float64),
            np.full(n_target, 1.0 / float(n_target), dtype=np.float64),
        ]
    )

    result = linprog(
        costs.astype(np.float64, copy=False),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            "Exact masked Maizels OT was infeasible or failed to solve: "
            f"{result.message}"
        )

    plan_mass = np.asarray(result.x, dtype=np.float64)
    plan_mass = np.where(plan_mass > mass_tol, plan_mass, 0.0)
    total_mass = float(plan_mass.sum())
    if total_mass <= 0.0:
        raise RuntimeError("Exact masked Maizels OT returned zero positive mass.")
    if abs(total_mass - 1.0) > 1e-6:
        plan_mass = plan_mass / total_mass

    source_mass = np.bincount(source_idx, weights=plan_mass, minlength=n_source)
    target_mass = np.bincount(target_idx, weights=plan_mass, minlength=n_target)
    source_target = np.full(n_source, 1.0 / float(n_source), dtype=np.float64)
    target_target = np.full(n_target, 1.0 / float(n_target), dtype=np.float64)
    stats = {
        "ot_edges": n_edges,
        "ot_positive_edges": int(np.count_nonzero(plan_mass)),
        "ot_objective": float(np.dot(plan_mass, costs)),
        "ot_total_mass": float(plan_mass.sum()),
        "ot_source_max_abs_residual": float(
            np.max(np.abs(source_mass - source_target))
        ),
        "ot_target_max_abs_residual": float(
            np.max(np.abs(target_mass - target_target))
        ),
    }
    return plan_mass, stats


def _solve_sparse_max_valid_partial_ot(
    n_source: int,
    n_target: int,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    costs: np.ndarray,
    *,
    mass_tol: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve maximum-valid-mass partial OT without relaxing the hard edge mask.

    One dummy source and target make the transport LP feasible. A penalty on
    unmatched endpoint mass first maximizes mass on valid real edges and then
    minimizes its transport cost. The valid mass is renormalized for sampling.
    """
    try:
        from scipy.optimize import linprog
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Partial Maizels OT coupling requires scipy.optimize.linprog."
        ) from exc

    n_edges = int(source_idx.shape[0])
    dummy_source = n_source
    dummy_target = n_target
    augmented_source_idx = np.concatenate(
        [
            source_idx,
            np.arange(n_source, dtype=np.int64),
            np.full(n_target, dummy_source, dtype=np.int64),
            np.asarray([dummy_source], dtype=np.int64),
        ]
    )
    augmented_target_idx = np.concatenate(
        [
            target_idx,
            np.full(n_source, dummy_target, dtype=np.int64),
            np.arange(n_target, dtype=np.int64),
            np.asarray([dummy_target], dtype=np.int64),
        ]
    )
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.shape[0] != costs.shape[0]:
        raise RuntimeError("Masked Maizels OT received non-finite edge costs.")
    unmatched_penalty = max(1.0, float(np.max(finite_costs)) + 1.0)
    augmented_costs = np.concatenate(
        [
            costs.astype(np.float64, copy=False),
            np.full(n_source + n_target, unmatched_penalty, dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        ]
    )

    augmented_n_source = n_source + 1
    augmented_n_target = n_target + 1
    augmented_n_edges = int(augmented_source_idx.shape[0])
    edge_ids = np.arange(augmented_n_edges, dtype=np.int64)
    rows = np.concatenate(
        [augmented_source_idx, augmented_n_source + augmented_target_idx]
    )
    cols = np.concatenate([edge_ids, edge_ids])
    data = np.ones(2 * augmented_n_edges, dtype=np.float64)
    a_eq = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(augmented_n_source + augmented_n_target, augmented_n_edges),
    ).tocsr()
    b_eq = np.concatenate(
        [
            np.full(n_source, 1.0 / float(n_source), dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.full(n_target, 1.0 / float(n_target), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        ]
    )

    result = linprog(
        augmented_costs,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            "Maximum-valid-mass partial Maizels OT failed to solve: "
            f"{result.message}"
        )

    raw_valid_mass = np.asarray(result.x[:n_edges], dtype=np.float64)
    raw_valid_mass = np.where(raw_valid_mass > mass_tol, raw_valid_mass, 0.0)
    retained_mass = float(raw_valid_mass.sum())
    if retained_mass <= 0.0:
        raise RuntimeError("Partial masked Maizels OT retained zero valid mass.")
    plan_mass = raw_valid_mass / retained_mass

    source_mass = np.bincount(source_idx, weights=plan_mass, minlength=n_source)
    target_mass = np.bincount(target_idx, weights=plan_mass, minlength=n_target)
    source_target = np.full(n_source, 1.0 / float(n_source), dtype=np.float64)
    target_target = np.full(n_target, 1.0 / float(n_target), dtype=np.float64)
    stats = {
        "ot_edges": n_edges,
        "ot_positive_edges": int(np.count_nonzero(plan_mass)),
        "ot_objective": float(np.dot(plan_mass, costs)),
        "ot_total_mass": float(plan_mass.sum()),
        "ot_source_max_abs_residual": float(
            np.max(np.abs(source_mass - source_target))
        ),
        "ot_target_max_abs_residual": float(
            np.max(np.abs(target_mass - target_target))
        ),
        "ot_solver_mode": "max_valid_partial",
        "ot_retained_valid_mass": retained_mass,
        "ot_unmatched_source_mass": max(0.0, 1.0 - retained_mass),
        "ot_unmatched_target_mass": max(0.0, 1.0 - retained_mass),
        "ot_source_cells_with_mass": int(np.count_nonzero(source_mass > mass_tol)),
        "ot_target_cells_with_mass": int(np.count_nonzero(target_mass > mass_tol)),
    }
    return plan_mass, stats


def _make_exact_ot_pair_pool_from_endpoint_arrays(
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create a pair pool by sampling from an exact OT plan."""
    pair_mode = _canonical_maizels_pair_mode(pair_mode)
    plain_ot = pair_mode == "ot_plain"
    class_to_id = class_to_id_map(class_names)
    source_type_ids_all = np.asarray(
        [class_to_id[str(ct)] for ct in source_types],
        dtype=np.int32,
    )
    target_type_ids_all = np.asarray(
        [class_to_id[str(ct)] for ct in target_types],
        dtype=np.int32,
    )
    lineage_transition_mode = lineage_transition_mode_from_config(cfg)
    endpoint_reachable = build_transition_reachable(
        "descendant",
        edges=transition_edges,
        class_names=class_names,
    )
    mass_tol = float(getattr(cfg.problem, "ot_mass_tolerance", 1e-12))
    verbose = bool(getattr(cfg.problem, "ot_verbose", True))
    metadata = _ot_cache_metadata(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        pair_mode=pair_mode,
        lineage_transition_mode=lineage_transition_mode,
        class_names=class_names,
        transition_edges=transition_edges,
    )
    cache_path = _ot_cache_path(cfg, metadata)
    cached = _load_cached_ot_plan(cache_path) if cache_path is not None else None

    if cached is not None:
        positive_source_idx, positive_target_idx, positive_mass, stats = cached
        if verbose:
            print(
                "Loaded cached Maizels exact OT plan: "
                f"{cache_path} "
                f"(positive_edges={positive_mass.shape[0]})"
            )
    else:
        if verbose and cache_path is not None:
            print(f"Maizels exact OT cache miss: {cache_path}")

        source_idx_edges, target_idx_edges, costs, stats = _collect_exact_ot_edges(
            cfg,
            source_x,
            source_types,
            source_type_ids_all,
            target_x,
            target_types,
            target_type_ids_all,
            pair_mode=pair_mode,
            endpoint_reachable=endpoint_reachable,
            lineage_transition_mode=lineage_transition_mode,
            transition_edges=transition_edges,
        )
        source_has_edge = (
            np.bincount(
                source_idx_edges,
                minlength=source_x.shape[0],
            )
            > 0
        )
        target_has_edge = (
            np.bincount(
                target_idx_edges,
                minlength=target_x.shape[0],
            )
            > 0
        )
        n_orphan_source = int((~source_has_edge).sum())
        n_orphan_target = int((~target_has_edge).sum())
        drop_orphans = bool(getattr(cfg.problem, "ot_drop_orphan_cells", True))
        if not source_has_edge.all() or not target_has_edge.all():
            if not drop_orphans:
                raise RuntimeError(
                    "Maizels exact OT is infeasible before solving: "
                    f"{n_orphan_source} source cells and {n_orphan_target} "
                    "target cells have no candidate edges. Set "
                    "problem.ot_drop_orphan_cells=True to solve exact OT on "
                    "the non-orphan subproblem."
                )
            if verbose:
                orphan_reason = "OT partners" if plain_ot else "hard-valid partners"
                print(
                    f"Dropping Maizels OT orphan cells with no {orphan_reason}: "
                    f"sources={n_orphan_source}, targets={n_orphan_target}"
                )

        active_source_idx = np.flatnonzero(source_has_edge).astype(np.int64)
        active_target_idx = np.flatnonzero(target_has_edge).astype(np.int64)
        source_remap = -np.ones(source_x.shape[0], dtype=np.int64)
        target_remap = -np.ones(target_x.shape[0], dtype=np.int64)
        source_remap[active_source_idx] = np.arange(active_source_idx.shape[0])
        target_remap[active_target_idx] = np.arange(active_target_idx.shape[0])
        remapped_source_edges = source_remap[source_idx_edges]
        remapped_target_edges = target_remap[target_idx_edges]
        active_edge_mask = (remapped_source_edges >= 0) & (remapped_target_edges >= 0)
        source_idx_edges = source_idx_edges[active_edge_mask]
        target_idx_edges = target_idx_edges[active_edge_mask]
        costs = costs[active_edge_mask]
        remapped_source_edges = remapped_source_edges[active_edge_mask]
        remapped_target_edges = remapped_target_edges[active_edge_mask]
        stats["ot_dropped_source_cells"] = n_orphan_source
        stats["ot_dropped_target_cells"] = n_orphan_target
        stats["ot_active_source_cells"] = int(active_source_idx.shape[0])
        stats["ot_active_target_cells"] = int(active_target_idx.shape[0])

        infeasible_fallback = str(
            getattr(cfg.problem, "ot_infeasible_fallback", "error")
        )
        if verbose:
            solver_description = (
                "maximum-valid-mass partial OT"
                if infeasible_fallback == "partial"
                else "exact OT"
            )
            print(
                f"Solving Maizels {solver_description} LP: "
                f"sources={active_source_idx.shape[0]}, "
                f"targets={active_target_idx.shape[0]}, "
                f"valid_edges={source_idx_edges.shape[0]}"
            )
        plan_mass, ot_stats = _solve_sparse_exact_ot(
            active_source_idx.shape[0],
            active_target_idx.shape[0],
            remapped_source_edges,
            remapped_target_edges,
            costs,
            mass_tol=mass_tol,
            infeasible_fallback=infeasible_fallback,
        )
        stats.update(ot_stats)

        positive = plan_mass > 0.0
        positive_source_idx = source_idx_edges[positive]
        positive_target_idx = target_idx_edges[positive]
        positive_mass = plan_mass[positive]
        positive_mass = positive_mass / positive_mass.sum()
        stats["ot_cache_hit"] = False
        stats["ot_cache_path"] = "" if cache_path is None else str(cache_path)

        if verbose:
            print(
                "Solved Maizels OT LP: "
                f"objective={stats['ot_objective']:.6g}, "
                f"positive_edges={stats['ot_positive_edges']}, "
                f"retained_valid_mass={stats.get('ot_retained_valid_mass', 1.0):.6g}"
            )

        if cache_path is not None:
            try:
                _save_cached_ot_plan(
                    cache_path,
                    positive_source_idx,
                    positive_target_idx,
                    positive_mass,
                    stats,
                    metadata,
                )
                if verbose:
                    print(f"Saved Maizels exact OT plan cache: {cache_path}")
            except OSError as exc:
                print(f"Failed to save Maizels exact OT cache {cache_path}: {exc}")

    sampled_edge_idx = rng.choice(
        positive_mass.shape[0],
        size=n_pairs,
        replace=True,
        p=positive_mass,
    )
    source_idx = positive_source_idx[sampled_edge_idx]
    target_idx = positive_target_idx[sampled_edge_idx]
    labels = np.stack(
        [source_type_ids_all[source_idx], target_type_ids_all[target_idx]],
        axis=1,
    ).astype(np.int32)

    paired = {
        "x0": source_x[source_idx].astype(np.float32),
        "x1": target_x[target_idx].astype(np.float32),
        "label": labels,
    }
    stats["ot_cache_enabled"] = cache_path is not None
    stats["ot_cache_path"] = "" if cache_path is None else str(cache_path)
    stats["sampled_pairs"] = int(paired["x0"].shape[0])
    stats["collected_accepted_pairs"] = int(stats.get("accepted_pairs", 0))
    stats["truncated_accepted_pairs"] = 0
    if "candidate_acceptance_rate" not in stats:
        stats["candidate_acceptance_rate"] = (
            stats["accepted_pairs"] / stats["candidate_pairs"]
            if stats["candidate_pairs"] > 0
            else 0.0
        )
    stats["source_n"] = int(source_x.shape[0])
    stats["target_n"] = int(target_x.shape[0])
    stats["source_counts"] = dict(Counter(source_types.tolist()))
    stats["target_counts"] = dict(Counter(target_types.tolist()))
    stats["endpoint_transition_mode"] = "none" if plain_ot else "descendant"
    stats["lineage_transition_mode"] = lineage_transition_mode
    stats["ot_cost"] = "whitened_sqeuclidean"
    return paired, stats


def _split_train_holdout_indices(
    n_items: int,
    *,
    holdout_fraction: float,
    holdout_n: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/holdout row indices for an endpoint pool."""
    if n_items <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    holdout_fraction = max(0.0, min(float(holdout_fraction), 1.0))
    holdout_n = int(holdout_n)
    if holdout_n <= 0 and holdout_fraction > 0.0:
        holdout_n = max(1, int(round(holdout_fraction * n_items)))
    holdout_n = max(0, min(holdout_n, max(n_items - 1, 0)))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_items)
    holdout_idx = np.sort(perm[:holdout_n]).astype(np.int64)
    train_idx = np.sort(perm[holdout_n:]).astype(np.int64)
    return train_idx, holdout_idx


def timepoint_pool_splits(
    cfg, dataset_location: str | None = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load Maizels pools while preserving the existing endpoint holdouts."""
    dataset_path = resolve_dataset_path(
        dataset_location or getattr(cfg.problem, "dataset_location", None)
    )
    data = load_pca50_dataset(dataset_path)
    configured_order = tuple(
        str(value)
        for value in getattr(cfg.problem, "timepoint_order", TIMEPOINTS)
    )
    available = set(str(value) for value in np.unique(data["timepoints"]))
    unknown = [timepoint for timepoint in configured_order if timepoint not in available]
    if unknown:
        raise ValueError(f"Unknown configured Maizels timepoints: {unknown}.")

    training_seed = int(getattr(getattr(cfg, "training", None), "seed", 0))
    split_seed = int(getattr(cfg.problem, "maizels_holdout_seed", training_seed + 701))
    holdout_fraction = float(getattr(cfg.problem, "maizels_holdout_fraction", 0.0))
    holdout_n = int(getattr(cfg.problem, "maizels_holdout_n", 0))

    result: Dict[str, Dict[str, np.ndarray]] = {}
    source_time = str(getattr(cfg.problem, "source_time", "D3"))
    target_time = str(getattr(cfg.problem, "target_time", "D8"))
    retained = set(retained_timepoints(cfg))
    for index, timepoint in enumerate(configured_order):
        x, cell_types = subset_time(data, timepoint)
        if timepoint == source_time:
            split_offset = 11
        elif timepoint == target_time:
            split_offset = 29
        elif timepoint in retained:
            split_offset = 101 * (index + 1)
        else:
            split_offset = None
        if split_offset is None:
            # Omitted evaluation days never enter training or validation.
            train_idx = np.arange(x.shape[0], dtype=np.int64)
            holdout_idx = np.empty((0,), dtype=np.int64)
        else:
            train_idx, holdout_idx = _split_train_holdout_indices(
                x.shape[0],
                holdout_fraction=holdout_fraction,
                holdout_n=holdout_n,
                seed=split_seed + split_offset,
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


def _endpoint_split_dict(source, target) -> Dict[str, np.ndarray]:
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


def endpoint_pool_splits(
    cfg, dataset_location: str | None = None
) -> Dict[str, np.ndarray]:
    """Load D3/D8 endpoint pools and apply the configured held-out split."""
    if uses_retained_intervals(cfg):
        pools = timepoint_pool_splits(cfg, dataset_location=dataset_location)
        source_time = str(getattr(cfg.problem, "source_time", "D3"))
        target_time = str(getattr(cfg.problem, "target_time", "D8"))
        return _endpoint_split_dict(pools[source_time], pools[target_time])

    dataset_path = resolve_dataset_path(
        dataset_location or getattr(cfg.problem, "dataset_location", None)
    )
    data = load_pca50_dataset(dataset_path)
    source_time = getattr(cfg.problem, "source_time", "D3")
    target_time = getattr(cfg.problem, "target_time", "D8")
    source_x, source_types = subset_time(data, source_time)
    target_x, target_types = subset_time(data, target_time)

    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError(
            f"Could not find Maizels endpoint pools for {source_time} -> {target_time}."
        )

    training_seed = int(getattr(getattr(cfg, "training", None), "seed", 0))
    split_seed = int(getattr(cfg.problem, "maizels_holdout_seed", training_seed + 701))
    holdout_fraction = float(getattr(cfg.problem, "maizels_holdout_fraction", 0.0))
    holdout_n = int(getattr(cfg.problem, "maizels_holdout_n", 0))
    source_train_idx, source_holdout_idx = _split_train_holdout_indices(
        source_x.shape[0],
        holdout_fraction=holdout_fraction,
        holdout_n=holdout_n,
        seed=split_seed + 11,
    )
    target_train_idx, target_holdout_idx = _split_train_holdout_indices(
        target_x.shape[0],
        holdout_fraction=holdout_fraction,
        holdout_n=holdout_n,
        seed=split_seed + 29,
    )

    source = {
        "x": source_x,
        "types": source_types,
        "train_x": source_x[source_train_idx],
        "train_types": source_types[source_train_idx],
        "holdout_x": source_x[source_holdout_idx],
        "holdout_types": source_types[source_holdout_idx],
        "train_idx": source_train_idx,
        "holdout_idx": source_holdout_idx,
    }
    target = {
        "x": target_x,
        "types": target_types,
        "train_x": target_x[target_train_idx],
        "train_types": target_types[target_train_idx],
        "holdout_x": target_x[target_holdout_idx],
        "holdout_types": target_types[target_holdout_idx],
        "train_idx": target_train_idx,
        "holdout_idx": target_holdout_idx,
    }
    return _endpoint_split_dict(source, target)


def _make_pair_pool_from_endpoint_arrays(
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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create independent or prior-filtered pairs from provided endpoint arrays."""
    pair_mode = _canonical_maizels_pair_mode(pair_mode)
    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError("Maizels source/target pair pools must both be non-empty.")

    if pair_mode in ("ot_plain", "ot_endpoint", "ot_endpoint_interpolant"):
        return _make_exact_ot_pair_pool_from_endpoint_arrays(
            cfg,
            source_x,
            source_types,
            target_x,
            target_types,
            n_pairs=n_pairs,
            rng=rng,
            pair_mode=pair_mode,
            class_names=class_names,
            transition_edges=transition_edges,
        )

    class_to_id = class_to_id_map(class_names)
    source_type_ids_all = np.asarray(
        [class_to_id[str(ct)] for ct in source_types], dtype=np.int32
    )
    target_type_ids_all = np.asarray(
        [class_to_id[str(ct)] for ct in target_types], dtype=np.int32
    )
    lineage_transition_mode = lineage_transition_mode_from_config(cfg)
    endpoint_reachable = build_transition_reachable(
        "descendant",
        edges=transition_edges,
        class_names=class_names,
    )

    accepted_source_idx: List[np.ndarray] = []
    accepted_target_idx: List[np.ndarray] = []
    stats = {
        "candidate_pairs": 0,
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "accepted_pairs": 0,
    }

    if pair_mode == "none":
        sidx, tidx = _sample_pair_indices(
            rng, source_x.shape[0], target_x.shape[0], n_pairs
        )
        accepted_source_idx.append(sidx)
        accepted_target_idx.append(tidx)
        stats["candidate_pairs"] = n_pairs
        stats["accepted_pairs"] = n_pairs
    elif pair_mode in ("endpoint", "endpoint_interpolant"):
        classifier_path = resolve_classifier_path(
            getattr(cfg.problem, "classifier_path", None)
        )
        chunk_size = int(getattr(cfg.problem, "rejection_chunk_size", 50_000))
        max_candidates = int(
            getattr(
                cfg.problem, "rejection_max_candidates", max(10 * n_pairs, n_pairs + 1)
            )
        )
        n_check_times = int(getattr(cfg.problem, "n_interpolant_check_times", 5))
        prob_threshold = float(getattr(cfg.problem, "classifier_prob_threshold", 0.85))
        margin_threshold = float(
            getattr(cfg.problem, "classifier_margin_threshold", 1.0)
        )
        classifier_batch_size = int(getattr(cfg.problem, "classifier_batch_size", 8192))

        while (
            stats["accepted_pairs"] < n_pairs
            and stats["candidate_pairs"] < max_candidates
        ):
            remaining = n_pairs - stats["accepted_pairs"]
            adaptive_chunk = max(2 * remaining, min(chunk_size, 4096))
            curr_chunk = min(
                chunk_size,
                adaptive_chunk,
                max_candidates - stats["candidate_pairs"],
            )
            sidx, tidx = _sample_pair_indices(
                rng,
                source_x.shape[0],
                target_x.shape[0],
                curr_chunk,
            )
            stats["candidate_pairs"] += curr_chunk
            src_types = source_types[sidx]
            dst_types = target_types[tidx]
            endpoint_ok = _filter_endpoint_pairs(
                src_types,
                dst_types,
                endpoint_reachable,
            )
            stats["endpoint_rejected"] += int((~endpoint_ok).sum())
            if not endpoint_ok.any():
                continue

            sidx_ok = sidx[endpoint_ok]
            tidx_ok = tidx[endpoint_ok]
            keep = np.ones(sidx_ok.shape[0], dtype=bool)
            if pair_mode == "endpoint_interpolant":
                validity = _check_candidate_interpolants(
                    source_x=source_x[sidx_ok],
                    source_type_ids=source_type_ids_all[sidx_ok],
                    target_x=target_x[tidx_ok],
                    target_type_ids=target_type_ids_all[tidx_ok],
                    classifier_path=classifier_path,
                    n_check_times=n_check_times,
                    prob_threshold=prob_threshold,
                    margin_threshold=margin_threshold,
                    classifier_batch_size=classifier_batch_size,
                    lineage_transition_mode=lineage_transition_mode,
                    transition_edges=transition_edges,
                    path_builder=getattr(
                        cfg.problem,
                        "interpolant_path_builder",
                        None,
                    ),
                )
                keep = np.asarray(validity["valid"], dtype=bool)
                stats["interpolant_rejected"] += int((~keep).sum())

            if keep.any():
                accepted_source_idx.append(sidx_ok[keep])
                accepted_target_idx.append(tidx_ok[keep])
                stats["accepted_pairs"] += int(keep.sum())

        if stats["accepted_pairs"] < n_pairs:
            raise RuntimeError(
                "Maizels rejection sampling failed to collect enough accepted pairs: "
                f"{stats['accepted_pairs']} accepted from {stats['candidate_pairs']} candidates."
            )
    else:
        raise ValueError(
            "problem.maizels_pair_mode must be one of 'none', 'endpoint', "
            "'endpoint_interpolant', 'ot'/'ot_plain', 'ot_endpoint', or "
            f"'ot_endpoint_interpolant', got {pair_mode!r}."
        )

    collected_source_idx = np.concatenate(accepted_source_idx)
    collected_target_idx = np.concatenate(accepted_target_idx)
    collected_accepted = int(collected_source_idx.shape[0])
    source_idx = collected_source_idx[:n_pairs]
    target_idx = collected_target_idx[:n_pairs]
    labels = np.stack(
        [source_type_ids_all[source_idx], target_type_ids_all[target_idx]],
        axis=1,
    ).astype(np.int32)

    paired = {
        "x0": source_x[source_idx].astype(np.float32),
        "x1": target_x[target_idx].astype(np.float32),
        "label": labels,
    }
    stats["accepted_pairs"] = int(paired["x0"].shape[0])
    stats["collected_accepted_pairs"] = collected_accepted
    stats["truncated_accepted_pairs"] = max(
        0, collected_accepted - stats["accepted_pairs"]
    )
    stats["candidate_acceptance_rate"] = (
        collected_accepted / stats["candidate_pairs"]
        if stats["candidate_pairs"] > 0
        else 0.0
    )
    stats["source_n"] = int(source_x.shape[0])
    stats["target_n"] = int(target_x.shape[0])
    stats["source_counts"] = dict(Counter(source_types.tolist()))
    stats["target_counts"] = dict(Counter(target_types.tolist()))
    stats["endpoint_transition_mode"] = "descendant"
    stats["lineage_transition_mode"] = lineage_transition_mode
    return paired, stats


def _make_minibatch_ot_pair_pool_from_endpoint_arrays(
    cfg,
    source_x: np.ndarray,
    source_types: np.ndarray,
    target_x: np.ndarray,
    target_types: np.ndarray,
    *,
    n_pairs: int,
    rng: np.random.Generator,
    pair_mode: str,
    sample_with_replacement: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Couple one Maizels batch with exact OT in the raw PCA space."""
    # Imported lazily because cite_multi imports this module for the shared
    # lineage and classifier utilities.
    from . import cite_multi

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
    stats["ot_cost"] = "raw_sqeuclidean"
    return paired, stats


def _add_time_bounds(
    cfg,
    paired: Dict[str, np.ndarray],
    source_time: str,
    target_time: str,
) -> Dict[str, np.ndarray]:
    """Append the absolute interval occupied by every paired example."""
    n_pairs = paired["x0"].shape[0]
    bounds = np.tile(
        np.asarray(
            [
                normalized_time(source_time, cfg),
                normalized_time(target_time, cfg),
            ],
            dtype=np.float32,
        ),
        (n_pairs, 1),
    )
    result = dict(paired)
    result["label"] = np.concatenate(
        [paired["label"].astype(np.float32), bounds], axis=1
    )
    return result


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
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    paired, stats = _make_pair_pool_from_endpoint_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=pair_mode,
    )
    paired = _add_time_bounds(cfg, paired, source_time, target_time)
    stats.update(
        {
            "source_time": str(source_time),
            "target_time": str(target_time),
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "sampled_pairs": int(paired["x0"].shape[0]),
        }
    )
    return paired, stats


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
    sample_with_replacement: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    paired, stats = _make_minibatch_ot_pair_pool_from_endpoint_arrays(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=pair_mode,
        sample_with_replacement=sample_with_replacement,
    )
    paired = _add_time_bounds(cfg, paired, source_time, target_time)
    stats.update(
        {
            "source_time": str(source_time),
            "target_time": str(target_time),
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "sampled_pairs": int(paired["x0"].shape[0]),
        }
    )
    return paired, stats


def _allocate_pairs(total: int, n_intervals: int) -> List[int]:
    """Allocate a total pair budget equally over adjacent intervals."""
    if total < n_intervals:
        raise ValueError(
            f"problem.n={total} must be at least the number of intervals ({n_intervals})."
        )
    counts = [total // n_intervals] * n_intervals
    for index in range(total % n_intervals):
        counts[index] += 1
    return counts


def make_minibatch_ot_training_pools(
    cfg, dataset_location: str | None = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load compact Maizels train-timepoint pools for dynamic minibatch OT.

    Each original training cell is retained once. Optimizer batches sample
    endpoints from these populations directly, avoiding the expanded
    ``problem.n`` independent-pair pool used by the compatibility path.
    """
    pair_mode = _canonical_maizels_pair_mode(
        str(getattr(cfg.problem, "maizels_pair_mode", "none"))
    )
    if not uses_minibatch_ot(cfg, pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")

    include_time_bounds = uses_retained_intervals(cfg)
    retained = retained_timepoints(cfg)
    if include_time_bounds:
        split_pools = timepoint_pool_splits(cfg, dataset_location)
    else:
        splits = endpoint_pool_splits(cfg, dataset_location=dataset_location)
        source_time = str(getattr(cfg.problem, "source_time", retained[0]))
        target_time = str(getattr(cfg.problem, "target_time", retained[-1]))
        retained = (source_time, target_time)
        split_pools = {
            source_time: {
                "x": splits["source_x"],
                "types": splits["source_types"],
                "train_x": splits["source_train_x"],
                "train_types": splits["source_train_types"],
                "holdout_x": splits["source_holdout_x"],
                "holdout_types": splits["source_holdout_types"],
            },
            target_time: {
                "x": splits["target_x"],
                "types": splits["target_types"],
                "train_x": splits["target_train_x"],
                "train_types": splits["target_train_types"],
                "holdout_x": splits["target_holdout_x"],
                "holdout_types": splits["target_holdout_types"],
            },
        }

    interval_names = list(zip(retained[:-1], retained[1:]))
    nominal_n = int(getattr(cfg.problem, "n", 100_000))
    counts = _allocate_pairs(nominal_n, len(interval_names))
    timepoints = {}
    for timepoint in retained:
        split = split_pools[timepoint]
        x = np.asarray(split["train_x"], dtype=np.float32)
        if x.shape[0] == 0:
            raise RuntimeError(f"Maizels training pool for {timepoint} is empty.")
        timepoints[timepoint] = {
            "x": x,
            "type_ids": _cell_type_ids(split["train_types"]),
        }

    intervals = []
    interval_stats = {}
    for (source_time, target_time), count in zip(interval_names, counts):
        source = split_pools[source_time]
        target = split_pools[target_time]
        interval = {
            "source_time": source_time,
            "target_time": target_time,
            "t_start": normalized_time(source_time, cfg),
            "t_end": normalized_time(target_time, cfg),
            "nominal_pairs": int(count),
        }
        intervals.append(interval)
        interval_stats[
            f"{source_time}_to_{target_time}".replace(".", "p")
        ] = {
            **interval,
            "sampled_pairs": int(count),
            "candidate_pairs": int(count),
            "accepted_pairs": int(count),
            "collected_accepted_pairs": int(count),
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
            "ot_minibatch_size": int(
                getattr(cfg.problem, "ot_minibatch_size", 128)
            ),
            "ot_cost": "raw_sqeuclidean",
        }

    stored_cells = sum(pool["x"].shape[0] for pool in timepoints.values())
    stored_bytes = sum(
        pool["x"].nbytes + pool["type_ids"].nbytes
        for pool in timepoints.values()
    )
    compact = {
        "timepoints": timepoints,
        "intervals": tuple(intervals),
        "nominal_n": nominal_n,
        "dimension": int(next(iter(timepoints.values()))["x"].shape[1]),
        "include_time_bounds": include_time_bounds,
    }
    stats = {
        "retained_timepoints": list(retained),
        "split": "train",
        "pair_mode": pair_mode,
        "sampled_pairs": nominal_n,
        "candidate_pairs": nominal_n,
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "candidate_acceptance_rate": 1.0,
        "pair_pool_mode": "direct_timepoint_pools",
        "coupling": "dynamic_minibatch_ot",
        "ot_minibatch_size": int(
            getattr(cfg.problem, "ot_minibatch_size", 128)
        ),
        "ot_cost": "raw_sqeuclidean",
        "stored_endpoint_cells": int(stored_cells),
        "stored_endpoint_bytes": int(stored_bytes),
        "expanded_pair_rows_avoided": nominal_n,
    }
    if include_time_bounds:
        stats["intervals"] = interval_stats
    else:
        only_stats = next(iter(interval_stats.values()))
        stats.update(
            {
                "source_total_n": only_stats["source_total_n"],
                "source_train_n": only_stats["source_train_n"],
                "source_holdout_n": only_stats["source_holdout_n"],
                "target_total_n": only_stats["target_total_n"],
                "target_train_n": only_stats["target_train_n"],
                "target_holdout_n": only_stats["target_holdout_n"],
            }
        )
    return compact, stats


def couple_minibatch_ot_timepoint_pools(
    cfg,
    pools: Dict[str, Any],
    n_pairs: int,
    *,
    seed: int,
    pair_mode: str | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Couple a balanced batch directly from compact Maizels pools."""
    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    pair_mode = _canonical_maizels_pair_mode(str(pair_mode))
    if not uses_minibatch_ot(cfg, pair_mode):
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")

    intervals = tuple(pools["intervals"])
    counts = _allocate_pairs(int(n_pairs), len(intervals))
    rng = np.random.default_rng(int(seed))
    paired_parts = []
    interval_stats = []
    for interval, interval_n in zip(intervals, counts):
        source_time = str(interval["source_time"])
        target_time = str(interval["target_time"])
        source = pools["timepoints"][source_time]
        target = pools["timepoints"][target_time]
        paired, stats = _make_minibatch_ot_pair_pool_from_endpoint_arrays(
            cfg,
            source["x"],
            source["type_ids"],
            target["x"],
            target["type_ids"],
            n_pairs=interval_n,
            rng=rng,
            pair_mode=pair_mode,
            sample_with_replacement=True,
        )
        if bool(pools.get("include_time_bounds", False)):
            paired = _add_time_bounds(cfg, paired, source_time, target_time)
        stats.update(
            {
                "source_time": source_time,
                "target_time": target_time,
                "t_start": float(interval["t_start"]),
                "t_end": float(interval["t_end"]),
                "sampled_pairs": int(paired["x0"].shape[0]),
            }
        )
        paired_parts.append(paired)
        interval_stats.append(stats)

    result = {
        key: np.concatenate([part[key] for part in paired_parts], axis=0)
        for key in ("x0", "x1", "label")
    }
    permutation = rng.permutation(result["x0"].shape[0])
    result = {key: value[permutation] for key, value in result.items()}
    summary = {
        "pair_mode": pair_mode,
        "coupling": "dynamic_minibatch_ot",
        "pair_pool_mode": "direct_timepoint_pools",
        "sampled_pairs": int(result["x0"].shape[0]),
        "ot_minibatch_size": int(getattr(cfg.problem, "ot_minibatch_size", 128)),
        "ot_cost": "raw_sqeuclidean",
    }
    if bool(pools.get("include_time_bounds", False)):
        summary["intervals"] = interval_stats
    else:
        summary.update(interval_stats[0])
        summary.update(
            {
                "coupling": "dynamic_minibatch_ot",
                "pair_pool_mode": "direct_timepoint_pools",
            }
        )
    return result, summary


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
    """Build equally weighted pairs over adjacent retained Maizels intervals."""
    split = str(split).lower()
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

    for (source_time, target_time), interval_n in zip(intervals, counts):
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
            n_pairs=interval_n,
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
            (
                f"{stats['source_time']}_to_{stats['target_time']}"
                .replace(".", "p")
            ): stats
            for stats in interval_stats
        },
    }
    return paired, aggregate


def couple_minibatch_ot_pair_pool(
    cfg,
    paired: Dict[str, np.ndarray],
    n_pairs: int,
    *,
    seed: int,
    pair_mode: str | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Construct a fresh Maizels optimizer batch with raw-cost minibatch OT."""
    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    pair_mode = _canonical_maizels_pair_mode(str(pair_mode))
    if pair_mode not in _OT_PAIR_MODES:
        raise ValueError(f"Pair mode {pair_mode!r} does not request minibatch OT.")

    labels = np.asarray(paired["label"])
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Maizels minibatch OT requires two-column pair labels.")
    source_ids = labels[:, 0].astype(np.int32)
    target_ids = labels[:, 1].astype(np.int32)
    if (
        np.any(source_ids < 0)
        or np.any(source_ids >= len(CLASS_NAMES))
        or np.any(target_ids < 0)
        or np.any(target_ids >= len(CLASS_NAMES))
    ):
        raise ValueError("Maizels minibatch OT received an invalid cell-type id.")

    class_names = np.asarray(CLASS_NAMES, dtype=object)
    if labels.shape[1] >= 4:
        interval_bounds = np.unique(labels[:, 2:4], axis=0)
        interval_bounds = interval_bounds[np.argsort(interval_bounds[:, 0])]
        counts = _allocate_pairs(int(n_pairs), int(interval_bounds.shape[0]))
        rng = np.random.default_rng(int(seed))
        paired_parts = []
        interval_stats = []
        for bounds, interval_n in zip(interval_bounds, counts):
            in_interval = np.all(
                np.isclose(labels[:, 2:4], bounds[None, :]), axis=1
            )
            if not in_interval.any():
                raise RuntimeError(
                    "Maizels candidate pool has no rows for interval "
                    f"{bounds.tolist()}."
                )
            interval_pairs, stats = _make_minibatch_ot_pair_pool_from_endpoint_arrays(
                cfg,
                np.asarray(paired["x0"])[in_interval],
                class_names[source_ids[in_interval]],
                np.asarray(paired["x1"])[in_interval],
                class_names[target_ids[in_interval]],
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
            stats.update({"t_start": float(bounds[0]), "t_end": float(bounds[1])})
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
            "pair_pool_mode": "independent_candidates",
            "sampled_pairs": int(result["x0"].shape[0]),
            "ot_minibatch_size": int(
                getattr(cfg.problem, "ot_minibatch_size", n_pairs)
            ),
            "intervals": interval_stats,
        }

    result, stats = _make_minibatch_ot_pair_pool_from_endpoint_arrays(
        cfg,
        np.asarray(paired["x0"]),
        class_names[source_ids],
        np.asarray(paired["x1"]),
        class_names[target_ids],
        n_pairs=int(n_pairs),
        rng=np.random.default_rng(int(seed)),
        pair_mode=pair_mode,
    )
    stats.update(
        {
            "coupling": "dynamic_minibatch_ot",
            "pair_pool_mode": "independent_candidates",
        }
    )
    return result, stats


def make_pair_pool(
    cfg, dataset_location: str | None = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create endpoint pairs or adjacent retained-interval training pairs."""
    if uses_retained_intervals(cfg):
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
        minibatch_ot = uses_minibatch_ot(cfg, pair_mode)
        paired, stats = _make_retained_interval_pair_pool(
            cfg,
            int(getattr(cfg.problem, "n", 100_000)),
            split="train",
            dataset_location=dataset_location,
            pair_mode="none" if minibatch_ot else str(pair_mode),
            seed=int(getattr(cfg.training, "seed", 0)) + 301,
        )
        if minibatch_ot:
            canonical_mode = _canonical_maizels_pair_mode(str(pair_mode))
            for interval_stats in stats["intervals"].values():
                interval_stats.update(
                    {
                        "pair_mode": canonical_mode,
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
                    "pair_mode": canonical_mode,
                    "pair_pool_mode": "independent_candidates",
                    "coupling": "dynamic_minibatch_ot",
                    "ot_minibatch_size": int(
                        getattr(cfg.problem, "ot_minibatch_size", 128)
                    ),
                    "ot_cost": "raw_sqeuclidean",
                }
            )
        return paired, stats

    splits = endpoint_pool_splits(cfg, dataset_location=dataset_location)
    n_pairs = int(getattr(cfg.problem, "n", 100_000))
    seed = int(getattr(cfg.training, "seed", 0))
    rng = np.random.default_rng(seed + 301)
    pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    minibatch_ot = uses_minibatch_ot(cfg, pair_mode)

    paired, stats = _make_pair_pool_from_endpoint_arrays(
        cfg,
        splits["source_train_x"],
        splits["source_train_types"],
        splits["target_train_x"],
        splits["target_train_types"],
        n_pairs=n_pairs,
        rng=rng,
        # Dynamic OT starts from independent source and target candidate pools.
        pair_mode="none" if minibatch_ot else pair_mode,
    )
    if minibatch_ot:
        stats.update(
            {
                "pair_mode": _canonical_maizels_pair_mode(str(pair_mode)),
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
            "source_total_n": int(splits["source_n"]),
            "target_total_n": int(splits["target_n"]),
            "source_train_n": int(splits["source_train_n"]),
            "source_holdout_n": int(splits["source_holdout_n"]),
            "target_train_n": int(splits["target_train_n"]),
            "target_holdout_n": int(splits["target_holdout_n"]),
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
    """Build held-out pairs over the same adjacent intervals used for training."""
    if not uses_retained_intervals(cfg):
        return make_heldout_pair_pool(
            cfg,
            n_pairs,
            dataset_location=dataset_location,
            pair_mode=pair_mode,
            seed=seed,
        )
    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    if seed is None:
        seed = int(getattr(cfg.training, "seed", 0)) + 2701
    return _make_retained_interval_pair_pool(
        cfg,
        int(n_pairs),
        split="heldout",
        dataset_location=dataset_location,
        pair_mode=str(pair_mode),
        seed=int(seed),
        minibatch_ot=uses_minibatch_ot(cfg, pair_mode),
    )


def make_heldout_pair_pool(
    cfg,
    n_pairs: int,
    *,
    dataset_location: str | None = None,
    pair_mode: str | None = None,
    seed: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create D3 -> D8 diagnostic pairs from endpoint cells excluded from training."""
    splits = endpoint_pool_splits(cfg, dataset_location=dataset_location)
    if splits["source_holdout_n"] == 0 or splits["target_holdout_n"] == 0:
        raise RuntimeError(
            "Maizels held-out diagnostics require non-empty source and target "
            "holdout pools. Increase problem.maizels_holdout_fraction or "
            "problem.maizels_holdout_n."
        )

    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    if seed is None:
        seed = int(getattr(getattr(cfg, "training", None), "seed", 0)) + 997
    rng = np.random.default_rng(int(seed))
    pair_builder = (
        _make_minibatch_ot_pair_pool_from_endpoint_arrays
        if uses_minibatch_ot(cfg, pair_mode)
        else _make_pair_pool_from_endpoint_arrays
    )
    paired, stats = pair_builder(
        cfg,
        splits["source_holdout_x"],
        splits["source_holdout_types"],
        splits["target_holdout_x"],
        splits["target_holdout_types"],
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=str(pair_mode),
    )
    if bool(getattr(cfg.problem, "pair_time_bounds_in_label", False)):
        paired = _add_time_bounds(
            cfg,
            paired,
            str(getattr(cfg.problem, "source_time", "D3")),
            str(getattr(cfg.problem, "target_time", "D8")),
        )
    stats.update(
        {
            "source_total_n": int(splits["source_n"]),
            "target_total_n": int(splits["target_n"]),
            "source_train_n": int(splits["source_train_n"]),
            "source_holdout_n": int(splits["source_holdout_n"]),
            "target_train_n": int(splits["target_train_n"]),
            "target_holdout_n": int(splits["target_holdout_n"]),
        }
    )
    return paired, stats


def make_endpoint_split_pair_pool(
    cfg,
    n_pairs: int,
    *,
    split: str,
    dataset_location: str | None = None,
    pair_mode: str | None = None,
    seed: int | None = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create diagnostic pairs from a named D3/D8 endpoint split."""
    splits = endpoint_pool_splits(cfg, dataset_location=dataset_location)
    split = str(split).lower()
    if split in ("holdout", "heldout"):
        source_x = splits["source_holdout_x"]
        source_types = splits["source_holdout_types"]
        target_x = splits["target_holdout_x"]
        target_types = splits["target_holdout_types"]
    elif split == "train":
        source_x = splits["source_train_x"]
        source_types = splits["source_train_types"]
        target_x = splits["target_train_x"]
        target_types = splits["target_train_types"]
    elif split == "all":
        source_x = splits["source_x"]
        source_types = splits["source_types"]
        target_x = splits["target_x"]
        target_types = splits["target_types"]
    else:
        raise ValueError(
            "split must be one of 'heldout', 'train', or 'all', " f"got {split!r}."
        )

    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError(f"Maizels endpoint split {split!r} is empty.")

    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    if seed is None:
        seed = int(getattr(getattr(cfg, "training", None), "seed", 0)) + 997

    pair_builder = (
        _make_minibatch_ot_pair_pool_from_endpoint_arrays
        if uses_minibatch_ot(cfg, pair_mode)
        else _make_pair_pool_from_endpoint_arrays
    )
    paired, stats = pair_builder(
        cfg,
        source_x,
        source_types,
        target_x,
        target_types,
        n_pairs=int(n_pairs),
        rng=np.random.default_rng(int(seed)),
        pair_mode=str(pair_mode),
    )
    stats.update(
        {
            "split": split,
            "source_total_n": int(splits["source_n"]),
            "target_total_n": int(splits["target_n"]),
            "source_train_n": int(splits["source_train_n"]),
            "source_holdout_n": int(splits["source_holdout_n"]),
            "target_train_n": int(splits["target_train_n"]),
            "target_holdout_n": int(splits["target_holdout_n"]),
            "split_source_n": int(source_x.shape[0]),
            "split_target_n": int(target_x.shape[0]),
        }
    )
    return paired, stats


def all_timepoint_data(dataset_location: str | None = None) -> Dict[str, np.ndarray]:
    return load_pca50_dataset(resolve_dataset_path(dataset_location))


def endpoint_pools(dataset_location: str | None, source_time: str, target_time: str):
    data = all_timepoint_data(dataset_location)
    source_x, source_types = subset_time(data, source_time)
    target_x, target_types = subset_time(data, target_time)
    return source_x, source_types, target_x, target_types

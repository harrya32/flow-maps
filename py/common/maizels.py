"""Utilities for the Maizels PCA50 trajectory experiment."""

from __future__ import annotations

import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_DATASET = (
    "/Users/harryamad/Desktop/Maizels2023aa/data/"
    "celltype_classification_pca50_dataset.csv.gz"
)
DEFAULT_CLASSIFIER = (
    "/Users/harryamad/Desktop/Maizels2023aa/models/"
    "celltype_classifier_pca50.pt"
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

TRANSITION_EDGES = [
    ("NMP", "Mesoderm"),
    ("NMP", "Early_Neural"),
    ("Early_Neural", "Neural"),
    ("Early_Neural", "pMN"),
    ("Early_Neural", "p3"),
    ("p3", "V3"),
    ("p3", "FP"),
    ("pMN", "MN"),
]

_DATA_CACHE: Dict[str, Dict[str, np.ndarray]] = {}
_CLASSIFIER_CACHE: Dict[str, Tuple[Any, List[str], np.ndarray, np.ndarray]] = {}
_JAX_CLASSIFIER_CACHE: Dict[str, Tuple[Dict[str, jnp.ndarray], List[str], jnp.ndarray, jnp.ndarray]] = {}


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


def build_reachable(edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES) -> Dict[str, Set[str]]:
    """Return the reflexive transitive closure of the cell-type transition graph."""
    nodes = set(CLASS_NAMES)
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


def build_direct_reachable(edges: Sequence[Tuple[str, str]] = TRANSITION_EDGES) -> Dict[str, Set[str]]:
    """Return reflexive one-edge reachability for strict stepwise checks."""
    nodes = set(CLASS_NAMES)
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
        "lineage_transition_mode must be 'descendant' or 'direct', "
        f"got {mode!r}."
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
) -> Dict[str, Set[str]]:
    """Return the reachability relation for the configured lineage mode."""
    mode = resolve_lineage_transition_mode(mode)
    if mode == "direct":
        return build_direct_reachable(edges)
    return build_reachable(edges)


def endpoint_valid(src_type: str, dst_type: str, reachable: Dict[str, Set[str]]) -> bool:
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
            obs_names.append(row[index_col] if index_col is not None else str(len(obs_names)))
            timepoints.append(row["timepoint"])
            cell_types.append(row["cell_annotation"])
            pcs.append([float(row[col]) for col in pc_cols])

    data = {
        "obs_names": np.asarray(obs_names, dtype=object),
        "x": np.asarray(pcs, dtype=np.float32),
        "timepoints": np.asarray(timepoints, dtype=object),
        "time_values": np.asarray([parse_timepoint(tp) for tp in timepoints], dtype=np.float32),
        "cell_types": np.asarray(cell_types, dtype=object),
    }
    _DATA_CACHE[cache_key] = data
    return data


def subset_time(data: Dict[str, np.ndarray], timepoint: str) -> Tuple[np.ndarray, np.ndarray]:
    mask = data["timepoints"] == timepoint
    return data["x"][mask], data["cell_types"][mask]


def class_to_id_map(class_names: Sequence[str] = CLASS_NAMES) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


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
            arrays = {key: raw[key].astype(np.float32) for key in raw.files if key.startswith("net.")}
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
            "The Maizels prior/classifier path requires PyTorch to load "
            f"{path}."
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


def classifier_index_lookup(class_names: Sequence[str]) -> np.ndarray:
    """Map canonical Maizels class ids to a classifier-specific class order."""
    by_name = {str(name): idx for idx, name in enumerate(class_names)}
    missing = [name for name in CLASS_NAMES if name not in by_name]
    if missing:
        raise KeyError(
            "Classifier is missing Maizels classes required for constraints: "
            f"{missing}"
        )
    return np.asarray([by_name[name] for name in CLASS_NAMES], dtype=np.int32)


def lineage_invalid_transition_matrix(
    class_names: Sequence[str],
    transition_mode: str | None = "descendant",
) -> np.ndarray:
    """Return matrix M where M[i, j]=1 iff i -> j is biologically invalid."""
    reachable = build_transition_reachable(transition_mode)
    invalid = np.zeros((len(class_names), len(class_names)), dtype=np.float32)
    for ii, src in enumerate(class_names):
        for jj, dst in enumerate(class_names):
            invalid[ii, jj] = 0.0 if endpoint_valid(str(src), str(dst), reachable) else 1.0
    return invalid


def lineage_soft_terms_from_probs(
    probs: jnp.ndarray,
    source_type_ids: jnp.ndarray,
    invalid_transition: jnp.ndarray,
    canonical_to_classifier: jnp.ndarray,
    target_type_ids: jnp.ndarray | None = None,
) -> Dict[str, jnp.ndarray]:
    """Differentiable lineage-validity terms from path classifier probabilities."""
    source_cls = jnp.take(canonical_to_classifier, source_type_ids.astype(jnp.int32))
    source_probs = jax.nn.one_hot(source_cls, probs.shape[-1], dtype=probs.dtype)
    invalid_transition = invalid_transition.astype(probs.dtype)

    start_invalid = jnp.einsum(
        "bi,ij,bj->b",
        source_probs,
        invalid_transition,
        probs[:, 0, :],
    )
    if probs.shape[1] > 1:
        transition_invalid = jnp.einsum(
            "bti,ij,btj->bt",
            probs[:, :-1, :],
            invalid_transition,
            probs[:, 1:, :],
        )
        transition_invalid_per_path = jnp.mean(transition_invalid, axis=1)
    else:
        transition_invalid = jnp.zeros((probs.shape[0], 0), dtype=probs.dtype)
        transition_invalid_per_path = jnp.zeros((probs.shape[0],), dtype=probs.dtype)

    if target_type_ids is None:
        final_invalid = jnp.zeros((probs.shape[0],), dtype=probs.dtype)
    else:
        target_cls = jnp.take(canonical_to_classifier, target_type_ids.astype(jnp.int32))
        target_probs = jax.nn.one_hot(target_cls, probs.shape[-1], dtype=probs.dtype)
        final_invalid = jnp.einsum(
            "bi,ij,bj->b",
            probs[:, -1, :],
            invalid_transition,
            target_probs,
        )

    return {
        "start_invalid_loss": jnp.mean(start_invalid),
        "transition_invalid_loss": jnp.mean(transition_invalid_per_path),
        "final_invalid_loss": jnp.mean(final_invalid),
        "path_invalid_loss": jnp.mean(
            start_invalid + transition_invalid_per_path + final_invalid
        ),
        "start_invalid_mass": jnp.mean(start_invalid),
        "transition_invalid_mass": (
            jnp.mean(transition_invalid)
            if transition_invalid.size > 0
            else jnp.asarray(0.0, dtype=probs.dtype)
        ),
        "final_invalid_mass": jnp.mean(final_invalid),
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
) -> Dict[str, np.ndarray | int]:
    """Classify path points and apply the Maizels transition prior."""
    model, class_names, scaler_mean, scaler_scale = load_classifier(classifier_path)
    reachable = build_transition_reachable(lineage_transition_mode)
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
) -> Dict[str, np.ndarray | int]:
    taus = np.linspace(0.0, 1.0, n_check_times + 2, dtype=np.float32)[1:-1]
    paths = np.stack([(1.0 - tau) * source_x + tau * target_x for tau in taus], axis=1)
    return check_paths_with_classifier(
        paths=paths,
        start_type_ids=source_type_ids,
        classifier_path=classifier_path,
        prob_threshold=prob_threshold,
        margin_threshold=margin_threshold,
        final_type_ids=target_type_ids,
        classifier_batch_size=classifier_batch_size,
        lineage_transition_mode=lineage_transition_mode,
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


def endpoint_pool_splits(cfg, dataset_location: str | None = None) -> Dict[str, np.ndarray]:
    """Load D3/D8 endpoint pools and apply the configured held-out split."""
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

    return {
        "source_x": source_x,
        "source_types": source_types,
        "target_x": target_x,
        "target_types": target_types,
        "source_train_x": source_x[source_train_idx],
        "source_train_types": source_types[source_train_idx],
        "target_train_x": target_x[target_train_idx],
        "target_train_types": target_types[target_train_idx],
        "source_holdout_x": source_x[source_holdout_idx],
        "source_holdout_types": source_types[source_holdout_idx],
        "target_holdout_x": target_x[target_holdout_idx],
        "target_holdout_types": target_types[target_holdout_idx],
        "source_train_idx": source_train_idx,
        "source_holdout_idx": source_holdout_idx,
        "target_train_idx": target_train_idx,
        "target_holdout_idx": target_holdout_idx,
        "source_n": int(source_x.shape[0]),
        "target_n": int(target_x.shape[0]),
        "source_train_n": int(source_train_idx.shape[0]),
        "source_holdout_n": int(source_holdout_idx.shape[0]),
        "target_train_n": int(target_train_idx.shape[0]),
        "target_holdout_n": int(target_holdout_idx.shape[0]),
    }


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
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create independent or prior-filtered pairs from provided endpoint arrays."""
    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError("Maizels source/target pair pools must both be non-empty.")

    class_to_id = class_to_id_map(CLASS_NAMES)
    source_type_ids_all = np.asarray([class_to_id[str(ct)] for ct in source_types], dtype=np.int32)
    target_type_ids_all = np.asarray([class_to_id[str(ct)] for ct in target_types], dtype=np.int32)
    lineage_transition_mode = lineage_transition_mode_from_config(cfg)
    endpoint_reachable = build_transition_reachable("descendant")

    accepted_source_idx: List[np.ndarray] = []
    accepted_target_idx: List[np.ndarray] = []
    stats = {
        "candidate_pairs": 0,
        "endpoint_rejected": 0,
        "interpolant_rejected": 0,
        "accepted_pairs": 0,
    }

    if pair_mode == "none":
        sidx, tidx = _sample_pair_indices(rng, source_x.shape[0], target_x.shape[0], n_pairs)
        accepted_source_idx.append(sidx)
        accepted_target_idx.append(tidx)
        stats["candidate_pairs"] = n_pairs
        stats["accepted_pairs"] = n_pairs
    elif pair_mode in ("endpoint", "endpoint_interpolant"):
        classifier_path = resolve_classifier_path(getattr(cfg.problem, "classifier_path", None))
        chunk_size = int(getattr(cfg.problem, "rejection_chunk_size", 50_000))
        max_candidates = int(getattr(cfg.problem, "rejection_max_candidates", max(10 * n_pairs, n_pairs + 1)))
        n_check_times = int(getattr(cfg.problem, "n_interpolant_check_times", 5))
        prob_threshold = float(getattr(cfg.problem, "classifier_prob_threshold", 0.85))
        margin_threshold = float(getattr(cfg.problem, "classifier_margin_threshold", 1.0))
        classifier_batch_size = int(getattr(cfg.problem, "classifier_batch_size", 8192))

        while stats["accepted_pairs"] < n_pairs and stats["candidate_pairs"] < max_candidates:
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
            f"or 'endpoint_interpolant', got {pair_mode!r}."
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
    stats["truncated_accepted_pairs"] = max(0, collected_accepted - stats["accepted_pairs"])
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


def make_pair_pool(cfg, dataset_location: str | None = None) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Create independent or prior-filtered D3 -> D8 training endpoint pairs."""
    splits = endpoint_pool_splits(cfg, dataset_location=dataset_location)
    n_pairs = int(getattr(cfg.problem, "n", 100_000))
    seed = int(getattr(cfg.training, "seed", 0))
    rng = np.random.default_rng(seed + 301)
    pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")

    paired, stats = _make_pair_pool_from_endpoint_arrays(
        cfg,
        splits["source_train_x"],
        splits["source_train_types"],
        splits["target_train_x"],
        splits["target_train_types"],
        n_pairs=n_pairs,
        rng=rng,
        pair_mode=pair_mode,
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
    paired, stats = _make_pair_pool_from_endpoint_arrays(
        cfg,
        splits["source_holdout_x"],
        splits["source_holdout_types"],
        splits["target_holdout_x"],
        splits["target_holdout_types"],
        n_pairs=int(n_pairs),
        rng=rng,
        pair_mode=str(pair_mode),
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
            "split must be one of 'heldout', 'train', or 'all', "
            f"got {split!r}."
        )

    if source_x.shape[0] == 0 or target_x.shape[0] == 0:
        raise RuntimeError(f"Maizels endpoint split {split!r} is empty.")

    if pair_mode is None:
        pair_mode = getattr(cfg.problem, "maizels_pair_mode", "none")
    if seed is None:
        seed = int(getattr(getattr(cfg, "training", None), "seed", 0)) + 997

    paired, stats = _make_pair_pool_from_endpoint_arrays(
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

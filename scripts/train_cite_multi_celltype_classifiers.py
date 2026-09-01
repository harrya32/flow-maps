#!/usr/bin/env python3
"""Train evaluation and leave-one-day-out classifiers for CITE and Multi.

Each H5AD has its own 100-dimensional PCA coordinate system, so the two
datasets need separate classifiers.  For every requested dataset this script
trains exactly three models:

* ``all_days`` uses days 2, 3, 4, and 7 and is reserved for flow evaluation;
* ``except_day3`` is the classifier used while training a day-3-held-out flow;
* ``except_day4`` is the classifier used while training a day-4-held-out flow.

Every model uses a stratified 90/10 train/validation split.  Its scaler is fit
only on the 90% training split, and the checkpoint with the lowest validation
cross-entropy is saved.  The MLP and export schema match the Maizels classifier
used by the flow code.  Both ``.pt`` and ``.npz`` files are written because the
hard diagnostics use PyTorch while differentiable flow constraints use JAX. A
training/validation loss-curve PNG is saved beside each checkpoint pair.

Outputs are placed under ``cite-classifiers`` and ``multi-classifiers`` in the
repository root by default.

Example
-------
python scripts/train_cite_multi_celltype_classifiers.py
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "CITE_MULTI_DATA_DIR",
        str(Path.home() / "Desktop" / "flow-maps-data"),
    )
).expanduser()
DEFAULT_VARIANTS = ("all_days", "except_day3", "except_day4")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    filename: str


DATASET_SPECS: Mapping[str, DatasetSpec] = {
    "cite": DatasetSpec("cite", "op_cite_inputs_0.h5ad"),
    "multi": DatasetSpec("multi", "op_train_multi_targets_0.h5ad"),
}


@dataclass(frozen=True)
class ClassifierVariant:
    key: str
    excluded_day: str | None
    usage: str


CLASSIFIER_VARIANTS: Mapping[str, ClassifierVariant] = {
    "all_days": ClassifierVariant(
        key="all_days",
        excluded_day=None,
        usage="flow_evaluation",
    ),
    "except_day3": ClassifierVariant(
        key="except_day3",
        excluded_day="3",
        usage="flow_training_when_day3_is_held_out",
    ),
    "except_day4": ClassifierVariant(
        key="except_day4",
        excluded_day="4",
        usage="flow_training_when_day4_is_held_out",
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the all-days and day-3/day-4 leave-one-out classifiers on "
            "the 100-PC CITE and Multi embeddings."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASET_SPECS),
        default=list(DATASET_SPECS),
        help="Datasets to train (default: cite multi).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing the H5AD files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--cite-path",
        type=Path,
        default=None,
        help="Optional explicit path to op_cite_inputs_0.h5ad.",
    )
    parser.add_argument(
        "--multi-path",
        type=Path,
        default=None,
        help="Optional explicit path to op_train_multi_targets_0.h5ad.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(CLASSIFIER_VARIANTS),
        default=list(DEFAULT_VARIANTS),
        help="Classifier variants to train (default: all three).",
    )
    parser.add_argument(
        "--output-root",
        "--model-dir",
        dest="output_root",
        type=Path,
        default=REPO_ROOT,
        help=(
            "Parent of cite-classifiers and multi-classifiers "
            f"(default: {REPO_ROOT})."
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=(
            "Optional report root. By default reports are stored in a reports "
            "subdirectory beside each dataset's checkpoints."
        ),
    )
    parser.add_argument("--pca-key", default="X_pca")
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--day-key", default="day")
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=100,
        help="Number of leading stored PCs to use (default: all 100).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=0,
        help=(
            "Reduce the learning rate after this many unimproved epochs; 0 "
            "disables scheduling (default: 0, matching Maizels)."
        ),
    )
    parser.add_argument("--lr-factor", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (default: 0.05).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs=2,
        metavar=("H1", "H2"),
        default=(128, 64),
        help="Widths of the two hidden layers (default: 128 64).",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.25,
        help=(
            "Exponent applied to inverse-frequency class weights. 1 reproduces "
            "Maizels; 0 disables weighting; values such as 0.5 temper very rare "
            "classes (default: 0.25; use 1 for the original Maizels rule)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help=(
            "Training device. Auto uses CUDA when available and CPU otherwise; "
            "MPS is opt-in because its BatchNorm training can be unstable."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print an epoch summary every N epochs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing checkpoints for the requested variants.",
    )
    args = parser.parse_args(argv)

    positive_ints = ("n_pcs", "batch_size", "max_epochs", "patience", "log_every")
    for name in positive_ints:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.lr_patience < 0:
        parser.error("--lr-patience must be non-negative")
    if not 0.0 < args.lr_factor < 1.0:
        parser.error("--lr-factor must be in (0, 1)")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be in [0, 1)")
    if any(width <= 0 for width in args.hidden_dims):
        parser.error("--hidden-dims values must be positive")
    if not 0.0 <= args.class_weight_power <= 1.0:
        parser.error("--class-weight-power must be in [0, 1]")
    if args.min_delta < 0.0:
        parser.error("--min-delta must be non-negative")
    return args


def import_dependencies() -> Dict[str, Any]:
    missing: List[str] = []
    modules: Dict[str, Any] = {}
    for import_name, package_name in (
        ("anndata", "anndata"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
    ):
        try:
            modules[import_name] = __import__(import_name)
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            f"Missing required package(s): {names}. The metric-flow-matching "
            "environment already includes these dependencies; alternatively "
            "install them in the active environment before rerunning."
        )
    return modules


def resolve_input_path(args: argparse.Namespace, dataset: str) -> Path:
    explicit = getattr(args, f"{dataset}_path")
    path = (
        explicit
        if explicit is not None
        else args.data_dir / DATASET_SPECS[dataset].filename
    )
    return path.expanduser().resolve()


def classifier_output_dir(output_root: Path, dataset: str) -> Path:
    """Return the dataset-specific checkpoint directory required by the configs."""
    return output_root.expanduser().resolve() / f"{dataset}-classifiers"


def checkpoint_stem(dataset: str, n_pcs: int, variant: str) -> str:
    if variant not in CLASSIFIER_VARIANTS:
        raise KeyError(f"Unknown classifier variant {variant!r}.")
    return f"celltype_classifier_{dataset}_pca{n_pcs}_{variant}"


def checkpoint_path(
    output_root: Path,
    dataset: str,
    n_pcs: int,
    variant: str,
    suffix: str = ".pt",
) -> Path:
    return classifier_output_dir(output_root, dataset) / (
        checkpoint_stem(dataset, n_pcs, variant) + suffix
    )


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    if requested == "mps":
        print(
            "warning: MPS is opt-in because BatchNorm running statistics can "
            "become numerically unstable; use --device cpu if validation "
            "collapses or export verification fails.",
            file=sys.stderr,
        )
    return torch.device(requested)


def seed_everything(np: Any, torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(
    ad: Any,
    np: Any,
    path: Path,
    pca_key: str,
    label_key: str,
    day_key: str,
    n_pcs: int,
) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run scripts/download_cite_multi.py first, or pass an explicit path."
        )

    adata = ad.read_h5ad(path, backed="r")
    try:
        if pca_key not in adata.obsm:
            raise KeyError(
                f"{path.name} has no adata.obsm[{pca_key!r}]; available keys: "
                f"{list(adata.obsm.keys())}"
            )
        stored_pcs = adata.obsm[pca_key]
        if stored_pcs.shape[1] < n_pcs:
            raise ValueError(
                f"{path.name} contains {stored_pcs.shape[1]} PCs, but {n_pcs} "
                "were requested"
            )
        if label_key not in adata.obs:
            raise KeyError(
                f"{path.name} has no adata.obs[{label_key!r}]; available keys: "
                f"{list(adata.obs.columns)}"
            )
        if day_key not in adata.obs:
            raise KeyError(
                f"{path.name} has no adata.obs[{day_key!r}]; available keys: "
                f"{list(adata.obs.columns)}"
            )

        label_series = adata.obs[label_key]
        missing = label_series.isna()
        stripped = label_series.astype("string").str.strip()
        missing = missing | stripped.eq("").fillna(True)
        if bool(missing.any()):
            raise ValueError(
                f"{path.name} has {int(missing.sum())} missing/empty {label_key!r} labels"
            )

        x = np.asarray(stored_pcs[:, :n_pcs], dtype=np.float32).copy()
        if not bool(np.isfinite(x).all()):
            raise ValueError(f"{path.name} contains non-finite PCA coordinates")

        labels = stripped.astype(str).to_numpy()
        cell_ids = np.asarray(adata.obs_names.astype(str))
        day_series = adata.obs[day_key]
        missing_days = day_series.isna()
        days = day_series.astype("string").str.strip()
        missing_days = missing_days | days.eq("").fillna(True)
        if bool(missing_days.any()):
            raise ValueError(
                f"{path.name} has {int(missing_days.sum())} missing/empty "
                f"{day_key!r} values"
            )
        days = days.astype(str).to_numpy()
        feature_names = [str(name) for name in adata.var_names]
        return {
            "x": x,
            "labels": labels,
            "cell_ids": cell_ids,
            "days": days,
            "feature_names": feature_names,
            "n_cells": int(adata.n_obs),
            "n_features": int(adata.n_vars),
            "stored_n_pcs": int(stored_pcs.shape[1]),
        }
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def select_variant_rows(
    np: Any,
    data: Mapping[str, Any],
    variant: ClassifierVariant,
) -> Any:
    """Return source-row indices after applying the variant's day exclusion."""
    if variant.excluded_day is None:
        return np.arange(data["n_cells"], dtype=np.int64)
    selected = np.flatnonzero(data["days"] != variant.excluded_day)
    excluded_count = int(data["n_cells"] - len(selected))
    if excluded_count == 0:
        raise ValueError(
            f"Cannot train {variant.key}: day {variant.excluded_day} is absent."
        )
    return selected.astype(np.int64, copy=False)


def make_model(
    nn: Any,
    n_features: int,
    n_classes: int,
    dropout: float,
    hidden_dims: Sequence[int],
) -> Any:
    hidden_1, hidden_2 = (int(value) for value in hidden_dims)

    class CellTypeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden_1),
                nn.BatchNorm1d(hidden_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_1, hidden_2),
                nn.BatchNorm1d(hidden_2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_2, n_classes),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return CellTypeMLP()


def make_loaders(
    np: Any,
    torch: Any,
    x: Any,
    y: Any,
    class_names: Sequence[str],
    seed: int,
    batch_size: int,
    num_workers: int,
    device: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    indices = np.arange(len(y))
    label_encoder = LabelEncoder()
    label_encoder.fit(np.asarray(class_names))
    y_encoded = label_encoder.transform(y).astype(np.int64, copy=False)

    class_counts_all = np.bincount(y_encoded, minlength=len(label_encoder.classes_))
    if bool((class_counts_all < 2).any()):
        too_small = {
            str(label_encoder.classes_[idx]): int(count)
            for idx, count in enumerate(class_counts_all)
            if count < 2
        }
        raise ValueError(
            "Every class needs at least two cells for the stratified 90/10 "
            f"split; too-small classes: {too_small}"
        )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=seed,
        stratify=y_encoded,
    )
    split_indices = {"train": train_idx, "validation": val_idx}

    scaler = StandardScaler()
    split_x = {
        "train": scaler.fit_transform(x[train_idx]).astype(np.float32),
        "validation": scaler.transform(x[val_idx]).astype(np.float32),
    }
    split_y = {
        "train": y_encoded[train_idx],
        "validation": y_encoded[val_idx],
    }

    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(seed)
    loaders: Dict[str, Any] = {}
    for split in ("train", "validation"):
        dataset = TensorDataset(
            torch.from_numpy(split_x[split]), torch.from_numpy(split_y[split])
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            # BatchNorm cannot train on a final singleton batch.  The default
            # dataset sizes do not hit this case, but custom batch sizes can.
            drop_last=split == "train" and len(dataset) % batch_size == 1,
            generator=generator if split == "train" else None,
        )

    prepared = {
        "label_encoder": label_encoder,
        "scaler": scaler,
        "split_indices": split_indices,
        "split_x": split_x,
        "split_y": split_y,
    }
    return loaders, prepared, class_counts_all


def run_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    optimizer: Any | None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def evaluate(
    np: Any,
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> Tuple[float, float, Any, Any]:
    truth: List[Any] = []
    predictions: List[Any] = []
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels_device)
            batch_size = int(labels.shape[0])
            predicted = logits.argmax(dim=1)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predicted == labels_device).sum().item())
            total_examples += batch_size
            truth.append(labels.numpy())
            predictions.append(predicted.cpu().numpy())
    return (
        total_loss / total_examples,
        total_correct / total_examples,
        np.concatenate(truth),
        np.concatenate(predictions),
    )


def state_dict_to_numpy(np: Any, state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.detach().cpu().numpy()
        for key, value in state_dict.items()
        if not key.endswith("num_batches_tracked")
    }


def numpy_logits(np: Any, arrays: Mapping[str, Any], x: Any) -> Any:
    def linear(values: Any, prefix: str) -> Any:
        return values @ arrays[f"{prefix}.weight"].T + arrays[f"{prefix}.bias"]

    def batch_norm(values: Any, prefix: str) -> Any:
        return (values - arrays[f"{prefix}.running_mean"]) / np.sqrt(
            arrays[f"{prefix}.running_var"] + 1e-5
        ) * arrays[f"{prefix}.weight"] + arrays[f"{prefix}.bias"]

    hidden = np.maximum(batch_norm(linear(x, "net.0"), "net.1"), 0.0)
    hidden = np.maximum(batch_norm(linear(hidden, "net.4"), "net.5"), 0.0)
    return linear(hidden, "net.8")


def save_plots(
    matplotlib: Any,
    np: Any,
    history: Any,
    confusion: Any,
    class_names: Sequence[str],
    prefix: Path,
    loss_curve_path: Path,
) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["validation_accuracy"], label="validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1.01))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(prefix.with_name(prefix.name + "_training.png"), dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(history["epoch"], history["train_loss"], label="training")
    axis.plot(history["epoch"], history["validation_loss"], label="validation")
    best_row = history.loc[history["validation_loss"].idxmin()]
    axis.scatter(
        [best_row["epoch"]],
        [best_row["validation_loss"]],
        color="black",
        marker="*",
        s=90,
        zorder=3,
        label=f"best validation (epoch {int(best_row['epoch'])})",
    )
    axis.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Classifier loss")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(loss_curve_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set(xlabel="Predicted", ylabel="True", title="Validation confusion matrix")
    threshold = float(confusion.max()) / 2.0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            value = int(confusion[row, column])
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > threshold else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(prefix.with_name(prefix.name + "_confusion_matrix.png"), dpi=180)
    plt.close(fig)


def train_one_variant(
    args: argparse.Namespace,
    dataset: str,
    variant: ClassifierVariant,
    input_path: Path,
    data: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    device: Any,
) -> Dict[str, Any]:
    matplotlib = dependencies["matplotlib"]
    np = dependencies["numpy"]
    pd = dependencies["pandas"]
    torch = dependencies["torch"]
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    from torch import nn

    run_label = f"{dataset}/{variant.key}"
    output_dir = classifier_output_dir(args.output_root, dataset)
    stem = checkpoint_stem(dataset, args.n_pcs, variant.key)
    pt_path = output_dir / f"{stem}.pt"
    npz_path = output_dir / f"{stem}.npz"
    report_dir = (
        output_dir / "reports"
        if args.report_dir is None
        else args.report_dir.expanduser().resolve() / f"{dataset}-classifiers"
    )
    report_prefix = report_dir / stem
    if not args.overwrite:
        existing = [path for path in (pt_path, npz_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing checkpoint(s): "
                + ", ".join(str(path) for path in existing)
                + ". Pass --overwrite to replace them."
            )

    selected_indices = select_variant_rows(np, data, variant)
    selected_x = data["x"][selected_indices]
    selected_labels = data["labels"][selected_indices]
    all_class_names = sorted(str(value) for value in np.unique(data["labels"]))
    loaders, prepared, class_counts_all = make_loaders(
        np,
        torch,
        selected_x,
        selected_labels,
        all_class_names,
        args.seed,
        args.batch_size,
        args.num_workers,
        device,
    )
    label_encoder = prepared["label_encoder"]
    class_names = [str(value) for value in label_encoder.classes_]
    split_sizes = {
        split: int(len(indices)) for split, indices in prepared["split_indices"].items()
    }
    counts_text = ", ".join(
        f"{name}={int(count)}" for name, count in zip(class_names, class_counts_all)
    )
    print(
        f"\n[{run_label}] {len(selected_indices)}/{data['n_cells']} cells, "
        f"{data['stored_n_pcs']} stored PCs; using {args.n_pcs} PCs on {device}",
        flush=True,
    )
    if variant.excluded_day is not None:
        print(
            f"[{run_label}] Excluded every day-{variant.excluded_day} cell.", flush=True
        )
    print(f"[{run_label}] Classes: {counts_text}", flush=True)
    print(
        f"[{run_label}] Split: train={split_sizes['train']}, "
        f"validation={split_sizes['validation']}",
        flush=True,
    )

    seed_everything(np, torch, args.seed)
    model = make_model(
        nn, args.n_pcs, len(class_names), args.dropout, args.hidden_dims
    ).to(device)
    train_counts = np.bincount(prepared["split_y"]["train"], minlength=len(class_names))
    inverse_frequency_weights = len(prepared["split_y"]["train"]) / (
        len(class_names) * train_counts
    )
    class_weights = inverse_frequency_weights**args.class_weight_power
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=1e-6,
        )
        if args.lr_patience > 0
        else None
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, Any] | None = None
    best_validation_metrics: Dict[str, float] | None = None
    early_stopping_loss = float("inf")
    epochs_without_improvement = 0
    history_rows: List[Dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_accuracy = run_epoch(
            torch, model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy, validation_true, validation_pred = (
            evaluate(np, torch, model, loaders["validation"], criterion, device)
        )
        validation_balanced_accuracy = float(
            balanced_accuracy_score(validation_true, validation_pred)
        )
        validation_macro_f1 = float(
            f1_score(validation_true, validation_pred, average="macro")
        )
        if scheduler is not None:
            scheduler.step(validation_loss)
        is_best = validation_loss < best_validation_loss
        if is_best:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_validation_metrics = {
                "loss": float(validation_loss),
                "accuracy": float(validation_accuracy),
                "balanced_accuracy": validation_balanced_accuracy,
                "macro_f1": validation_macro_f1,
            }
        improved_for_patience = validation_loss < early_stopping_loss - args.min_delta
        if improved_for_patience:
            early_stopping_loss = float(validation_loss)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "validation_balanced_accuracy": validation_balanced_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "is_best": is_best,
            }
        )
        if epoch == 1 or epoch % args.log_every == 0 or is_best:
            marker = " *" if is_best else ""
            print(
                f"[{run_label}] epoch {epoch:03d}/{args.max_epochs}: "
                f"train loss={train_loss:.4f} acc={train_accuracy:.4f}; "
                f"val loss={validation_loss:.4f} acc={validation_accuracy:.4f} "
                f"bal_acc={validation_balanced_accuracy:.4f} "
                f"macro_f1={validation_macro_f1:.4f}{marker}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(
                f"[{run_label}] Early stopping after {epoch} epochs "
                f"(best epoch {best_epoch}).",
                flush=True,
            )
            break

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    (
        selected_validation_loss,
        selected_validation_accuracy,
        y_true,
        y_pred,
    ) = evaluate(np, torch, model, loaders["validation"], criterion, device)
    metrics = {
        "best_epoch": int(best_epoch),
        "validation_loss": float(selected_validation_loss),
        "validation_accuracy": float(selected_validation_accuracy),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "validation_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    model = model.to("cpu").eval()
    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    numpy_state = state_dict_to_numpy(np, cpu_state)
    verification_x = prepared["split_x"]["validation"][
        : min(256, split_sizes["validation"])
    ]
    with torch.no_grad():
        torch_values = model(torch.from_numpy(verification_x)).numpy()
    numpy_values = numpy_logits(np, numpy_state, verification_x)
    finite_torch = bool(np.isfinite(torch_values).all())
    finite_numpy = bool(np.isfinite(numpy_values).all())
    export_max_abs_error = (
        float(np.max(np.abs(torch_values - numpy_values)))
        if finite_torch and finite_numpy
        else float("nan")
    )
    if not finite_torch or not finite_numpy:
        device_hint = (
            " This is a known failure mode of MPS BatchNorm; rerun this "
            "variant with --device cpu."
            if device.type == "mps"
            else ""
        )
        raise RuntimeError(
            "Classifier export produced non-finite logits "
            f"(PyTorch finite={finite_torch}, NumPy finite={finite_numpy})."
            + device_hint
        )
    if not np.allclose(torch_values, numpy_values, rtol=1e-5, atol=1e-5):
        raise RuntimeError(
            "NumPy checkpoint verification failed: maximum absolute logit error "
            f"was {export_max_abs_error:.3g}"
        )

    elapsed_seconds = float(time.monotonic() - started)
    metadata = {
        "dataset": dataset,
        "variant": variant.key,
        "usage": variant.usage,
        "excluded_day": variant.excluded_day,
        "included_days": sorted(
            str(value) for value in np.unique(data["days"][selected_indices])
        ),
        "source_h5ad": str(input_path),
        "pca_key": args.pca_key,
        "label_key": args.label_key,
        "day_key": args.day_key,
        "class_names": class_names,
        # Kept for compatibility with the original Maizels checkpoint schema.
        "selected_genes": data["feature_names"],
        "n_pcs": int(args.n_pcs),
        "stored_n_pcs": int(data["stored_n_pcs"]),
        "n_source_cells": int(data["n_cells"]),
        "n_cells": int(len(selected_indices)),
        "n_excluded_cells": int(data["n_cells"] - len(selected_indices)),
        "n_features": int(data["n_features"]),
        "scaler_mean": prepared["scaler"].mean_.astype(float).tolist(),
        "scaler_scale": prepared["scaler"].scale_.astype(float).tolist(),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "split_sizes": split_sizes,
        "split_fractions": {"train": 0.90, "validation": 0.10},
        "architecture": [
            int(args.n_pcs),
            int(args.hidden_dims[0]),
            int(args.hidden_dims[1]),
            len(class_names),
        ],
        "dropout": float(args.dropout),
        "label_smoothing": float(args.label_smoothing),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "lr_patience": int(args.lr_patience),
        "lr_factor": float(args.lr_factor),
        "weight_decay": float(args.weight_decay),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "class_weights": class_weights.astype(float).tolist(),
        "class_weight_power": float(args.class_weight_power),
        "selection_metric": "validation_loss",
        "best_validation_loss": float(best_validation_loss),
        "best_validation_metrics": best_validation_metrics,
        "validation_metrics": metrics,
        "numpy_export_max_abs_error": export_max_abs_error,
        "elapsed_seconds": elapsed_seconds,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": cpu_state, "metadata": metadata}, pt_path)
    np.savez_compressed(
        npz_path,
        class_names=np.asarray(class_names),
        scaler_mean=prepared["scaler"].mean_.astype(np.float32),
        scaler_scale=prepared["scaler"].scale_.astype(np.float32),
        n_pcs=np.asarray(args.n_pcs, dtype=np.int64),
        dataset=np.asarray(dataset),
        variant=np.asarray(variant.key),
        excluded_day=np.asarray(variant.excluded_day or ""),
        **numpy_state,
    )

    history = pd.DataFrame(history_rows)
    history.to_csv(
        report_prefix.with_name(report_prefix.name + "_history.csv"), index=False
    )
    metrics_row = {
        "dataset": dataset,
        "variant": variant.key,
        "excluded_day": variant.excluded_day,
        "n_pcs": int(args.n_pcs),
        "class_weight_power": float(args.class_weight_power),
        "label_smoothing": float(args.label_smoothing),
        "selection_metric": "validation_loss",
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(
        report_prefix.with_name(report_prefix.name + "_validation_metrics.csv"),
        index=False,
    )
    pd.DataFrame(report).transpose().to_csv(
        report_prefix.with_name(
            report_prefix.name + "_validation_classification_report.csv"
        )
    )
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(
        report_prefix.with_name(report_prefix.name + "_validation_confusion_matrix.csv")
    )
    split_name = np.full(data["n_cells"], "excluded", dtype="<U10")
    for name, local_indices in prepared["split_indices"].items():
        split_name[selected_indices[local_indices]] = name
    pd.DataFrame(
        {
            "cell_id": data["cell_ids"],
            "day": data["days"],
            "cell_type": data["labels"],
            "split": split_name,
        }
    ).to_csv(
        report_prefix.with_name(report_prefix.name + "_splits.csv.gz"),
        index=False,
        compression="gzip",
    )
    loss_curve_path = output_dir / f"{stem}_loss_curve.png"
    save_plots(
        matplotlib,
        np,
        history,
        confusion,
        class_names,
        report_prefix,
        loss_curve_path,
    )

    print(
        f"[{run_label}] Validation loss={metrics['validation_loss']:.4f}, "
        f"accuracy={metrics['validation_accuracy']:.4f}, "
        f"balanced accuracy={metrics['validation_balanced_accuracy']:.4f}, "
        f"macro F1={metrics['validation_macro_f1']:.4f}",
        flush=True,
    )
    print(f"[{run_label}] Saved {pt_path}", flush=True)
    print(f"[{run_label}] Saved {npz_path}", flush=True)
    print(f"[{run_label}] Saved {loss_curve_path}", flush=True)
    print(f"[{run_label}] Reports: {report_dir}", flush=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
    try:
        dependencies = import_dependencies()
        device = choose_device(dependencies["torch"], args.device)
        all_metrics: Dict[str, Dict[str, Any]] = {}
        for dataset in args.datasets:
            input_path = resolve_input_path(args, dataset)
            print(f"\n[{dataset}] Reading {input_path}", flush=True)
            data = load_dataset(
                dependencies["anndata"],
                dependencies["numpy"],
                input_path,
                args.pca_key,
                args.label_key,
                args.day_key,
                args.n_pcs,
            )
            all_metrics[dataset] = {}
            for variant_name in args.variants:
                variant = CLASSIFIER_VARIANTS[variant_name]
                all_metrics[dataset][variant_name] = train_one_variant(
                    args,
                    dataset,
                    variant,
                    input_path,
                    data,
                    dependencies,
                    device,
                )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("\nSummary")
    for dataset, variants in all_metrics.items():
        for variant_name, metrics in variants.items():
            print(
                f"  {dataset}/{variant_name}: "
                f"val_loss={metrics['validation_loss']:.4f}, "
                f"accuracy={metrics['validation_accuracy']:.4f}, "
                "balanced_accuracy="
                f"{metrics['validation_balanced_accuracy']:.4f}, "
                f"macro_f1={metrics['validation_macro_f1']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Observed-only axis-aligned classifier used by the lineage loss."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from .dataset import CLASS_NAMES


def sha256(path: Union[str, Path]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_indices(labels, fraction=0.8, seed=21):
    """Stratify each observed marginal by hard class."""
    rng = np.random.default_rng(seed)
    training, validation, singletons = [], [], []
    for time_index, marginal in enumerate(labels):
        train_ids, validation_ids = [], []
        for label in np.unique(marginal):
            indices = rng.permutation(np.flatnonzero(marginal == label))
            count = max(1, int(round((1.0 - fraction) * len(indices)))) if len(indices) > 1 else 0
            validation_ids.extend(indices[:count])
            train_ids.extend(indices[count:])
            if len(indices) == 1:
                singletons.append({"time_index": time_index, "class": int(label)})
        training.append(np.asarray(sorted(train_ids), dtype=np.int64))
        validation.append(np.asarray(sorted(validation_ids), dtype=np.int64))
    return training, validation, singletons


class AxisHierarchyClassifier(nn.Module):
    """Four learned logistic gates defining the five-class hierarchy."""

    def __init__(self, boundaries, slopes):
        super().__init__()
        self.boundaries = nn.Parameter(torch.as_tensor(boundaries, dtype=torch.float32).clone())
        self.log_slopes = nn.Parameter(torch.log(torch.as_tensor(slopes, dtype=torch.float32)))

    def forward(self, points):
        values = points[..., [0, 1, 0, 1]]
        logits = (values - self.boundaries) * self.log_slopes.clamp(-3, 9).exp()
        yes, no = functional.logsigmoid(logits), functional.logsigmoid(-logits)
        return torch.stack(
            [
                no[..., 0],
                yes[..., 0] + no[..., 1],
                yes[..., 0] + yes[..., 1] + no[..., 2],
                yes[..., 0] + yes[..., 1] + yes[..., 2] + no[..., 3],
                yes[..., 0] + yes[..., 1] + yes[..., 2] + yes[..., 3],
            ],
            dim=-1,
        )

    def probabilities(self, points):
        return self(points).exp()


def initialize_from_labeled_cells(points, labels):
    boundaries, slopes, intervals = [], [], []
    for axis, selected, positive in [
        (0, np.ones(len(labels), dtype=bool), labels != 0),
        (1, labels != 0, labels != 1),
        (0, labels >= 2, labels >= 3),
        (1, labels >= 3, labels == 4),
    ]:
        negative_values = points[selected & ~positive, axis]
        positive_values = points[selected & positive, axis]
        if len(negative_values) == 0 or len(positive_values) == 0:
            raise ValueError("Every classifier gate needs both outcomes in the training cells.")
        lower, upper = float(negative_values.max()), float(positive_values.min())
        if lower >= upper:
            raise ValueError("Hard labels are not separable by the declared axis hierarchy.")
        boundaries.append((lower + upper) / 2.0)
        slopes.append(8.0 / max(float(np.std(np.r_[negative_values, positive_values])), 0.02))
        intervals.append([lower, upper])
    return AxisHierarchyClassifier(boundaries, slopes), intervals


def train_classifier(training_file: Path, output: Path, settings, seed=21):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    data = np.load(training_file, allow_pickle=False)
    points, labels = data["points"], data["labels"]
    expected = np.eye(len(CLASS_NAMES), dtype=np.float32)[labels]
    if not np.array_equal(data["one_hot"], expected):
        raise ValueError("Classifier targets must be exact hard-label one-hot vectors.")
    train_ids, validation_ids, singletons = split_indices(
        labels, float(settings["train_fraction"]), int(seed)
    )
    train_x = np.concatenate([values[index] for values, index in zip(points, train_ids)])
    train_y = np.concatenate([values[index] for values, index in zip(labels, train_ids)])
    validation_x = np.concatenate([values[index] for values, index in zip(points, validation_ids)])
    validation_y = np.concatenate([values[index] for values, index in zip(labels, validation_ids)])
    model, gaps = initialize_from_labeled_cells(train_x, train_y)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.long)
    validation_x_tensor = torch.as_tensor(validation_x, dtype=torch.float32)
    validation_y_tensor = torch.as_tensor(validation_y, dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(settings["learning_rate"]))
    best, best_step, best_state, stale = float("inf"), 0, None, 0
    history = []
    stop_reason = "fixed_safety_cap"
    for step in range(1, int(settings["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.nll_loss(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % int(settings["eval_every"]) == 0:
            with torch.no_grad():
                log_probabilities = model(validation_x_tensor)
                validation_loss = float(functional.nll_loss(log_probabilities, validation_y_tensor))
                accuracy = float((log_probabilities.argmax(-1) == validation_y_tensor).float().mean())
            if validation_loss < best - 1e-6:
                best, best_step = validation_loss, step
                best_state, stale = copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            history.append(
                {
                    "step": step,
                    "train_ce": float(loss.detach()),
                    "validation_ce": validation_loss,
                    "validation_accuracy": accuracy,
                    "boundaries": model.boundaries.detach().tolist(),
                }
            )
            if stale >= int(settings["patience"]):
                stop_reason = "validation_ce_early_stopping"
                break
    if best_state is None:
        raise RuntimeError("No classifier checkpoint was selected.")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "model": model.state_dict(),
            "class_names": CLASS_NAMES,
            "selected_step": best_step,
            "training_data_sha256": sha256(training_file),
        },
        output / "classifier.pt",
    )
    split_arrays = {
        f"{split}_{index}": values
        for split, bank in (("train", train_ids), ("validation", validation_ids))
        for index, values in enumerate(bank)
    }
    np.savez_compressed(output / "splits.npz", **split_arrays)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    manifest = {
        "status": "completed",
        "stop_reason": stop_reason,
        "steps": step,
        "selected_step": best_step,
        "validation_ce": best,
        "validation_accuracy": float(
            (model(validation_x_tensor).argmax(-1) == validation_y_tensor).float().mean()
        ),
        "learned_boundaries": model.boundaries.detach().tolist(),
        "learned_slopes": model.log_slopes.detach().exp().tolist(),
        "initial_training_separation_intervals": gaps,
        "training_singletons": singletons,
        "seconds": time.perf_counter() - started,
        "settings": settings,
        "checkpoint_sha256": sha256(output / "classifier.pt"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_classifier(path: Union[str, Path], device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if tuple(saved["class_names"]) != CLASS_NAMES:
        raise ValueError("Classifier class order does not match the benchmark hierarchy.")
    model = AxisHierarchyClassifier(np.zeros(4), np.ones(4))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def predict(model, points, batch_size=8192):
    shape = points.shape[:-1]
    flat = np.asarray(points).reshape(-1, 2)
    device = next(model.parameters()).device
    output = []
    for start in range(0, len(flat), batch_size):
        batch = torch.as_tensor(flat[start : start + batch_size], dtype=torch.float32, device=device)
        output.append(model.probabilities(batch).cpu().numpy())
    return np.concatenate(output).reshape(*shape, len(CLASS_NAMES))

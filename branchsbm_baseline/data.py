"""Endpoint-labelled data adapter for the original BranchSBM implementation."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, TensorDataset


def _as_float_tensor(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float32))


def _sample_rows(
    values: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == 0:
        raise ValueError("Cannot resample an empty BranchSBM population.")
    indices = rng.choice(
        values.shape[0],
        size=int(size),
        replace=values.shape[0] < int(size),
    )
    return values[indices]


def _limited_rows(
    values: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if max_points <= 0 or values.shape[0] <= max_points:
        return values
    indices = rng.choice(values.shape[0], size=max_points, replace=False)
    return values[indices]


class BranchEndpointDataModule(pl.LightningDataModule):
    """Present labelled endpoint marginals in BranchSBM's native batch format.

    The wrapped flow-maps data module performs the benchmark split first.  Each
    terminal cell type is then independently resampled to the source-pool size,
    exactly as BranchSBM's branched single-cell loaders resample terminal
    clusters.  Consequently, oversampling a rare branch cannot copy a raw cell
    from training into validation.
    """

    def __init__(
        self,
        base_datamodule,
        *,
        retained_timepoints: Sequence[str],
        class_order: Iterable[str],
        seed: int,
        batch_size: int,
        metric_max_points_per_cluster: int = 0,
    ):
        super().__init__()
        retained = tuple(str(value) for value in retained_timepoints)
        if len(retained) < 2:
            raise ValueError("BranchSBM requires source and terminal timepoints.")
        if not hasattr(base_datamodule, "timepoint_splits"):
            raise ValueError(
                "BranchSBM requires cell-type-aware timepoint_splits from the "
                "flow-maps data module."
            )

        self.base_datamodule = base_datamodule
        self.retained_timepoints = retained
        self.source_timepoint = retained[0]
        self.terminal_timepoint = retained[-1]
        self.timepoint_splits = {
            str(timepoint): dict(pool)
            for timepoint, pool in base_datamodule.timepoint_splits.items()
        }
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.metric_max_points_per_cluster = int(metric_max_points_per_cluster)
        self.num_timesteps = 2
        self.times = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self.data_type = getattr(base_datamodule, "data_type", "scrna")
        self.data_name = getattr(base_datamodule, "data_name", "")
        self.whiten = bool(getattr(base_datamodule, "whiten", False))
        if hasattr(base_datamodule, "scaler"):
            self.scaler = base_datamodule.scaler

        self._prepare_branches(tuple(str(value) for value in class_order))
        self._copy_evaluation_state()

    def _copy_evaluation_state(self) -> None:
        for name in (
            "all_timepoint_data",
            "cfg",
            "classifier_path",
            "eval_pairs",
            "eval_pair_stats",
            "splits",
        ):
            if hasattr(self.base_datamodule, name):
                setattr(self, name, getattr(self.base_datamodule, name))

    def _prepare_branches(self, class_order: Sequence[str]) -> None:
        source_pool = self.timepoint_splits[self.source_timepoint]
        terminal_pool = self.timepoint_splits[self.terminal_timepoint]
        terminal_types_all = np.asarray(terminal_pool["types"], dtype=object)
        terminal_train_x = np.asarray(terminal_pool["train_x"], dtype=np.float32)
        terminal_holdout_x = np.asarray(
            terminal_pool["holdout_x"], dtype=np.float32
        )
        terminal_types_train = np.asarray(terminal_pool["train_types"], dtype=object)
        terminal_types_holdout = np.asarray(
            terminal_pool["holdout_types"], dtype=object
        )

        present = set(str(value) for value in terminal_types_all)
        unknown = present - set(class_order)
        if unknown:
            raise ValueError(
                "Terminal cell types are absent from the canonical class order: "
                f"{sorted(unknown)}."
            )

        # Branch zero is the method's distinguished mass-carrying trunk.  Use
        # the most abundant terminal type, with canonical order as a stable
        # tie-breaker, while retaining every non-empty terminal cell type.
        full_counts = Counter(str(value) for value in terminal_types_all)
        canonical_index = {name: index for index, name in enumerate(class_order)}
        self.branch_labels = tuple(
            sorted(
                present,
                key=lambda name: (-full_counts[name], canonical_index[name]),
            )
        )
        self.num_branches = len(self.branch_labels)
        if self.num_branches == 0:
            raise ValueError("The terminal day has no cell types for BranchSBM.")

        (
            terminal_train_x,
            terminal_types_train,
            terminal_holdout_x,
            terminal_types_holdout,
            self.terminal_split_repairs,
        ) = self._ensure_terminal_split_coverage(
            terminal_train_x,
            terminal_types_train,
            terminal_holdout_x,
            terminal_types_holdout,
        )
        train_counts = Counter(str(value) for value in terminal_types_train)
        holdout_counts = Counter(str(value) for value in terminal_types_holdout)
        self.timepoint_splits[self.terminal_timepoint].update(
            {
                "train_x": terminal_train_x,
                "train_types": terminal_types_train,
                "holdout_x": terminal_holdout_x,
                "holdout_types": terminal_types_holdout,
            }
        )
        missing_train = [name for name in self.branch_labels if train_counts[name] == 0]
        missing_holdout = [
            name for name in self.branch_labels if holdout_counts[name] == 0
        ]
        if missing_train or missing_holdout:
            raise ValueError(
                "Every terminal type must occur on both sides of the existing "
                "benchmark split. Missing from train="
                f"{missing_train}, missing from validation={missing_holdout}."
            )

        # Match the original BranchSBM loaders: branch mass and inverse-size
        # loss weighting come from the unresampled full terminal distribution.
        self.cluster_sizes = [full_counts[name] for name in self.branch_labels]
        terminal_total = float(sum(self.cluster_sizes))
        self.branch_weights = np.asarray(
            [count / terminal_total for count in self.cluster_sizes],
            dtype=np.float32,
        )
        self.branch_metadata = [
            {
                "branch_index": index,
                "cell_type": name,
                "full_count": int(full_counts[name]),
                "train_count": int(train_counts[name]),
                "holdout_count": int(holdout_counts[name]),
                "target_mass": float(self.branch_weights[index]),
            }
            for index, name in enumerate(self.branch_labels)
        ]

        source_train = np.asarray(source_pool["train_x"], dtype=np.float32)
        source_holdout = np.asarray(source_pool["holdout_x"], dtype=np.float32)
        if min(source_train.shape[0], source_holdout.shape[0]) < self.batch_size:
            raise ValueError(
                "The source train and validation pools must each contain at "
                f"least batch_size={self.batch_size} cells."
            )

        rng_train = np.random.default_rng(self.seed + 101)
        rng_holdout = np.random.default_rng(self.seed + 211)
        self.train_dataloaders = {
            "x0": self._weighted_loader(
                source_train,
                weight=1.0,
                shuffle=True,
                drop_last=True,
            )
        }
        self.val_dataloaders = {
            "x0": self._weighted_loader(
                source_holdout,
                weight=1.0,
                shuffle=False,
                drop_last=True,
            )
        }

        for branch_index, label in enumerate(self.branch_labels, start=1):
            train_branch = terminal_train_x[terminal_types_train == label]
            holdout_branch = terminal_holdout_x[terminal_types_holdout == label]
            sampled_train = _sample_rows(
                train_branch, source_train.shape[0], rng_train
            )
            sampled_holdout = _sample_rows(
                holdout_branch, source_holdout.shape[0], rng_holdout
            )
            mass = float(self.branch_weights[branch_index - 1])
            key = "x1" if self.num_branches == 1 else f"x1_{branch_index}"
            self.train_dataloaders[key] = self._weighted_loader(
                sampled_train,
                weight=mass,
                shuffle=True,
                drop_last=True,
            )
            self.val_dataloaders[key] = self._weighted_loader(
                sampled_holdout,
                weight=mass,
                shuffle=False,
                drop_last=True,
            )

        # BranchSBM's RBF metric has two clusters: source and all retained
        # post-source observations.  The latter includes the retained middle
        # marginal as requested, but never an omitted evaluation day.
        metric_rng = np.random.default_rng(self.seed + 307)
        metric_source = _limited_rows(
            source_train,
            self.metric_max_points_per_cluster,
            metric_rng,
        )
        metric_post_source = np.concatenate(
            [
                np.asarray(
                    (
                        terminal_train_x
                        if timepoint == self.terminal_timepoint
                        else self.timepoint_splits[timepoint]["train_x"]
                    ),
                    dtype=np.float32,
                )
                for timepoint in self.retained_timepoints[1:]
            ],
            axis=0,
        )
        metric_post_source = _limited_rows(
            metric_post_source,
            self.metric_max_points_per_cluster,
            metric_rng,
        )
        self.metric_group_sizes = (
            int(metric_source.shape[0]),
            int(metric_post_source.shape[0]),
        )
        self.metric_samples_dataloaders = [
            DataLoader(
                _as_float_tensor(metric_source),
                batch_size=metric_source.shape[0],
                shuffle=False,
                drop_last=False,
            ),
            DataLoader(
                _as_float_tensor(metric_post_source),
                batch_size=metric_post_source.shape[0],
                shuffle=False,
                drop_last=False,
            ),
        ]

        self._test_source = _as_float_tensor(source_holdout)

    def _ensure_terminal_split_coverage(
        self,
        train_x,
        train_types,
        holdout_x,
        holdout_types,
    ):
        """Repair only a missing rare-type side while keeping pools disjoint."""

        train_x = np.asarray(train_x, dtype=np.float32)
        holdout_x = np.asarray(holdout_x, dtype=np.float32)
        train_types = np.asarray(train_types, dtype=object)
        holdout_types = np.asarray(holdout_types, dtype=object)
        repairs = []
        rng = np.random.default_rng(self.seed + 43)

        def move_one(label, source_x, source_types, target_x, target_types, direction):
            candidates = np.flatnonzero(source_types == label)
            if candidates.size <= 1:
                raise ValueError(
                    f"Terminal type {label!r} cannot be represented in both "
                    "training and validation without duplicating a raw cell."
                )
            selected = int(rng.choice(candidates))
            moved_x = source_x[selected : selected + 1]
            moved_type = source_types[selected : selected + 1]
            source_x = np.delete(source_x, selected, axis=0)
            source_types = np.delete(source_types, selected, axis=0)
            target_x = np.concatenate([target_x, moved_x], axis=0)
            target_types = np.concatenate([target_types, moved_type], axis=0)
            repairs.append({"cell_type": label, "direction": direction})
            return source_x, source_types, target_x, target_types

        for label in self.branch_labels:
            if not np.any(train_types == label):
                holdout_x, holdout_types, train_x, train_types = move_one(
                    label,
                    holdout_x,
                    holdout_types,
                    train_x,
                    train_types,
                    "validation_to_train",
                )
            if not np.any(holdout_types == label):
                train_x, train_types, holdout_x, holdout_types = move_one(
                    label,
                    train_x,
                    train_types,
                    holdout_x,
                    holdout_types,
                    "train_to_validation",
                )
        return train_x, train_types, holdout_x, holdout_types, repairs

    def _weighted_loader(
        self,
        values: np.ndarray,
        *,
        weight: float,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        x = _as_float_tensor(values)
        weights = torch.full((x.shape[0], 1), float(weight), dtype=torch.float32)
        return DataLoader(
            TensorDataset(x, weights),
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader(self):
        loaders = {
            "train_samples": CombinedLoader(
                self.train_dataloaders,
                mode="min_size",
            ),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders,
                mode="min_size",
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def val_dataloader(self):
        loaders = {
            "val_samples": CombinedLoader(
                self.val_dataloaders,
                mode="min_size",
            ),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders,
                mode="min_size",
            ),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        # Evaluation callbacks read the named benchmark pools directly.  This
        # one-batch loader only gives Lightning a test loop on which to invoke
        # on_test_start after restoring the selected checkpoint.
        return DataLoader(
            TensorDataset(self._test_source),
            batch_size=self._test_source.shape[0],
            shuffle=False,
            drop_last=False,
        )

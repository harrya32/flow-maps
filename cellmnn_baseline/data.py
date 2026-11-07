"""Shared cell holdouts with the upstream Cell-MNN population sampler."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import cite_multi, maizels
from lib.data.data_loading import MnnDataset
from mfm.dataloaders.maizels_data import SCHEDULES, _timepoint_values

from . import REPO_ROOT


class PC1Scaler:
    """Appendix G.1's single global scale, fitted on retained training cells."""

    def fit(self, x):
        self.scale_ = float(np.std(np.asarray(x, dtype=np.float64)[:, 0]))
        if not np.isfinite(self.scale_) or self.scale_ <= 0:
            raise ValueError(
                "Training PC1 must have a finite, positive standard deviation."
            )
        return self

    def transform(self, x):
        return np.asarray(x, dtype=np.float32) / self.scale_

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32) * self.scale_


class PopulationDataset(MnnDataset):
    """Reuse upstream sampling for any retained schedule, with finite epochs."""

    def __init__(self, arrays, times, batch_size, *, batches=None):
        # Upstream requires an omitted interior entry, even for endpoint training.
        # An empty excluded entry lets its sampler use exactly our retained pools.
        marker = (float(times[0]) + float(times[1])) / 2.0
        super().__init__(
            X=[arrays[0], np.empty((0, arrays[0].shape[1]), np.float32), *arrays[1:]],
            days=[times[0], marker, *times[1:]],
            batch_size=int(batch_size),
            skip_day_idx=1,
            device=torch.device("cpu"),
        )
        self.batches = (
            max(1, int(batches)) if batches is not None else max(1, super().__len__())
        )

    def __len__(self):
        return self.batches

    def __iter__(self):
        iterator = super().__iter__()
        for _ in range(self.batches):
            yield next(iterator)


def _maizels_path(location):
    if location:
        path = maizels.resolve_dataset_path(location).expanduser()
        nested = Path(location).expanduser() / "data" / path.name
        return (nested if not path.is_file() and nested.is_file() else path).resolve()
    local = REPO_ROOT / "celltype_classification_pca50_dataset.csv.gz"
    return (local if local.is_file() else Path(maizels.DEFAULT_DATASET)).resolve()


def prepare_data(args, config, heldout_day, seed):
    """Omitted days never enter normalization, training, or validation loss."""
    is_maizels = args.dataset_name == "maizels"
    if is_maizels:
        path = _maizels_path(args.dataset_location)
        retained = tuple(SCHEDULES[args.maizels_schedule])
        order = maizels.TIMEPOINTS
        time_map = dict(zip(order, _timepoint_values(retained, args.maizels_time_mode)))
        time_mode = args.maizels_time_mode
        source, target = "D3", "D8"
    else:
        path = cite_multi.resolve_dataset_path(args.dataset_location, args.dataset_name)
        order = cite_multi.TIMEPOINTS
        retained = tuple(day for day in order if day != str(heldout_day))
        time_map = cite_multi.normalized_time_map(args.cite_multi_time_mode)
        time_mode = args.cite_multi_time_mode
        source, target = "2", "7"
    cfg = SimpleNamespace(
        problem=SimpleNamespace(
            dataset_name=args.dataset_name,
            dataset_location=str(path),
            source_time=source,
            target_time=target,
            heldout_timepoint=None if is_maizels else str(heldout_day),
            retained_timepoints=list(retained),
            timepoint_order=list(order),
            timepoint_values=[time_map[day] for day in order],
            cite_multi_time_mode=time_mode,
            cite_multi_train_fraction=1.0 - float(config["val_split"]),
            maizels_schedule=args.maizels_schedule,
            maizels_time_mode=time_mode,
            maizels_holdout_fraction=float(config["val_split"]),
            maizels_holdout_n=0,
            maizels_holdout_seed=701,
        ),
        training=SimpleNamespace(seed=int(seed)),
    )
    loader = (
        maizels.timepoint_pool_splits
        if is_maizels
        else cite_multi.timepoint_pool_splits
    )
    pools = loader(cfg)
    if any(
        len(pools[day][key]) == 0
        for day in retained
        for key in ("train_x", "holdout_x")
    ):
        raise ValueError(
            "Every retained day needs non-empty training and validation cells."
        )
    arrays = [pools[day]["train_x"] for day in retained]
    scaler = PC1Scaler().fit(np.concatenate(arrays)) if config["normalize"] else None
    # Keep their per-day ODE/kinetic-energy units, while respecting our clock.
    parse = maizels.parse_timepoint
    time_origin = parse(source)
    time_scale = parse(target) - time_origin
    model_time_map = {day: time_origin + time_scale * time_map[day] for day in order}
    retained_times = [model_time_map[day] for day in retained]

    def make_dataset(key, batches):
        values = [np.asarray(pools[day][key], dtype=np.float32) for day in retained]
        if scaler is not None:
            values = [scaler.transform(x) for x in values]
        return PopulationDataset(
            values, retained_times, config["batch_size"], batches=batches
        )

    train = make_dataset("train_x", 1 if args.smoke_test else config["train_batches"])
    val = make_dataset("holdout_x", 1 if args.smoke_test else config["val_batches"])
    return SimpleNamespace(
        cfg=cfg,
        dataset_name=args.dataset_name,
        path=path,
        heldout_day=str(heldout_day),
        retained=retained,
        time_map=time_map,
        model_time_map=model_time_map,
        time_origin=time_origin,
        time_scale=time_scale,
        time_mode=time_mode,
        timepoint_splits=pools,
        scaler=scaler,
        train_dataset=train,
        val_dataset=val,
        train_loader=DataLoader(train, batch_size=None),
        val_loader=DataLoader(val, batch_size=None),
    )

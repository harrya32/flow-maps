"""Use shared benchmark cell splits with upstream independent or OT tuples."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from torch.utils.data import DataLoader, Subset

from common import cite_multi, maizels
from mfm.dataloaders.maizels_data import SCHEDULES, _timepoint_values
from experiments.citeseq.data import (
    CiteSeqMultiMarginalDataset,
    MaxStdScaler,
    compute_ot_alignments,
)

from . import REPO_ROOT


def _maizels_path(location):
    if location:
        path = maizels.resolve_dataset_path(location).expanduser()
        nested = Path(location).expanduser() / "data" / path.name
        return (nested if not path.is_file() and nested.is_file() else path).resolve()
    local = REPO_ROOT / "celltype_classification_pca50_dataset.csv.gz"
    return (local if local.is_file() else Path(maizels.DEFAULT_DATASET)).resolve()


def prepare_data(
    dataset_name,
    heldout_day,
    time_mode,
    dataset_location,
    config,
    *,
    smoke_test=False,
    maizels_schedule="d3_d3p8_d8",
):
    """Raw benchmark pools for evaluation; normalize only retained training cells."""
    is_maizels = dataset_name == "maizels"
    if is_maizels:
        path = _maizels_path(dataset_location)
        order = maizels.TIMEPOINTS
        retained = tuple(SCHEDULES[maizels_schedule])
        time_map = dict(zip(order, _timepoint_values(retained, time_mode)))
        source, target = "D3", "D8"
        heldout_day = None
    else:
        dataset_name = cite_multi.canonical_dataset_name(dataset_name)
        heldout_day = str(heldout_day)
        time_mode = cite_multi.canonical_time_mode(time_mode)
        path = cite_multi.resolve_dataset_path(dataset_location, dataset_name)
        order = cite_multi.TIMEPOINTS
        retained = tuple(day for day in order if day != heldout_day)
        time_map = cite_multi.normalized_time_map(time_mode)
        source, target = "2", "7"
    cfg = SimpleNamespace(
        problem=SimpleNamespace(
            dataset_name=dataset_name,
            dataset_location=str(path),
            source_time=source,
            target_time=target,
            heldout_timepoint=heldout_day,
            retained_timepoints=list(retained),
            cite_multi_time_mode=time_mode,
            timepoint_order=list(order),
            timepoint_values=list(time_map.values()),
            cite_multi_train_fraction=1.0 - float(config["val_split"]),
            maizels_schedule=maizels_schedule,
            maizels_time_mode=time_mode,
            maizels_holdout_fraction=float(config["val_split"]),
            maizels_holdout_n=0,
            # Match the suite's fixed cell split for both representations.
            maizels_holdout_seed=701,
        ),
        training=SimpleNamespace(seed=int(config["seed"])),
    )
    loader = (
        maizels.timepoint_pool_splits
        if is_maizels
        else cite_multi.timepoint_pool_splits
    )
    pools = loader(cfg)
    # Upstream OT logging expects CITE indices 0..3. Maizels' retained
    # tuple columns use contiguous IDs; their physical times stay in time_map.
    tuple_order = (
        (*retained, *(day for day in order if day not in retained))
        if is_maizels
        else order
    )
    day_indices = {day: index for index, day in enumerate(tuple_order)}
    train_arrays = [pools[day]["train_x"] for day in retained]
    if any(
        len(pools[day][key]) == 0
        for day in retained
        for key in ("train_x", "holdout_x")
    ):
        raise ValueError(
            "Every retained day needs non-empty training and validation cells."
        )

    scaler = None
    if config["normalize"]:
        scaler = MaxStdScaler().fit(np.concatenate(train_arrays))
        if not np.isfinite(scaler.scale_) or scaler.scale_ <= 0:
            raise ValueError(
                "The training population has an invalid normalization scale."
            )

    def make_dataset(split_key):
        arrays = [pools[day][split_key] for day in retained]
        if smoke_test and config["ot_coupling"]:
            # Keep the dense startup OT solves small for a one-batch smoke run.
            split_seed = 1703 if split_key == "train_x" else 2803
            rng = np.random.default_rng(int(config["seed"]) + split_seed)
            arrays = [
                array[
                    rng.choice(
                        len(array),
                        size=min(len(array), int(config["batch_size"])),
                        replace=False,
                    )
                ]
                for array in arrays
            ]
        pcs = np.concatenate(arrays).astype(np.float32)
        if scaler is not None:
            pcs = scaler.transform(pcs).astype(np.float32)
        labels = np.concatenate(
            [
                np.full(len(array), day_indices[day], dtype=np.int64)
                for day, array in zip(retained, arrays)
            ]
        )
        alignments = None
        if config["ot_coupling"]:
            alignments = compute_ot_alignments(
                pcs,
                labels,
                [day_indices[day] for day in retained],
                method=config["ot_method"],
                reg=float(config.get("ot_reg", 0.01)),
            )
            if alignments is None:
                raise ImportError("Precomputed OT coupling requires POT (import ot).")
        return CiteSeqMultiMarginalDataset(
            pcs,
            labels,
            holdout_times=[day_indices[day] for day in order if day not in retained],
            ot_alignments=alignments,
        )

    train_dataset = make_dataset("train_x")
    val_dataset = make_dataset("holdout_x")

    def make_loader(dataset, shuffle):
        if smoke_test:
            dataset = Subset(
                dataset, range(min(len(dataset), int(config["batch_size"])))
            )
        return DataLoader(
            dataset, batch_size=int(config["batch_size"]), shuffle=shuffle
        )

    return SimpleNamespace(
        cfg=cfg,
        dataset_name=dataset_name,
        heldout_day=heldout_day,
        path=path,
        retained=retained,
        day_indices=day_indices,
        time_mode=time_mode,
        source_day=source,
        target_day=target,
        time_map=time_map,
        timepoint_splits=pools,
        scaler=scaler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=make_loader(train_dataset, True),
        val_loader=make_loader(val_dataset, False),
    )

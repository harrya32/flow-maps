"""Use shared benchmark cell splits with upstream independent or OT tuples."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from torch.utils.data import DataLoader, Subset

from common import cite_multi
from experiments.citeseq.data import (
    CiteSeqMultiMarginalDataset,
    MaxStdScaler,
    compute_ot_alignments,
)


def prepare_data(
    dataset_name, heldout_day, time_mode, dataset_location, config, *, smoke_test=False
):
    """Keep raw PCA100 pools for evaluation and normalize only training tuples."""
    dataset_name = cite_multi.canonical_dataset_name(dataset_name)
    heldout_day = str(heldout_day)
    time_mode = cite_multi.canonical_time_mode(time_mode)
    path = cite_multi.resolve_dataset_path(dataset_location, dataset_name)
    time_map = cite_multi.normalized_time_map(time_mode)
    cfg = SimpleNamespace(
        problem=SimpleNamespace(
            dataset_name=dataset_name,
            dataset_location=str(path),
            source_time="2",
            target_time="7",
            heldout_timepoint=heldout_day,
            cite_multi_time_mode=time_mode,
            timepoint_order=list(cite_multi.TIMEPOINTS),
            timepoint_values=list(time_map.values()),
            cite_multi_train_fraction=1.0 - float(config["val_split"]),
            # Match configs.cite_multi_pca100, including its fixed cell split.
            maizels_holdout_seed=701,
        ),
        training=SimpleNamespace(seed=int(config["seed"])),
    )
    retained = cite_multi.retained_timepoints(cfg)
    pools = cite_multi.timepoint_pool_splits(cfg)
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
                np.full(len(array), cite_multi.TIMEPOINTS.index(day), dtype=np.int64)
                for day, array in zip(retained, arrays)
            ]
        )
        alignments = None
        if config["ot_coupling"]:
            alignments = compute_ot_alignments(
                pcs,
                labels,
                [cite_multi.TIMEPOINTS.index(day) for day in retained],
                method=config["ot_method"],
                reg=float(config.get("ot_reg", 0.01)),
            )
            if alignments is None:
                raise ImportError("Precomputed OT coupling requires POT (import ot).")
        return CiteSeqMultiMarginalDataset(
            pcs,
            labels,
            holdout_times=[cite_multi.TIMEPOINTS.index(heldout_day)],
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
        time_map=time_map,
        timepoint_splits=pools,
        scaler=scaler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_loader=make_loader(train_dataset, True),
        val_loader=make_loader(val_dataset, False),
    )

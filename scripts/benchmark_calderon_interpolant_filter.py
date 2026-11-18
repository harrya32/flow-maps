#!/usr/bin/env python3
"""Compare complete versus early-rejection filtering on identical OT blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

import jax
import numpy as np
from ml_collections import ConfigDict

from common.calderon import training_dataset, load_endpoint_experiment, sha256_file
from configs.calderon_constrained_pca100 import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error("--blocks must be positive")
    cfg = get_config()
    data = load_endpoint_experiment(cfg)
    datasets = []
    for chunk in (50, 10):
        config = ConfigDict(cfg.to_dict())
        config.problem.interpolant_filter_backend = "reference"
        config.problem.interpolant_check_chunk_size = chunk
        ds = training_dataset(config, data, "train")
        datasets.append(ds)
    model = datasets[0].datasets[0].classifier
    for ds in datasets:
        batch = ds.sample_device_batch(cfg.optimization.bs, jax.random.PRNGKey(999))
        jax.block_until_ready(batch["x0"])
    durations, point_counts = [], []
    for i in range(args.blocks):
        outputs, stats, elapsed, points = [], [], [], []
        for ds in datasets:
            before = ds.interpolant_classifier_points
            start = time.monotonic()
            batch = ds.sample_device_batch(cfg.optimization.bs, jax.random.PRNGKey(1000 + i))
            jax.block_until_ready(batch["x0"])
            elapsed.append(time.monotonic() - start)
            points.append(ds.interpolant_classifier_points - before)
            outputs.append({name: np.asarray(value) for name, value in batch.items()})
            stats.append(ds.last_pair_stats)
        for name in outputs[0]:
            np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
        assert stats[0] == stats[1], (stats[0], stats[1])
        durations.append(elapsed)
        point_counts.append(points)
    durations, point_counts = np.asarray(durations), np.asarray(point_counts)
    result = {"status": "passed", "devices": [str(d) for d in jax.devices()],
              "n_identical_blocks": args.blocks, "minibatch_size": cfg.problem.ot_minibatch_size,
              "training_windows": list(data.training_times),
              "training_intervals": [list(interval) for interval in datasets[0].intervals],
              "seeds": list(range(1000, 1000 + args.blocks)),
              "optimizer_pairs_and_solver_statistics_identical": True,
              "whole_filter_mean_seconds": float(durations[:, 0].mean()),
              "early_rejection_mean_seconds": float(durations[:, 1].mean()),
              "speed_ratio": float(durations[:, 0].mean() / durations[:, 1].mean()),
              "whole_filter_points": int(point_counts[:, 0].sum()),
              "early_rejection_points": int(point_counts[:, 1].sum()),
              "early_rejection_point_fraction": float(point_counts[:, 1].sum() / point_counts[:, 0].sum()),
              "every_retained_interpolant_checks_all_50_points": True,
              "export_sha256": model.export_sha256, "manifest_sha256": model.manifest_sha256,
              "sampler_sha256": sha256_file(ROOT / "py/common/calderon.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

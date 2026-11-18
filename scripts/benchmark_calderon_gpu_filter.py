#!/usr/bin/env python3
"""Compare reference and fused JAX filtering on the same Calderon OT blocks."""

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

from common.calderon import load_endpoint_experiment, sha256_file, training_dataset
from configs.calderon_constrained_pca100 import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--parity-pairs-per-interval", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.blocks < 1 or args.parity_pairs_per_interval < 1:
        parser.error("Block and parity-pair counts must be positive.")
    cfg = get_config()
    print("Loading the three-time Calderon training populations.", flush=True)
    data = load_endpoint_experiment(cfg)
    datasets = []
    for backend in ("reference", "jax"):
        config = ConfigDict(cfg.to_dict())
        config.problem.interpolant_filter_backend = backend
        datasets.append(training_dataset(config, data, "train"))

    warmup_start = time.monotonic()
    # Warm every possible pair bucket before taking steady-state timings.
    decoder = datasets[0].datasets[0].classifier
    for dataset in datasets:
        # The fused backend also uses this path for its near-tie fallback.
        model = dataset.datasets[0].classifier
        size = 32
        while size < model.batch_size:
            model.predict_proba(np.zeros((size, 100), dtype=np.float32))
            size *= 2
        model.predict_proba(np.zeros((model.batch_size, 100), dtype=np.float32))
    gpu = datasets[1].datasets[0]
    size = 1
    while size <= gpu.jax_interpolant_filter.pair_batch_size:
        print(f"Compiling the fused filter for {size} pairs.", flush=True)
        gpu.check_interpolants(gpu.source["x"][:size], gpu.source["type_ids"][:size],
                               gpu.target["x"][:size], gpu.target["type_ids"][:size])
        size *= 2
    for dataset in datasets:
        batch = dataset.sample_device_batch(cfg.optimization.bs, jax.random.PRNGKey(999))
        jax.block_until_ready(batch["x0"])
    warmup_seconds = time.monotonic() - warmup_start

    # Arbitrary pairs probe decisions independently of the OT-selected edges.
    parity = []
    rng = np.random.default_rng(1701)
    for reference, fused in zip(datasets[0].datasets, datasets[1].datasets):
        n = args.parity_pairs_per_interval
        source_rows = rng.choice(len(reference.source["x"]), n, replace=False)
        target_rows = rng.choice(len(reference.target["x"]), n, replace=False)
        inputs = (reference.source["x"][source_rows], reference.source["type_ids"][source_rows],
                  reference.target["x"][target_rows], reference.target["type_ids"][target_rows])
        start = time.monotonic()
        expected = reference.check_interpolants(*inputs)
        reference_seconds = time.monotonic() - start
        start = time.monotonic()
        actual = fused.check_interpolants(*inputs)
        jax_seconds = time.monotonic() - start
        for name in ("valid", "confident", "rejected_at_checkpoint", "rejected_at_final"):
            np.testing.assert_array_equal(expected[name], actual[name], err_msg=name)
        parity.append({"n_pairs": n, "n_interior_points": n * cfg.problem.n_interpolant_check_times,
                       "reference_fallback_pairs": actual["reference_fallback_pairs"],
                       "reference_seconds": reference_seconds, "jax_seconds": jax_seconds})

    profiles = [{"seconds": 0., "calls": 0, "pairs": 0} for _ in datasets]
    for dataset, profile in zip(datasets, profiles):
        for interval in dataset.datasets:
            def measured(*inputs, _checker=interval.interpolant_checker, _profile=profile, **kwargs):
                started = time.monotonic()
                result = _checker(*inputs, **kwargs)
                _profile["seconds"] += time.monotonic() - started
                _profile["calls"] += 1
                _profile["pairs"] += len(kwargs["source_x"] if kwargs else inputs[0])
                return result
            interval.interpolant_checker = measured

    dispatches_before = datasets[1].filter_dispatches
    durations, point_counts, fallback_pairs, filter_durations = [], [], [], []
    for index in range(args.blocks):
        outputs, stats, elapsed, points = [], [], [], []
        fallback_before = sum(d.interpolant_reference_fallback_pairs for d in datasets[1].datasets)
        # Alternate timing order to reduce order/temperature effects.
        order = (0, 1) if index % 2 == 0 else (1, 0)
        runs = {}
        for which in order:
            dataset = datasets[which]
            before = dataset.interpolant_classifier_points
            filter_before = profiles[which]["seconds"]
            started = time.monotonic()
            batch = dataset.sample_device_batch(cfg.optimization.bs, jax.random.PRNGKey(1000 + index))
            jax.block_until_ready(batch["x0"])
            dispatch_stats = dataset.last_filter_dispatch_stats
            filter_seconds = (dispatch_stats["seconds"] if dispatch_stats is not None
                              else profiles[which]["seconds"] - filter_before)
            runs[which] = (time.monotonic() - started,
                           dataset.interpolant_classifier_points - before,
                           {name: np.asarray(value) for name, value in batch.items()}, dataset.last_pair_stats,
                           filter_seconds)
        block_filter_durations = []
        for which in (0, 1):
            seconds, count, output, statistics, filter_seconds = runs[which]
            elapsed.append(seconds)
            points.append(count)
            outputs.append(output)
            stats.append(statistics)
            block_filter_durations.append(filter_seconds)
        for name in outputs[0]:
            np.testing.assert_array_equal(outputs[0][name], outputs[1][name], err_msg=name)
        assert stats[0] == stats[1], (stats[0], stats[1])
        assert points[0] == points[1]
        durations.append(elapsed)
        filter_durations.append(block_filter_durations)
        point_counts.append(points)
        fallback_pairs.append(sum(d.interpolant_reference_fallback_pairs for d in datasets[1].datasets) - fallback_before)
        print(f"Block {index + 1}: reference {elapsed[0]:.3f}s, fused JAX {elapsed[1]:.3f}s.", flush=True)

    durations, point_counts = np.asarray(durations), np.asarray(point_counts)
    result = {
        "status": "passed", "devices": [str(device) for device in jax.devices()],
        "training_windows": list(data.training_times),
        "training_intervals": [list(interval) for interval in datasets[0].intervals],
        "n_identical_blocks": args.blocks, "optimizer_batch_size": cfg.optimization.bs,
        "seeds": list(range(1000, 1000 + args.blocks)),
        "optimizer_pairs_and_solver_statistics_identical": True,
        "arbitrary_pair_decisions_identical": True, "arbitrary_pair_checks": parity,
        "warmup_seconds": warmup_seconds,
        "reference_mean_seconds": float(durations[:, 0].mean()),
        "jax_mean_seconds": float(durations[:, 1].mean()),
        "speedup_reference_over_jax": float(durations[:, 0].mean() / durations[:, 1].mean()),
        "per_block_seconds_reference_jax": durations.tolist(),
        "filter_mean_seconds_reference_jax": np.mean(filter_durations, axis=0).tolist(),
        "filter_profiles_reference_jax": profiles,
        "jax_worker_profile_seconds_can_overlap": cfg.problem.batch_interpolant_checks_across_intervals,
        "jax_gpu_dispatches": datasets[1].filter_dispatches - dispatches_before,
        "batch_interpolant_checks_across_intervals": cfg.problem.batch_interpolant_checks_across_intervals,
        "logical_classifier_points_per_backend": int(point_counts[:, 0].sum()),
        "reference_fallback_pairs": int(sum(fallback_pairs)),
        "reference_fallback_extra_points": int(sum(fallback_pairs) * cfg.problem.n_interpolant_check_times),
        "every_retained_interpolant_checks_all_50_points": cfg.problem.n_interpolant_check_times == 50,
        "decision_tolerance": cfg.problem.gpu_filter_decision_tolerance,
        "constraint_metrics_enabled": cfg.logging.constraint_metrics_enabled,
        "export_sha256": decoder.export_sha256, "manifest_sha256": decoder.manifest_sha256,
        "sampler_sha256": sha256_file(ROOT / "py/common/calderon.py"),
        "filter_sha256": sha256_file(ROOT / "py/common/calderon_filter.py"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

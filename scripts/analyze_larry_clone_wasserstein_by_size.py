#!/usr/bin/env python3
"""Stratify saved LARRY SSFM clone Wasserstein scores by clone size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PY_DIR = SCRIPT_DIR.parent / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import flax.serialization
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np

from common import larry
from common import larry_stochastic_eval
from common import maizels_stochastic_eval
from common import stochastic_flow_map
from common import wasserstein


SOURCE_BIN_ORDER = ("1", "2", "3", "4+")
TARGET_BIN_ORDER = ("1", "2", "3-4", "5-9", "10+")
TOTAL_BIN_ORDER = ("2", "3-4", "5-9", "10+")


def _checkpoint_spec(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must have the form LABEL=PATH")
    label, path_text = value.split("=", 1)
    path = Path(path_text).expanduser().resolve()
    if not label.strip():
        raise argparse.ArgumentTypeError("checkpoint label cannot be empty")
    if path.is_dir():
        path = path / "best.msgpack"
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return label.strip(), path


def _load_checkpoint(path: Path):
    payload = flax.serialization.msgpack_restore(path.read_bytes())
    config_path = path.with_name("config.json")
    if config_path.is_file():
        cfg = ml_collections.ConfigDict(json.loads(config_path.read_text()))
    elif "config" in payload:
        cfg = ml_collections.ConfigDict(payload["config"])
    else:
        raise FileNotFoundError(
            f"Neither {config_path} nor a config in {path} was found."
        )
    pc_scale = np.asarray(payload["pc_scale"], dtype=np.float32)
    model = stochastic_flow_map.StrongStochasticFlowMapMLP(
        data_dim=int(cfg.problem.d),
        n_coefficients=int(cfg.ssfm.n_coefficients),
        hidden_dim=int(cfg.ssfm.hidden_dim),
        n_hidden=int(cfg.ssfm.n_hidden),
        rescale=float(np.sqrt(np.mean(np.square(pc_scale)))),
        uncertainty_hidden_dim=int(cfg.ssfm.uncertainty_hidden_dim),
    )
    params = jax.tree_util.tree_map(jnp.asarray, payload["ema_params"])
    return model, params, cfg, payload


def _source_size_bin(value: int) -> str:
    return str(value) if value <= 3 else "4+"


def _target_size_bin(value: int) -> str:
    if value <= 2:
        return str(value)
    if value <= 4:
        return "3-4"
    if value <= 9:
        return "5-9"
    return "10+"


def _total_size_bin(value: int) -> str:
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 9:
        return "5-9"
    return "10+"


def _clone_records(
    label: str,
    checkpoint: Path,
    target_times: Iterable[str],
    *,
    noise_draws: int | None,
    samples_per_source: int | None,
) -> List[Dict[str, object]]:
    model, params, cfg, payload = _load_checkpoint(checkpoint)
    data = larry.all_timepoint_data(
        cfg.problem.dataset_location,
        representation=str(cfg.problem.larry_representation),
        n_pcs=int(cfg.problem.n_pcs),
    )
    clone_ids = np.asarray(data["clone_ids"], dtype=np.int64)
    source_time = larry.format_timepoint(cfg.evaluation.clone_source_time)
    source_mask = (data["timepoints"] == source_time) & (clone_ids >= 0)
    source_x = np.asarray(data["x"][source_mask], dtype=np.float32)
    source_ids = clone_ids[source_mask]
    source_unique, source_counts_array = np.unique(source_ids, return_counts=True)
    source_counts = dict(zip(source_unique.tolist(), source_counts_array.tolist()))

    records: List[Dict[str, object]] = []
    base_seed = int(cfg.evaluation.clone_seed)
    for target_index, target_time_value in enumerate(target_times):
        target_time = larry.format_timepoint(target_time_value)
        target_mask = (data["timepoints"] == target_time) & (clone_ids >= 0)
        target_x = np.asarray(data["x"][target_mask], dtype=np.float32)
        target_ids = clone_ids[target_mask]
        target_unique, target_counts_array = np.unique(target_ids, return_counts=True)
        target_counts = dict(zip(target_unique.tolist(), target_counts_array.tolist()))
        min_source = max(1, int(cfg.evaluation.clone_min_source_cells))
        min_target = max(1, int(cfg.evaluation.clone_min_target_cells))
        eligible = np.asarray(
            sorted(
                clone_id
                for clone_id in set(source_counts).intersection(target_counts)
                if source_counts[clone_id] >= min_source
                and target_counts[clone_id] >= min_target
            ),
            dtype=np.int64,
        )
        if eligible.size == 0:
            continue

        target_seed = base_seed + 1009 * target_index
        rng = np.random.default_rng(target_seed)
        target_groups = larry_stochastic_eval._clone_target_groups(
            target_x,
            target_ids,
            eligible,
            maximum=int(cfg.evaluation.clone_max_target_cells),
            rng=rng,
        )
        n_samples_per_source = int(
            cfg.evaluation.clone_samples_per_source
            if samples_per_source is None
            else samples_per_source
        )
        if n_samples_per_source <= 0:
            raise ValueError("samples_per_source must be positive")
        repeated_source, source_spans, evaluated_source_counts = (
            larry_stochastic_eval._repeated_clone_sources(
                source_x,
                source_ids,
                eligible,
                n_samples_per_source,
            )
        )
        n_draws = int(
            cfg.evaluation.clone_n_noise_draws
            if noise_draws is None
            else noise_draws
        )
        if n_draws <= 0:
            raise ValueError("noise_draws must be positive")
        batch_size = int(cfg.evaluation.clone_batch_size)
        use_euler_maruyama = maizels_stochastic_eval.uses_euler_maruyama_evaluation(
            cfg
        )
        key = jax.random.PRNGKey(target_seed)
        distances_by_draw = []
        for _ in range(n_draws):
            key, draw_key = jax.random.split(key)
            sampler_args = (
                model,
                params,
                repeated_source,
                larry.normalized_time(source_time, cfg),
                larry.normalized_time(target_time, cfg),
                draw_key,
            )
            if use_euler_maruyama:
                prediction = larry_stochastic_eval._euler_maruyama_samples_in_batches(
                    *sampler_args,
                    n_coefficients=int(cfg.ssfm.n_coefficients),
                    n_steps=maizels_stochastic_eval.euler_maruyama_n_steps(cfg),
                    batch_size=batch_size,
                )
            else:
                prediction = larry_stochastic_eval._composed_samples_in_batches(
                    *sampler_args,
                    n_coefficients=int(cfg.ssfm.n_coefficients),
                    n_steps=int(cfg.evaluation.clone_flowmap_n_steps),
                    batch_size=batch_size,
                )
            distances_by_draw.append(
                [
                    wasserstein.exact_emd(
                        prediction[source_spans[int(clone_id)]],
                        target_groups[int(clone_id)],
                    )
                    for clone_id in eligible
                ]
            )

        distances = np.asarray(distances_by_draw, dtype=np.float64)
        mean_distances = np.mean(distances, axis=0)
        for clone_index, clone_id in enumerate(eligible.tolist()):
            source_size = int(evaluated_source_counts[clone_index])
            target_size = int(target_groups[int(clone_id)].shape[0])
            total_size = source_size + target_size
            records.append(
                {
                    "method": label,
                    "checkpoint": str(checkpoint),
                    "variant": str(cfg.logging.comparison_mode),
                    "training_seed": int(cfg.training.seed),
                    "best_step": int(np.asarray(payload["best_step"])),
                    "source_time": source_time,
                    "target_time": target_time,
                    "clone_id": int(clone_id),
                    "source_size": source_size,
                    "target_size": target_size,
                    "total_size": total_size,
                    "source_size_bin": _source_size_bin(source_size),
                    "target_size_bin": _target_size_bin(target_size),
                    "total_size_bin": _total_size_bin(total_size),
                    "samples_per_source": n_samples_per_source,
                    "noise_draws": n_draws,
                    "wasserstein_w1": float(mean_distances[clone_index]),
                    "wasserstein_w1_noise_sd": float(
                        np.std(distances[:, clone_index])
                    ),
                }
            )
        print(
            f"{label}: {source_time}->{target_time}, {eligible.size} clones, "
            f"macro W1={np.mean(mean_distances):.6g}",
            flush=True,
        )
    return records


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str, str], List[float]] = defaultdict(list)
    measures = (
        ("source_size", "source_size_bin"),
        ("target_size", "target_size_bin"),
        ("total_size", "total_size_bin"),
    )
    for row in rows:
        for measure, bin_field in measures:
            grouped[
                (
                    str(row["target_time"]),
                    measure,
                    str(row[bin_field]),
                    str(row["method"]),
                )
            ].append(float(row["wasserstein_w1"]))

    result = []
    for (target_time, measure, size_bin, method), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        sd = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        result.append(
            {
                "target_time": target_time,
                "size_measure": measure,
                "size_bin": size_bin,
                "method": method,
                "n_clones": int(array.size),
                "w1_mean": float(np.mean(array)),
                "w1_sd": sd,
                "w1_se": sd / math.sqrt(array.size),
                "w1_p10": float(np.quantile(array, 0.10)),
                "w1_q25": float(np.quantile(array, 0.25)),
                "w1_median": float(np.median(array)),
                "w1_q75": float(np.quantile(array, 0.75)),
                "w1_p90": float(np.quantile(array, 0.90)),
                "w1_min": float(np.min(array)),
                "w1_max": float(np.max(array)),
            }
        )
    return result


def _plot(
    rows: List[Dict[str, object]],
    output_path: Path,
    *,
    size_measure: str,
    method_order: List[str],
) -> None:
    bin_field = f"{size_measure}_bin"
    if size_measure == "source_size":
        bin_order = SOURCE_BIN_ORDER
        x_label = "Number of observed D2 cells in clone"
    elif size_measure == "target_size":
        bin_order = TARGET_BIN_ORDER
        x_label = "Number of observed target-day cells in clone"
    else:
        bin_order = TOTAL_BIN_ORDER
        x_label = "Number of observed D2 + target-day cells in clone"

    target_times = sorted({str(row["target_time"]) for row in rows})
    fig, axes = plt.subplots(
        len(target_times),
        1,
        figsize=(10.5, 4.1 * len(target_times)),
        squeeze=False,
        sharey=True,
    )
    palette = ("#4C78A8", "#F58518", "#54A24B", "#B279A2")
    width = 0.22 if len(method_order) >= 3 else 0.30
    offsets = (np.arange(len(method_order)) - (len(method_order) - 1) / 2.0) * width
    for axis_index, target_time in enumerate(target_times):
        ax = axes[axis_index, 0]
        for method_index, method in enumerate(method_order):
            values = []
            positions = []
            for bin_index, size_bin in enumerate(bin_order):
                group = [
                    float(row["wasserstein_w1"])
                    for row in rows
                    if str(row["target_time"]) == target_time
                    and str(row["method"]) == method
                    and str(row[bin_field]) == size_bin
                ]
                if group:
                    values.append(group)
                    positions.append(bin_index + offsets[method_index])
            boxes = ax.boxplot(
                values,
                positions=positions,
                widths=width * 0.82,
                whis=(10, 90),
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            color = palette[method_index % len(palette)]
            for box in boxes["boxes"]:
                box.set(facecolor=color, edgecolor=color, alpha=0.35)
            for median in boxes["medians"]:
                median.set(color=color, linewidth=2.0)
            for part in ("whiskers", "caps"):
                for artist in boxes[part]:
                    artist.set(color=color, linewidth=1.0)
            ax.plot([], [], color=color, linewidth=7, alpha=0.35, label=method)
        ax.set_title(f"D2 to {target_time}")
        ax.set_xticks(range(len(bin_order)), bin_order)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Clone-conditioned Wasserstein-1")
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        if axis_index == 0:
            ax.legend(frameon=False, ncol=len(method_order), loc="upper right")
    fig.suptitle("LARRY clonal Wasserstein distributions by clone size")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=_checkpoint_spec,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--target-times", default="D4,D6")
    parser.add_argument(
        "--noise-draws",
        type=int,
        default=None,
        help="Override each checkpoint's configured clone noise-draw count.",
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=None,
        help="Override each checkpoint's generated samples per D2 source cell.",
    )
    parser.add_argument(
        "--out-dir", default="outputs/larry_clone_wasserstein_by_size"
    )
    args = parser.parse_args()
    target_times = tuple(
        larry.format_timepoint(value)
        for value in args.target_times.split(",")
        if value.strip()
    )
    if not target_times:
        parser.error("target-times cannot be empty")
    if args.noise_draws is not None and args.noise_draws <= 0:
        parser.error("noise-draws must be positive")
    if args.samples_per_source is not None and args.samples_per_source <= 0:
        parser.error("samples-per-source must be positive")

    rows: List[Dict[str, object]] = []
    method_order = []
    for label, checkpoint in args.checkpoint:
        if label not in method_order:
            method_order.append(label)
        rows.extend(
            _clone_records(
                label,
                checkpoint,
                target_times,
                noise_draws=args.noise_draws,
                samples_per_source=args.samples_per_source,
            )
        )

    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_clone_wasserstein.csv", rows)
    _write_csv(output_dir / "summary_by_clone_size.csv", _summaries(rows))
    for measure in ("source_size", "target_size", "total_size"):
        _plot(
            rows,
            output_dir / f"clonal_w1_by_{measure}.png",
            size_measure=measure,
            method_order=method_order,
        )
    print(f"Saved clone-size analysis to {output_dir}")


if __name__ == "__main__":
    main()

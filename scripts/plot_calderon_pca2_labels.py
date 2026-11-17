#!/usr/bin/env python3
"""Plot the all-times Calderon PCA by released branch and tissue labels."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/calderon2022/matplotlib"))

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_to_lab(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    xyz = linear @ np.asarray([[.4124564, .3575761, .1804375],
                              [.2126729, .7151522, .0721750],
                              [.0193339, .1191920, .9503041]]).T
    xyz /= np.asarray([.95047, 1., 1.08883])
    delta = 6 / 29
    transformed = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.column_stack([116 * transformed[:, 1] - 16,
                            500 * (transformed[:, 0] - transformed[:, 1]),
                            200 * (transformed[:, 1] - transformed[:, 2])])


def categorical_palette(n):
    """Choose visible categorical colours greedily by CIELAB separation."""
    rgb = np.asarray([colorsys.hsv_to_rgb(hue, saturation, value)
                      for hue in np.linspace(0, 1, 96, endpoint=False)
                      for saturation in (.55, .75, .95)
                      for value in (.65, .82, .97)])
    lab = rgb_to_lab(rgb)
    keep = (lab[:, 0] >= 38) & (lab[:, 0] <= 76) & (np.linalg.norm(lab[:, 1:], axis=1) >= 32)
    rgb, lab = rgb[keep], lab[keep]
    seed = rgb_to_lab(np.asarray([to_rgb("#246BA6")]))[0]
    chosen = [int(np.argmin(np.sum((lab - seed) ** 2, axis=1)))]
    distances = np.full(len(rgb), np.inf)
    while len(chosen) < n:
        distances = np.minimum(distances, np.sum((lab - lab[chosen[-1]]) ** 2, axis=1))
        distances[chosen] = -np.inf
        chosen.append(int(np.argmax(distances)))
    return rgb[chosen]


def load_data(path):
    with h5py.File(path, "r") as source:
        xy = source["obsm/X_pca"][:, :2]
        variance = source["uns/pca/variance_ratio"][:2]
        labels = {}
        for target in ("branch_id", "broad_tissue"):
            group = source[f"obs/{target}"]
            names = group["categories"].asstr()[:]
            codes = group["codes"][:]
            if len(codes) != len(xy) or (codes < 0).any() or (codes >= len(names)).any():
                raise ValueError(f"Missing or invalid {target} labels")
            labels[target] = (names, codes)
        for key in ("hvg_fit", "pca_fit"):
            if source[f"uns/preprocessing/{key}"].asstr()[()] != "all_cells":
                raise ValueError("This plot expects the all-cell fitted representation")
        if int(source["uns/pca/n_fit_cells"][()]) != len(xy):
            raise ValueError("PCA was not fitted on every plotted cell")
        windows = source["obs/day/categories"].asstr()[:].tolist()
    if not np.isfinite(xy).all() or not np.isfinite(variance).all():
        raise ValueError("Nonfinite PCA coordinates or variance ratios")
    return xy, variance, labels, windows


def draw(xy, variance, names, codes, order, limits, palette, target, args):
    branch = target == "branch_id"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#72777E", "axes.labelcolor": "#24292F",
                         "xtick.color": "#50565D", "ytick.color": "#50565D"})
    fig = plt.figure(figsize=(12.6, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1, .70), left=.065, right=.985,
                           bottom=.12, top=.85, wspace=.04)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis("off")
    ax.scatter(xy[order, 0], xy[order, 1], c=palette[codes[order]], s=args.point_size,
               alpha=args.alpha, linewidths=0, rasterized=True)
    ax.set(xlabel=f"PC1 ({100 * variance[0]:.2f}% variance)",
           ylabel=f"PC2 ({100 * variance[1]:.2f}% variance)",
           xlim=limits[0], ylim=limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#E3E6E9", linewidth=.55)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                      markersize=6, label=name) for name, color in zip(names, palette)]
    legend_ax.legend(handles=handles, loc="center left", frameon=False,
                     ncol=3 if branch else 2, fontsize=9 if branch else 10,
                     title=f"{'Branch ID' if branch else 'Broad tissue'} ({len(names)})",
                     title_fontsize=12, columnspacing=.9, handletextpad=.35,
                     labelspacing=.65, borderaxespad=0)
    fig.suptitle(f"Calderon atlas · {'released branch IDs' if branch else 'authors’ broad tissue labels'}",
                 x=.065, y=.962, ha="left", fontsize=19, fontweight="medium", color="#22272E")
    fig.text(.065, .909,
             f"All {len(xy):,} lineage-assigned cells · 0–18 h · HVG3000/PCA100 fitted across all times",
             fontsize=11, color="#50565D")
    fig.text(.065, .047,
             f"PC1 + PC2 explain {100 * variance.sum():.2f}% of scaled HVG variance. "
             "Same cells, axes and shuffled draw order in both plots; no subsampling.",
             fontsize=9.5, color="#60666D")
    base = args.output_dir / f"all_times_pca2_{target}"
    outputs = []
    for extension in ("png", "svg"):
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/calderon2022/all_times/lineage_assigned_hvg3000_pca100_all_times.h5ad")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/calderon2022/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--point-size", type=float, default=.7)
    parser.add_argument("--alpha", type=float, default=.55)
    args = parser.parse_args()
    args.data, args.output_dir = args.data.resolve(), args.output_dir.resolve()
    if args.point_size <= 0 or not 0 < args.alpha <= 1:
        parser.error("Point size must be positive and alpha must lie in (0, 1]")
    initial_hash = sha256(args.data)
    xy, variance, labels, windows = load_data(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(args.seed).permutation(len(xy))
    bounds = np.stack([xy.min(axis=0), xy.max(axis=0)])
    margin = .025 * (bounds[1] - bounds[0])
    limits = [(float(bounds[0, dim] - margin[dim]), float(bounds[1, dim] + margin[dim])) for dim in (0, 1)]
    metadata = {"source_h5ad": str(args.data), "source_h5ad_sha256": initial_hash,
                "n_plotted_cells": len(xy), "subsampled": False,
                "coordinates": "obsm/X_pca[:, :2]", "fit_population": "all_lineage_assigned_cells",
                "variance_ratio": variance.tolist(), "time_windows": windows,
                "draw_order_seed": args.seed, "same_draw_order_for_both_labels": True,
                "axis_limits": limits, "point_size": args.point_size,
                "alpha": args.alpha, "targets": {}}
    for target, (names, codes) in labels.items():
        palette = categorical_palette(len(names) - int("Unknown" in names))
        if "Unknown" in names:
            palette = np.insert(palette, int(np.flatnonzero(names == "Unknown")[0]), to_rgb("#73777D"), axis=0)
        counts = np.bincount(codes, minlength=len(names))
        if not np.all(counts > 0) or len(set(map(to_hex, palette))) != len(names):
            raise ValueError("Expected a unique colour and plotted cells for every class")
        outputs = draw(xy, variance, names, codes, order, limits, palette, target, args)
        metadata["targets"][target] = {"n_classes": len(names), "outputs": outputs,
                                      "classes": [{"label": name, "n_cells": int(count), "color": to_hex(color)}
                                                  for name, count, color in zip(names, counts, palette)]}
        print(f"Plotted {target}: {len(xy):,} cells, {len(names)} classes", flush=True)
    if sha256(args.data) != initial_hash:
        raise ValueError("Source dataset changed during plotting")
    metadata["source_unchanged"] = True
    path = args.output_dir / "all_times_pca2_labels.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved figures and provenance: {path}", flush=True)


if __name__ == "__main__":
    main()

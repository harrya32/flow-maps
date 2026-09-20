#!/usr/bin/env python3
"""Plot LARRY SPRING2D classifier regions with annotated cells overlaid.

The left panel shows the classifier trained only on D2 and D6 and used for
biological-prior filtering/constraints.  The right panel shows the all-days
classifier reserved for evaluation.  Both panels overlay the same cells and
use a common colour assignment for true and predicted cell types.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import larry  # noqa: E402


CLASS_COLOURS: Dict[str, str] = {
    "Baso": "#8c564b",
    "Ccr7_DC": "#17becf",
    "Eos": "#bcbd22",
    "Erythroid": "#d62728",
    "Lymphoid": "#9467bd",
    "Mast": "#e377c2",
    "Meg": "#ff7f0e",
    "Monocyte": "#1f77b4",
    "Neutrophil": "#2ca02c",
    "Undifferentiated": "#7f7f7f",
    "pDC": "#aec7e8",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compare the D2/D6 and all-days LARRY classifiers in SPRING2D.")
    )
    parser.add_argument(
        "--dataset-location",
        type=Path,
        default=larry.DEFAULT_DATA_DIR,
        help=(
            "Directory containing stateFate_inVitro_spring2d.h5ad, or the "
            "explicit H5AD path."
        ),
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        default=larry.DEFAULT_CLASSIFIER_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "larry-classifiers"
            / "reports"
            / "larry_spring2d_classifier_decision_boundaries.png"
        ),
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=700,
        help="Number of grid points along the longer axis.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Maximum cells to overlay; zero plots all cells.",
    )
    parser.add_argument("--padding", type=float, default=0.035)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=240)
    args = parser.parse_args(argv)
    if args.grid_size < 50:
        parser.error("--grid-size must be at least 50")
    if args.max_points < 0:
        parser.error("--max-points must be non-negative")
    if args.padding < 0:
        parser.error("--padding must be non-negative")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def classifier_paths(classifier_dir: Path) -> Tuple[Path, Path]:
    training = larry.classifier_checkpoint_path(
        training_timepoints=("D2", "D6"),
        classifier_dir=classifier_dir,
        representation=larry.SPRING2D_REPRESENTATION,
    )
    evaluation = larry.classifier_checkpoint_path(
        all_days=True,
        classifier_dir=classifier_dir,
        representation=larry.SPRING2D_REPRESENTATION,
    )
    for path in (training, evaluation):
        npz_path = path.with_suffix(".npz")
        if not path.is_file() and not npz_path.is_file():
            raise FileNotFoundError(f"Missing LARRY classifier checkpoint: {path}")
    return training, evaluation


def make_grid(
    x: np.ndarray,
    *,
    grid_size: int,
    padding: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.min(x, axis=0)
    upper = np.max(x, axis=0)
    extent = upper - lower
    lower = lower - padding * extent
    upper = upper + padding * extent
    aspect = float(extent[1] / max(extent[0], np.finfo(np.float32).eps))
    nx = int(grid_size)
    ny = max(50, int(round(grid_size * aspect)))
    grid_x = np.linspace(lower[0], upper[0], nx, dtype=np.float32)
    grid_y = np.linspace(lower[1], upper[1], ny, dtype=np.float32)
    xx, yy = np.meshgrid(grid_x, grid_y)
    points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    return xx, yy, points


def predict_grid(
    classifier_path: Path,
    points: np.ndarray,
    grid_shape: Tuple[int, int],
) -> np.ndarray:
    model, class_names, scaler_mean, scaler_scale = larry.load_classifier(
        classifier_path
    )
    if set(class_names) != set(larry.CLASS_NAMES):
        raise ValueError(
            f"{classifier_path.name} classes do not match the LARRY annotations."
        )
    predicted, _, _ = larry.classifier_predictions(
        model,
        class_names,
        scaler_mean,
        scaler_scale,
        points,
        batch_size=65_536,
    )
    class_to_id = {name: index for index, name in enumerate(larry.CLASS_NAMES)}
    return np.asarray(
        [class_to_id[str(value)] for value in predicted], dtype=np.int16
    ).reshape(grid_shape)


def sampled_overlay(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    maximum: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if maximum <= 0 or maximum >= x.shape[0]:
        return x, labels
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(x.shape[0], size=maximum, replace=False))
    return x[indices], labels[indices]


def draw_panel(
    axis,
    xx: np.ndarray,
    yy: np.ndarray,
    regions: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    cmap: ListedColormap,
    norm: BoundaryNorm,
) -> None:
    axis.pcolormesh(
        xx,
        yy,
        regions,
        cmap=cmap,
        norm=norm,
        shading="nearest",
        alpha=0.23,
        rasterized=True,
        zorder=0,
    )

    # Plot abundant populations first so rarer annotated types remain visible.
    counts = {name: int(np.sum(labels == name)) for name in larry.CLASS_NAMES}
    for name in sorted(larry.CLASS_NAMES, key=counts.get, reverse=True):
        mask = labels == name
        axis.scatter(
            x[mask, 0],
            x[mask, 1],
            s=1.1,
            c=CLASS_COLOURS[name],
            alpha=0.42,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    # One-vs-rest contours avoid treating integer class IDs as ordered values.
    for class_id in np.unique(regions):
        axis.contour(
            xx,
            yy,
            (regions == class_id).astype(np.float32),
            levels=(0.5,),
            colors=("#252525",),
            linewidths=0.48,
            alpha=0.8,
            zorder=2,
        )

    axis.set_title(title, fontsize=12, pad=9)
    axis.set_xlabel("SPRING 1 (scaled)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(float(xx.min()), float(xx.max()))
    axis.set_ylim(float(yy.min()), float(yy.max()))
    axis.spines[["top", "right"]].set_visible(False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data = larry.all_timepoint_data(
        args.dataset_location,
        representation=larry.SPRING2D_REPRESENTATION,
    )
    x = np.asarray(data["x"], dtype=np.float32)
    labels = np.asarray(data["cell_types"], dtype=object)
    x_overlay, labels_overlay = sampled_overlay(
        x,
        labels,
        maximum=int(args.max_points),
        seed=int(args.seed),
    )

    training_path, evaluation_path = classifier_paths(args.classifier_dir)
    xx, yy, grid_points = make_grid(
        x,
        grid_size=int(args.grid_size),
        padding=float(args.padding),
    )
    training_regions = predict_grid(training_path, grid_points, xx.shape)
    evaluation_regions = predict_grid(evaluation_path, grid_points, xx.shape)

    colours = [CLASS_COLOURS[name] for name in larry.CLASS_NAMES]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(
        np.arange(-0.5, len(larry.CLASS_NAMES) + 0.5),
        ncolors=len(larry.CLASS_NAMES),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 6.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    draw_panel(
        axes[0],
        xx,
        yy,
        training_regions,
        x_overlay,
        labels_overlay,
        title="Training prior classifier (D2 + D6)",
        cmap=cmap,
        norm=norm,
    )
    draw_panel(
        axes[1],
        xx,
        yy,
        evaluation_regions,
        x_overlay,
        labels_overlay,
        title="Evaluation classifier (D2 + D4 + D6)",
        cmap=cmap,
        norm=norm,
    )
    axes[0].set_ylabel("SPRING 2 (scaled)")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=CLASS_COLOURS[name],
            markeredgewidth=0,
            label=name,
        )
        for name in larry.CLASS_NAMES
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=9,
        columnspacing=1.4,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(
        "LARRY SPRING2D cell-type classifier decision regions",
        fontsize=14,
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Background: classifier prediction; points: annotated cell type "
            f"({x_overlay.shape[0]:,} cells)"
        ),
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.89, bottom=0.16, wspace=0.08)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(args.dpi), bbox_inches="tight", facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    figure.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved classifier-boundary plot to {output}")
    print(f"Saved vector copy to {pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

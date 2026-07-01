"""
Plot 2D box-avoiding quadratic Bezier interpolants.
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow_maps_mpl_cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "flow_maps_xdg_cache")
)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np


SOURCE_MEAN = np.array([-3.0, 0.0], dtype=np.float32)
TARGET_MEAN = np.array([3.0, 0.0], dtype=np.float32)
BOX_XLIM = (-1.5, 1.5)
BOX_YLIM = (-1.0, 1.0)


def sample_pairs(
    n_samples: int,
    *,
    std: float,
    height: float,
    reject: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0_chunks = []
    x1_chunks = []
    control_chunks = []
    sign_chunks = []
    total = 0
    reject_times = np.linspace(0.0, 1.0, 41, dtype=np.float32)

    while total < n_samples:
        remaining = n_samples - total
        n_draw = remaining if not reject else min(max(2 * remaining, 4096), 65_536)
        x0 = SOURCE_MEAN + rng.normal(scale=std, size=(n_draw, 2))
        x1 = TARGET_MEAN + rng.normal(scale=std, size=(n_draw, 2))
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_draw)
        controls = 0.5 * (x0 + x1)
        controls[:, 1] += height * signs

        if reject:
            paths = bezier_paths(reject_times, x0, x1, controls)
            keep = ~np.any(in_forbidden_box(paths), axis=0)
        else:
            keep = np.ones((n_draw,), dtype=bool)

        if not np.any(keep):
            continue

        take = min(remaining, int(np.sum(keep)))
        x0_chunks.append(x0[keep][:take])
        x1_chunks.append(x1[keep][:take])
        control_chunks.append(controls[keep][:take])
        sign_chunks.append(signs[keep][:take])
        total += take

    return (
        np.concatenate(x0_chunks, axis=0),
        np.concatenate(x1_chunks, axis=0),
        np.concatenate(control_chunks, axis=0),
        np.concatenate(sign_chunks, axis=0),
    )


def bezier_paths(
    times: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    t = times[:, None, None]
    return (1.0 - t) ** 2 * x0[None, :, :] + 2.0 * t * (
        1.0 - t
    ) * controls[None, :, :] + t**2 * x1[None, :, :]


def in_forbidden_box(points: np.ndarray) -> np.ndarray:
    return (
        (points[..., 0] >= BOX_XLIM[0])
        & (points[..., 0] <= BOX_XLIM[1])
        & (points[..., 1] >= BOX_YLIM[0])
        & (points[..., 1] <= BOX_YLIM[1])
    )


def draw_forbidden_box(ax) -> None:
    ax.add_patch(
        Rectangle(
            (BOX_XLIM[0], BOX_YLIM[0]),
            BOX_XLIM[1] - BOX_XLIM[0],
            BOX_YLIM[1] - BOX_YLIM[0],
            facecolor="none",
            edgecolor="crimson",
            linestyle="--",
            linewidth=1.6,
            zorder=10,
            label="infeasible box",
        )
    )


def plot_process(
    times: np.ndarray,
    paths: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    controls: np.ndarray,
    output_path: Path,
) -> None:
    del controls
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(times)))

    for idx, (time, color) in enumerate(zip(times, colors)):
        ax.scatter(
            paths[idx, :, 0],
            paths[idx, :, 1],
            s=5,
            alpha=0.22,
            color=color,
            linewidths=0,
            label=f"t={time:.2f}",
        )

    n_lines = min(180, x0.shape[0])
    line_paths = np.swapaxes(paths[:, :n_lines, :], 0, 1)
    segments = np.stack([line_paths[:, :-1, :], line_paths[:, 1:, :]], axis=2)
    ax.add_collection(
        LineCollection(
            segments.reshape((-1, 2, 2)),
            colors="black",
            linewidths=0.45,
            alpha=0.13,
        )
    )

    ax.scatter(x0[:, 0], x0[:, 1], s=5, alpha=0.22, color="black", label="p0")
    ax.scatter(x1[:, 0], x1[:, 1], s=5, alpha=0.22, color="C0", label="p1")
    draw_forbidden_box(ax)
    ax.set_title("Box-avoiding quadratic Bezier interpolants")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        markerscale=2.0,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and plot box-avoiding Bezier interpolants."
    )
    parser.add_argument("--n_samples", type=int, default=1600)
    parser.add_argument("--std", type=float, default=0.25)
    parser.add_argument("--height", type=float, default=4.0)
    parser.add_argument(
        "--reject",
        action="store_true",
        help="Enable rejection filtering for paths that touch the box.",
    )
    parser.add_argument(
        "--no_reject",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        help="Times in [0, 1] to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs/box_avoiding_bezier_process.png"),
        help="Path to write the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    times = np.asarray(args.times, dtype=np.float32)
    x0, x1, controls, signs = sample_pairs(
        args.n_samples,
        std=args.std,
        height=args.height,
        reject=args.reject and not args.no_reject,
        rng=rng,
    )
    paths = bezier_paths(times, x0, x1, controls)
    plot_process(times, paths, x0, x1, controls, args.output)

    path_violation_rate = np.mean(np.any(in_forbidden_box(paths), axis=0))
    central_times = np.linspace(0.25, 0.75, 11, dtype=np.float32)
    central_paths = bezier_paths(central_times, x0, x1, controls)
    central_path_rate = np.mean(np.any(in_forbidden_box(central_paths), axis=0))
    midpoint = bezier_paths(np.array([0.5], dtype=np.float32), x0, x1, controls)[0]
    midpoint_rate = np.mean(in_forbidden_box(midpoint))
    print(f"Wrote {args.output}")
    print(f"upper_branch_fraction={np.mean(signs > 0):.3f}")
    print(f"midpoint_forbidden_rate={midpoint_rate:.6f}")
    print(f"path_forbidden_rate_on_plotted_times={path_violation_rate:.6f}")
    print(f"central_path_forbidden_rate={central_path_rate:.6f}")


if __name__ == "__main__":
    main()

"""
Plot the one-source/two-target Gaussian fork process and forbidden box.
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
from matplotlib.patches import Rectangle
import numpy as np


P0 = np.array([0.0, 0.0], dtype=np.float32)
TARGET_MEANS = np.array([[-2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
BOX_XLIM = (-0.3, 0.3)
BOX_YLIM = (0.7, 1.3)


def sample_pairs(
    n_samples: int,
    *,
    std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = rng.integers(0, 2, size=n_samples)
    x0 = P0 + rng.normal(scale=std, size=(n_samples, 2))
    x1 = TARGET_MEANS[labels] + rng.normal(scale=std, size=(n_samples, 2))
    return x0, x1, labels


def interpolate(times: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    t = times[:, None, None]
    return (1.0 - t) * x0[None, :, :] + t * x1[None, :, :]


def in_forbidden_box(points: np.ndarray) -> np.ndarray:
    return (
        (points[:, 0] >= BOX_XLIM[0])
        & (points[:, 0] <= BOX_XLIM[1])
        & (points[:, 1] >= BOX_YLIM[0])
        & (points[:, 1] <= BOX_YLIM[1])
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
            linewidth=1.5,
            zorder=10,
            label="forbidden B",
        )
    )


def plot_process(
    times: np.ndarray,
    paths: np.ndarray,
    x0: np.ndarray,
    x1: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(times)))

    for idx, (time, color) in enumerate(zip(times, colors)):
        ax.scatter(
            paths[idx, :, 0],
            paths[idx, :, 1],
            s=6,
            alpha=0.25,
            color=color,
            linewidths=0,
            label=f"t={time:.2f}",
        )

    n_lines = min(80, x0.shape[0])
    for sample_idx in range(n_lines):
        ax.plot(
            paths[:, sample_idx, 0],
            paths[:, sample_idx, 1],
            color="black",
            alpha=0.10,
            linewidth=0.7,
        )

    ax.scatter(x0[:, 0], x0[:, 1], s=5, alpha=0.20, color="black")
    ax.scatter(x1[:, 0], x1[:, 1], s=5, alpha=0.20, color="C0")
    draw_forbidden_box(ax)
    ax.set_title("Fork Gaussian linear interpolants")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", frameon=False, markerscale=2.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and plot the fork Gaussian synthetic process."
    )
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--std", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Times in [0, 1] to visualize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs/fork_gaussian_process.png"),
        help="Path to write the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    times = np.asarray(args.times, dtype=np.float32)
    x0, x1, labels = sample_pairs(args.n_samples, std=args.std, rng=rng)
    paths = interpolate(times, x0, x1)
    plot_process(times, paths, x0, x1, args.output)

    midpoints = 0.5 * x0 + 0.5 * x1
    midpoint_rate = np.mean(in_forbidden_box(midpoints))
    left_frac = np.mean(labels == 0)
    print(f"Wrote {args.output}")
    print(f"left_target_fraction={left_frac:.3f}")
    print(f"interpolant_midpoint_forbidden_rate={midpoint_rate:.6f}")


if __name__ == "__main__":
    main()

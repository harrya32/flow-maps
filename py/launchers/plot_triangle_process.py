"""
Plot a synthetic triangular Gaussian temporal process.

The process independently couples x0 ~ N((0, 0), sigma^2 I) and
x1 ~ N((3, 0), sigma^2 I), then uses the triangular interpolant
x_t = (1 - t) x0 + t x1 + (0, 6 min(t, 1 - t)). Each p_t is Gaussian,
with a covariance that contracts toward t=0.5 and expands again.
"""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "flow_maps_mpl_cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "flow_maps_xdg_cache")
)

import matplotlib.pyplot as plt
import numpy as np


P0 = np.array([0.0, 0.0])
PMID = np.array([1.5, 3.0])
P1 = np.array([3.0, 0.0])


def triangle_mean(t: np.ndarray) -> np.ndarray:
    """Return the piecewise-linear mean at times in [0, 1]."""
    t = np.asarray(t, dtype=np.float32)
    if np.any((t < 0.0) | (t > 1.0)):
        raise ValueError("All times must be in [0, 1].")

    t_col = t[..., None]
    first_half = t_col <= 0.5
    before_mid = P0 + (t_col / 0.5) * (PMID - P0)
    after_mid = PMID + ((t_col - 0.5) / 0.5) * (P1 - PMID)
    return np.where(first_half, before_mid, after_mid)


def triangle_velocity(t: np.ndarray) -> np.ndarray:
    """Return the piecewise-constant velocity away from the kink at t=0.5."""
    t = np.asarray(t, dtype=np.float32)
    if np.any((t < 0.0) | (t > 1.0)):
        raise ValueError("All times must be in [0, 1].")

    first_velocity = 2.0 * (PMID - P0)
    second_velocity = 2.0 * (P1 - PMID)
    return np.where((t[..., None] <= 0.5), first_velocity, second_velocity)


def triangle_bump(t: np.ndarray) -> np.ndarray:
    """Return the deterministic vertical bump added to endpoint interpolation."""
    t = np.asarray(t, dtype=np.float32)
    bump = np.zeros((*t.shape, 2), dtype=np.float32)
    bump[..., 1] = 6.0 * np.minimum(t, 1.0 - t)
    return bump


def sample_marginals(
    times: Iterable[float],
    *,
    n_samples: int,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent Gaussian clouds from p_t for each requested time."""
    times = np.asarray(list(times), dtype=np.float32)
    means = triangle_mean(times)
    marginal_stds = std * np.sqrt((1.0 - times) ** 2 + times**2)
    noise = rng.normal(
        scale=marginal_stds[:, None, None],
        size=(times.shape[0], n_samples, 2),
    )
    return means[:, None, :] + noise


def sample_trajectories(
    times: Iterable[float],
    *,
    n_samples: int,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample trajectories from independent endpoint couplings."""
    times = np.asarray(list(times), dtype=np.float32)
    x0 = P0 + rng.normal(scale=std, size=(n_samples, 2))
    x1 = P1 + rng.normal(scale=std, size=(n_samples, 2))
    t_col = times[:, None, None]
    return (1.0 - t_col) * x0[None, :, :] + t_col * x1[None, :, :] + triangle_bump(
        times
    )[:, None, :]


def plot_process(
    times: np.ndarray,
    clouds: np.ndarray,
    trajectories: np.ndarray,
    output_path: Path,
) -> None:
    """Create a scatter plot of marginals plus a few coupled trajectories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.05, 0.95, len(times)))

    for idx, (time, color) in enumerate(zip(times, colors)):
        ax.scatter(
            clouds[idx, :, 0],
            clouds[idx, :, 1],
            s=6,
            alpha=0.28,
            color=color,
            linewidths=0,
            label=f"t={time:.2f}",
        )

    center_times = np.linspace(0.0, 1.0, 201, dtype=np.float32)
    centerline = triangle_mean(center_times)
    ax.plot(centerline[:, 0], centerline[:, 1], color="black", linewidth=2.0)
    ax.scatter(
        [P0[0], PMID[0], P1[0]],
        [P0[1], PMID[1], P1[1]],
        color="black",
        s=36,
        zorder=4,
    )

    n_lines = min(16, trajectories.shape[1])
    for sample_idx in range(n_lines):
        ax.plot(
            trajectories[:, sample_idx, 0],
            trajectories[:, sample_idx, 1],
            color="black",
            alpha=0.18,
            linewidth=0.8,
        )

    ax.set_title("Triangular Gaussian temporal process")
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
        description="Sample and plot a triangular Gaussian temporal process."
    )
    parser.add_argument("--n_samples", type=int, default=600)
    parser.add_argument("--std", type=float, default=0.18)
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
        default=Path("figs/triangle_temporal_process.png"),
        help="Path to write the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    times = np.asarray(args.times, dtype=np.float32)

    clouds = sample_marginals(
        times,
        n_samples=args.n_samples,
        std=args.std,
        rng=rng,
    )
    trajectories = sample_trajectories(
        times,
        n_samples=args.n_samples,
        std=args.std,
        rng=rng,
    )
    plot_process(times, clouds, trajectories, args.output)

    print(f"Wrote {args.output}")
    for time, mean, velocity in zip(times, triangle_mean(times), triangle_velocity(times)):
        marginal_std = args.std * np.sqrt((1.0 - time) ** 2 + time**2)
        print(
            f"t={time:.2f}: mean=({mean[0]:.3f}, {mean[1]:.3f}), "
            f"E[velocity]=({velocity[0]:.3f}, {velocity[1]:.3f}), "
            f"marginal_std={marginal_std:.3f}"
        )


if __name__ == "__main__":
    main()

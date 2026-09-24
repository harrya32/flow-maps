"""Recreate the paper-style conditional lineage-violation illustration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from .dataset import CLASS_NAMES, hard_labels, simulate
from .models import load_fields, load_map, sample_map, sample_sde
from .training import select_device


def _allowed_transitions() -> np.ndarray:
    descendants = {
        "root": set(CLASS_NAMES),
        "lower": {"lower"},
        "upper": {"upper", "upper_down", "upper_up"},
        "upper_down": {"upper_down"},
        "upper_up": {"upper_up"},
    }
    allowed = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=bool)
    for source, source_name in enumerate(CLASS_NAMES):
        for target_name in descendants[source_name]:
            allowed[source, CLASS_NAMES.index(target_name)] = True
    return allowed


def _violation_mask(paths: np.ndarray, dataset: dict) -> np.ndarray:
    labels = hard_labels(paths, dataset)
    allowed = _allowed_transitions()
    return (~allowed[labels[:, :-1], labels[:, 1:]]).any(axis=1)


def _observation_times(start: float, end: float, spacing: float) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("time-spacing must be positive.")
    indices = np.arange(
        np.floor(start / spacing + 1e-8) + 1,
        round(end / spacing) + 1,
    )
    return np.unique(np.round(np.r_[start, indices * spacing, end], 10))


def _load_config(root: Path) -> dict:
    path = root / "resolved_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared experiment configuration: {path}")
    return json.loads(path.read_text())


def _compatible_sf2m_root(root: Path, supplied: Path | None, config: dict, seed: int, prior: str) -> Path:
    candidates = []
    if supplied is not None:
        candidates.append(supplied.resolve())
    else:
        candidates.extend([root, root.parent / "synthetic_curved_v5_ground_truth"])
        candidates.extend(sorted(root.parent.glob("synthetic_curved_v5*")))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        checkpoint = candidate / "models" / f"seed_{seed}" / prior / "sf2m" / "model.pt"
        if not checkpoint.exists():
            continue
        other = _load_config(candidate)
        shared = ("dataset", "learning_split", "network", "priors", "sf2m")
        if all(other.get(key) == config.get(key) for key in shared):
            return candidate
        if supplied is not None:
            raise ValueError(
                f"The supplied SF2M root {candidate} does not share the dataset/model configuration."
            )
    raise FileNotFoundError(
        "No compatible SF2M seed checkpoint was found. Supply one with --sf2m-root."
    )


def _conditioning_state(root: Path, config: dict, start: float, target: np.ndarray) -> tuple[int, np.ndarray]:
    starts = np.asarray(config["evaluation"]["start_times"], dtype=np.float64)
    matches = np.flatnonzero(np.isclose(starts, start))
    if len(matches) != 1:
        raise ValueError(
            f"start time {start:g} is not uniquely present in evaluation.start_times: {starts.tolist()}"
        )
    condition_index = int(matches[0])
    reference = root / "evaluation" / "reference" / f"conditional_{condition_index}.npz"
    if not reference.exists():
        raise FileNotFoundError(f"Missing conditional reference bank: {reference}")
    with np.load(reference, allow_pickle=False) as saved:
        states = saved["states"].copy()
    state_index = int(np.square(states - target).sum(axis=1).argmin())
    return state_index, states[state_index]


def _sample_paths(args, root: Path, sf2m_root: Path, config: dict, state: np.ndarray, device):
    count = int(args.n_paths)
    if count <= 0:
        raise ValueError("n-paths must be positive.")
    dataset = config["dataset"]
    times = _observation_times(args.start_time, float(dataset["end_time"]), args.time_spacing)
    initial = np.repeat(state[None], count, axis=0)
    sf2m_checkpoint = sf2m_root / "models" / f"seed_{args.seed}" / args.prior / "sf2m" / "model.pt"
    real_clift_checkpoint = (
        root / "models" / f"seed_{args.seed}" / args.prior / "real_clift" / "model.pt"
    )
    if not real_clift_checkpoint.exists():
        raise FileNotFoundError(f"Missing real_clift checkpoint: {real_clift_checkpoint}")
    sf2m = load_fields(sf2m_checkpoint, device)
    real_clift = load_map(real_clift_checkpoint, device)
    paths = {
        "truth": simulate(
            initial,
            args.start_time,
            times,
            args.sampling_seed,
            dataset,
            steps_per_unit=args.steps_per_unit,
        ),
        "sf2m": sample_sde(
            sf2m,
            initial,
            args.start_time,
            times,
            args.steps_per_unit,
            args.sampling_seed,
        ),
        "real_clift": sample_map(
            real_clift,
            initial,
            args.start_time,
            times,
            args.steps_per_unit,
            args.sampling_seed,
        ),
    }
    if not all(np.isfinite(values).all() for values in paths.values()):
        raise FloatingPointError("A sampled trajectory contains a non-finite value.")
    return times, paths


def _plot(root: Path, config: dict, args, state: np.ndarray, times: np.ndarray, paths: dict) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str((root.parent / ".cache" / "matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    dataset = config["dataset"]
    palette = ["#527DB5", "#248E89", "#C78B24", "#8865B8", "#CE6173"]
    valid_color, invalid_color = "#248E89", "#C33E3E"
    with np.load(root / "dataset" / "training_data.npz", allow_pickle=False) as saved:
        observed = saved["points"].copy()
    rng = np.random.default_rng(int(args.background_seed))
    backdrop = np.concatenate(
        [
            points[rng.choice(len(points), min(int(args.background_per_time), len(points)), replace=False)]
            for points in observed[1:]
        ]
    )
    backdrop_labels = hard_labels(backdrop, dataset)
    invalid = {name: _violation_mask(values, dataset) for name, values in paths.items()}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": "#243448",
            "axes.titlecolor": "#243448",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.3, 5.5), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.014, right=0.988, top=0.85, bottom=0.12, wspace=0.065)
    xmin, xmax, ymin, ymax = 0.46, 2.56, -0.76, 1.16
    scale = float(dataset["coordinate_scale"])
    labels = dataset["labels"]
    first_x = float(labels["first_x_boundary"]) * scale
    second_x = float(labels["second_x_boundary"]) * scale
    lower_y = float(labels["lower_y_boundary"]) * scale
    children_y = float(labels["children_y_boundary"]) * scale
    boxes = [
        (0, xmin, ymin, first_x - xmin, ymax - ymin),
        (1, first_x, ymin, xmax - first_x, lower_y - ymin),
        (2, first_x, lower_y, second_x - first_x, ymax - lower_y),
        (3, second_x, lower_y, xmax - second_x, children_y - lower_y),
        (4, second_x, children_y, xmax - second_x, ymax - children_y),
    ]
    panels = (
        ("truth", "Ground truth"),
        ("sf2m", r"SF$^2$M"),
        ("real_clift", "Real CLIFT"),
    )
    for axis, (name, title) in zip(axes, panels):
        for label, x, y, width, height in boxes:
            axis.add_patch(
                Rectangle((x, y), width, height, facecolor=palette[label], alpha=0.045, edgecolor="none")
            )
        for label in (1, 2, 3, 4):
            points = backdrop[backdrop_labels == label]
            axis.scatter(*points.T, color=palette[label], s=3, alpha=0.16, edgecolors="none", zorder=1)
        boundary_style = dict(color="#7D8A9B", linewidth=0.7, linestyle=(0, (4, 4)), alpha=0.6)
        axis.plot([first_x, first_x], [ymin, ymax], **boundary_style)
        axis.plot([second_x, second_x], [lower_y, ymax], **boundary_style)
        axis.plot([first_x, xmax], [lower_y, lower_y], **boundary_style)
        axis.plot([second_x, xmax], [children_y, children_y], **boundary_style)
        axis.text(0.78, 1.07, "Upper parent", fontsize=9, color=palette[2])
        axis.text(1.70, 1.07, "Upper child A", fontsize=9, color=palette[4])
        axis.text(1.70, 0.47, "Upper child B", fontsize=9, color=palette[3])
        axis.text(0.78, -0.69, "Lower lineage", fontsize=9, color=palette[1])
        for mask, color, alpha, width in (
            (~invalid[name], valid_color, 0.40, 0.8),
            (invalid[name], invalid_color, 0.80, 1.15),
        ):
            axis.add_collection(
                LineCollection(paths[name][mask], colors=color, alpha=alpha, linewidths=width, zorder=4)
            )
            endpoints = paths[name][mask, -1]
            axis.scatter(*endpoints.T, s=9, color=color, alpha=0.8, edgecolors="none", zorder=6)
        axis.scatter(
            *state,
            s=115,
            marker="*",
            facecolor="#243448",
            edgecolor="white",
            linewidth=0.65,
            zorder=8,
        )
        axis.set_title(title, fontsize=17, pad=34)
        axis.text(
            0.5,
            1.035,
            f"{int(invalid[name].sum())}/{len(paths[name])} shown paths violate lineage",
            transform=axis.transAxes,
            horizontalalignment="center",
            fontsize=10,
            color="#546477",
        )
        axis.set(xlim=(xmin, xmax), ylim=(ymin, ymax), xticks=[], yticks=[])
        axis.set_aspect("equal")
        for spine in axis.spines.values():
            spine.set_visible(False)
    legend = [
        Line2D([0], [0], color=valid_color, linewidth=2, label="Lineage-consistent path"),
        Line2D([0], [0], color=invalid_color, linewidth=2, label="Path with a lineage violation"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#243448",
            markersize=11,
            label=rf"Common start: $t_0={args.start_time:.2f}$",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=2,
    )
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"conditional_trajectories.{extension}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor="white",
        )
    plt.close(figure)
    np.savez_compressed(
        output / "conditional_trajectories_paths.npz",
        times=times,
        state=state,
        truth=paths["truth"],
        sf2m=paths["sf2m"],
        real_clift=paths["real_clift"],
    )
    manifest = {
        "conditioning_time": float(args.start_time),
        "conditioning_state": state.tolist(),
        "n_paths": int(args.n_paths),
        "time_spacing": float(args.time_spacing),
        "steps_per_unit": int(args.steps_per_unit),
        "sampling_seed": int(args.sampling_seed),
        "violation_definition": "Any forbidden consecutive hard-label transition on the dense output grid.",
        "violations": {name: int(mask.sum()) for name, mask in invalid.items()},
    }
    (output / "conditional_trajectories_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared real_clift experiment root.")
    parser.add_argument("--sf2m-root", type=Path, help="Compatible experiment root containing the SF2M checkpoint.")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--prior", default="ground_truth_ot")
    parser.add_argument("--start-time", type=float, default=0.55)
    parser.add_argument("--state-x", type=float, default=0.585)
    parser.add_argument("--state-y", type=float, default=-0.17)
    parser.add_argument("--n-paths", type=int, default=48)
    parser.add_argument("--time-spacing", type=float, default=0.005)
    parser.add_argument("--steps-per-unit", type=int, default=4160)
    parser.add_argument("--sampling-seed", type=int, default=922801)
    parser.add_argument("--background-seed", type=int, default=922901)
    parser.add_argument("--background-per-time", type=int, default=800)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config = _load_config(root)
    sf2m_root = _compatible_sf2m_root(root, args.sf2m_root, config, args.seed, args.prior)
    state_index, state = _conditioning_state(
        root,
        config,
        args.start_time,
        np.asarray([args.state_x, args.state_y], dtype=np.float64),
    )
    torch.set_num_threads(1)
    device = select_device(args.device)
    times, paths = _sample_paths(args, root, sf2m_root, config, state, device)
    manifest = _plot(root, config, args, state, times, paths)
    print(
        f"Saved {root / 'evaluation' / 'conditional_trajectories.pdf'}\n"
        f"state_index={state_index}, state={state.tolist()}\n"
        f"violations={manifest['violations']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

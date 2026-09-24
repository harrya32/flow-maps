"""Run controlled synthetic-process variants for SF2M, SSFM, and real CLIFT.

The study changes only the data-generating process.  Both methods retain their
ordinary objectives and share the same snapshots, train/validation cells, OT
plans, network architecture, optimizer budget, and evaluation banks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config, validate_config
from .evaluation import evaluate_all
from .process_diagnostics import diagnose_config
from .run import _prepared_config, prepare
from .training import train_all


PROFILES = (
    "baseline",
    "close_only",
    "ot_ambiguous_moderate",
    "ot_ambiguous_strong",
    "committed_swap_moderate",
    "committed_swap_balanced",
    "committed_swap_strong",
    "early_committed_swap",
)


def _set_progress_endpoints(dataset, at_one, at_two):
    """Encode branch-specific x means at t=1 and t=2 as ramp coefficients."""
    scale = float(dataset["coordinate_scale"])
    base_one = (
        float(dataset["progress_mean"]["constant"])
        + float(dataset["progress_mean"]["linear"])
        + float(dataset["progress_mean"]["quadratic"])
    ) * scale
    base_two = (
        float(dataset["progress_mean"]["constant"])
        + 2.0 * float(dataset["progress_mean"]["linear"])
        + 4.0 * float(dataset["progress_mean"]["quadratic"])
    ) * scale
    coefficients = {}
    for branch, x_one, x_two in zip(
        ("lower", "upper_down", "upper_up"), at_one, at_two
    ):
        before = (float(x_one) - base_one) / scale
        total = (float(x_two) - base_two) / scale
        coefficients[branch] = {
            "x_pre": before,
            "x_post": total - before,
        }
    dataset["progress_coefficients"] = coefficients


def build_profile(profile: str, *, prototype: bool = True):
    """Return one frozen process-study configuration."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown process profile {profile!r}; choose from {PROFILES}.")
    config = load_config(DEFAULT_CONFIG)
    config["version"] = f"curved_v5_process_study_{profile}"
    config["priors"] = {"ground_truth_ot": {"kind": "ground_truth"}}
    # This is the ordinary classifier-to-classifier lineage loss used before
    # the rejected source-label experiment.  It remains fixed across profiles.
    config["real_clift"]["constraint_weight"] = 1.0

    if profile in {"close_only", "ot_ambiguous_moderate", "ot_ambiguous_strong"}:
        dataset = config["dataset"]
        # Compared with the paper process, commitment happens cleanly and the
        # fate clouds are closer.  The moderate/strong profiles then add only
        # branch-specific progress, creating an OT shortcut without obscuring
        # the true cell-state labels.
        dataset["mixture_weights"] = [0.5, 0.25, 0.25]
        dataset["ramps"] = {
            "a": [0.2, 0.55],
            "c": [1.05, 1.5],
            "h": [1.0, 1.8],
            "x_pre": [0.2, 0.8],
            "x_post": [1.0, 2.0],
        }
        dataset["phenotype_coefficients"] = {
            "lower": {"a": -0.9, "h": -1.35},
            "upper_down": {"a": 0.9},
            "upper_up": {"a": 0.9, "c": 2.35},
        }
        dataset["std_knots"] = [0.0, 0.2, 0.45, 0.8, 1.0, 1.3, 1.6, 2.0]
        dataset["std_values"] = [[0.2, 0.15]] * len(dataset["std_knots"])
        dataset["labels"] = {
            "first_x_boundary": 2.25,
            "second_x_boundary": 9.0,
            "lower_y_boundary": 0.0,
            "children_y_boundary": 2.0,
        }
        dataset["true_diffusion"] = {
            "kind": "ground_truth",
            "progress_diffusion": 0.04,
            "diffusion_floor": 0.002,
            "first_diffusion_peak": 0.1,
            "second_diffusion_peak": 0.1,
            "first_branch_time": 0.32,
            "second_branch_time": 1.25,
            "first_diffusion_width": 0.08,
            "second_diffusion_width": 0.08,
        }
        endpoints = {
            "close_only": ((1.0, 1.0, 1.0), (2.2, 2.2, 2.2)),
            "ot_ambiguous_moderate": ((0.85, 1.05, 1.25), (2.4, 2.05, 2.2)),
            "ot_ambiguous_strong": ((0.8, 1.1, 1.3), (2.45, 2.05, 2.2)),
        }
        _set_progress_endpoints(dataset, *endpoints[profile])

    if profile in {
        "committed_swap_moderate",
        "committed_swap_balanced",
        "committed_swap_strong",
        "early_committed_swap",
    }:
        dataset = config["dataset"]
        # The lower fate and both upper children are already spatially
        # identifiable before the second observed interval.  Their phenotype
        # coordinates stay separated, but the two child branches exchange
        # their relative progress ordering.  Consequently Euclidean OT takes
        # a lineage-forbidden sibling shortcut even though oracle paths do not.
        dataset["mixture_weights"] = [0.5, 0.25, 0.25]
        dataset["ramps"] = {
            "a": [0.08, 0.25],
            "c": [0.25, 0.45],
            "e": [1.0, 1.6],
            "h": [1.0, 1.8],
            "x_pre": [0.35, 0.95],
            "x_post": [1.0, 2.0],
        }
        dataset["phenotype_coefficients"] = {
            "lower": {"a": -1.1, "h": -1.15},
            "upper_down": {"a": 0.9},
            "upper_up": {"a": 0.9, "c": 1.8, "e": 0.5},
        }
        dataset["std_knots"] = [0.0, 0.15, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0]
        dataset["std_values"] = [[0.15, 0.12]] * len(dataset["std_knots"])
        dataset["labels"] = {
            "first_x_boundary": 2.25,
            "second_x_boundary": 6.2,
            "lower_y_boundary": 0.0,
            "children_y_boundary": 1.8,
        }
        dataset["true_diffusion"] = {
            "kind": "ground_truth",
            "progress_diffusion": 0.03,
            "diffusion_floor": 0.03,
            "first_diffusion_peak": 0.08,
            "second_diffusion_peak": 0.08,
            "first_branch_time": 0.17,
            "second_branch_time": 0.35,
            "first_diffusion_width": 0.06,
            "second_diffusion_width": 0.07,
        }
        endpoints = {
            "committed_swap_moderate": ((1.0, 1.3, 1.7), (2.0, 2.4, 2.15)),
            "committed_swap_balanced": ((1.0, 1.3, 1.7), (2.0, 2.44, 2.13)),
            "committed_swap_strong": ((1.0, 1.3, 1.7), (2.0, 2.5, 2.1)),
            "early_committed_swap": ((1.0, 1.3, 1.7), (2.0, 2.4, 2.15)),
        }
        _set_progress_endpoints(dataset, *endpoints[profile])
        if profile == "early_committed_swap":
            dataset["labels"]["first_x_boundary"] = 1.5
            dataset["std_values"] = [[0.1, 0.12]] * len(dataset["std_knots"])

    if prototype:
        evaluation = config["evaluation"]
        evaluation.update(
            {
                "states_per_time": 12,
                "candidate_rollouts": 256,
                "reference_rollouts": 1024,
                "reference_sw2_rollouts": 512,
                "reference_steps_per_unit": 500,
                "population_samples": 400,
                "population_replicates": 2,
                "marginal_candidate_seeds": evaluation["marginal_candidate_seeds"][:2],
                "marginal_map_steps_per_unit": 512,
            }
        )
        config["version"] += "_prototype"
    validate_config(config)
    return config


def _write_cross_profile_summary(parent, profiles, seeds, methods):
    rows = []
    for profile in profiles:
        root = parent / profile
        diagnostics_path = root / "process_diagnostics.json"
        diagnostics = (
            json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else None
        )
        endpoint_rates = {}
        if diagnostics is not None:
            endpoint_rates = {
                interval: values["endpoint_forbidden_rate"]
                for interval, values in diagnostics["intervals"].items()
            }
        for seed in seeds:
            by_method = {}
            for method in methods:
                path = (
                    root
                    / "evaluation"
                    / "per_fit"
                    / f"seed_{seed}"
                    / "ground_truth_ot"
                    / method
                    / "metrics.json"
                )
                if path.exists():
                    by_method[method] = json.loads(path.read_text())
            for method, result in by_method.items():
                row = {
                    "profile": profile,
                    "seed": seed,
                    "method": method,
                    **result["summary"],
                    **result["invalidity"],
                    "oracle_any_forbidden_path_rate": None
                    if diagnostics is None
                    else diagnostics["oracle_paths_observation_grid"]["any_forbidden_path_rate"],
                }
                for interval, rate in endpoint_rates.items():
                    row[f"ot_endpoint_forbidden_{interval}"] = rate
                rows.append(row)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path = parent / "process_study_summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "evaluate", "all"))
    parser.add_argument("--output-parent", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        default="committed_swap_moderate",
        help=f"Comma-separated subset of: {','.join(PROFILES)}",
    )
    parser.add_argument("--seeds", default="21")
    parser.add_argument(
        "--methods",
        default="sf2m,ssfm,real_clift",
        help="Comma-separated method subset (default: sf2m,ssfm,real_clift).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--full-evaluation",
        action="store_true",
        help="Use the paper Monte Carlo/integration budget instead of the screening budget.",
    )
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    profiles = tuple(value.strip() for value in arguments.profiles.split(",") if value.strip())
    unknown = set(profiles) - set(PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    seeds = [int(value.strip()) for value in arguments.seeds.split(",") if value.strip()]
    methods = tuple(
        value.strip() for value in arguments.methods.split(",") if value.strip()
    )
    parent = arguments.output_parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        root = parent / profile
        expected = build_profile(profile, prototype=not arguments.full_evaluation)
        if arguments.stage in ("prepare", "all") and not root.exists():
            prepare(root, expected)
        config = _prepared_config(root, None, [])
        if config != expected:
            raise ValueError(
                f"Prepared configuration at {root} does not match profile {profile!r}."
            )
        diagnostic_path = root / "process_diagnostics.json"
        if not diagnostic_path.exists():
            diagnostic_path.write_text(
                json.dumps(diagnose_config(config), indent=2, allow_nan=False) + "\n"
            )
        if arguments.stage in ("train", "all"):
            train_all(
                root,
                config,
                methods=methods,
                seeds=seeds,
                priors=("ground_truth_ot",),
                device=arguments.device,
            )
        if arguments.stage in ("evaluate", "all"):
            evaluate_all(
                root,
                config,
                methods=methods,
                seeds=seeds,
                priors=("ground_truth_ot",),
                device=arguments.device,
            )
    _write_cross_profile_summary(parent, profiles, seeds, methods)


if __name__ == "__main__":
    main()

"""Configuration loading, dotted overrides, and validation."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Union


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "paper.json"


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_override(config: dict[str, Any], expression: str) -> None:
    """Apply ``section.key=value`` to a nested configuration in place."""
    if "=" not in expression:
        raise ValueError(f"Override must have the form key=value, got {expression!r}.")
    dotted_key, raw_value = expression.split("=", 1)
    keys = [part for part in dotted_key.split(".") if part]
    if not keys:
        raise ValueError("Override key cannot be empty.")
    target: dict[str, Any] = config
    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            raise KeyError(f"Unknown configuration section {'.'.join(keys[:-1])!r}.")
        target = target[key]
    if keys[-1] not in target:
        raise KeyError(f"Unknown configuration key {dotted_key!r}.")
    target[keys[-1]] = _parse_value(raw_value)


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Load a JSON configuration and apply optional dotted overrides."""
    source = DEFAULT_CONFIG if path is None else Path(path)
    config = json.loads(source.read_text())
    for expression in overrides:
        apply_override(config, expression)
    validate_config(config)
    return copy.deepcopy(config)


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "version",
        "dataset",
        "classifier",
        "learning_split",
        "priors",
        "network",
        "sf2m",
        "strong_map",
        "real_clift",
        "evaluation",
        "seeds",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    dataset = config["dataset"]
    start, end, step = (
        float(dataset["start_time"]),
        float(dataset["end_time"]),
        float(dataset["snapshot_step"]),
    )
    if not start < end or step <= 0:
        raise ValueError("Dataset time interval and snapshot_step must be positive.")
    train_times = [float(value) for value in dataset["training_times"]]
    if (
        train_times != sorted(train_times)
        or any(right <= left for left, right in zip(train_times, train_times[1:]))
        or train_times[0] != start
        or train_times[-1] != end
    ):
        raise ValueError("training_times must be ordered and include the time endpoints.")
    if int(dataset["oracle_particles"]) <= 0 or int(dataset["cells_per_time"]) <= 0:
        raise ValueError("Dataset particle and cell counts must be positive.")
    if int(dataset["cells_per_time"]) > int(dataset["oracle_particles"]):
        raise ValueError("dataset.cells_per_time cannot exceed dataset.oracle_particles.")
    if int(dataset["steps_per_unit"]) <= 0 or float(dataset["coordinate_scale"]) <= 0:
        raise ValueError("Dataset integration rate and coordinate scale must be positive.")
    weights = [float(value) for value in dataset["mixture_weights"]]
    if len(weights) != 3 or any(value <= 0 for value in weights):
        raise ValueError("Exactly three positive mixture weights are required.")
    if abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("Mixture weights must sum to one.")
    if len(dataset["std_knots"]) != len(dataset["std_values"]):
        raise ValueError("std_knots and std_values must have equal length.")
    if any(len(row) != 2 for row in dataset["std_values"]):
        raise ValueError("Each standard-deviation entry must contain x and y values.")
    if any(right <= left for left, right in zip(dataset["std_knots"], dataset["std_knots"][1:])):
        raise ValueError("std_knots must be strictly increasing.")
    if float(dataset["std_knots"][0]) > start or float(dataset["std_knots"][-1]) < end:
        raise ValueError("The standard-deviation schedule must cover the full time interval.")
    if any(float(value) <= 0 for row in dataset["std_values"] for value in row):
        raise ValueError("All configured standard deviations must be positive.")
    count = int(round((end - start) / step))
    if not math.isclose(start + count * step, end, abs_tol=1e-9):
        raise ValueError("snapshot_step must divide the configured time interval.")
    grid = [start + index * step for index in range(count + 1)]
    if any(not any(math.isclose(value, point, abs_tol=1e-9) for point in grid) for value in train_times):
        raise ValueError("Every training time must lie on the snapshot grid.")
    if not any(not any(math.isclose(point, value, abs_tol=1e-9) for value in train_times) for point in grid):
        raise ValueError("At least one snapshot time must remain held out for evaluation.")
    ramp_names = set(dataset["ramps"])
    if any(len(bounds) != 2 or float(bounds[1]) <= float(bounds[0]) for bounds in dataset["ramps"].values()):
        raise ValueError("Each phenotype ramp must contain strictly increasing bounds.")
    if set(dataset["phenotype_coefficients"]) != {"lower", "upper_down", "upper_up"}:
        raise ValueError("Phenotype coefficients must define lower, upper_down, and upper_up.")
    for branch, coefficients in dataset["phenotype_coefficients"].items():
        unknown = set(coefficients) - ramp_names
        if unknown:
            raise ValueError(f"Branch {branch!r} uses unknown ramps: {sorted(unknown)}")
    progress_coefficients = dataset.get("progress_coefficients", {})
    if progress_coefficients and set(progress_coefficients) != {"lower", "upper_down", "upper_up"}:
        raise ValueError(
            "Progress coefficients, when provided, must define lower, upper_down, and upper_up."
        )
    for branch, coefficients in progress_coefficients.items():
        unknown = set(coefficients) - ramp_names
        if unknown:
            raise ValueError(f"Progress branch {branch!r} uses unknown ramps: {sorted(unknown)}")

    truth = dataset["true_diffusion"]
    if truth.get("kind") == "constant":
        if float(truth["sigma"]) <= 0:
            raise ValueError("A constant ground-truth diffusion needs a positive sigma.")
    elif truth.get("kind") == "ground_truth":
        amplitudes = (
            "progress_diffusion",
            "diffusion_floor",
            "first_diffusion_peak",
            "second_diffusion_peak",
        )
        if any(float(truth[key]) < 0 for key in amplitudes) or float(truth["progress_diffusion"]) == 0:
            raise ValueError("Ground-truth diffusion amplitudes must be nonnegative with positive progress noise.")
        if float(truth["diffusion_floor"]) == 0 and not any(
            float(truth[key]) > 0 for key in ("first_diffusion_peak", "second_diffusion_peak")
        ):
            raise ValueError("The phenotype diffusion must be nondegenerate.")
        if any(float(truth[key]) <= 0 for key in ("first_diffusion_width", "second_diffusion_width")):
            raise ValueError("Ground-truth diffusion widths must be positive.")
    else:
        raise ValueError("dataset.true_diffusion.kind must be ground_truth or constant.")

    for name, prior in config["priors"].items():
        if prior.get("kind") not in {"ground_truth", "constant"}:
            raise ValueError(f"Unknown prior kind for {name!r}.")
        if prior["kind"] == "constant" and float(prior["sigma"]) <= 0:
            raise ValueError("Constant diffusion sigma must be positive.")

    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("At least one unique training seed is required.")
    for section in ("classifier", "sf2m", "strong_map", "real_clift", "evaluation"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    if not 0.0 < float(config["classifier"]["train_fraction"]) < 1.0:
        raise ValueError("classifier.train_fraction must be strictly between zero and one.")
    for key in ("max_steps", "patience", "eval_every"):
        if int(config["classifier"][key]) <= 0:
            raise ValueError(f"classifier.{key} must be positive.")
    for key in ("train_cells_per_time", "validation_cells_per_time"):
        if int(config["learning_split"][key]) <= 0:
            raise ValueError(f"learning_split.{key} must be positive.")
    for key in ("width", "depth", "time_frequencies"):
        if int(config["network"][key]) <= 0:
            raise ValueError(f"network.{key} must be positive.")
    for section, keys in (
        ("sf2m", ("base_steps", "continuation_steps", "batch_size", "eval_every", "continuation_eval_every", "validation_batch_size")),
        (
            "strong_map",
            (
                "base_steps",
                "continuation_steps",
                "batch_size",
                "composition_batch_size",
                "rollout_batch_size",
                "eval_every",
                "continuation_eval_every",
                "validation_bridge_batch_size",
                "validation_composition_batch_size",
                "validation_fate_batch_size",
                "noise_bins",
                "fate_map_steps",
            ),
        ),
    ):
        if any(int(config[section][key]) <= 0 for key in keys):
            raise ValueError(f"All configured step and batch counts in {section} must be positive.")
    if not 0.0 <= float(config["strong_map"]["ema_decay"]) < 1.0:
        raise ValueError("strong_map.ema_decay must lie in [0, 1).")
    if float(config["strong_map"]["anchor_step"]) < 0:
        raise ValueError("strong_map.anchor_step cannot be negative.")
    horizons = [float(value) for value in config["strong_map"]["horizons"]]
    if not horizons or any(value <= 0 or value > end - start for value in horizons):
        raise ValueError("strong_map.horizons must be positive and no longer than the experiment.")
    fate_range = [float(value) for value in config["strong_map"]["fate_start_range"]]
    fate_end = float(config["strong_map"]["fate_end_time"])
    minimum_horizon = float(config["strong_map"]["fate_minimum_horizon"])
    if len(fate_range) != 2 or fate_range[0] > fate_range[1]:
        raise ValueError("strong_map.fate_start_range must contain ordered lower and upper bounds.")
    if fate_range[0] < train_times[-2] or fate_range[1] + minimum_horizon > fate_end + 1e-9:
        raise ValueError("The fate start range must lie in the final training interval and leave the minimum horizon.")
    if fate_end > end + 1e-9:
        raise ValueError("strong_map.fate_end_time cannot exceed the dataset end time.")

    real_clift = config["real_clift"]
    integer_keys = (
        "steps",
        "batch_size",
        "validation_batch_size",
        "constraint_batch_size",
        "eval_every",
        "noise_bins",
    )
    if any(int(real_clift[key]) <= 0 for key in integer_keys):
        raise ValueError("All real_clift step and batch counts must be positive.")
    if int(real_clift["batch_size"]) < 2 or int(real_clift["validation_batch_size"]) < 2:
        raise ValueError("real_clift batches must contain at least two examples.")
    local_fraction = float(real_clift["local_fraction"])
    if not 0.0 < local_fraction < 1.0:
        raise ValueError("real_clift.local_fraction must lie strictly between zero and one.")
    off_diagonal_count = int(real_clift["batch_size"]) - int(
        int(real_clift["batch_size"]) * local_fraction
    )
    if int(real_clift["constraint_batch_size"]) > off_diagonal_count:
        raise ValueError("real_clift.constraint_batch_size cannot exceed its off-diagonal batch.")
    local_step = float(real_clift["local_step_fraction"])
    maximum_horizon = float(real_clift["max_horizon_fraction"])
    time_epsilon = float(real_clift["time_eps_fraction"])
    if local_step <= 0.0 or maximum_horizon < local_step:
        raise ValueError("real_clift horizons must satisfy 0 < local_step <= max_horizon.")
    if time_epsilon < 0.0 or maximum_horizon > 1.0 - 2.0 * time_epsilon:
        raise ValueError("real_clift.max_horizon_fraction must leave both endpoint epsilons.")
    if not 0.0 < float(real_clift["horizon_curriculum_fraction"]) <= 1.0:
        raise ValueError("real_clift.horizon_curriculum_fraction must lie in (0, 1].")
    if float(real_clift["horizon_curriculum_power"]) <= 0.0:
        raise ValueError("real_clift.horizon_curriculum_power must be positive.")
    if not 0.0 <= float(real_clift["ema_decay"]) < 1.0:
        raise ValueError("real_clift.ema_decay must lie in [0, 1).")
    if int(real_clift["warmup_steps"]) < 0 or int(real_clift["warmup_steps"]) > int(real_clift["steps"]):
        raise ValueError("real_clift.warmup_steps must lie between zero and steps.")
    if any(
        float(real_clift[key]) < 0.0
        for key in ("constraint_weight", "anchor_step", "weight_decay")
    ):
        raise ValueError("real_clift weights and anchor_step must be non-negative.")
    if float(real_clift["minimum_learning_rate"]) < 0.0 or float(real_clift["learning_rate"]) <= 0.0:
        raise ValueError("real_clift learning rates must be non-negative with a positive maximum.")
    if float(real_clift["minimum_learning_rate"]) > float(real_clift["learning_rate"]):
        raise ValueError("real_clift.minimum_learning_rate cannot exceed learning_rate.")
    if float(real_clift["gradient_clip"]) <= 0.0:
        raise ValueError("real_clift.gradient_clip must be positive.")
    if any(
        float(real_clift[key]) < 0.0
        for key in ("local_loss_horizon_power", "semigroup_loss_horizon_power")
    ):
        raise ValueError("real_clift horizon-normalization powers must be non-negative.")
    if any(not 0.0 < float(real_clift[key]) < 1.0 for key in ("adam_beta1", "adam_beta2")):
        raise ValueError("real_clift Adam beta values must lie in (0, 1).")
    evaluation = config["evaluation"]
    starts = [float(value) for value in evaluation["start_times"]]
    if not starts or any(value < start or value >= end for value in starts):
        raise ValueError("Evaluation start times must lie inside [start_time, end_time).")
    conditional_seeds = evaluation.get("conditional_candidate_seeds")
    if conditional_seeds is not None and len(conditional_seeds) != len(starts):
        raise ValueError("evaluation.conditional_candidate_seeds must match start_times in length.")
    replicates = int(evaluation["population_replicates"])
    marginal_seeds = evaluation.get("marginal_candidate_seeds")
    if marginal_seeds is not None and len(marginal_seeds) != replicates:
        raise ValueError("evaluation.marginal_candidate_seeds must match population_replicates in length.")
    missing_rates = set(config["priors"]) - set(evaluation["conditional_sde_steps_per_unit"])
    if missing_rates:
        raise ValueError(f"Missing conditional SDE integration rates for priors: {sorted(missing_rates)}")
    if any(int(value) <= 0 for value in evaluation["conditional_sde_steps_per_unit"].values()):
        raise ValueError("Conditional SDE integration rates must be positive.")
    positive_counts = (
        "states_per_time",
        "candidate_rollouts",
        "reference_rollouts",
        "reference_sw2_rollouts",
        "sw2_directions",
        "reference_steps_per_unit",
        "population_samples",
        "population_replicates",
        "marginal_sde_steps_per_unit",
        "marginal_map_steps_per_unit",
        "conditional_map_steps_per_unit",
    )
    if any(int(evaluation[key]) <= 0 for key in positive_counts):
        raise ValueError("Evaluation sample counts and integration rates must be positive.")
    if int(evaluation["reference_sw2_rollouts"]) > int(evaluation["reference_rollouts"]):
        raise ValueError("reference_sw2_rollouts cannot exceed reference_rollouts.")


def write_config(path: Union[str, Path], config: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(config, indent=2) + "\n")

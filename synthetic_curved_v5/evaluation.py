"""Common-bank evaluation for the configurable curved-v5 benchmark.

The evaluator mirrors the quantities reported for the main-body synthetic
experiment: conditional terminal-fate TV, time-averaged conditional sliced W2,
and held-out marginal W1.  Oracle functions are imported only here, never by
the fitting code.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import ndtri
from scipy.stats import qmc
import torch

from .classifier import load_classifier, predict, sha256
from .dataset import (
    CLASS_NAMES,
    exact_marginal,
    hard_labels,
    means_and_derivatives,
    simulate,
    snapshot_times,
    std_and_derivative,
    terminal_probabilities,
)
from .models import load_fields, load_map, sample_map, sample_sde, valid_transition_matrix
from .training import METHODS, select_device


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _state_bank(time_value, count, seed, dataset):
    """Deterministic low-discrepancy sample from the analytic marginal."""
    dimensions = 3
    exponent = int(np.ceil(np.log2(count)))
    uniforms = qmc.Sobol(dimensions, scramble=True, seed=int(seed)).random_base2(exponent)[:count]
    weights = np.asarray(dataset["mixture_weights"], dtype=np.float64)
    components = np.searchsorted(np.cumsum(weights), uniforms[:, 0], side="right")
    means, _ = means_and_derivatives(float(time_value), dataset)
    standard_deviations, _ = std_and_derivative(float(time_value), dataset)
    normal = ndtri(uniforms[:, 1:].clip(1e-10, 1.0 - 1e-10))
    return (means[components] + standard_deviations * normal).astype(np.float32)


def _future_times(start, dataset):
    grid = snapshot_times(dataset)
    later = grid[grid > float(start) + 1e-10]
    return np.r_[float(start), later]


def _directions(count):
    angles = np.arange(int(count), dtype=np.float64) * np.pi / int(count)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def conditional_metrics(candidate, reference, times, dataset, directions, reference_limit):
    """Metrics for arrays shaped state x rollout x time x coordinate."""
    candidate_count = candidate.shape[1]
    reference_count = min(reference.shape[1], int(reference_limit))
    unit = _directions(directions)
    candidate_fates = terminal_probabilities(candidate[:, :, -1], dataset).mean(1)
    reference_fates = terminal_probabilities(reference[:, :, -1], dataset).mean(1)
    candidate_quantiles = (np.arange(candidate_count) + 0.5) / candidate_count
    reference_quantiles = (np.arange(reference_count) + 0.5) / reference_count
    rows = []
    for state_index in range(len(candidate)):
        by_time = []
        for time_index in range(1, len(times)):
            candidate_projection = np.sort(
                candidate[state_index, :, time_index].astype(np.float64) @ unit.T,
                axis=0,
            )
            reference_projection = np.sort(
                reference[state_index, :reference_count, time_index].astype(np.float64) @ unit.T,
                axis=0,
            )
            if candidate_count != reference_count:
                reference_projection = np.stack(
                    [
                        np.interp(candidate_quantiles, reference_quantiles, reference_projection[:, column])
                        for column in range(len(unit))
                    ],
                    axis=-1,
                )
            by_time.append(float(np.sqrt(np.mean((candidate_projection - reference_projection) ** 2))))
        if len(times) == 1:
            integrated = 0.0
        else:
            trapezoid = getattr(np, "trapezoid", None)
            if trapezoid is None:  # NumPy 1.x compatibility.
                trapezoid = np.trapz
            integrated = float(trapezoid(np.r_[0.0, by_time], times) / (times[-1] - times[0]))
        rows.append(
            {
                "state_index": state_index,
                "start_time": float(times[0]),
                "fate_tv": float(0.5 * np.abs(candidate_fates[state_index] - reference_fates[state_index]).sum()),
                "conditional_sw2": integrated,
                "per_future_time_sw2": by_time,
                "candidate_fates": candidate_fates[state_index].tolist(),
                "reference_fates": reference_fates[state_index].tolist(),
            }
        )
    return rows


def exact_empirical_wasserstein(candidate, reference):
    """Exact equal-mass empirical W1 and W2 via a Hungarian assignment."""
    if len(candidate) != len(reference):
        raise ValueError("Exact empirical Wasserstein comparison requires equal sample counts.")
    distance = cdist(np.asarray(candidate, dtype=np.float64), np.asarray(reference, dtype=np.float64))
    rows, columns = linear_sum_assignment(distance)
    matched = distance[rows, columns]
    return float(matched.mean()), float(np.sqrt(np.mean(matched**2)))


def population_metrics(paths, marginals, times, classifier, dataset):
    rows = []
    training_times = np.asarray(dataset["training_times"], dtype=np.float64)
    for index, time_value in enumerate(times):
        w1, w2 = exact_empirical_wasserstein(paths[:, index], marginals[index])
        rows.append(
            {
                "time": float(time_value),
                "role": "observed" if np.any(np.isclose(time_value, training_times)) else "held_out",
                "w1": w1,
                "w2": w2,
            }
        )
    allowed = valid_transition_matrix().cpu().numpy().astype(bool)
    hard = hard_labels(paths, dataset)
    learned = predict(classifier, paths).argmax(-1)
    hard_invalid = ~allowed[hard[:, :-1], hard[:, 1:]]
    learned_invalid = ~allowed[learned[:, :-1], learned[:, 1:]]
    invalidity = {
        "hard_consecutive_invalidity": float(hard_invalid.mean()),
        "hard_any_invalid_path": float(hard_invalid.max(1).mean()),
        "classifier_consecutive_invalidity": float(learned_invalid.mean()),
        "classifier_any_invalid_path": float(learned_invalid.max(1).mean()),
        "classifier_vs_hard_accuracy": float((hard == learned).mean()),
    }
    return rows, invalidity


def prepare_references(root, config):
    root = Path(root)
    output = root / "evaluation" / "reference"
    manifest = output / "manifest.json"
    if manifest.exists():
        return output
    output.mkdir(parents=True, exist_ok=True)
    settings, dataset = config["evaluation"], config["dataset"]
    started = time.perf_counter()
    for index, start in enumerate(settings["start_times"]):
        states = _state_bank(start, int(settings["states_per_time"]), int(settings["seed"]) + index, dataset)
        times = _future_times(start, dataset)
        initial = np.repeat(states, int(settings["reference_rollouts"]), axis=0)
        paths = simulate(
            initial,
            float(start),
            times,
            int(settings["seed"]) + 100 + index,
            dataset,
            steps_per_unit=int(settings["reference_steps_per_unit"]),
        ).reshape(len(states), int(settings["reference_rollouts"]), len(times), 2)
        np.savez_compressed(
            output / f"conditional_{index}.npz",
            states=states,
            times=times,
            paths=paths,
            fate_probabilities=terminal_probabilities(paths[:, :, -1], dataset).mean(1),
        )
        print(f"reference start={start}: {len(states)} states", flush=True)
    times = snapshot_times(dataset)
    rng = np.random.default_rng(int(settings["seed"]) + 300)
    for replicate in range(int(settings["population_replicates"])):
        marginals = np.stack(
            [exact_marginal(time_value, int(settings["population_samples"]), rng, dataset) for time_value in times]
        )
        initial = exact_marginal(times[0], int(settings["population_samples"]), rng, dataset)
        truth = simulate(
            initial,
            float(times[0]),
            times,
            int(settings["seed"]) + 302 + replicate,
            dataset,
            steps_per_unit=int(settings["reference_steps_per_unit"]),
        )
        np.savez_compressed(
            output / f"population_{replicate}.npz",
            initial=initial,
            times=times,
            marginals=marginals,
            truth_paths=truth,
        )
    _write_json(
        manifest,
        {
            "status": "completed",
            "seconds": time.perf_counter() - started,
            "settings": settings,
            "files": {path.name: sha256(path) for path in sorted(output.glob("*.npz"))},
        },
    )
    return output


def _sampler(method, checkpoint, prior_name, settings, device):
    if method == "sf2m":
        model = load_fields(checkpoint, device)
        conditional_rate = int(settings["conditional_sde_steps_per_unit"][prior_name])
        marginal_rate = int(settings["marginal_sde_steps_per_unit"])
        return model, sample_sde, conditional_rate, marginal_rate
    model = load_map(checkpoint, device)
    return (
        model,
        sample_map,
        int(settings["conditional_map_steps_per_unit"]),
        int(settings["marginal_map_steps_per_unit"]),
    )


def _candidate_seed(settings, key, index, fallback):
    values = settings.get(key)
    return int(fallback if values is None else values[index])


def evaluate_fit(root, config, seed, prior_name, method, device):
    root = Path(root)
    output = root / "evaluation" / "per_fit" / f"seed_{seed}" / prior_name / method
    checkpoint = root / "models" / f"seed_{seed}" / prior_name / method / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    metrics_file = output / "metrics.json"
    if metrics_file.exists():
        return json.loads(metrics_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    settings, dataset = config["evaluation"], config["dataset"]
    reference = prepare_references(root, config)
    model, sampler, conditional_rate, marginal_rate = _sampler(
        method, checkpoint, prior_name, settings, device
    )
    classifier = load_classifier(root / "classifier" / "classifier.pt", device)
    conditional_rows, population_rows, invalidity_replicates = [], [], []
    sampling_seconds = 0.0
    for index, start in enumerate(settings["start_times"]):
        with np.load(reference / f"conditional_{index}.npz", allow_pickle=False) as saved:
            states = saved["states"].copy()
            times = saved["times"].copy()
            reference_paths = saved["paths"].copy()
        initial = np.repeat(states, int(settings["candidate_rollouts"]), axis=0)
        tick = time.perf_counter()
        paths = sampler(
            model,
            initial,
            float(start),
            times,
            conditional_rate,
            _candidate_seed(settings, "conditional_candidate_seeds", index, int(settings["seed"]) + 500 + index),
        )
        sampling_seconds += time.perf_counter() - tick
        paths = paths.reshape(len(states), int(settings["candidate_rollouts"]), len(times), 2)
        if not np.isfinite(paths).all():
            raise FloatingPointError(f"Non-finite conditional paths for seed {seed}, {prior_name}, {method}.")
        conditional_rows.extend(
            conditional_metrics(
                paths,
                reference_paths,
                times,
                dataset,
                int(settings["sw2_directions"]),
                int(settings["reference_sw2_rollouts"]),
            )
        )
        if settings.get("save_paths", True):
            np.savez_compressed(output / f"conditional_{index}.npz", states=states, times=times, paths=paths)
    for replicate in range(int(settings["population_replicates"])):
        with np.load(reference / f"population_{replicate}.npz", allow_pickle=False) as saved:
            initial = saved["initial"].copy()
            times = saved["times"].copy()
            marginals = saved["marginals"].copy()
        tick = time.perf_counter()
        paths = sampler(
            model,
            initial,
            float(times[0]),
            times,
            marginal_rate,
            _candidate_seed(settings, "marginal_candidate_seeds", replicate, int(settings["seed"]) + 600 + replicate),
        )
        sampling_seconds += time.perf_counter() - tick
        if not np.isfinite(paths).all():
            raise FloatingPointError(f"Non-finite population paths for seed {seed}, {prior_name}, {method}.")
        rows, invalidity = population_metrics(paths, marginals, times, classifier, dataset)
        population_rows.extend([{**row, "replicate": replicate} for row in rows])
        invalidity_replicates.append(invalidity)
        if settings.get("save_paths", True):
            np.savez_compressed(output / f"population_{replicate}.npz", times=times, paths=paths)
    invalidity = {
        key: float(np.mean([row[key] for row in invalidity_replicates]))
        for key in invalidity_replicates[0]
    }
    heldout_draws = [
        float(np.mean([row["w1"] for row in population_rows if row["replicate"] == replicate and row["role"] == "held_out"]))
        for replicate in range(int(settings["population_replicates"]))
    ]
    observed_draws = [
        float(
            np.mean(
                [
                    row["w1"]
                    for row in population_rows
                    if row["replicate"] == replicate and row["role"] == "observed" and row["time"] > dataset["start_time"]
                ]
            )
        )
        for replicate in range(int(settings["population_replicates"]))
    ]
    summary = {
        "fate_tv": float(np.mean([row["fate_tv"] for row in conditional_rows])),
        "conditional_sw2": float(np.mean([row["conditional_sw2"] for row in conditional_rows])),
        "heldout_w1": float(np.mean(heldout_draws)),
        "observed_w1": float(np.mean(observed_draws)),
    }
    result = {
        "status": "completed",
        "seed": int(seed),
        "prior": prior_name,
        "method": method,
        "checkpoint_sha256": sha256(checkpoint),
        "summary": summary,
        "conditional_rows": conditional_rows,
        "population_rows": population_rows,
        "population_draw_heldout_w1": heldout_draws,
        "population_draw_observed_w1": observed_draws,
        "invalidity": invalidity,
        "conditional_steps_per_unit": conditional_rate,
        "marginal_steps_per_unit": marginal_rate,
        "sampling_seconds": sampling_seconds,
        "device": str(device),
    }
    _write_json(metrics_file, result)
    print(
        f"evaluated seed={seed} prior={prior_name} method={method}: "
        f"fate_tv={summary['fate_tv']:.5g}, sw2={summary['conditional_sw2']:.5g}, "
        f"heldout_w1={summary['heldout_w1']:.5g}",
        flush=True,
    )
    return result


def _fit_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "seed_values": values.tolist(),
    }


def _collect_results(root):
    results = {}
    for path in sorted((Path(root) / "evaluation" / "per_fit").glob("seed_*/*/*/metrics.json")):
        value = json.loads(path.read_text())
        key = f"seed_{value['seed']}/{value['prior']}/{value['method']}"
        results[key] = value
    return results


def aggregate(root):
    root = Path(root)
    results = _collect_results(root)
    metrics = ("fate_tv", "conditional_sw2", "heldout_w1", "observed_w1")
    groups = []
    combinations = sorted({(value["prior"], value["method"]) for value in results.values()})
    for prior, method in combinations:
        selected = sorted(
            (value for value in results.values() if value["prior"] == prior and value["method"] == method),
            key=lambda value: value["seed"],
        )
        group = {"prior": prior, "method": method, "seeds": [value["seed"] for value in selected]}
        for metric in metrics:
            group[metric] = _fit_statistics([value["summary"][metric] for value in selected])
        groups.append(group)
    paired = {}
    for prior in sorted({value["prior"] for value in results.values()}):
        for guided_method in ("clift", "real_clift"):
            baselines = (
                ("sf2m", "ssfm", "clift")
                if guided_method == "real_clift"
                else ("sf2m", "ssfm")
            )
            for baseline in baselines:
                common = sorted(
                    set(
                        value["seed"]
                        for value in results.values()
                        if value["prior"] == prior and value["method"] == guided_method
                    )
                    & set(
                        value["seed"]
                        for value in results.values()
                        if value["prior"] == prior and value["method"] == baseline
                    )
                )
                if not common:
                    continue
                comparison = {}
                for metric in metrics:
                    guided = np.asarray(
                        [
                            results[f"seed_{seed}/{prior}/{guided_method}"]["summary"][metric]
                            for seed in common
                        ]
                    )
                    control = np.asarray(
                        [
                            results[f"seed_{seed}/{prior}/{baseline}"]["summary"][metric]
                            for seed in common
                        ]
                    )
                    delta = guided - control
                    denominator = float(control.mean())
                    comparison[metric] = {
                        "difference": _fit_statistics(delta),
                        "percent_change": None
                        if abs(denominator) < 1e-15
                        else float(100.0 * delta.mean() / denominator),
                    }
                paired[f"{prior}/{guided_method}_vs_{baseline}"] = comparison
    summary = {"per_fit": results, "groups": groups, "paired_comparisons": paired}
    output = root / "evaluation"
    _write_json(output / "summary.json", summary)
    columns = ["prior", "method", "n_seeds"]
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for group in groups:
            row = {"prior": group["prior"], "method": group["method"], "n_seeds": len(group["seeds"])}
            for metric in metrics:
                row[f"{metric}_mean"] = group[metric]["mean"]
                row[f"{metric}_std"] = group[metric]["std"]
            writer.writerow(row)
    _plot_summary(output / "summary.png", groups)
    _plot_conditional_comparison(output / "conditional_paths.png", root, results)
    return summary


def _plot_summary(path, groups):
    if not groups:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("fate_tv", "Fate TV"), ("conditional_sw2", "Conditional SW2"), ("heldout_w1", "Held-out marginal W1"))
    labels = [f"{row['prior']}\n{row['method'].upper()}" for row in groups]
    colors = {
        "sf2m": "#78869B",
        "ssfm": "#3F8E8A",
        "clift": "#C66A52",
        "real_clift": "#8F5AA8",
    }
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 4.6))
    for axis, (metric, title) in zip(axes, metrics):
        mean = [row[metric]["mean"] for row in groups]
        error = [row[metric]["std"] for row in groups]
        axis.bar(np.arange(len(groups)), mean, yerr=error, color=[colors[row["method"]] for row in groups], capsize=3)
        axis.set_xticks(np.arange(len(groups)), labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_conditional_comparison(path, root, results):
    """Plot one common conditioning state for every available method/prior."""
    if not results:
        return
    reference_file = Path(root) / "evaluation" / "reference" / "conditional_0.npz"
    if not reference_file.exists():
        return
    methods = [method for method in METHODS if any(value["method"] == method for value in results.values())]
    priors = sorted({value["prior"] for value in results.values()})
    candidates = {}
    for prior in priors:
        for method in methods:
            choices = sorted(
                (value for value in results.values() if value["prior"] == prior and value["method"] == method),
                key=lambda value: value["seed"],
            )
            if not choices:
                continue
            selected = choices[0]
            candidate_file = (
                Path(root)
                / "evaluation"
                / "per_fit"
                / f"seed_{selected['seed']}"
                / prior
                / method
                / "conditional_0.npz"
            )
            if not candidate_file.exists():
                return
            with np.load(candidate_file, allow_pickle=False) as saved:
                candidates[(prior, method)] = (selected["seed"], saved["paths"].copy())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(reference_file, allow_pickle=False) as saved:
        reference_paths = saved["paths"][0].copy()
        reference_time = float(saved["times"][0])
    all_paths = [reference_paths] + [value[1][0] for value in candidates.values()]
    points = np.concatenate([values.reshape(-1, 2) for values in all_paths])
    lower, upper = np.quantile(points, [0.002, 0.998], axis=0)
    padding = np.maximum(0.08 * (upper - lower), 0.03)
    figure, axes = plt.subplots(
        len(priors),
        1 + len(methods),
        figsize=(3.7 * (1 + len(methods)), 3.4 * len(priors)),
        squeeze=False,
    )
    colors = {
        "oracle": "#3B4351",
        "sf2m": "#667FA8",
        "ssfm": "#2F8B82",
        "clift": "#CB6750",
        "real_clift": "#8F5AA8",
    }
    maximum_paths = 64
    for row, prior in enumerate(priors):
        panels = [("Oracle", reference_paths, "oracle", None)]
        for method in methods:
            if (prior, method) in candidates:
                seed, values = candidates[(prior, method)]
                panels.append((method.upper(), values[0], method, seed))
            else:
                panels.append((method.upper(), None, method, None))
        for column, (title, values, method, seed) in enumerate(panels):
            axis = axes[row, column]
            if values is not None:
                for trajectory in values[:maximum_paths]:
                    axis.plot(trajectory[:, 0], trajectory[:, 1], color=colors[method], lw=0.6, alpha=0.18)
                axis.scatter(values[:maximum_paths, -1, 0], values[:maximum_paths, -1, 1], s=7, color=colors[method], alpha=0.55)
                axis.scatter(values[0, 0, 0], values[0, 0, 1], s=30, color="black", marker="x", zorder=5)
            axis.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
            axis.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
            axis.set_aspect("equal")
            axis.grid(alpha=0.12)
            suffix = "" if seed is None else f" (seed {seed})"
            axis.set_title(title + suffix)
            if column == 0:
                axis.set_ylabel(prior)
            if row == len(priors) - 1:
                axis.set_xlabel("developmental progress")
    figure.suptitle(f"Conditional paths from t={reference_time:g}; first common state")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def evaluate_all(root, config, methods=METHODS, seeds=None, priors=None, device="auto"):
    root = Path(root)
    methods = tuple(methods)
    if not methods:
        raise ValueError("At least one method must be selected.")
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    seeds = [int(value) for value in (config["seeds"] if seeds is None else seeds)]
    unknown_seeds = set(seeds) - set(config["seeds"])
    if unknown_seeds:
        raise ValueError(f"Evaluation seeds are not registered in the config: {sorted(unknown_seeds)}")
    priors = list(config["priors"] if priors is None else priors)
    if not priors:
        raise ValueError("At least one prior must be selected.")
    unknown_priors = set(priors) - set(config["priors"])
    if unknown_priors:
        raise ValueError(f"Unknown priors: {sorted(unknown_priors)}")
    torch.set_num_threads(1)
    selected_device = select_device(device)
    prepare_references(root, config)
    for seed in seeds:
        for prior_name in priors:
            for method in methods:
                evaluate_fit(root, config, seed, prior_name, method, selected_device)
    return aggregate(root)

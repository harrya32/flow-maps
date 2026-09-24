"""Evaluation utilities for the isolated Maizels stochastic experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import maizels
from . import ssfm_brownian
from . import wasserstein


def _sample_indices(rng: np.random.Generator, n: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or maximum >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=maximum, replace=False)).astype(np.int64)


def _time_tag(timepoint: str) -> str:
    return str(timepoint).replace(".", "p")


def uses_euler_maruyama_evaluation(cfg) -> bool:
    """Whether this checkpoint learned only the local Euler--Maruyama map."""
    return float(cfg.ssfm.local_fraction) >= 1.0


def euler_maruyama_n_steps(cfg) -> int:
    """Return the fixed number of Euler--Maruyama evaluation steps."""
    value = int(getattr(cfg.evaluation, "euler_maruyama_n_steps", 50))
    if value <= 0:
        raise ValueError("evaluation.euler_maruyama_n_steps must be positive.")
    return value


def sample_pushforward(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
) -> np.ndarray:
    """Apply one direct SSFM draw to a population."""
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    n = x_jax.shape[0]
    s = jnp.full((n,), float(start_time), dtype=x_jax.dtype)
    t = jnp.full((n,), float(end_time), dtype=x_jax.dtype)
    horizon = t - s
    coefficients = ssfm_brownian.sample_legendre_coefficients(
        key,
        horizon,
        n_coefficients=n_coefficients,
        data_dim=x_jax.shape[-1],
    )
    prediction, _ = model.apply({"params": params}, s, t, x_jax, coefficients)
    return np.asarray(jax.device_get(prediction), dtype=np.float32)


def sample_composed_pushforward(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Compose stochastic flow maps over equal subintervals.

    Each subinterval receives an independent Brownian segment. This is the
    stochastic analogue of the deterministic experiment's multi-step
    ``flowmap`` sampler; it never evaluates an Euler discretization of a
    velocity field.
    """
    prediction, _ = _sample_composed(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=False,
    )
    return prediction


def sample_composed_path(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Return the states after every step of a composed stochastic map."""
    _, path = _sample_composed(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=True,
    )
    return path


def sample_euler_maruyama_pushforward(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Roll the learned local drift/diffusion update to the requested time."""
    prediction, _ = _sample_euler_maruyama(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=False,
    )
    return prediction


def sample_euler_maruyama_path(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
) -> np.ndarray:
    """Return every state in a local-model Euler--Maruyama rollout."""
    _, path = _sample_euler_maruyama(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=n_coefficients,
        n_steps=n_steps,
        return_path=True,
    )
    return path


def _sample_euler_maruyama(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
    return_path: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Integrate with the same learned local EM update used by training.

    A local SSFM call has the explicit Euler--Maruyama form
    ``x_next = x + dt * drift + dW * diffusion``.  Unlike flow-map sampling,
    this routine never asks the network to predict a whole non-local interval;
    it advances through ``n_steps`` equal local updates.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if float(end_time) < float(start_time):
        raise ValueError("end_time must be greater than or equal to start_time.")
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    n = x_jax.shape[0]
    times = jnp.linspace(
        float(start_time), float(end_time), n_steps + 1, dtype=x_jax.dtype
    )
    step_keys = jax.random.split(key, n_steps)

    def step(current, inputs):
        left, right, step_key = inputs
        s = jnp.full((n,), left, dtype=x_jax.dtype)
        t = jnp.full((n,), right, dtype=x_jax.dtype)
        coefficients = ssfm_brownian.sample_legendre_coefficients(
            step_key,
            t - s,
            n_coefficients=n_coefficients,
            data_dim=x_jax.shape[-1],
        )
        prediction, _ = model.apply({"params": params}, s, t, current, coefficients)
        return prediction, prediction if return_path else None

    prediction, states = jax.lax.scan(
        step,
        x_jax,
        (times[:-1], times[1:], step_keys),
    )
    prediction_np = np.asarray(jax.device_get(prediction), dtype=np.float32)
    if states is None:
        return prediction_np, None
    path_np = np.asarray(jax.device_get(jnp.swapaxes(states, 0, 1)), dtype=np.float32)
    return prediction_np, path_np


def sample_evaluation_path(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    cfg,
    *,
    n_steps: Optional[int] = None,
) -> np.ndarray:
    """Use EM for local-only models and composed maps for full SSFMs."""
    if uses_euler_maruyama_evaluation(cfg):
        return sample_euler_maruyama_path(
            model,
            params,
            x,
            start_time,
            end_time,
            key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=euler_maruyama_n_steps(cfg),
        )
    return sample_composed_path(
        model,
        params,
        x,
        start_time,
        end_time,
        key,
        n_coefficients=int(cfg.ssfm.n_coefficients),
        n_steps=int(cfg.evaluation.lineage_n_steps if n_steps is None else n_steps),
    )


def _sample_composed(
    model,
    params,
    x: np.ndarray,
    start_time: float,
    end_time: float,
    key: jnp.ndarray,
    *,
    n_coefficients: int,
    n_steps: int,
    return_path: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run equal-subinterval stochastic maps, optionally retaining the path."""
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    x_jax = jnp.asarray(x, dtype=jnp.float32)
    n = x_jax.shape[0]
    times = jnp.linspace(
        float(start_time), float(end_time), n_steps + 1, dtype=x_jax.dtype
    )
    step_keys = jax.random.split(key, n_steps)

    def step(current, inputs):
        left, right, step_key = inputs
        s = jnp.full((n,), left, dtype=x_jax.dtype)
        t = jnp.full((n,), right, dtype=x_jax.dtype)
        coefficients = ssfm_brownian.sample_legendre_coefficients(
            step_key,
            t - s,
            n_coefficients=n_coefficients,
            data_dim=x_jax.shape[-1],
        )
        prediction, _ = model.apply({"params": params}, s, t, current, coefficients)
        return prediction, prediction if return_path else None

    prediction, states = jax.lax.scan(
        step,
        x_jax,
        (times[:-1], times[1:], step_keys),
    )
    prediction_np = np.asarray(jax.device_get(prediction), dtype=np.float32)
    if states is None:
        return prediction_np, None
    path_np = np.asarray(jax.device_get(jnp.swapaxes(states, 0, 1)), dtype=np.float32)
    return prediction_np, path_np


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    result = (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * (x @ y.T)
    )
    return np.maximum(result, 0.0)


def rbf_mmd2(x: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Biased RBF MMD² with a reproducible median-distance bandwidth."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y], axis=0)
    indices = _sample_indices(rng, combined.shape[0], 512)
    calibration = _squared_distances(combined[indices], combined[indices])
    positive = calibration[calibration > 0]
    bandwidth2 = float(np.median(positive)) if positive.size else 1.0
    bandwidth2 = max(bandwidth2, 1e-8)

    def kernel(a, b):
        return np.exp(-_squared_distances(a, b) / (2.0 * bandwidth2))

    value = kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean()
    return float(max(value, 0.0))


def _interval_source(cfg, timepoint: str) -> Tuple[str, str]:
    return maizels.retained_interval_for_timepoint(cfg, timepoint)


def _heldout_timepoints(cfg) -> Sequence[str]:
    return tuple(str(value) for value in cfg.problem.evaluation_timepoints)


def _mean_or_absent(metrics: Dict[str, float], key: str, values: Iterable[float]):
    values = list(values)
    if values:
        metrics[key] = float(np.mean(values))


def distribution_metrics(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_source_points: Optional[int] = None,
    max_target_points: Optional[int] = None,
    timepoints: Optional[Sequence[str]] = None,
    n_noise_draws: Optional[int] = None,
) -> Tuple[
    Dict[str, float],
    Dict[str, Tuple[np.ndarray, ...]],
]:
    """Compare valid model samples with omitted-day populations.

    Full SSFMs are evaluated both directly and by map composition.  A model
    trained with ``local_fraction == 1`` is evaluated only by rolling its
    local Euler--Maruyama update; it has no trained non-local flow map.
    """
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_source_points = int(
        cfg.evaluation.max_source_points
        if max_source_points is None
        else max_source_points
    )
    max_target_points = int(
        cfg.evaluation.max_target_points
        if max_target_points is None
        else max_target_points
    )
    n_draws = int(
        cfg.evaluation.n_noise_draws if n_noise_draws is None else n_noise_draws
    )
    if n_draws <= 0:
        raise ValueError("evaluation.n_noise_draws must be positive.")
    use_euler_maruyama = uses_euler_maruyama_evaluation(cfg)
    flowmap_n_steps = int(cfg.evaluation.flowmap_n_steps)
    if not use_euler_maruyama and flowmap_n_steps <= 0:
        raise ValueError("evaluation.flowmap_n_steps must be positive.")
    em_n_steps = euler_maruyama_n_steps(cfg) if use_euler_maruyama else None

    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    val_times = set(str(value) for value in cfg.problem.hparam_val_times)
    samplers = ("euler_maruyama",) if use_euler_maruyama else ("direct", "flowmap")
    all_emd = {name: [] for name in samplers}
    val_emd = {name: [] for name in samplers}
    test_emd = {name: [] for name in samplers}
    metrics: Dict[str, float] = {}
    plot_data: Dict[str, Tuple[np.ndarray, ...]] = {}

    selected_timepoints = (
        _heldout_timepoints(cfg)
        if timepoints is None
        else tuple(str(value) for value in timepoints)
    )
    unavailable = sorted(set(selected_timepoints) - set(_heldout_timepoints(cfg)))
    if unavailable:
        raise ValueError(
            f"Requested stochastic evaluation times are not held out: {unavailable}."
        )
    for timepoint in selected_timepoints:
        source_time, _ = _interval_source(cfg, timepoint)
        source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
        target_all = np.asarray(pools[timepoint]["x"], dtype=np.float32)
        source = source_all[
            _sample_indices(rng, source_all.shape[0], max_source_points)
        ]
        target = target_all[
            _sample_indices(rng, target_all.shape[0], max_target_points)
        ]
        start = maizels.normalized_time(source_time, cfg)
        end = maizels.normalized_time(timepoint, cfg)
        draw_emd = {name: [] for name in samplers}
        draw_mmd = {name: [] for name in samplers}
        first_prediction = {}
        for draw in range(n_draws):
            if use_euler_maruyama:
                key, rollout_key = jax.random.split(key)
                predictions = {
                    "euler_maruyama": sample_euler_maruyama_pushforward(
                        model,
                        params,
                        source,
                        start,
                        end,
                        rollout_key,
                        n_coefficients=int(cfg.ssfm.n_coefficients),
                        n_steps=em_n_steps,
                    )
                }
            else:
                key, direct_key, flowmap_key = jax.random.split(key, 3)
                predictions = {
                    "direct": sample_pushforward(
                        model,
                        params,
                        source,
                        start,
                        end,
                        direct_key,
                        n_coefficients=int(cfg.ssfm.n_coefficients),
                    ),
                    "flowmap": sample_composed_pushforward(
                        model,
                        params,
                        source,
                        start,
                        end,
                        flowmap_key,
                        n_coefficients=int(cfg.ssfm.n_coefficients),
                        n_steps=flowmap_n_steps,
                    ),
                }
            for sampler, prediction in predictions.items():
                if sampler not in first_prediction:
                    first_prediction[sampler] = prediction
                draw_emd[sampler].append(wasserstein.exact_emd(prediction, target))
                draw_mmd[sampler].append(rbf_mmd2(prediction, target, seed=seed + draw))

        tag = _time_tag(timepoint)
        for sampler in samplers:
            emd = float(np.mean(draw_emd[sampler]))
            metrics[f"final_eval/{tag}_{sampler}_emd"] = emd
            metrics[f"final_eval/{tag}_{sampler}_emd_std_over_noise"] = float(
                np.std(draw_emd[sampler])
            )
            metrics[f"final_eval/{tag}_{sampler}_rbf_mmd2"] = float(
                np.mean(draw_mmd[sampler])
            )
            all_emd[sampler].append(emd)
            (val_emd if timepoint in val_times else test_emd)[sampler].append(emd)

        # Sampler-neutral aliases keep sweep/multiseed consumers stable.
        primary_sampler = "euler_maruyama" if use_euler_maruyama else "direct"
        metrics[f"final_eval/{tag}_ssfm_emd"] = metrics[
            f"final_eval/{tag}_{primary_sampler}_emd"
        ]
        metrics[f"final_eval/{tag}_ssfm_emd_std_over_noise"] = metrics[
            f"final_eval/{tag}_{primary_sampler}_emd_std_over_noise"
        ]
        metrics[f"final_eval/{tag}_ssfm_rbf_mmd2"] = metrics[
            f"final_eval/{tag}_{primary_sampler}_rbf_mmd2"
        ]
        if use_euler_maruyama:
            plot_data[timepoint] = (target, first_prediction["euler_maruyama"])
        else:
            plot_data[timepoint] = (
                target,
                first_prediction["direct"],
                first_prediction["flowmap"],
            )

    for sampler in samplers:
        _mean_or_absent(metrics, f"final_eval/{sampler}_mean_emd", all_emd[sampler])
        _mean_or_absent(
            metrics,
            f"final_eval/{sampler}_mean_emd_hparam_val_times",
            val_emd[sampler],
        )
        _mean_or_absent(
            metrics,
            f"final_eval/{sampler}_mean_emd_test_times",
            test_emd[sampler],
        )

    # Preserve the sampler-neutral aggregate keys used by sweep scripts.
    primary_sampler = "euler_maruyama" if use_euler_maruyama else "direct"
    for suffix in ("mean_emd", "mean_emd_hparam_val_times", "mean_emd_test_times"):
        sampler_key = f"final_eval/{primary_sampler}_{suffix}"
        if sampler_key in metrics:
            metrics[f"final_eval/ssfm_{suffix}"] = metrics[sampler_key]
    return metrics, plot_data


def _observed_time_tag(timepoint: str) -> str:
    tag = _time_tag(timepoint)
    return tag if str(timepoint).upper().startswith("D") else f"day{tag}"


def _observed_draw_key(
    seed: int,
    draw: int,
    interval_index: int,
    protocol: int,
):
    key = jax.random.PRNGKey(int(seed))
    key = jax.random.fold_in(key, 73_019)
    key = jax.random.fold_in(key, int(draw))
    key = jax.random.fold_in(key, int(interval_index))
    return jax.random.fold_in(key, int(protocol))


def observed_distribution_metrics(
    model,
    params,
    cfg,
    *,
    data_backend=maizels,
    seed: Optional[int] = None,
    max_source_points: Optional[int] = None,
    max_target_points: Optional[int] = None,
    n_noise_draws: Optional[int] = None,
) -> Dict[str, float]:
    """Score retained marginals and a source-to-final stochastic rollout.

    Observed-marginal metrics restart from the real population at the nearest
    retained left endpoint.  The rollout metric instead starts at the original
    source once and carries samples through every retained interval.  Full
    SSFMs use composed-map sampling; local-only models use Euler--Maruyama.
    """
    if not bool(getattr(cfg.evaluation, "observed_marginal_emd_enabled", True)):
        return {}
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_source_points = int(
        cfg.evaluation.max_source_points
        if max_source_points is None
        else max_source_points
    )
    max_target_points = int(
        cfg.evaluation.max_target_points
        if max_target_points is None
        else max_target_points
    )
    n_draws = int(
        cfg.evaluation.n_noise_draws if n_noise_draws is None else n_noise_draws
    )
    if n_draws <= 0:
        raise ValueError("evaluation.n_noise_draws must be positive.")

    retained = tuple(str(value) for value in data_backend.retained_timepoints(cfg))
    if len(retained) < 2:
        return {}
    pools = data_backend.timepoint_pool_splits(
        cfg,
        dataset_location=cfg.problem.dataset_location,
    )
    rng = np.random.default_rng(seed)

    def selected_population(timepoint: str, maximum: int) -> np.ndarray:
        population = np.asarray(pools[timepoint]["x"], dtype=np.float32)
        return population[_sample_indices(rng, population.shape[0], maximum)]

    sources = {
        timepoint: selected_population(timepoint, max_source_points)
        for timepoint in retained[:-1]
    }
    targets = {
        timepoint: selected_population(timepoint, max_target_points)
        for timepoint in retained[1:]
    }

    use_euler_maruyama = uses_euler_maruyama_evaluation(cfg)
    sampler = "euler_maruyama" if use_euler_maruyama else "flowmap"
    n_steps = (
        euler_maruyama_n_steps(cfg)
        if use_euler_maruyama
        else int(cfg.evaluation.flowmap_n_steps)
    )
    if n_steps <= 0:
        raise ValueError("Observed-marginal evaluation requires positive steps.")

    def pushforward(
        x: np.ndarray,
        source_time: str,
        target_time: str,
        key,
    ) -> np.ndarray:
        start = float(data_backend.normalized_time(source_time, cfg))
        end = float(data_backend.normalized_time(target_time, cfg))
        if use_euler_maruyama:
            return sample_euler_maruyama_pushforward(
                model,
                params,
                x,
                start,
                end,
                key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=n_steps,
            )
        return sample_composed_pushforward(
            model,
            params,
            x,
            start,
            end,
            key,
            n_coefficients=int(cfg.ssfm.n_coefficients),
            n_steps=n_steps,
        )

    interval_emd = {
        (source_time, target_time): []
        for source_time, target_time in zip(retained[:-1], retained[1:])
    }
    rollout_emd = []
    for draw in range(n_draws):
        rollout = sources[retained[0]]
        final_interval_emd = None
        for interval_index, (source_time, target_time) in enumerate(
            zip(retained[:-1], retained[1:])
        ):
            prediction = pushforward(
                sources[source_time],
                source_time,
                target_time,
                _observed_draw_key(seed, draw, interval_index, protocol=0),
            )
            emd = wasserstein.exact_emd(prediction, targets[target_time])
            interval_emd[(source_time, target_time)].append(emd)
            final_interval_emd = emd
            if interval_index == 0:
                rollout = prediction
            else:
                rollout = pushforward(
                    rollout,
                    source_time,
                    target_time,
                    _observed_draw_key(seed, draw, interval_index, protocol=1),
                )

        if len(retained) == 2:
            rollout_emd.append(float(final_interval_emd))
        else:
            rollout_emd.append(
                wasserstein.exact_emd(rollout, targets[retained[-1]])
            )

    metrics: Dict[str, float] = {}
    interval_means = []
    for (source_time, target_time), values in interval_emd.items():
        transition_tag = (
            f"{_observed_time_tag(source_time)}_to_"
            f"{_observed_time_tag(target_time)}"
        )
        mean_emd = float(np.mean(values))
        std_emd = float(np.std(values))
        metrics[
            f"final_eval/observed_{transition_tag}_{sampler}_emd"
        ] = mean_emd
        metrics[
            f"final_eval/observed_{transition_tag}_{sampler}_emd_std_over_noise"
        ] = std_emd
        metrics[
            f"final_eval/observed_{transition_tag}_evaluation_emd"
        ] = mean_emd
        interval_means.append(mean_emd)
        print(
            "Observed stochastic EMD evaluation: "
            f"{source_time}->{target_time}, sampler={sampler}, "
            f"steps={n_steps}, source_n={sources[source_time].shape[0]}, "
            f"target_n={targets[target_time].shape[0]}, draws={n_draws}, "
            f"EMD={mean_emd:.8g}"
        )

    metrics[f"final_eval/observed_{sampler}_mean_emd"] = float(
        np.mean(interval_means)
    )
    metrics["final_eval/observed_evaluation_mean_emd"] = metrics[
        f"final_eval/observed_{sampler}_mean_emd"
    ]
    mean_rollout_emd = float(np.mean(rollout_emd))
    rollout_tag = (
        f"{_observed_time_tag(retained[0])}_to_"
        f"{_observed_time_tag(retained[-1])}"
    )
    metrics[
        f"final_eval/{rollout_tag}_rollout_{sampler}_emd"
    ] = mean_rollout_emd
    metrics[
        f"final_eval/{rollout_tag}_rollout_{sampler}_emd_std_over_noise"
    ] = float(np.std(rollout_emd))
    metrics[
        f"final_eval/{rollout_tag}_rollout_evaluation_emd"
    ] = mean_rollout_emd
    print(
        "Source-to-final stochastic EMD evaluation: "
        f"{retained[0]}->{retained[-1]}, sampler={sampler}, "
        f"source_n={rollout.shape[0]}, target_n={targets[retained[-1]].shape[0]}, "
        f"draws={n_draws}, EMD={mean_rollout_emd:.8g}"
    )
    return metrics


def pushforward_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
    max_points: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, ...]]:
    """Sample plotting populations without running EMD or MMD calculations."""
    seed = int(cfg.evaluation.seed if seed is None else seed)
    max_points = int(cfg.logging.maizels.plot_bs if max_points is None else max_points)
    use_euler_maruyama = uses_euler_maruyama_evaluation(cfg)
    flowmap_n_steps = int(cfg.evaluation.flowmap_n_steps)
    if max_points < 0:
        raise ValueError("Plot max_points must be non-negative (zero means all).")
    if not use_euler_maruyama and flowmap_n_steps <= 0:
        raise ValueError("evaluation.flowmap_n_steps must be positive.")

    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    plot_data = {}
    for timepoint in _heldout_timepoints(cfg):
        source_time, _ = _interval_source(cfg, timepoint)
        source_all = np.asarray(pools[source_time]["x"], dtype=np.float32)
        target_all = np.asarray(pools[timepoint]["x"], dtype=np.float32)
        source = source_all[_sample_indices(rng, source_all.shape[0], max_points)]
        target = target_all[_sample_indices(rng, target_all.shape[0], max_points)]
        start = maizels.normalized_time(source_time, cfg)
        end = maizels.normalized_time(timepoint, cfg)
        if use_euler_maruyama:
            key, rollout_key = jax.random.split(key)
            rollout = sample_euler_maruyama_pushforward(
                model,
                params,
                source,
                start,
                end,
                rollout_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=euler_maruyama_n_steps(cfg),
            )
            plot_data[timepoint] = (target, rollout)
        else:
            key, direct_key, flowmap_key = jax.random.split(key, 3)
            direct = sample_pushforward(
                model,
                params,
                source,
                start,
                end,
                direct_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
            )
            flowmap = sample_composed_pushforward(
                model,
                params,
                source,
                start,
                end,
                flowmap_key,
                n_coefficients=int(cfg.ssfm.n_coefficients),
                n_steps=flowmap_n_steps,
            )
            plot_data[timepoint] = (target, direct, flowmap)
    return plot_data


def lineage_metrics(
    model, params, cfg, *, seed: Optional[int] = None
) -> Dict[str, float]:
    """Score valid sampled paths from held-out D3 cells with both classifiers."""
    seed = int(cfg.evaluation.seed + 211 if seed is None else seed)
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source = pools[str(cfg.problem.source_time)]
    source_x_all = np.asarray(source["holdout_x"], dtype=np.float32)
    source_types_all = np.asarray(source["holdout_types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(
        rng,
        source_x_all.shape[0],
        int(cfg.evaluation.lineage_max_source_points),
    )
    source_x = source_x_all[indices]
    if source_x.shape[0] == 0:
        return {}
    type_to_id = maizels.class_to_id_map(maizels.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types_all[indices]],
        dtype=np.int32,
    )
    key = jax.random.PRNGKey(seed)
    classifiers = {
        "schedule_classifier": Path(cfg.problem.classifier_path),
        "full_data_classifier": Path(cfg.logging.maizels.full_data_classifier_path),
    }
    classifiers = {
        name: path if path.suffix == ".npz" else path.with_suffix(".npz")
        for name, path in classifiers.items()
    }
    available_classifiers = {}
    for name, path in classifiers.items():
        if path.is_file():
            available_classifiers[name] = path
        else:
            print(
                f"Skipping {name} stochastic lineage metrics because the "
                f"NumPy classifier checkpoint is absent: {path}"
            )
    if not available_classifiers:
        return {}

    n_steps = int(cfg.evaluation.lineage_n_steps)
    use_euler_maruyama = uses_euler_maruyama_evaluation(cfg)
    if not use_euler_maruyama and n_steps <= 0:
        raise ValueError("evaluation.lineage_n_steps must be positive.")
    sampler_tag = "euler_maruyama" if use_euler_maruyama else "flowmap"
    valid_fractions = {name: [] for name in available_classifiers}
    for _ in range(int(cfg.evaluation.n_noise_draws)):
        key, draw_key = jax.random.split(key)
        path = sample_evaluation_path(
            model,
            params,
            source_x,
            maizels.normalized_time(cfg.problem.source_time, cfg),
            maizels.normalized_time(cfg.problem.target_time, cfg),
            draw_key,
            cfg,
            n_steps=n_steps,
        )
        for name, classifier_path in available_classifiers.items():
            validity = maizels.check_paths_with_classifier(
                path,
                source_ids,
                classifier_path,
                prob_threshold=float(cfg.problem.classifier_prob_threshold),
                margin_threshold=float(cfg.problem.classifier_margin_threshold),
                classifier_batch_size=int(cfg.problem.classifier_batch_size),
                lineage_transition_mode=maizels.lineage_transition_mode_from_config(
                    cfg
                ),
            )
            valid_fractions[name].append(float(np.mean(validity["valid"])))

    metrics = {"final_eval/lineage_eval_source_count": float(source_x.shape[0])}
    for name, values in valid_fractions.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        prefix = f"final_eval/{name}"
        metrics[f"{prefix}/stochastic_path_valid_fraction"] = mean
        metrics[f"{prefix}/stochastic_path_valid_fraction_std_over_noise"] = std
        metrics[f"{prefix}/{sampler_tag}_valid_trajectory_pct"] = 100.0 * mean
        metrics[f"{prefix}/{sampler_tag}_invalid_trajectory_pct"] = 100.0 * (1.0 - mean)
        metrics[f"{prefix}/{sampler_tag}_valid_trajectory_pct_std_over_noise"] = (
            100.0 * std
        )

    # Match the deterministic metric convention: unprefixed validity uses the
    # schedule-specific classifier; the all-day classifier has its own prefix.
    if "schedule_classifier" in valid_fractions:
        metrics["final_eval/stochastic_path_valid_fraction"] = metrics[
            "final_eval/schedule_classifier/stochastic_path_valid_fraction"
        ]
        metrics["final_eval/stochastic_path_valid_fraction_std_over_noise"] = metrics[
            "final_eval/schedule_classifier/stochastic_path_valid_fraction_std_over_noise"
        ]
        metrics[f"final_eval/{sampler_tag}_valid_trajectory_pct"] = metrics[
            f"final_eval/schedule_classifier/{sampler_tag}_valid_trajectory_pct"
        ]
        metrics[f"final_eval/{sampler_tag}_invalid_trajectory_pct"] = metrics[
            f"final_eval/schedule_classifier/{sampler_tag}_invalid_trajectory_pct"
        ]
    return metrics


def full_data_trajectory_plot_data(
    model,
    params,
    cfg,
    *,
    seed: Optional[int] = None,
):
    """Generate held-out D3 trajectories and full-classifier validity labels."""
    configured = Path(cfg.logging.maizels.full_data_classifier_path)
    classifier_path = (
        configured if configured.suffix == ".npz" else configured.with_suffix(".npz")
    )
    if not classifier_path.is_file():
        print(
            "Skipping stochastic validity trajectory plot because the full-data "
            f"NumPy classifier checkpoint is absent: {classifier_path}"
        )
        return None

    seed = int(cfg.evaluation.seed + 503 if seed is None else seed)
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    source_pool = pools[str(cfg.problem.source_time)]
    target_pool = pools[str(cfg.problem.target_time)]
    source_all = np.asarray(source_pool["holdout_x"], dtype=np.float32)
    source_types_all = np.asarray(source_pool["holdout_types"])
    rng = np.random.default_rng(seed)
    indices = _sample_indices(
        rng,
        source_all.shape[0],
        int(cfg.evaluation.lineage_max_source_points),
    )
    source = source_all[indices]
    if source.shape[0] == 0:
        return None
    target = np.asarray(target_pool["holdout_x"], dtype=np.float32)
    type_to_id = maizels.class_to_id_map(maizels.CLASS_NAMES)
    source_ids = np.asarray(
        [type_to_id[str(value)] for value in source_types_all[indices]],
        dtype=np.int32,
    )
    use_euler_maruyama = uses_euler_maruyama_evaluation(cfg)
    generated = sample_evaluation_path(
        model,
        params,
        source,
        maizels.normalized_time(cfg.problem.source_time, cfg),
        maizels.normalized_time(cfg.problem.target_time, cfg),
        jax.random.PRNGKey(seed),
        cfg,
        n_steps=int(cfg.evaluation.lineage_n_steps),
    )
    validity = maizels.check_paths_with_classifier(
        generated,
        source_ids,
        classifier_path,
        prob_threshold=float(cfg.problem.classifier_prob_threshold),
        margin_threshold=float(cfg.problem.classifier_margin_threshold),
        classifier_batch_size=int(cfg.problem.classifier_batch_size),
        lineage_transition_mode=maizels.lineage_transition_mode_from_config(cfg),
    )
    paths = np.concatenate([source[:, None, :], generated], axis=1)
    return {
        "paths": paths,
        "valid": np.asarray(validity["valid"], dtype=bool),
        "source": source,
        "target": target,
        "sampler_label": (
            "Euler--Maruyama rollout"
            if use_euler_maruyama
            else "Composed SSFM trajectories"
        ),
    }


def save_full_data_trajectory_plot(plot_data, output_path: Path) -> None:
    """Plot stochastic trajectories by full-data-classifier lineage validity."""
    if plot_data is None:
        return
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    paths = np.asarray(plot_data["paths"], dtype=np.float32)
    valid = np.asarray(plot_data["valid"], dtype=bool)
    source = np.asarray(plot_data["source"], dtype=np.float32)
    target = np.asarray(plot_data["target"], dtype=np.float32)

    def segments(selected):
        selected_paths = paths[selected, :, :2]
        if selected_paths.shape[0] == 0:
            return np.empty((0, 2, 2), dtype=np.float32)
        return np.stack(
            [selected_paths[:, :-1], selected_paths[:, 1:]], axis=2
        ).reshape((-1, 2, 2))

    figure, axis = plt.subplots(figsize=(7.0, 6.0))
    axis.scatter(
        target[:, 0],
        target[:, 1],
        s=5,
        alpha=0.18,
        color="tab:blue",
        label="held-out D8",
    )
    valid_segments = segments(valid)
    invalid_segments = segments(~valid)
    if valid_segments.shape[0]:
        axis.add_collection(
            LineCollection(
                valid_segments,
                colors="black",
                linewidths=0.4,
                alpha=0.25,
                label="valid",
            )
        )
    if invalid_segments.shape[0]:
        axis.add_collection(
            LineCollection(
                invalid_segments,
                colors="crimson",
                linewidths=0.8,
                alpha=0.7,
                label="invalid",
            )
        )
    axis.scatter(
        source[:, 0],
        source[:, 1],
        s=8,
        alpha=0.6,
        color="tab:green",
        label="held-out D3",
    )
    axis.autoscale_view()
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_title(
        f"{plot_data.get('sampler_label', 'Composed SSFM trajectories')}: "
        f"{100.0 * float(np.mean(valid)):.1f}% lineage-valid\n"
        "(full-data classifier)"
    )
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_pushforward_plot(
    plot_data: Dict[str, Tuple[np.ndarray, ...]],
    output_path: Path,
    *,
    flowmap_n_steps: int,
    euler_maruyama: Optional[bool] = None,
    euler_maruyama_n_steps: int = 50,
) -> None:
    """Save held-out populations with only samplers valid for the model."""
    if not plot_data:
        return
    import matplotlib.pyplot as plt

    timepoints = list(plot_data)
    n_columns = len(plot_data[timepoints[0]])
    if n_columns not in (2, 3):
        raise ValueError(f"Expected two or three plot populations, got {n_columns}.")
    if any(len(plot_data[timepoint]) != n_columns for timepoint in timepoints):
        raise ValueError("Every held-out timepoint must use the same plot samplers.")
    if euler_maruyama is not None and bool(euler_maruyama) != (n_columns == 2):
        raise ValueError("Plot populations do not match the configured sampler mode.")
    figure, axes = plt.subplots(
        len(timepoints),
        n_columns,
        figsize=(4.0 * n_columns, max(2.6 * len(timepoints), 3.0)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, timepoint in enumerate(timepoints):
        populations = plot_data[timepoint]
        for column, population in enumerate(populations):
            axes[row, column].scatter(
                population[:, 0], population[:, 1], s=3, alpha=0.35
            )
        axes[row, 0].set_ylabel(timepoint)
        axes[row, 0].set_title("Actual" if row == 0 else "")
        if n_columns == 2:
            axes[row, 1].set_title(
                f"Euler--Maruyama rollout ({int(euler_maruyama_n_steps)} steps)"
                if row == 0
                else ""
            )
        else:
            axes[row, 1].set_title("Direct SSFM" if row == 0 else "")
            axes[row, 2].set_title(
                f"Composed SSFM ({flowmap_n_steps} steps)" if row == 0 else ""
            )
    for axis in axes[-1]:
        axis.set_xlabel("PC1")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def final_evaluation(model, params, cfg, output_dir: Path) -> Dict[str, float]:
    """Run distribution and lineage evaluation for one checkpoint."""
    distribution, plot_data = distribution_metrics(model, params, cfg)
    metrics = dict(distribution)
    metrics.update(observed_distribution_metrics(model, params, cfg))
    metrics.update(lineage_metrics(model, params, cfg))
    if bool(cfg.evaluation.save_plot):
        save_pushforward_plot(
            plot_data,
            Path(output_dir) / "heldout_pushforwards.png",
            flowmap_n_steps=int(cfg.evaluation.flowmap_n_steps),
            euler_maruyama=uses_euler_maruyama_evaluation(cfg),
            euler_maruyama_n_steps=euler_maruyama_n_steps(cfg),
        )
        trajectory_data = full_data_trajectory_plot_data(model, params, cfg)
        if trajectory_data is not None:
            save_full_data_trajectory_plot(
                trajectory_data,
                Path(output_dir) / "heldout_d3_trajectory_validity.png",
            )
    return metrics

#!/usr/bin/env python3
"""Train direct strong stochastic flow maps on the Maizels PCA50 data.

This launcher is intentionally independent of ``launchers/learn.py``.  Slurm
IDs select plain SSFM, endpoint-filtered SSFM, or endpoint-filtered SSFM with
the differentiable lineage constraint.
"""

from __future__ import annotations

# isort: off
import os
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PY_DIR = SCRIPT_DIR.parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

# Keep TensorFlow (if imported indirectly elsewhere) away from the accelerator.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# isort: on

import argparse
import importlib
import inspect
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm.auto import tqdm

from common import maizels
from common import maizels_stochastic_eval
from common import maizels_stochastic_training
from common import stochastic_flow_map


class SSFMTrainState(train_state.TrainState):
    """Optimizer state; EMA parameters are kept as a separate JAX pytree."""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a direct strong stochastic flow map on Maizels PCA50."
    )
    parser.add_argument(
        "--cfg-path", "--cfg_path", default="configs.maizels_stochastic"
    )
    parser.add_argument("--slurm-id", "--slurm_id", type=int, required=True)
    parser.add_argument("--dataset-location", "--dataset_location", default="")
    parser.add_argument("--output-folder", "--output_folder", default="")
    parser.add_argument("--classifier-path", "--classifier_path", default=None)
    parser.add_argument(
        "--maizels-schedule",
        "--maizels_schedule",
        choices=("d3_d8", "d3_d3p8_d8"),
        default="d3_d3p8_d8",
    )
    parser.add_argument(
        "--maizels-time-mode",
        "--maizels_time_mode",
        choices=("real_time", "equal_time"),
        default="real_time",
    )
    parser.add_argument("--hparam-val-times", "--hparam_val_times", default="D3.4,D6")
    parser.add_argument("--learning-rate", "--learning_rate", type=float)
    parser.add_argument(
        "--diffusion-scale", "--diffusion_scale", type=float, default=None
    )
    parser.add_argument("--gamma-scale", "--gamma_scale", type=float, default=None)
    parser.add_argument(
        "--constraint-weight", "--constraint_weight", type=float, default=None
    )
    parser.add_argument(
        "--entropy-weight", "--entropy_weight", type=float, default=None
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-steps", "--total_steps", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--n-pairs", "--n_pairs", type=int, default=None)
    parser.add_argument(
        "--validation-frequency",
        "--validation_frequency",
        "--eval-frequency",
        "--eval_frequency",
        dest="validation_frequency",
        type=int,
        default=None,
        help="Steps between held-out objective checks (default: 100).",
    )
    parser.add_argument(
        "--eval-noise-draws", "--eval_noise_draws", type=int, default=None
    )
    parser.add_argument(
        "--eval-max-points", "--eval_max_points", type=int, default=None
    )
    parser.add_argument(
        "--eval-flowmap-steps",
        "--eval_flowmap_steps",
        type=int,
        default=None,
        help="Number of maps composed for final flow-map evaluation (default: 50).",
    )
    parser.add_argument(
        "--visual-frequency",
        "--visual_frequency",
        type=int,
        default=None,
        help="Training steps between pushforward plots; zero disables them.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        "--early_stopping_patience",
        type=int,
        default=None,
    )
    parser.add_argument("--final-metrics-path", "--final_metrics_path", default=None)
    parser.add_argument(
        "--wandb-mode",
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    return parser.parse_args(argv)


def _build_config(args: argparse.Namespace):
    module = importlib.import_module(args.cfg_path)
    arguments = {
        "dataset_location": args.dataset_location,
        "output_folder": args.output_folder,
        "classifier_path": args.classifier_path,
        "maizels_schedule": args.maizels_schedule,
        "maizels_time_mode": args.maizels_time_mode,
        "hparam_val_times": args.hparam_val_times,
        "learning_rate": args.learning_rate,
        "constraint_weight": args.constraint_weight,
        "entropy_weight": args.entropy_weight,
        "diffusion_scale": args.diffusion_scale,
        "gamma_scale": args.gamma_scale,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "n_pairs": args.n_pairs,
    }
    supported = inspect.signature(module.get_config).parameters
    kwargs = {
        key: value
        for key, value in arguments.items()
        if value is not None and key in supported
    }
    cfg = module.get_config(args.slurm_id, **kwargs)
    if args.validation_frequency is not None:
        if args.validation_frequency <= 0:
            raise ValueError("validation_frequency must be positive.")
        cfg.optimization.early_stopping.check_freq = int(args.validation_frequency)
    if args.eval_noise_draws is not None:
        if args.eval_noise_draws <= 0:
            raise ValueError("eval_noise_draws must be positive.")
        cfg.evaluation.n_noise_draws = int(args.eval_noise_draws)
    if args.eval_max_points is not None:
        if args.eval_max_points < 0:
            raise ValueError("eval_max_points must be non-negative (zero means all).")
        cfg.evaluation.max_source_points = int(args.eval_max_points)
        cfg.evaluation.max_target_points = int(args.eval_max_points)
    if args.eval_flowmap_steps is not None:
        if args.eval_flowmap_steps <= 0:
            raise ValueError("eval_flowmap_steps must be positive.")
        cfg.evaluation.flowmap_n_steps = int(args.eval_flowmap_steps)
    if args.visual_frequency is not None:
        if args.visual_frequency < 0:
            raise ValueError("visual_frequency must be non-negative.")
        cfg.logging.visual_freq = int(args.visual_frequency)
    return cfg


def _ensure_pair_time_bounds(pairs: Dict[str, np.ndarray], cfg):
    labels = np.asarray(pairs["label"])
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("Maizels SSFM pairs require source and target type IDs.")
    if labels.shape[1] >= 4:
        return pairs
    bounds = np.tile(
        np.asarray(
            [
                maizels.normalized_time(cfg.problem.source_time, cfg),
                maizels.normalized_time(cfg.problem.target_time, cfg),
            ],
            dtype=np.float32,
        ),
        (labels.shape[0], 1),
    )
    result = dict(pairs)
    result["label"] = np.concatenate([labels.astype(np.float32), bounds], axis=1)
    return result


def _feature_scale(arrays) -> np.ndarray:
    arrays = tuple(np.asarray(values, dtype=np.float32) for values in arrays)
    if not arrays or any(values.ndim != 2 for values in arrays):
        raise ValueError("Feature-scale inputs must be non-empty matrices.")
    count = float(sum(values.shape[0] for values in arrays))
    total = np.zeros((arrays[0].shape[1],), dtype=np.float64)
    total_square = np.zeros_like(total)
    # Chunked accumulation avoids making two additional float64 copies of the
    # full PCA50 population.
    for values in arrays:
        for start in range(0, values.shape[0], 50_000):
            chunk = values[start : start + 50_000]
            total += np.sum(chunk, axis=0, dtype=np.float64)
            total_square += np.einsum("ij,ij->j", chunk, chunk, dtype=np.float64)
    mean = total / count
    second = total_square / count
    return np.sqrt(np.maximum(second - mean * mean, 1e-8)).astype(np.float32)


def _training_population_feature_scale(cfg) -> np.ndarray:
    """Use the same unique retained cells to calibrate all three variants."""
    pools = maizels.timepoint_pool_splits(
        cfg, dataset_location=cfg.problem.dataset_location
    )
    return _feature_scale(
        pools[str(timepoint)]["train_x"]
        for timepoint in cfg.problem.retained_timepoints
    )


def _learning_rate_schedule(cfg):
    learning_rate = float(cfg.optimization.learning_rate)
    warmup = max(1, int(cfg.optimization.warmup_steps))
    total = max(warmup + 1, int(cfg.optimization.total_steps))

    def schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_factor = jnp.minimum((step + 1.0) / float(warmup), 1.0)
        decay_progress = jnp.clip(
            (step - float(warmup)) / float(max(total - warmup, 1)), 0.0, 1.0
        )
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_progress))
        return learning_rate * warmup_factor * cosine

    return schedule


def _create_model_and_state(cfg, pc_scale: np.ndarray):
    data_dim = int(cfg.problem.d)
    rescale = float(np.sqrt(np.mean(np.square(pc_scale))))
    model = stochastic_flow_map.StrongStochasticFlowMapMLP(
        data_dim=data_dim,
        n_coefficients=int(cfg.ssfm.n_coefficients),
        hidden_dim=int(cfg.ssfm.hidden_dim),
        n_hidden=int(cfg.ssfm.n_hidden),
        rescale=rescale,
        uncertainty_hidden_dim=int(cfg.ssfm.uncertainty_hidden_dim),
    )
    init_key = jax.random.PRNGKey(int(cfg.training.seed))
    dummy_n = 2
    variables = model.init(
        init_key,
        jnp.zeros((dummy_n,), dtype=jnp.float32),
        jnp.ones((dummy_n,), dtype=jnp.float32),
        jnp.zeros((dummy_n, data_dim), dtype=jnp.float32),
        jnp.zeros(
            (dummy_n, int(cfg.ssfm.n_coefficients), data_dim),
            dtype=jnp.float32,
        ),
    )
    schedule = _learning_rate_schedule(cfg)
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.optimization.clip)),
        optax.adamw(
            learning_rate=schedule,
            b1=float(cfg.optimization.b1),
            b2=float(cfg.optimization.b2),
            weight_decay=float(cfg.optimization.weight_decay),
        ),
    )
    state = SSFMTrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )
    return model, state, variables["params"], schedule


def _make_objective_steps(model, cfg, pc_scale, classifier):
    loss_fn = maizels_stochastic_training.make_loss_fn(
        model,
        cfg,
        jnp.asarray(pc_scale),
        classifier=classifier,
    )
    ema_decay = float(cfg.ssfm.ema_decay)

    @jax.jit
    def train_step(state, ema_params, batch, key):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, ema_params, batch, key, state.step
        )
        state = state.apply_gradients(grads=grads)
        ema_params = stochastic_flow_map.ema_update(state.params, ema_params, ema_decay)
        grad_norm = optax.global_norm(grads)
        metrics = dict(metrics)
        metrics["training/grad_norm"] = grad_norm
        return state, ema_params, metrics

    @jax.jit
    def validation_step(params, ema_params, batch, key, step):
        _, metrics = loss_fn(params, ema_params, batch, key, step)
        return metrics

    return train_step, validation_step


def _sample_batch(
    training_data: Dict[str, object],
    rng: np.random.Generator,
    batch_size: int,
    cfg,
) -> Dict[str, jnp.ndarray]:
    if maizels.uses_minibatch_ot(cfg):
        paired, _ = maizels.couple_minibatch_ot_timepoint_pools(
            cfg,
            training_data,
            int(batch_size),
            seed=int(rng.integers(0, np.iinfo(np.int64).max)),
            pair_mode=str(cfg.problem.maizels_pair_mode),
        )
        paired = _ensure_pair_time_bounds(paired, cfg)
        return {
            key: jnp.asarray(np.asarray(value))
            for key, value in paired.items()
            if key in ("x0", "x1", "label")
        }

    indices = rng.integers(0, training_data["x0"].shape[0], size=batch_size)
    return {
        key: jnp.asarray(np.asarray(value)[indices])
        for key, value in training_data.items()
        if key in ("x0", "x1", "label")
    }


def _checkpoint_payload(state, ema_params, cfg, pc_scale, best_metric, best_step):
    return {
        "state": state,
        "ema_params": ema_params,
        "pc_scale": jnp.asarray(pc_scale),
        "best_validation_loss": jnp.asarray(best_metric),
        "best_step": jnp.asarray(best_step, dtype=jnp.int32),
        "config": cfg.to_dict(),
    }


def _save_checkpoint(
    path: Path,
    state,
    ema_params,
    cfg,
    pc_scale,
    best_metric: float,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = jax.device_get(
        _checkpoint_payload(state, ema_params, cfg, pc_scale, best_metric, best_step)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(flax.serialization.to_bytes(payload))
    os.replace(temporary, path)


def _json_scalars(values: Dict[str, object]) -> Dict[str, float]:
    result = {}
    for key, value in values.items():
        array = np.asarray(jax.device_get(value))
        if array.size == 1:
            result[key] = float(array.reshape(()))
    return result


def _wandb_setup(cfg, mode: Optional[str]):
    try:
        import wandb
    except ModuleNotFoundError:
        return None
    kwargs = {
        "project": str(cfg.logging.wandb_project),
        "entity": str(cfg.logging.wandb_entity),
        "name": str(cfg.logging.wandb_name),
        "config": cfg.to_dict(),
    }
    if mode is not None:
        kwargs["mode"] = mode
    return wandb.init(**kwargs)


def _save_periodic_pushforward_plot(model, params, cfg, run_dir, step, wandb_run):
    """Save fixed-seed population and lineage plots for a training checkpoint."""
    plot_data = maizels_stochastic_eval.pushforward_plot_data(model, params, cfg)
    plot_path = (
        Path(run_dir) / "plots" / f"heldout_pushforwards_step_{int(step):07d}.png"
    )
    maizels_stochastic_eval.save_pushforward_plot(
        plot_data,
        plot_path,
        flowmap_n_steps=int(cfg.evaluation.flowmap_n_steps),
    )
    saved_paths = [plot_path]
    trajectory_data = maizels_stochastic_eval.full_data_trajectory_plot_data(
        model, params, cfg
    )
    trajectory_path = (
        Path(run_dir)
        / "plots"
        / f"heldout_d3_trajectory_validity_step_{int(step):07d}.png"
    )
    if trajectory_data is not None:
        maizels_stochastic_eval.save_full_data_trajectory_plot(
            trajectory_data, trajectory_path
        )
        saved_paths.append(trajectory_path)
    if wandb_run is not None:
        import wandb

        payload = {"visualization/heldout_pushforwards": wandb.Image(str(plot_path))}
        if trajectory_data is not None:
            payload["visualization/heldout_d3_trajectory_validity"] = wandb.Image(
                str(trajectory_path)
            )
        wandb_run.log(payload, step=int(step))
    return tuple(saved_paths)


def _make_validation_batch(cfg):
    """Construct the fixed held-out pair batch used by early stopping."""
    validation_cfg = cfg.logging.maizels
    pair_mode = str(validation_cfg.validation_pair_mode)
    if pair_mode == "same_as_training":
        pair_mode = str(cfg.problem.maizels_pair_mode)
    pairs, stats = maizels.make_validation_pair_pool(
        cfg,
        max(2, int(validation_cfg.validation_bs)),
        dataset_location=cfg.problem.dataset_location,
        pair_mode=pair_mode,
        seed=int(validation_cfg.validation_seed),
    )
    pairs = _ensure_pair_time_bounds(pairs, cfg)
    batch = {
        key: jnp.asarray(np.asarray(value))
        for key, value in pairs.items()
        if key in ("x0", "x1", "label")
    }
    return batch, stats


def _validation_loss_metrics(
    validation_step, params, ema_params, batch, key, step: int
):
    """Evaluate exactly the training objective on fixed held-out pairs/noise."""
    raw_metrics = validation_step(
        params,
        ema_params,
        batch,
        key,
        jnp.asarray(step, dtype=jnp.int32),
    )
    scalars = _json_scalars(raw_metrics)
    metrics = {f"validation/{name}": value for name, value in scalars.items()}
    metrics["validation_loss"] = scalars["loss"]
    return metrics, metrics["validation_loss"]


def train(
    cfg, final_metrics_path: Optional[str] = None, wandb_mode=None
) -> Dict[str, float]:
    """Train, select by held-out objective loss, then run final evaluation."""
    print("Loading Maizels stochastic training pairs.")
    dynamic_minibatch_ot = maizels.uses_minibatch_ot(cfg)
    if dynamic_minibatch_ot:
        training_data, pair_stats = maizels.make_minibatch_ot_training_pools(
            cfg, dataset_location=cfg.problem.dataset_location
        )
        loaded_description = (
            f"{int(training_data['stored_endpoint_cells']):,} unique endpoint cells"
            if "stored_endpoint_cells" in training_data
            else f"{int(pair_stats['stored_endpoint_cells']):,} unique endpoint cells"
        )
    else:
        training_data, pair_stats = maizels.make_pair_pool(
            cfg, dataset_location=cfg.problem.dataset_location
        )
        training_data = _ensure_pair_time_bounds(training_data, cfg)
        loaded_description = f"{training_data['x0'].shape[0]:,} pairs"
    pc_scale = _training_population_feature_scale(cfg)
    print(
        f"Loaded {loaded_description} in {int(cfg.problem.d)} dimensions; "
        f"mode={cfg.problem.maizels_pair_mode}, "
        f"coupling={'minibatch_ot' if dynamic_minibatch_ot else 'precomputed'}, "
        f"retained={list(cfg.problem.retained_timepoints)}."
    )
    print("Pair construction: " + json.dumps(pair_stats, default=str))
    validation_batch, validation_stats = _make_validation_batch(cfg)
    print(
        "Loaded fixed held-out validation pairs: "
        f"n={validation_batch['x0'].shape[0]}, "
        f"mode={cfg.problem.maizels_pair_mode}."
    )
    print("Validation pair construction: " + json.dumps(validation_stats, default=str))

    classifier = None
    if bool(cfg.constraints.enabled):
        classifier = maizels_stochastic_training.setup_lineage_classifier(cfg)
        print(
            f"Loaded differentiable lineage classifier: {cfg.problem.classifier_path}"
        )

    model, state, ema_params, schedule = _create_model_and_state(cfg, pc_scale)
    train_step, validation_step = _make_objective_steps(
        model, cfg, pc_scale, classifier
    )
    n_parameters = sum(
        int(value.size) for value in jax.tree_util.tree_leaves(state.params)
    )
    print(
        f"Initialized direct SSFM with {n_parameters:,} parameters on "
        f"{jax.device_count()} JAX device(s); diffusion_scale="
        f"{float(cfg.ssfm.diffusion_scale):g}."
    )

    output_root = Path(cfg.logging.output_folder or "outputs/maizels_stochastic")
    run_dir = output_root.expanduser().resolve() / str(cfg.logging.output_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str) + "\n"
    )
    wandb_run = _wandb_setup(cfg, wandb_mode)

    numpy_rng = np.random.default_rng(int(cfg.training.seed) + 101)
    key = jax.random.PRNGKey(int(cfg.training.seed) + 211)
    total_steps = int(cfg.optimization.total_steps)
    scalar_freq = max(1, int(cfg.logging.scalar_freq))
    save_freq = max(1, int(cfg.logging.save_freq))
    visual_freq = int(cfg.logging.visual_freq)
    early_cfg = cfg.optimization.early_stopping
    validation_frequency = max(1, int(early_cfg.check_freq))
    patience = int(early_cfg.patience)
    min_delta = float(early_cfg.min_delta)
    warmup_steps = int(early_cfg.warmup_steps)
    validation_key = jax.random.PRNGKey(int(cfg.logging.maizels.validation_seed) + 31)
    best_metric = math.inf
    best_step = 0
    best_params = None
    checks_without_improvement = 0
    started = time.monotonic()

    progress = tqdm(range(total_steps), desc="Maizels SSFM")
    try:
        for step_index in progress:
            batch = _sample_batch(
                training_data,
                numpy_rng,
                int(cfg.optimization.bs),
                cfg,
            )
            key, step_key = jax.random.split(key)
            state, ema_params, step_metrics = train_step(
                state, ema_params, batch, step_key
            )
            step = step_index + 1

            if step == 1 or step % scalar_freq == 0:
                scalars = _json_scalars(step_metrics)
                scalars["training/learning_rate"] = float(
                    jax.device_get(schedule(state.step))
                )
                progress.set_postfix(loss=f"{scalars['loss']:.4g}")
                if wandb_run is not None:
                    wandb_run.log(scalars, step=step)

            if step % save_freq == 0:
                _save_checkpoint(
                    run_dir / "latest.msgpack",
                    state,
                    ema_params,
                    cfg,
                    pc_scale,
                    best_metric,
                    best_step,
                )

            if (
                bool(cfg.evaluation.save_plot)
                and visual_freq > 0
                and step % visual_freq == 0
            ):
                plot_paths = _save_periodic_pushforward_plot(
                    model,
                    ema_params,
                    cfg,
                    run_dir,
                    step,
                    wandb_run,
                )

            should_validate = step >= warmup_steps and (
                step % validation_frequency == 0 or step == total_steps
            )
            if should_validate:
                validation, objective = _validation_loss_metrics(
                    validation_step,
                    state.params,
                    ema_params,
                    validation_batch,
                    validation_key,
                    step,
                )
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)
                if objective < best_metric - min_delta:
                    best_metric = objective
                    best_step = step
                    best_params = jax.device_get(ema_params)
                    checks_without_improvement = 0
                    _save_checkpoint(
                        run_dir / "best.msgpack",
                        state,
                        ema_params,
                        cfg,
                        pc_scale,
                        best_metric,
                        best_step,
                    )
                else:
                    checks_without_improvement += 1
                if patience > 0 and checks_without_improvement >= patience:
                    print(
                        f"Early stopping at step {step}; best validation loss "
                        f"{best_metric:.6g} at step {best_step}."
                    )
                    break
    except KeyboardInterrupt:
        print("Interrupted; saving the latest stochastic checkpoint.")
        _save_checkpoint(
            run_dir / "latest.msgpack",
            state,
            ema_params,
            cfg,
            pc_scale,
            best_metric,
            best_step,
        )
        if wandb_run is not None:
            wandb_run.finish(exit_code=130)
        raise

    _save_checkpoint(
        run_dir / "latest.msgpack",
        state,
        ema_params,
        cfg,
        pc_scale,
        best_metric,
        best_step,
    )
    if best_params is None:
        validation, best_metric = _validation_loss_metrics(
            validation_step,
            state.params,
            ema_params,
            validation_batch,
            validation_key,
            int(state.step),
        )
        best_step = int(state.step)
        best_params = jax.device_get(ema_params)
        _save_checkpoint(
            run_dir / "best.msgpack",
            state,
            ema_params,
            cfg,
            pc_scale,
            best_metric,
            best_step,
        )

    print(f"Evaluating best stochastic checkpoint from step {best_step}.")
    final_metrics = maizels_stochastic_eval.final_evaluation(
        model, best_params, cfg, run_dir
    )
    final_metrics.update(
        {
            "training/best_step": float(best_step),
            "training/best_validation_loss": float(best_metric),
            "training/wall_seconds": float(time.monotonic() - started),
            "training/diffusion_scale": float(cfg.ssfm.diffusion_scale),
            "training/gamma_scale": float(cfg.ssfm.gamma_scale),
        }
    )
    metrics_path = (
        Path(final_metrics_path).expanduser().resolve()
        if final_metrics_path
        else run_dir / "final_metrics.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(final_metrics, indent=2, sort_keys=True) + "\n")
    if wandb_run is not None:
        wandb_run.log(final_metrics, step=int(state.step))
        wandb_run.summary["best_validation_loss"] = best_metric
        wandb_run.summary["best_step"] = best_step
        wandb_run.finish()
    print(f"Final stochastic metrics: {metrics_path}")
    return final_metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = _build_config(args)
    train(
        cfg,
        final_metrics_path=args.final_metrics_path,
        wandb_mode=args.wandb_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

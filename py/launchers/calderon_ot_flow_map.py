#!/usr/bin/env python3
"""Train/evaluate plain, filtered, or constrained Calderon OT flow maps.

Use --stage prepare to inspect populations without training, --smoke for a
short end-to-end check, and --stage evaluate to reload a saved best checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "py"))

import numpy as np

from common import calderon


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs.calderon_pca100")
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-output-dir", type=Path,
                        help="Separate result directory for --stage evaluate; preserves the original evaluation")
    parser.add_argument("--dataset-location")
    parser.add_argument("--classifier-manifest", help="Training tissue classifier for filtering/constraints")
    parser.add_argument("--classifier-export", help="Frozen JAX export of the training classifier")
    parser.add_argument("--interpolant-filter-backend", choices=("jax", "reference"),
                        help="Training filter implementation; filtered/constrained defaults use fused JAX")
    parser.add_argument("--evaluation-classifier-manifest", help="All-times reference classifier for violation evaluation")
    parser.add_argument("--evaluation-classifier-export", help="Frozen JAX export for the all-times logging classifier")
    parser.add_argument("--classifier-python", help="Python with the existing PyTorch classifier dependencies")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--constraint-weight", type=float,
                        help="Override the training/validation constraint weight in a constrained config")
    parser.add_argument("--flowmap-steps", type=int, help="Composed steps per W1 query/observed interval (default 50)")
    parser.add_argument("--checking-points", type=int,
                        help="Evenly spaced violation checks after t=0 through t=1; even count (default 50)")
    parser.add_argument("--eval-source-cap", type=int, help="Violation source cap (default 512); zero uses every held-out source")
    parser.add_argument("--marginal-cap", type=int, help="W1 samples per source/reference population (default 2048); zero means uncapped")
    parser.add_argument("--reduced-evaluation", action="store_true",
                        help="For a saved run: flowmap/Euler only, no MMD, and 2048 W1 cells per population")
    parser.add_argument("--validation-bs", type=int, help="Fixed Maizels validation-pair batch size (default 1024)")
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--prob-threshold", type=float)
    parser.add_argument("--margin-threshold", type=float)
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a saved native checkpoint; defaults to best.msgpack")
    parser.add_argument("--smoke", action="store_true", help="Two steps, small network, and capped evaluation")
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"),
                        help="W&B mode; new defaults use Maizels's normal online logging")
    return parser.parse_args()


def configure(args):
    import jax
    from ml_collections import ConfigDict
    if args.checkpoint is not None and args.stage != "evaluate":
        raise ValueError("--checkpoint applies to --stage evaluate.")
    if args.evaluation_output_dir is not None and args.stage != "evaluate":
        raise ValueError("--evaluation-output-dir applies to --stage evaluate.")
    if getattr(args, "reduced_evaluation", False) and args.stage != "evaluate":
        raise ValueError("--reduced-evaluation applies to --stage evaluate; new runs already use these defaults.")
    if args.stage == "evaluate":
        if args.output_dir is None:
            raise ValueError("--stage evaluate requires --output-dir from a saved run.")
        if args.evaluation_output_dir is not None and args.evaluation_output_dir.resolve() == args.output_dir.resolve():
            raise ValueError("--evaluation-output-dir must differ from the saved training directory.")
        if args.smoke or any(value is not None for value in
                             (args.dataset_location, args.classifier_manifest, args.classifier_export,
                              args.evaluation_classifier_manifest, args.seed,
                              args.steps, args.batch_size, args.learning_rate,
                              getattr(args, "interpolant_filter_backend", None),
                              getattr(args, "constraint_weight", None))):
            raise ValueError("Evaluation reloads the saved training config; training/data overrides are unavailable.")
        cfg = ConfigDict(json.loads((args.output_dir / "config.json").read_text()))
        cfg.logging.output_folder = str(args.output_dir.resolve())
    else:
        module = importlib.import_module(args.config)
        cfg = module.get_config(dataset_location=args.dataset_location or "",
                                output_folder=str(args.output_dir) if args.output_dir else "",
                                classifier_path=args.classifier_manifest or "",
                                classifier_python=args.classifier_python or "",
                                seed=0 if args.seed is None else args.seed,
                                evaluation_classifier_path=args.evaluation_classifier_manifest or "")
        if getattr(args, "constraint_weight", None) is not None:
            if not cfg.constraints.enabled:
                raise ValueError("--constraint-weight requires a constrained config, such as configs.calderon_constrained_pca100.")
            cfg.constraints.weight = args.constraint_weight
        if args.classifier_export:
            cfg.classifier.jax_export_path = str(Path(args.classifier_export).expanduser().resolve())
        if getattr(args, "interpolant_filter_backend", None) is not None:
            cfg.problem.interpolant_filter_backend = args.interpolant_filter_backend
        if args.smoke:
            cfg.optimization.total_steps = 2
            cfg.optimization.bs = 32
            cfg.optimization.early_stopping.check_freq = 1
            cfg.optimization.early_stopping.patience = 0
            cfg.network.n_hidden = 2
            cfg.network.n_neurons = 64
            cfg.logging.scalar_freq = 1
            cfg.evaluation.validation_batches = 1
            cfg.evaluation.validation_batch_size = 32
            cfg.logging.maizels.validation_bs = 32
            cfg.logging.visual_freq = 0
            cfg.evaluation.flowmap_n_steps = 8
            cfg.evaluation.checkpoint_times = [.25, .5, .75, 1.]
            cfg.evaluation.flow_batch_size = 32
            cfg.evaluation.source_max_points = 32
            cfg.evaluation.marginal_max_points = 64
            if args.output_dir is None:
                cfg.logging.output_folder += "_smoke"
        for name, value in (("total_steps", args.steps), ("bs", args.batch_size), ("learning_rate", args.learning_rate)):
            if value is not None:
                cfg.optimization[name] = value
        cfg.problem.ot_minibatch_size = cfg.optimization.bs
        cfg.optimization.total_samples = cfg.optimization.bs * cfg.optimization.total_steps
        cfg.optimization.decay_steps = cfg.optimization.total_steps
    if getattr(args, "reduced_evaluation", False):
        if not calderon.uses_maizels_protocol(cfg):
            raise ValueError("--reduced-evaluation requires a saved Maizels-protocol Calderon run.")
        from configs.calderon_pca100 import use_reduced_evaluation
        use_reduced_evaluation(cfg)
    if args.classifier_python:
        cfg.classifier.python = args.classifier_python
    if args.evaluation_classifier_export:
        cfg.classifier.evaluation_jax_export_path = str(Path(args.evaluation_classifier_export).expanduser().resolve())
    for name, value in (("flowmap_n_steps", args.flowmap_steps), ("source_max_points", args.eval_source_cap),
                        ("marginal_max_points", args.marginal_cap), ("validation_batches", args.validation_batches)):
        if value is not None:
            cfg.evaluation[name] = value
    if args.checking_points is not None:
        if args.checking_points < 2 or args.checking_points % 2:
            raise ValueError("--checking-points must be even and at least two so t=0.5 is checked.")
        cfg.evaluation.checkpoint_times = (np.arange(1, args.checking_points + 1) / args.checking_points).tolist()
    for name, value in (("prob_threshold", args.prob_threshold), ("margin_threshold", args.margin_threshold)):
        if value is not None:
            cfg.classifier[name] = value
    if calderon.uses_maizels_protocol(cfg):
        if args.validation_bs is not None:
            cfg.evaluation.validation_batch_size = args.validation_bs
        cfg.logging.maizels.validation_bs = cfg.evaluation.validation_batch_size
        cfg.logging.maizels.flowmap_n_steps = cfg.evaluation.flowmap_n_steps
        cfg.logging.maizels.distribution_eval_flowmap_n_steps = cfg.evaluation.flowmap_n_steps
        cfg.logging.maizels.check_n_times = len(cfg.evaluation.checkpoint_times)
        cfg.logging.maizels.trajectory_eval_source_max_points = cfg.evaluation.source_max_points
        cfg.logging.maizels.distribution_eval_points_per_time = cfg.evaluation.marginal_max_points
        cfg.logging.maizels.distribution_eval_source_max_points = cfg.evaluation.marginal_max_points
        del cfg.logging.maizels.prob_threshold
        del cfg.logging.maizels.margin_threshold
        cfg.logging.maizels.prob_threshold = cfg.classifier.prob_threshold
        cfg.logging.maizels.margin_threshold = cfg.classifier.margin_threshold
    cfg.training.ndevices = jax.device_count()
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.training.conditional:
        raise ValueError("This launcher trains a flow map unconditional on tissue labels.")
    time_aware = bool(cfg.problem.pair_time_bounds_in_label)
    if time_aware:
        if (len(calderon.training_windows(cfg)) < 3 or not cfg.problem.interp_uses_labels
                or cfg.problem.interp_type != "time_rescaled_linear"):
            raise ValueError("Multiple training marginals require time-rescaled interpolation and pair-time labels.")
    if cfg.constraints.enabled:
        if cfg.constraints.type != "maizels_lineage_path" or cfg.constraints.path_mode != "loss_points_nll":
            raise ValueError("Expected the native Maizels loss-point transition NLL constraint.")
        if not cfg.problem.interp_uses_labels or cfg.problem.maizels_pair_mode != "ot_endpoint_interpolant":
            raise ValueError("The constrained variant requires endpoint IDs and interpolant-filtered OT.")
        if (not math.isfinite(cfg.constraints.weight) or cfg.constraints.weight < 0
                or cfg.constraints.classifier_temperature != 1.):
            raise ValueError("Use a finite nonnegative constraint weight and the frozen ensemble's calibrated probabilities.")
        if (not calderon.uses_maizels_protocol(cfg)
                and cfg.evaluation.get("validation_objective", "") != "plain_ot_lsd_without_constraint"):
            raise ValueError("Select constrained checkpoints with the shared plain-OT LSD validation objective.")
    elif cfg.problem.interp_uses_labels and not time_aware:
        raise ValueError("Endpoint loss labels are enabled only for the constrained variant.")
    if cfg.training.loss_type != "lsd" or not 0 < cfg.optimization.diag_fraction < 1:
        raise ValueError("This launcher expects the Maizels LSD flow-map objective.")
    if cfg.problem.maizels_pair_mode not in ("ot_plain", "ot_endpoint_interpolant") or cfg.problem.maizels_ot_coupling != "minibatch_ot":
        raise ValueError("Expected plain or endpoint/interpolant-filtered dynamic minibatch OT.")
    if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
        if cfg.problem.n_interpolant_check_times < 1 or cfg.problem.classifier_batch_size < 1:
            raise ValueError("Training interpolant checks and inference batch size must be positive.")
        chunk = cfg.problem.get("interpolant_check_chunk_size", cfg.problem.n_interpolant_check_times)
        if not 1 <= chunk <= cfg.problem.n_interpolant_check_times:
            raise ValueError("Interpolant check chunk size must lie between one and the full check count.")
        backend = cfg.problem.get("interpolant_filter_backend", "reference")
        if backend not in ("jax", "reference"):
            raise ValueError("Interpolant filter backend must be jax or reference.")
        if backend == "jax":
            if chunk != cfg.problem.n_interpolant_check_times:
                raise ValueError("The JAX filter uses the full check grid; use the reference backend for early rejection.")
            if not 0 < cfg.problem.get("gpu_filter_decision_tolerance", 1e-4) <= 1:
                raise ValueError("GPU filter decision tolerance must be positive and at most one.")
        for name in ("classifier_prob_threshold", "classifier_margin_threshold"):
            if not 0 <= cfg.problem[name] <= 1:
                raise ValueError("Training classifier thresholds must lie in [0,1].")
        if (not calderon.uses_maizels_protocol(cfg)
                and cfg.evaluation.get("validation_pair_mode", "ot_plain") != "ot_plain"):
            raise ValueError("Use the fixed plain-OT endpoint validation objective shared with the baseline.")
    if cfg.problem.d != 100 or cfg.network.output_dim != 100 or cfg.network.network_type != "mlp":
        raise ValueError("Expected the native PCA100 MLP flow map.")
    if cfg.optimization.total_steps <= 0 or cfg.optimization.bs < 2 or cfg.optimization.learning_rate <= 0:
        raise ValueError("steps, batch size and learning rate must be positive (batch >=2).")
    if cfg.training.seed < 0 or cfg.problem.split_seed < 0:
        raise ValueError("Seeds must be nonnegative.")
    if cfg.optimization.bs % cfg.training.ndevices:
        raise ValueError("Batch size must be divisible by the JAX device count.")
    if cfg.evaluation.validation_batches <= 0 or cfg.evaluation.flow_batch_size <= 0:
        raise ValueError("Validation batches and flow batch size must be positive.")
    if cfg.evaluation.source_max_points < 0 or cfg.evaluation.marginal_max_points < 0:
        raise ValueError("Population caps must be nonnegative; zero means all cells.")
    times = np.asarray(cfg.evaluation.checkpoint_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or (np.diff(times) <= 0).any() or times[0] <= 0 or times[-1] > 1:
        raise ValueError("Classifier checks must be finite, strictly increasing times in (0,1].")
    if not np.isclose(times[-1], 1, atol=1e-12, rtol=0) or np.isclose(times, .5, atol=1e-12, rtol=0).sum() != 1:
        raise ValueError("Classifier checks must include t=0.5 and end at t=1 for both W1 evaluations.")
    if cfg.evaluation.flowmap_n_steps <= 0:
        raise ValueError("Composed step count must be positive.")
    marginal_times = np.array([calderon.normalized_window_time(window, cfg)
                               for window in (*calderon.training_windows(cfg)[1:], *calderon.heldout_windows(cfg))])
    marginal_indices = marginal_times * cfg.evaluation.flowmap_n_steps
    if calderon.uses_maizels_protocol(cfg):
        if cfg.evaluation.validation_pair_mode != "same_as_training" or cfg.evaluation.validation_objective != "same_as_training":
            raise ValueError("Maizels validation must retain the training pairs and objective.")
        if cfg.evaluation.validation_batches != 1 or cfg.evaluation.validation_batch_size < 2:
            raise ValueError("Use one fixed Maizels validation batch of at least two pairs.")
        if cfg.evaluation.ema_factor is not None:
            raise ValueError("The Maizels protocol evaluates instantaneous parameters.")
        if cfg.classifier.prob_threshold != 0 or cfg.classifier.margin_threshold != 0:
            raise ValueError("Maizels result reporting uses raw classifier argmax predictions with zero thresholds.")
        if not np.allclose(times, np.arange(1, len(times) + 1) / len(times)):
            raise ValueError("Maizels violation checks must be uniformly spaced across the full trajectory.")
    else:
        if not np.allclose(times * cfg.evaluation.flowmap_n_steps, np.round(times * cfg.evaluation.flowmap_n_steps)):
            raise ValueError("Composed step grid must include every classifier checkpoint exactly.")
        if ((marginal_times <= 0).any() or (marginal_times > 1).any()
                or not np.allclose(marginal_indices, np.round(marginal_indices))):
            raise ValueError("Composed step grid must include all selected-time test and intermediate W1 marginals exactly.")
    if cfg.evaluation.ema_factor is not None and cfg.evaluation.ema_factor not in cfg.training.ema_facs:
        raise ValueError("Evaluation EMA must be one of the trained EMA factors.")
    early = cfg.optimization.early_stopping
    if early.check_freq <= 0 or early.patience < 0 or early.warmup_steps < 0 or early.min_delta < 0:
        raise ValueError("Invalid endpoint validation/early-stopping settings.")
    for name in ("prob_threshold", "margin_threshold"):
        if not 0 <= cfg.classifier[name] <= 1:
            raise ValueError(f"classifier.{name} must lie in [0,1].")


def make_source_sampler(x):
    import functools
    import jax
    import jax.numpy as jnp
    population = jnp.asarray(x, dtype=jnp.float32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def sample(n_samples, key):
        indices = jax.random.randint(key, (n_samples,), 0, population.shape[0])
        return population[indices]
    return sample


def setup_state(cfg, data):
    import jax
    import jax.numpy as jnp
    from ml_collections import FrozenConfigDict
    from common import dist_utils, interpolant, loss_args, losses, state_utils, updates
    frozen = FrozenConfigDict(cfg)
    source = data.pool(data.source_time, "train")
    ds = calderon.training_dataset(frozen, data, "train")
    key = jax.random.PRNGKey(cfg.training.seed)
    state, net, schedule, key = state_utils.setup_training_state(frozen, jnp.asarray(source["x"][0]), key)
    interp = interpolant.setup_interpolant(frozen)
    loss = losses.setup_loss(frozen, net, interp)
    statics = state_utils.StaticArgs(net=net, schedule=schedule, loss=loss,
                                    get_loss_fn_args=loss_args.get_loss_fn_args,
                                    train_step=updates.setup_train_step(frozen),
                                    update_ema_params=updates.setup_ema_update(frozen), ds=ds, interp=interp,
                                    sample_rho0=make_source_sampler(source["x"]))
    return frozen, statics, dist_utils.safe_replicate(frozen, state), key


def evaluation_params(cfg, state):
    from common import dist_utils
    state = dist_utils.safe_unreplicate(cfg, state)
    return state.params if cfg.evaluation.ema_factor is None else state.ema_params[cfg.evaluation.ema_factor]


def make_validation_evaluator(cfg, statics, data):
    import jax
    from ml_collections import ConfigDict, FrozenConfigDict
    from common import dist_utils, losses
    if calderon.uses_maizels_protocol(cfg):
        from common import calderon_logging, logging
        calderon_logging.register(cfg, data)
        def evaluate(state):
            step = dist_utils.safe_unreplicate(cfg, state).step
            value = float(jax.device_get(logging.compute_maizels_validation_metrics(cfg, statics, state, step)["validation_loss"]))
            if not math.isfinite(value):
                raise RuntimeError("Nonfinite Maizels validation loss.")
            return value
        return evaluate
    single = ConfigDict(cfg.to_dict())
    single.training.ndevices = 1
    single.problem.maizels_pair_mode = cfg.evaluation.get("validation_pair_mode", cfg.problem.maizels_pair_mode)
    if cfg.constraints.enabled:
        single.constraints.enabled = False
        single.problem.interp_uses_labels = bool(single.problem.pair_time_bounds_in_label)
    single = FrozenConfigDict(single)
    source = data.pool(data.source_time, "validation")
    validation_statics = statics._replace(
        ds=calderon.training_dataset(single, data, "validation"),
        sample_rho0=make_source_sampler(source["x"]),
    )
    validation_loss = losses.setup_loss(single, statics.net, statics.interp) if cfg.constraints.enabled else statics.loss
    objective = jax.jit(validation_loss)

    def evaluate(state):
        state = dist_utils.safe_unreplicate(cfg, state)
        state = state.replace(params=state.ema_params[cfg.evaluation.ema_factor])
        key = jax.random.PRNGKey(cfg.evaluation.validation_seed)
        values = []
        for _ in range(cfg.evaluation.validation_batches):
            arguments, key = validation_statics.get_loss_fn_args(single, validation_statics, state, key)
            values.append(float(jax.device_get(objective(state.params, *arguments))))
        value = float(np.mean(values))
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite endpoint validation loss.")
        return value
    return evaluate


def save_checkpoint(path, cfg, state):
    from flax.serialization import to_bytes
    from common import dist_utils
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(to_bytes(dist_utils.safe_unreplicate(cfg, state)))
    temporary.replace(path)


def load_checkpoint(path, cfg, state):
    from flax.serialization import from_bytes
    from common import dist_utils
    loaded = from_bytes(dist_utils.safe_unreplicate(cfg, state), Path(path).read_bytes())
    return dist_utils.safe_replicate(cfg, loaded)


def train(cfg, statics, state, key, data, output, tracker=None):
    import jax
    from common import logging as native_logging
    validate = make_validation_evaluator(cfg, statics, data)
    early = cfg.optimization.early_stopping
    best_loss = math.inf
    best_step = None
    failures = 0
    started = time.monotonic()
    filtered = cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant"
    pair_fields = ("endpoint_rejected", "interpolant_checked", "interpolant_rejected", "ot_refinements", "ot_mean_retained_valid_mass") if filtered else ()
    fields = ("step", "training_loss", "validation_loss", "learning_rate", "seconds", *pair_fields)
    maizels_protocol = calderon.uses_maizels_protocol(cfg)
    with (output / "training_history.csv").open("w", newline="") as handle:
        history = csv.DictWriter(handle, fieldnames=fields)
        history.writeheader()
        for step in range(1, cfg.optimization.total_steps + 1):
            step_started = time.monotonic()
            arguments, key = statics.get_loss_fn_args(cfg, statics, state, key)
            state, loss, grads = statics.train_step(state, statics.loss, arguments)
            state = statics.update_ema_params(state)
            value = float(np.mean(jax.device_get(loss)))
            if not math.isfinite(value):
                raise RuntimeError(f"Nonfinite training loss at step {step}.")
            check = (step >= early.warmup_steps and step % early.check_freq == 0) or step == cfg.optimization.total_steps
            if maizels_protocol:
                key, logged = native_logging.log_metrics(cfg, statics, state, grads, loss, arguments, key,
                    time.monotonic() - step_started, compute_validation=check)
                validation_loss = float(jax.device_get(logged["validation_loss"])) if check else None
            else:
                validation_loss = validate(state) if check else None
            if check:
                if validation_loss < best_loss - early.min_delta:
                    best_loss, best_step, failures = validation_loss, step, 0
                    save_checkpoint(output / "best.msgpack", cfg, state)
                    if maizels_protocol:
                        native_logging.save_state(state, cfg, checkpoint_label="best")
                else:
                    failures += 1
            if step == 1 or check or step % cfg.logging.scalar_freq == 0:
                row = {"step": step, "training_loss": value, "validation_loss": validation_loss,
                       "learning_rate": float(statics.schedule(step - 1)), "seconds": time.monotonic() - started}
                if filtered:
                    row.update({name: statics.ds.last_pair_stats[name] for name in pair_fields})
                    write_json(output / "pair_filtering_summary.json", statics.ds.pair_filter_summary())
                history.writerow(row)
                handle.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
                if tracker and not maizels_protocol:
                    tracker.log({name: item for name, item in row.items() if item is not None}, step=step)
            if cfg.logging.save_freq > 0 and step % cfg.logging.save_freq == 0:
                save_checkpoint(output / "last.msgpack", cfg, state)
            if check and early.patience > 0 and failures >= early.patience:
                print(f"Early stopping at step {step}; best endpoint validation at {best_step}.", flush=True)
                break
    save_checkpoint(output / "last.msgpack", cfg, state)
    write_json(output / "training_summary.json", {
        "completed_steps": step, "best_step": best_step, "best_validation_loss": best_loss,
        "selection": ("One fixed Maizels validation batch with the training pairing/objective and instantaneous parameters"
                      if maizels_protocol else "Fixed validation OT batches over adjacent training intervals, evaluated with the final sampling EMA"),
        "reference_window_used_for_selection": False, "test_endpoints_used_for_selection": False,
        "intermediate_windows_used_for_selection": False, "test_cells_used_for_selection": False,
        "training_windows": list(data.training_times), "intermediate_windows": list(data.heldout_times),
        "seconds": time.monotonic() - started,
        "validation_pair_mode": cfg.evaluation.get("validation_pair_mode", cfg.problem.maizels_pair_mode),
        "validation_objective": ("same_as_training" if maizels_protocol else
                                 "plain_ot_lsd_without_constraint" if cfg.constraints.enabled else "plain_ot_lsd"),
    })
    if filtered:
        write_json(output / "pair_filtering_summary.json", statics.ds.pair_filter_summary())
    # Final metrics intentionally go through the saved checkpoint loader.
    return load_checkpoint(output / "best.msgpack", cfg, state)


def rollout_checkpoints(apply_flow_map, params, source_x, evaluation_cfg):
    import jax
    import jax.numpy as jnp
    from common import flow_map
    steps = int(evaluation_cfg.flowmap_n_steps)
    indices = jnp.asarray(np.round(np.asarray(evaluation_cfg.checkpoint_times) * steps).astype(np.int32))

    @jax.jit
    def sample(batch):
        # The MLP ignores labels; dummy zeros keep the common vmap interface.
        trajectories = flow_map.batch_sample_trajectory(apply_flow_map, params, batch, steps,
                                                        jnp.zeros(batch.shape[0], dtype=jnp.int32))
        return trajectories[:, indices, :]

    batch_size = int(evaluation_cfg.flow_batch_size)
    paths = np.empty((len(source_x), len(indices), source_x.shape[1]), dtype=np.float32)
    for start in range(0, len(source_x), batch_size):
        batch = source_x[start:start + batch_size]
        n_real = len(batch)
        if n_real < batch_size:
            batch = np.pad(batch, ((0, batch_size - n_real), (0, 0)), mode="edge")
        paths[start:start + n_real] = np.asarray(jax.device_get(sample(jnp.asarray(batch))))[:n_real]
    if not np.isfinite(paths).all():
        raise RuntimeError("Nonfinite generated trajectory.")
    return paths


def check_classifier_python(cfg):
    try:
        subprocess.run([cfg.classifier.python, str(ROOT / "scripts/evaluate_calderon_tissue_paths.py"),
                        "--check-only", "--manifest", calderon.evaluation_manifest_path(cfg),
                        "--allowed-pairs", cfg.classifier.allowed_pairs_path, "--threads", str(cfg.classifier.threads)],
                       check=True, capture_output=True, text=True, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", str(error))
        raise RuntimeError(
            "Classifier worker Python is unavailable or lacks its dependencies. "
            "Pass --classifier-python with the interpreter used for the Calderon classifiers.\n" + detail
        ) from error


def prepare_logging_classifiers(cfg):
    """Copy frozen oracle weights for the native Maizels diagnostics."""
    if not calderon.uses_maizels_protocol(cfg):
        return
    export = Path(cfg.classifier.evaluation_jax_export_path)
    if export.exists():
        with np.load(export, allow_pickle=False) as stored:
            metadata = json.loads(stored["metadata"].item())
        if metadata["manifest_sha256"] != calderon.sha256_file(calderon.evaluation_manifest_path(cfg)):
            raise ValueError("Logging oracle export differs from the selected evaluation classifier.")
        return
    subprocess.run([cfg.classifier.python, str(ROOT / "scripts/export_calderon_tissue_ensemble.py"),
                    "--manifest", calderon.evaluation_manifest_path(cfg), "--output", str(export),
                    "--weights-only"], check=True, cwd=ROOT)


def evaluate_run(cfg, statics, state, data, output, checkpoint, evaluation_output=None):
    if calderon.uses_maizels_protocol(cfg):
        from common.calderon_evaluation import evaluate
        return evaluate(cfg, statics, state, data, output, checkpoint, evaluation_output)
    import jax
    from common import dist_utils, wasserstein
    evaluation = cfg.evaluation
    indices = data.splits[data.source_time]["test"]
    selected = calderon.select_population_indices(len(indices), evaluation.source_max_points, evaluation.seed)
    indices = indices[selected]
    source = data.marginals[data.source_time]
    source_x = source.x[indices]
    source_meta = source.metadata.iloc[indices]
    print(f"Rolling out {len(source_x)} held-out source cells at {evaluation.flowmap_n_steps} composed steps.", flush=True)
    result_root = output if evaluation_output is None else Path(evaluation_output)
    result_root.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "evaluation/evaluation_inputs.npz"
    trajectory_comparison = None
    populations = list(data.evaluation_populations())
    marginal_times = {window: data.normalized_time(window) for _, window, _, _ in populations}
    combined_times = set(evaluation.checkpoint_times) | set(marginal_times.values())
    baseline = None
    if result_root.resolve() != output.resolve() and baseline_path.exists():
        with np.load(baseline_path, allow_pickle=False) as stored:
            baseline = {name: stored[name] for name in stored.files}
        baseline_steps = baseline["checkpoint_times"] * evaluation.flowmap_n_steps
        baseline_on_grid = bool(np.allclose(baseline_steps, np.round(baseline_steps)))
        if baseline_on_grid:
            combined_times.update(baseline["checkpoint_times"].tolist())
    combined_times = sorted(combined_times)
    rollout_cfg = SimpleNamespace(**evaluation.to_dict())
    rollout_cfg.checkpoint_times = combined_times
    combined_paths = rollout_checkpoints(statics.net.apply, evaluation_params(cfg, state), source_x, rollout_cfg)
    paths = combined_paths[:, [combined_times.index(t) for t in evaluation.checkpoint_times]]
    if baseline is not None:
        same_sources = np.array_equal(source_meta.cell_id.to_numpy(dtype=str), baseline["source_cell_ids"])
        trajectory_comparison = {"baseline_inputs_sha256": calderon.sha256_file(baseline_path),
                                 "same_source_cell_ids": bool(same_sources),
                                 "baseline_checks_on_composition_grid": baseline_on_grid,
                                 "baseline_checkpoint_times": baseline["checkpoint_times"].tolist(),
                                 "same_coordinates_at_baseline_checks": None, "max_absolute_coordinate_difference": None}
        if same_sources and baseline_on_grid:
            recreated = combined_paths[:, [combined_times.index(t) for t in baseline["checkpoint_times"]]]
            trajectory_comparison["same_coordinates_at_baseline_checks"] = bool(np.array_equal(recreated, baseline["paths"]))
            trajectory_comparison["max_absolute_coordinate_difference"] = float(np.max(np.abs(recreated - baseline["paths"])))
            print("Comparison with saved trajectories: " + json.dumps(trajectory_comparison), flush=True)
    generated_indices = calderon.select_population_indices(len(paths), evaluation.marginal_max_points, evaluation.seed + 3)
    print("Computing exact PCA-space W1 at intermediate marginals and selected-time test populations.", flush=True)
    distribution, sampled_populations = {}, []
    for index, (metric_key, window, eligible, kind) in enumerate(populations):
        marginal = data.marginals[window]
        chosen = eligible[calderon.select_population_indices(len(eligible), evaluation.marginal_max_points,
                                                             evaluation.seed + index + 1)]
        time_index = combined_times.index(marginal_times[window])
        distribution[metric_key] = {
            "window": window, "normalized_time": marginal_times[window], "population": kind,
            "w1": wasserstein.exact_emd(combined_paths[generated_indices, time_index], marginal.x[chosen]),
            "n_generated": len(generated_indices), "n_reference": len(chosen), "n_available_reference": len(eligible),
            "sampler": "composed_flow_map",
        }
        sampled_populations.append((metric_key, window, chosen, kind))
    reference = data.marginals[data.reference_time]
    reference_indices = next(chosen for _, window, chosen, _ in sampled_populations if window == data.reference_time)
    target = data.marginals[data.target_time]
    target_indices = next(chosen for _, window, chosen, _ in sampled_populations if window == data.target_time)
    hours = cfg.problem.source_hours + np.asarray(evaluation.checkpoint_times) * (cfg.problem.target_hours - cfg.problem.source_hours)
    evaluation_dir = result_root / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    write_json(evaluation_dir / "distribution_metrics.json", distribution)
    write_json(evaluation_dir / "evaluation_config.json", cfg.to_dict())
    if trajectory_comparison is not None:
        write_json(evaluation_dir / "trajectory_comparison.json", trajectory_comparison)
    input_path = evaluation_dir / "evaluation_inputs.npz"
    inputs = dict(paths=paths, source_x=source_x,
                  source_cell_ids=source_meta.cell_id.to_numpy(dtype=str),
                  source_tissues=source_meta.broad_tissue.to_numpy(dtype=str),
                  reference_x=reference.x[reference_indices],
                  reference_cell_ids=reference.metadata.iloc[reference_indices].cell_id.to_numpy(dtype=str),
                  reference_tissues=reference.metadata.iloc[reference_indices].broad_tissue.to_numpy(dtype=str),
                  target_test_x=target.x[target_indices],
                  target_test_cell_ids=target.metadata.iloc[target_indices].cell_id.to_numpy(dtype=str),
                  target_test_tissues=target.metadata.iloc[target_indices].broad_tissue.to_numpy(dtype=str),
                  class_names=np.asarray(data.class_names), checkpoint_times=np.asarray(evaluation.checkpoint_times),
                  checkpoint_hours=hours, generated_w1_indices=generated_indices)
    if len(data.training_times) > 2:
        inputs.update(marginal_windows=np.asarray([window for _, window, _, _ in sampled_populations]),
                      marginal_metric_keys=np.asarray([key for key, _, _, _ in sampled_populations]),
                      marginal_populations=np.asarray([kind for _, _, _, kind in sampled_populations]),
                      marginal_times=np.asarray([marginal_times[window] for _, window, _, _ in sampled_populations]),
                      marginal_step_indices=np.asarray([round(marginal_times[window] * evaluation.flowmap_n_steps)
                                                        for _, window, _, _ in sampled_populations], dtype=np.int32),
                      marginal_generated_x=np.stack([combined_paths[:, combined_times.index(marginal_times[window])]
                                                     for _, window, _, _ in sampled_populations], axis=1))
        for i, (_, window, chosen, _) in enumerate(sampled_populations):
            marginal = data.marginals[window]
            inputs[f"marginal_{i}_x"] = marginal.x[chosen]
            inputs[f"marginal_{i}_cell_ids"] = marginal.metadata.iloc[chosen].cell_id.to_numpy(dtype=str)
            inputs[f"marginal_{i}_tissues"] = marginal.metadata.iloc[chosen].broad_tissue.to_numpy(dtype=str)
    np.savez_compressed(input_path, **inputs)
    command = [cfg.classifier.python, str(ROOT / "scripts/evaluate_calderon_tissue_paths.py"),
               "--input", str(input_path), "--manifest", calderon.evaluation_manifest_path(cfg),
               "--allowed-pairs", cfg.classifier.allowed_pairs_path, "--output-dir", str(evaluation_dir),
               "--prob-threshold", str(cfg.classifier.prob_threshold), "--margin-threshold", str(cfg.classifier.margin_threshold),
               "--batch-size", str(cfg.classifier.batch_size), "--threads", str(cfg.classifier.threads)]
    subprocess.run(command, check=True, cwd=ROOT)
    tissue_metrics = json.loads((evaluation_dir / "tissue_trajectory_metrics.json").read_text())
    metrics = {
        "method": ("Native Maizels constrained endpoint/interpolant-filtered OT flow map, LSD and lineage transition NLL"
                   if cfg.constraints.enabled else
                   "Native Maizels endpoint/interpolant-filtered minibatch-OT flow map, LSD self-distillation"
                   if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant" else
                   "Native Maizels plain minibatch-OT flow map, LSD self-distillation"),
        "source_window": data.source_time, "target_window": data.target_time,
        "training_windows": list(data.training_times), "intermediate_windows": list(data.heldout_times),
        "training_classifier_manifest": cfg.classifier.manifest_path,
        "evaluation_classifier_manifest": calderon.evaluation_manifest_path(cfg),
        "n_violation_checkpoints": len(evaluation.checkpoint_times),
        "w1_checkpoint_times": list(marginal_times.values()),
        "cohort_branch_prefix": cfg.problem.cohort_branch_prefix,
        "sampler": "composed_flow_map", "flowmap_n_steps": evaluation.flowmap_n_steps,
        "ema_factor": evaluation.ema_factor, "n_test_sources_available": len(data.splits[data.source_time]["test"]),
        "n_test_sources_evaluated": len(source_x), "distribution": distribution, "tissue_trajectories": tissue_metrics,
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": calderon.sha256_file(checkpoint),
        "training_step": int(jax.device_get(dist_utils.safe_unreplicate(cfg, state).step)),
        "training_config_sha256": calderon.sha256_file(output / "config.json"),
        "provenance": data.provenance,
    }
    if trajectory_comparison is not None:
        metrics["trajectory_comparison"] = trajectory_comparison
    if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
        metrics["training_interpolant_filter"] = {
            "n_interior_checks": cfg.problem.n_interpolant_check_times,
            "times": np.linspace(0., 1., cfg.problem.n_interpolant_check_times + 2)[1:-1].tolist(),
            "prob_threshold": cfg.problem.classifier_prob_threshold,
            "margin_threshold": cfg.problem.classifier_margin_threshold,
            "ot_infeasible_fallback": cfg.problem.ot_minibatch_infeasible_fallback,
            "classifier_export_sha256": calderon.sha256_file(cfg.classifier.jax_export_path),
            "early_rejection_chunk_size": cfg.problem.get("interpolant_check_chunk_size", cfg.problem.n_interpolant_check_times),
            "backend": cfg.problem.get("interpolant_filter_backend", "reference"),
            "decision_tolerance": cfg.problem.get("gpu_filter_decision_tolerance", None),
        }
    if cfg.constraints.enabled:
        metrics["training_lineage_constraint"] = cfg.constraints.to_dict()
    write_json(result_root / "metrics.json", metrics)
    write_json(evaluation_dir / "evaluation_config.json", cfg.to_dict())
    print(json.dumps({"distribution": distribution, "raw_trajectory_violation_pct":
                      tissue_metrics["raw"]["violation_pct_all_sources"]}, indent=2), flush=True)
    return metrics


def main():
    args = parse_args()
    cfg = configure(args)
    output = Path(cfg.logging.output_folder)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage != "evaluate" and any((output / name).exists() for name in ("best.msgpack", "last.msgpack", "training_history.csv")):
        raise FileExistsError("This run already has training artifacts. Choose another --output-dir or use --stage evaluate.")
    data = calderon.load_endpoint_experiment(cfg)
    training_populations = [data.pool(window, "train")["x"] for window in data.training_times]
    rescale = float(np.std(np.concatenate(training_populations), dtype=np.float64))
    if not math.isfinite(rescale) or rescale <= 0:
        raise ValueError("Training marginals must have positive finite variance.")
    if args.stage == "evaluate":
        prepared = json.loads((output / "data_summary.json").read_text())
        for name in ("dataset_sha256", "classifier_manifest_sha256", "evaluation_classifier_manifest_sha256",
                     "shared_split_manifest_sha256", "allowed_pairs_sha256", "training_classifier_export_sha256"):
            if name in prepared["provenance"] and prepared["provenance"][name] != data.provenance[name]:
                raise ValueError(f"Saved experiment input changed: {name}.")
        if not np.isclose(cfg.network.rescale, rescale, rtol=1e-8):
            raise ValueError("Saved training rescale differs from the recreated marginal splits.")
    else:
        cfg.network.rescale = rescale
        cfg.problem.n = sum(len(x) for x in training_populations)
        data.split_table().to_csv(output / "cell_splits.csv.gz", index=False)
        write_json(output / "data_summary.json", {"marginals": data.summary(), "provenance": data.provenance})
        write_json(output / "config.json", cfg.to_dict())
    print(json.dumps(data.summary(), indent=2), flush=True)
    print(f"Artifacts: {output}", flush=True)
    if args.stage == "prepare":
        return
    check_classifier_python(cfg)
    prepare_logging_classifiers(cfg)
    cfg, statics, state, key = setup_state(cfg, data)
    tracker = None
    if calderon.uses_maizels_protocol(cfg) or args.wandb not in (None, "disabled"):
        import wandb
        tracker = wandb.init(project=cfg.logging.wandb_project, entity=cfg.logging.get("wandb_entity"),
                             name=cfg.logging.wandb_name, config=cfg.to_dict(),
                             mode=args.wandb, dir=str(output))
    try:
        if args.stage == "train":
            state = train(cfg, statics, state, key, data, output, tracker)
            checkpoint = output / "best.msgpack"
        else:
            checkpoint = args.checkpoint or output / "best.msgpack"
            state = load_checkpoint(checkpoint, cfg, state)
        evaluate_run(cfg, statics, state, data, output, checkpoint, args.evaluation_output_dir)
    finally:
        if tracker:
            tracker.finish()


if __name__ == "__main__":
    main()

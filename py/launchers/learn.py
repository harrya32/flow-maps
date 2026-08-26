"""
Nicholas M. Boffi
10/5/25

Main training loop for self-distillation of flow maps.
"""

# isort: off
import os
import pathlib
import sys

# Set up path for imports FIRST
script_dir = os.path.dirname(os.path.abspath(__file__))
py_dir = os.path.join(script_dir, "..")
sys.path.append(py_dir)

# Suppress TensorFlow logging before any TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# Force TensorFlow to use CPU only for data loading - no GPU ops
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")  # Hide all GPUs from TensorFlow
# isort: on
#
import argparse
import importlib
import inspect
import time
from typing import Dict, Tuple

import common.datasets as datasets
import common.dist_utils as dist_utils
import common.fid_utils as fid_utils
import common.interpolant as interpolant
import common.logging as logging
import common.loss_args as loss_args
import common.losses as losses
import common.state_utils as state_utils
import common.updates as updates
import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
import wandb
from ml_collections import config_dict  # type: ignore
from tqdm.auto import tqdm as tqdm

Parameters = Dict[str, Dict]
mpl.rc_file(f"{pathlib.Path(__file__).resolve().parent}/matplotlibrc")


def train_loop(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: np.ndarray,
) -> None:
    """Carry out the training loop."""

    pbar = tqdm(range(cfg.optimization.total_steps))
    scalar_freq = max(1, int(getattr(cfg.logging, "scalar_freq", 1)))
    progress_freq = max(1, int(getattr(cfg.logging, "progress_freq", scalar_freq)))
    visual_freq = int(getattr(cfg.logging, "visual_freq", 0))
    save_freq = int(getattr(cfg.logging, "save_freq", 0))
    fid_freq = int(getattr(cfg.logging, "fid_freq", 0))
    early_cfg = getattr(cfg.optimization, "early_stopping", None)
    early_patience = int(getattr(early_cfg, "patience", 0))
    early_enabled = early_patience > 0
    early_check_freq = int(getattr(early_cfg, "check_freq", 1))
    early_warmup_steps = int(getattr(early_cfg, "warmup_steps", 0))
    early_min_delta = float(getattr(early_cfg, "min_delta", 0.0))
    early_metric = str(getattr(early_cfg, "metric", "validation_loss"))
    early_mode = str(getattr(early_cfg, "mode", "min")).lower()
    early_save_best = bool(getattr(early_cfg, "save_best", True))
    if early_patience < 0:
        raise ValueError("early_stopping.patience must be non-negative")
    if early_enabled and early_check_freq <= 0:
        raise ValueError("early_stopping.check_freq must be positive")
    if early_enabled and early_warmup_steps < 0:
        raise ValueError("early_stopping.warmup_steps must be non-negative")
    if early_enabled and early_min_delta < 0:
        raise ValueError("early_stopping.min_delta must be non-negative")
    if early_enabled and early_mode not in {"min", "max"}:
        raise ValueError("early_stopping.mode must be 'min' or 'max'")
    best_metric = np.inf if early_mode == "min" else -np.inf
    best_step = None
    best_params_for_evaluation = None
    checks_without_improvement = 0

    if early_enabled:
        print(
            f"Early stopping enabled: metric={early_metric}, "
            f"patience={early_patience} checks, check_freq={early_check_freq} steps, "
            f"min_delta={early_min_delta:g}, warmup_steps={early_warmup_steps}."
        )

    for step_idx in pbar:
        # construct loss function arguments
        start_time = time.time()
        loss_fn_args, prng_key = statics.get_loss_fn_args(
            cfg, statics, train_state, prng_key
        )

        # take a step on the loss
        train_state, loss_value, grads = statics.train_step(
            train_state, statics.loss, loss_fn_args
        )
        end_time = time.time()

        # compute update to EMA params
        train_state = statics.update_ema_params(train_state)

        step_num = step_idx + 1
        should_log_scalars = (step_num % scalar_freq) == 0
        should_visualize = visual_freq > 0 and (step_num % visual_freq) == 0
        should_save = save_freq > 0 and (step_num % save_freq) == 0
        should_fid = fid_freq > 0 and (step_num % fid_freq) == 0
        should_check_early_stopping = (
            early_enabled
            and step_num >= early_warmup_steps
            and (step_num % early_check_freq) == 0
        )

        metrics = {}
        if (
            should_log_scalars
            or should_visualize
            or should_save
            or should_fid
            or should_check_early_stopping
        ):
            prng_key, metrics = logging.log_metrics(
                cfg,
                statics,
                train_state,
                grads,
                loss_value,
                loss_fn_args,
                prng_key,
                end_time - start_time,
            )

        if should_check_early_stopping:
            if early_metric not in metrics:
                raise ValueError(
                    f"Early-stopping metric {early_metric!r} was not produced. "
                    "For Maizels validation loss, ensure "
                    "logging.maizels.validation_enabled=True."
                )
            current_metric = float(jax.device_get(metrics[early_metric]))
            improvement = (
                current_metric < best_metric - early_min_delta
                if early_mode == "min"
                else current_metric > best_metric + early_min_delta
            )
            if improvement:
                best_metric = current_metric
                best_step = step_num
                checks_without_improvement = 0
                best_params_for_evaluation = jax.device_get(
                    logging.get_params_for_sampling(
                        cfg,
                        train_state,
                        param_type="visual",
                    )
                )
                if early_save_best:
                    logging.save_state(train_state, cfg, checkpoint_label="best")
            else:
                checks_without_improvement += 1

            if wandb.run is not None:
                wandb.run.summary["early_stopping/best_metric"] = best_metric
                wandb.run.summary["early_stopping/best_step"] = best_step
                wandb.run.summary["early_stopping/checks_without_improvement"] = (
                    checks_without_improvement
                )
            if checks_without_improvement >= early_patience:
                print(
                    f"Early stopping at step {step_num}: {early_metric}="
                    f"{current_metric:.6g}; best={best_metric:.6g} at step "
                    f"{best_step}."
                )
                break

        if (step_num % progress_freq) == 0:
            loss_for_progress = dist_utils.safe_index(cfg, jnp.array(loss_value))
            pbar.set_postfix(loss=float(jax.device_get(loss_for_progress)))

        # guard against sigterm/sigint
        logging.register_signal_handlers(cfg, train_state)

    # dump one final time
    logging.save_state(train_state, cfg)

    maizels_cfg = getattr(cfg.logging, "maizels", None)
    if maizels_cfg is not None and bool(getattr(maizels_cfg, "enabled", False)):
        final_step = int(
            jax.device_get(dist_utils.safe_index(cfg, train_state.step))
        )
        if best_params_for_evaluation is None:
            params_for_evaluation = logging.get_params_for_sampling(
                cfg,
                train_state,
                param_type="visual",
            )
            evaluation_step = final_step
            evaluation_metric = None
            print(
                "No early-stopping best snapshot was selected; evaluating the "
                f"final model at step {final_step}."
            )
        else:
            params_for_evaluation = best_params_for_evaluation
            evaluation_step = int(best_step)
            evaluation_metric = float(best_metric)
            print(f"Evaluating the best model from step {evaluation_step}.")

        logging.log_maizels_final_evaluation(
            cfg,
            train_state,
            params_for_evaluation,
            best_step=evaluation_step,
            best_metric=evaluation_metric,
        )


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description="Direct flow map learning.")
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--slurm_id", type=int)
    parser.add_argument("--dataset_location", type=str)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument(
        "--dataset_name",
        choices=("cite", "multi"),
        default=None,
        help="Dataset variant for configs that expose a dataset_name option.",
    )
    parser.add_argument(
        "--heldout_day",
        choices=("3", "4"),
        default=None,
        help="Internal day omitted by leave-one-timepoint-out configs.",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default=None,
        help="Optional cell-type classifier override for lineage-aware configs.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help=(
            "Stop after this many validation checks without improvement. "
            "Zero disables early stopping."
        ),
    )
    parser.add_argument(
        "--maizels_ot_coupling",
        choices=("global_ot", "minibatch_ot"),
        default=None,
        help="Use cached global OT or fresh per-step minibatch OT for Maizels.",
    )
    return parser.parse_args()


def setup_config_dict():
    args = parse_command_line_arguments()
    cfg_module = importlib.import_module(args.cfg_path)
    get_config = cfg_module.get_config
    supported = inspect.signature(get_config).parameters
    optional = {
        "dataset_name": args.dataset_name,
        "heldout_day": args.heldout_day,
        "classifier_path": args.classifier_path,
        "early_stopping_patience": args.early_stopping_patience,
        "maizels_ot_coupling": args.maizels_ot_coupling,
    }
    kwargs = {
        name: value
        for name, value in optional.items()
        if value is not None and name in supported
    }
    return get_config(args.slurm_id, args.dataset_location, args.output_folder, **kwargs)


def setup_state(cfg: config_dict.ConfigDict, prng_key: jnp.ndarray) -> Tuple[
    config_dict.ConfigDict,
    state_utils.StaticArgs,
    state_utils.EMATrainState,
    jnp.ndarray,
]:
    """Construct static arguments and training state objects."""
    # define dataset
    cfg, ds, prng_key = datasets.setup_target(cfg, prng_key)
    ex_input = next(ds)
    if isinstance(ex_input, dict) and "image" in ex_input:  # handle image datasets
        ex_input = ex_input["image"][0]
    elif isinstance(ex_input, dict) and "x1" in ex_input:  # paired low-d datasets
        ex_input = ex_input["x1"][0]
    else:
        ex_input = ex_input[0]
    interp = interpolant.setup_interpolant(cfg)
    cfg = config_dict.FrozenConfigDict(cfg)

    # define training state
    train_state, net, schedule, prng_key = state_utils.setup_training_state(
        cfg,
        ex_input,
        prng_key,
    )

    # define the loss
    loss = losses.setup_loss(cfg, net, interp)

    # initialize FID network if FID computation is enabled
    inception_fn = None
    if hasattr(cfg.logging, "fid_freq") and cfg.logging.fid_freq > 0:
        print("Initializing Inception network for FID computation...")
        inception_fn = fid_utils.get_fid_network()
        print("Inception network initialized.")

    # define static object
    statics = state_utils.StaticArgs(
        net=net,
        schedule=schedule,
        loss=loss,
        get_loss_fn_args=loss_args.get_loss_fn_args,
        train_step=updates.setup_train_step(cfg),
        update_ema_params=updates.setup_ema_update(cfg),
        ds=ds,
        interp=interp,
        sample_rho0=datasets.setup_base(cfg, ex_input),
        inception_fn=inception_fn,
    )

    train_state = dist_utils.safe_replicate(cfg, train_state)

    return cfg, statics, train_state, prng_key


if __name__ == "__main__":
    print("Entering main. Setting up config dict and PRNG key.")
    cfg = setup_config_dict()

    # Populate JAX device information for single-node multi-GPU training
    cfg.training.ndevices = jax.device_count()
    print(f"Initialized with {cfg.training.ndevices} local GPUs")

    prng_key = jax.random.PRNGKey(cfg.training.seed)

    # Set up weights and biases tracking
    print("Setting up wandb.")
    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        name=cfg.logging.wandb_name,
        config=cfg.to_dict(),
    )

    print("Config dict set up. Setting up static arguments and training state.")
    cfg, statics, train_state, prng_key = setup_state(cfg, prng_key)

    train_loop(cfg, statics, train_state, prng_key)

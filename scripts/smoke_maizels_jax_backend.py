#!/usr/bin/env python3
"""Run a short, real Maizels OT flow-map backend smoke test."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "py"))

import jax  # noqa: E402

from configs.maizels_pca50_constraint_sweep import get_config  # noqa: E402
from launchers.learn import setup_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and run a few genuine three-marginal Maizels OT flow-map "
            "training steps on the selected JAX backend."
        )
    )
    parser.add_argument(
        "--dataset-location",
        default="/Users/harryamad/Desktop/Maizels2023aa/data",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1")
    if args.pool_size < 256:
        raise ValueError("--pool-size must be at least 256")

    # Constraint-sweep ID 4 is the unconstrained OT flow-map variant. This
    # keeps the smoke test focused on the flow-map/JVP and minibatch-OT path.
    backend = jax.default_backend()
    devices = jax.devices()
    cfg = get_config(
        4,
        args.dataset_location,
        "",
        early_stopping_patience=0,
        maizels_ot_coupling="minibatch_ot",
        maizels_schedule="d3_d3p8_d8",
        maizels_time_mode="real_time",
    )
    cfg.problem.n = args.pool_size
    cfg.optimization.total_steps = args.steps
    cfg.optimization.total_samples = cfg.optimization.bs * args.steps
    cfg.optimization.decay_steps = max(args.steps, 1)
    cfg.logging.fid_freq = 0
    cfg.logging.maizels.enabled = False
    cfg.logging.maizels.validation_enabled = False
    cfg.training.ndevices = jax.device_count()

    print(f"JAX {jax.__version__}")
    print(f"backend={backend} devices={devices}")
    print(
        "configuration="
        f"{cfg.logging.wandb_name} schedule={cfg.problem.maizels_schedule} "
        f"coupling={cfg.problem.maizels_ot_coupling} batch={cfg.optimization.bs}"
    )

    setup_started = time.perf_counter()
    key = jax.random.PRNGKey(cfg.training.seed)
    cfg, statics, state, key = setup_state(cfg, key)
    print(f"setup_seconds={time.perf_counter() - setup_started:.3f}")

    durations = []
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        loss_args, key = statics.get_loss_fn_args(cfg, statics, state, key)
        state, loss, _ = statics.train_step(state, statics.loss, loss_args)
        state = statics.update_ema_params(state)
        jax.block_until_ready(state)
        duration = time.perf_counter() - step_started
        loss_value = float(jax.device_get(loss))
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss_value}")
        durations.append(duration)
        print(f"step={step} loss={loss_value:.6g} seconds={duration:.3f}")

    if len(durations) > 1:
        print(f"post_compile_mean_seconds={sum(durations[1:]) / (len(durations) - 1):.3f}")
    print("status=PASS")


if __name__ == "__main__":
    main()

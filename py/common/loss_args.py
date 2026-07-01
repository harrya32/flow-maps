"""
Nicholas M. Boffi
10/5/25

Code for setting up arguments for loss functions.
"""

import functools
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import state_utils
from . import dist_utils


def compute_constraint_anneal_scale(
    cfg: config_dict.ConfigDict, step: jnp.ndarray
) -> jnp.ndarray:
    """Compute multiplicative annealing factor for constraint terms."""
    if not hasattr(cfg, "constraints") or not getattr(cfg.constraints, "enabled", False):
        return jnp.asarray(1.0, dtype=jnp.float32)

    anneal_cfg = getattr(cfg.constraints, "anneal", None)
    if anneal_cfg is None or not getattr(anneal_cfg, "enabled", False):
        return jnp.asarray(1.0, dtype=jnp.float32)

    start_step = float(getattr(anneal_cfg, "start_step", 0))
    default_end = float(getattr(cfg.optimization, "total_steps", 1))
    end_step = float(getattr(anneal_cfg, "end_step", default_end))
    power = float(getattr(anneal_cfg, "power", 1.0))

    denom = max(end_step - start_step, 1.0)
    frac = jnp.clip((jnp.asarray(step, dtype=jnp.float32) - start_step) / denom, 0.0, 1.0)
    return frac**power


def compute_two_stage_scale(
    cfg: config_dict.ConfigDict, step: jnp.ndarray
) -> jnp.ndarray:
    """Compute stage-2 interpolation factor in [0, 1]."""
    two_stage_cfg = getattr(cfg.training, "two_stage", None)
    if two_stage_cfg is None or not getattr(two_stage_cfg, "enabled", False):
        return jnp.asarray(0.0, dtype=jnp.float32)

    start_step = float(getattr(two_stage_cfg, "start_step", 0))
    ramp_steps = float(getattr(two_stage_cfg, "ramp_steps", 0))
    power = float(getattr(two_stage_cfg, "power", 1.0))

    step_f = jnp.asarray(step, dtype=jnp.float32)
    if ramp_steps <= 0:
        return (step_f >= start_step).astype(jnp.float32)

    frac = jnp.clip((step_f - start_step) / max(ramp_steps, 1.0), 0.0, 1.0)
    return frac**power


def select_teacher_params(
    cfg: config_dict.ConfigDict, train_state: state_utils.EMATrainState
):
    """Select teacher parameters for self-distillation."""
    ema_fac = getattr(cfg.training, "teacher_ema_factor", None)
    if ema_fac is None:
        return train_state.params

    ema_params = train_state.ema_params
    if ema_fac in ema_params:
        return ema_params[ema_fac]

    if len(ema_params) == 0:
        return train_state.params

    closest_ema_fac = min(
        ema_params.keys(), key=lambda k: abs(float(k) - float(ema_fac))
    )
    return ema_params[closest_ema_fac]


def safe_resize(curr_bs: int, bs: int, x: jnp.ndarray) -> jnp.ndarray:
    """Resize the input array to the current batch size."""
    if curr_bs < bs:
        x = x[:curr_bs]
    return x


def _sample_diagonal(
    key: jnp.ndarray, bs: int, tmin: float, tmax: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample points on the diagonal (s=t)."""
    s = jax.random.uniform(key, shape=(bs,), minval=tmin, maxval=tmax)
    return s, s


def _sample_triangle(
    key1: jnp.ndarray,
    key2: jnp.ndarray,
    bs: int,
    tmin: float,
    tmax: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample uniformly from upper triangle."""
    temp1 = jax.random.uniform(key1, shape=(bs,), minval=tmin, maxval=tmax)
    temp2 = jax.random.uniform(key2, shape=(bs,), minval=tmin, maxval=tmax)
    s = jnp.minimum(temp1, temp2)
    t = jnp.maximum(temp1, temp2)
    return s, t


def _get_diag_offdiag_bs(cfg: config_dict.ConfigDict, bs: int) -> Tuple[int, int]:
    """Get diagonal and off-diagonal batch sizes."""
    if hasattr(cfg.optimization, "diag_fraction"):
        diag_bs = max(1, int(bs * cfg.optimization.diag_fraction))
    elif hasattr(cfg.optimization, "diag_bs"):
        diag_bs = cfg.optimization.diag_bs
    else:
        raise ValueError("Either diag_fraction or diag_bs must be specified")

    offdiag_bs = bs - diag_bs

    return diag_bs, offdiag_bs


def _concat_diag_offdiag(
    s_diag: jnp.ndarray,
    t_diag: jnp.ndarray,
    s_offdiag: jnp.ndarray,
    t_offdiag: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Concatenate diagonal and off-diagonal samples."""
    sbatch = jnp.concatenate([s_diag, s_offdiag])
    tbatch = jnp.concatenate([t_diag, t_offdiag])
    return sbatch, tbatch


@functools.partial(jax.jit, static_argnums=(1, 2, 3, 4))
def get_loss_fn_args_randomness(
    prng_key: jnp.ndarray,
    cfg: config_dict.ConfigDict,
    sample_rho0: Callable,
    diag_bs: int,
    offdiag_bs: int,
) -> Tuple:
    """Draw random values needed for each loss function iteration."""
    (
        tkey,
        skey,
        ukey,
        x0key,
        tkey2,
    ) = jax.random.split(prng_key, num=5)
    x0batch = sample_rho0(cfg.optimization.bs, x0key)

    bs = cfg.optimization.bs
    tmin = cfg.training.tmin
    tmax = cfg.training.tmax

    # If offdiag_bs is 0, use full batch on diagonal
    if offdiag_bs == 0:
        sbatch, tbatch = _sample_diagonal(skey, bs, tmin, tmax)
    else:
        # sample diagonal and off-diagonal points
        s_diag, t_diag = (
            _sample_diagonal(skey, diag_bs, tmin, tmax)
            if diag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )
        s_offdiag, t_offdiag = (
            _sample_triangle(tkey, tkey2, offdiag_bs, tmin, tmax)
            if offdiag_bs > 0
            else (jnp.array([]), jnp.array([]))
        )

        sbatch, tbatch = _concat_diag_offdiag(s_diag, t_diag, s_offdiag, t_offdiag)

    if cfg.training.psd_type == "midpoint":
        ubatch = 0.5 * (sbatch + tbatch)
        hbatch = None  # Not used for midpoint interpolation
    elif cfg.training.psd_type == "uniform":
        minval = 0.0
        maxval = 1.0

        hbatch = jax.random.uniform(
            ukey, shape=(cfg.optimization.bs,), minval=minval, maxval=maxval
        )

        ubatch = hbatch * sbatch + (1 - hbatch) * tbatch
    elif cfg.training.psd_type == None:
        ubatch = None
        hbatch = None
    else:
        raise ValueError(f"Unknown psd_type: {cfg.training.psd_type}")

    dropout_keys = jax.random.split(tkey, num=cfg.optimization.bs).reshape(
        (cfg.optimization.bs, -1)
    )
    prng_key = jax.random.split(dropout_keys[0])[0]
    return (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    )


def get_batch(
    cfg: config_dict.ConfigDict, statics: state_utils.StaticArgs, prng_key: jnp.ndarray
) -> Tuple[Optional[jnp.ndarray], jnp.ndarray, Optional[jnp.ndarray], jnp.ndarray]:
    """Extract a batch based on the structure expected for image
    or non-image datasets."""
    is_image_dataset = (cfg.problem.target in ["cifar10", "celeb_a"]) or (
        "afhq" in cfg.problem.target
    )

    batch = next(statics.ds)
    x0batch = None
    if is_image_dataset:
        x1batch = batch["image"]
        label_batch = batch["label"]
    elif isinstance(batch, dict) and "x1" in batch:
        x0batch = batch.get("x0")
        x1batch = batch["x1"]
        label_batch = batch.get("label")
    else:
        x1batch = batch
        label_batch = None

    # add droput to randomly replace fraction cfg.class_dropout of labels by num_classes
    # if not conditional, we don't need the labels
    interp_uses_labels = bool(getattr(cfg.problem, "interp_uses_labels", False))
    if not cfg.training.conditional and not interp_uses_labels:
        label_batch = None

    elif cfg.training.conditional and cfg.training.class_dropout > 0:
        assert cfg.network.use_cfg  # class dropout doesn't make sense without cfg
        mask = jax.random.bernoulli(
            prng_key, cfg.training.class_dropout, shape=(cfg.optimization.bs,)
        )
        mask = mask > 0
        label_batch = label_batch.at[mask].set(cfg.problem.num_classes)
        prng_key = jax.random.split(prng_key)[0]

    return x0batch, x1batch, label_batch, prng_key


def get_loss_fn_args(
    cfg: config_dict.ConfigDict,
    statics: state_utils.StaticArgs,
    train_state: state_utils.EMATrainState,
    prng_key: jnp.ndarray,
) -> Tuple:

    # Determine batch sizes based on splitting configuration
    bs = cfg.optimization.bs

    # Normal batch splitting
    diag_bs, offdiag_bs = _get_diag_offdiag_bs(cfg, bs)

    # drew randomness needed for the objective
    (
        tbatch,
        sbatch,
        ubatch,
        hbatch,
        x0batch,
        dropout_keys,
        prng_key,
    ) = get_loss_fn_args_randomness(
        prng_key,
        cfg,
        statics.sample_rho0,
        diag_bs,
        offdiag_bs,
    )

    # grab next batch of samples and labels
    paired_x0batch, x1batch, label_batch, prng_key = get_batch(cfg, statics, prng_key)
    if paired_x0batch is not None:
        x0batch = paired_x0batch

    # set up the teacher (uses current params for self-distillation)
    teacher_params = select_teacher_params(cfg, train_state)
    step = dist_utils.safe_index(cfg, train_state.step)
    constraint_scale = compute_constraint_anneal_scale(cfg, step)
    stage2_scale = compute_two_stage_scale(cfg, step)
    constraint_scale_batch = jnp.full(
        (cfg.optimization.bs,), constraint_scale, dtype=jnp.float32
    )
    stage2_scale_batch = jnp.full((cfg.optimization.bs,), stage2_scale, dtype=jnp.float32)

    # for training flow map
    loss_fn_args = (
        x0batch,
        x1batch,
        label_batch,
        sbatch,
        tbatch,
        ubatch,
        hbatch,
        dropout_keys,
        constraint_scale_batch,
        stage2_scale_batch,
    )
    loss_fn_args = dist_utils.replicate_loss_fn_args(cfg, loss_fn_args)
    loss_fn_args = (teacher_params, *loss_fn_args)

    return loss_fn_args, prng_key

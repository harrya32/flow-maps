"""Direct strong stochastic flow-map experiments for CITE-seq and Multiome.

The data split, PCA100 representation, classifiers, lineage graph, and
leave-one-day-out protocol come from :mod:`configs.cite_multi_pca100`.  This
module only replaces the deterministic objective with the isolated direct
SSFM objective used by :mod:`configs.maizels_stochastic`.
"""

from __future__ import annotations

from typing import Optional

import ml_collections

from . import cite_multi_pca100


VARIANTS = (
    # name, endpoint pair mode, differentiable lineage loss, minibatch OT
    ("standard_ssfm", "none", False, False),
    ("bio_prior_ssfm", "endpoint", False, False),
    ("bio_prior_constrained_ssfm", "endpoint", True, False),
    ("bio_prior_minibatch_ot_ssfm", "ot_endpoint", False, True),
    (
        "bio_prior_minibatch_ot_constrained_ssfm",
        "ot_endpoint",
        True,
        True,
    ),
    ("minibatch_ot_ssfm", "ot_plain", False, True),
    (
        "bio_prior_minibatch_ot_rollout_constrained_ssfm",
        "ot_endpoint",
        True,
        True,
    ),
)


def get_hparam_sweep_spec(slurm_id: int) -> dict:
    """Declare the grid dimensions relevant to one stochastic variant."""
    name, _, constrained, minibatch_ot = VARIANTS[int(slurm_id) % len(VARIANTS)]
    return {
        "variant_name": name,
        "learning_rate": True,
        "diffusion_scale": True,
        "constraint_weight": bool(constrained),
        "entropy_weight": bool(constrained),
        "minibatch_ot": bool(minibatch_ot),
        "launcher": "cite_multi_stochastic.py",
        "objective_sampler": "ssfm",
    }


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
    dataset_name: str | None = None,
    heldout_day: str | int | None = None,
    classifier_path: Optional[str] = None,
    full_data_classifier_path: Optional[str] = None,
    learning_rate: Optional[float] = None,
    constraint_weight: Optional[float] = None,
    entropy_weight: Optional[float] = None,
    diffusion_scale: Optional[float] = None,
    gamma_scale: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    seed: Optional[int] = None,
    total_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_pairs: Optional[int] = None,
) -> ml_collections.ConfigDict:
    """Return one of seven CITE/Multi direct-SSFM configurations."""
    variant_name, pair_mode, constrained, minibatch_ot = VARIANTS[
        int(slurm_id) % len(VARIANTS)
    ]
    rollout_constrained = "rollout_constrained" in variant_name
    cfg = cite_multi_pca100.get_config(
        1,
        dataset_location=dataset_location,
        output_folder=output_folder,
        dataset_name=dataset_name,
        heldout_day=heldout_day,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
    )

    cfg.problem.maizels_pair_mode = pair_mode
    # A Gaussian-noised stochastic interpolant is not represented by its
    # endpoint straight line.  Every stochastic bio-prior variant therefore
    # filters endpoint cell types only.
    cfg.problem.n_interpolant_check_times = 0
    if minibatch_ot:
        cfg.problem.maizels_ot_coupling = "minibatch_ot"
    cfg.problem.n = int(500_000 if n_pairs is None else n_pairs)
    if cfg.problem.n <= 0:
        raise ValueError("n_pairs must be positive.")

    cfg.optimization.bs = int(128 if batch_size is None else batch_size)
    cfg.optimization.total_steps = int(10_000 if total_steps is None else total_steps)
    if cfg.optimization.bs < 2:
        raise ValueError("batch_size must be at least 2.")
    if cfg.optimization.total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if learning_rate is not None and float(learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive.")
    cfg.optimization.learning_rate = float(
        3e-4 if learning_rate is None else learning_rate
    )
    cfg.optimization.total_samples = cfg.optimization.bs * cfg.optimization.total_steps
    cfg.optimization.decay_steps = cfg.optimization.total_steps
    cfg.optimization.schedule_type = "cosine"
    cfg.optimization.warmup_steps = min(
        1_000, max(1, cfg.optimization.total_steps // 100)
    )
    cfg.optimization.clip = 1.0
    cfg.optimization.weight_decay = 1e-4
    cfg.optimization.b1 = 0.9
    cfg.optimization.b2 = 0.99

    cfg.ssfm = ml_collections.ConfigDict()
    cfg.ssfm.n_coefficients = 3
    cfg.ssfm.gamma_scale = float(0.2 if gamma_scale is None else gamma_scale)
    cfg.ssfm.diffusion_scale = float(
        0.2 if diffusion_scale is None else diffusion_scale
    )
    if cfg.ssfm.gamma_scale <= 0.0:
        raise ValueError("gamma_scale must be positive.")
    if cfg.ssfm.diffusion_scale < 0.0:
        raise ValueError("diffusion_scale must be non-negative.")
    cfg.ssfm.local_fraction = 1.0 if rollout_constrained else 0.75
    cfg.ssfm.local_step_fraction = 0.02
    cfg.ssfm.time_eps_fraction = 0.005
    cfg.ssfm.max_horizon_fraction = 0.98
    cfg.ssfm.log_uniform_horizon = True
    cfg.ssfm.horizon_curriculum = True
    cfg.ssfm.horizon_curriculum_fraction = 0.5
    cfg.ssfm.horizon_curriculum_power = 1.0
    cfg.ssfm.ema_decay = 0.999
    cfg.ssfm.hidden_dim = 512
    cfg.ssfm.n_hidden = 3
    cfg.ssfm.uncertainty_hidden_dim = 64

    cfg.constraints.enabled = constrained
    cfg.constraints.type = "cite_multi_ssfm_lineage_path"
    cfg.constraints.path_mode = (
        "stochastic_rollout_endpoint_nll"
        if rollout_constrained
        else "direct_offdiagonal_endpoint_nll"
    )
    cfg.constraints.path_n_times = 2
    cfg.constraints.constraint_batch_size = max(1, min(64, cfg.optimization.bs // 4))
    # Zero reuses constraint_batch_size, matching the effective batch used by
    # the corresponding 75/25 direct-constraint variant.
    cfg.constraints.stochastic_rollout_batch_size = 0
    cfg.constraints.stochastic_rollout_max_step = 0.01
    cfg.constraints.stochastic_rollout_max_steps = 0
    cfg.constraints.stochastic_rollout_loss_scope = "endpoints"
    cfg.constraints.weight = float(
        10.0 if constraint_weight is None else constraint_weight
    )
    cfg.constraints.lambda_start = 0.0
    cfg.constraints.lambda_transition = 1.0
    cfg.constraints.lambda_final = 0.0
    cfg.constraints.classifier_temperature = 1.0
    cfg.constraints.loss_point_entropy_weight = float(
        0.01 if entropy_weight is None else entropy_weight
    )
    if constraint_weight is not None and not constrained:
        raise ValueError(
            "constraint_weight is relevant only to a constrained stochastic variant."
        )
    if entropy_weight is not None and not constrained:
        raise ValueError(
            "entropy_weight is relevant only to a constrained stochastic variant."
        )
    if cfg.constraints.weight < 0.0:
        raise ValueError("constraint_weight must be non-negative.")
    if cfg.constraints.loss_point_entropy_weight < 0.0:
        raise ValueError("entropy_weight must be non-negative.")

    cfg.evaluation = ml_collections.ConfigDict()
    cfg.evaluation.n_noise_draws = 3
    # Zero means the full source and held-out populations, matching the
    # deterministic CITE/Multi EMD protocol.
    cfg.evaluation.max_source_points = 0
    cfg.evaluation.max_target_points = 0
    cfg.evaluation.flowmap_n_steps = 100
    cfg.evaluation.euler_maruyama_n_steps = 50
    cfg.evaluation.lineage_max_source_points = 0
    cfg.evaluation.lineage_n_steps = 100
    cfg.evaluation.seed = int(cfg.training.seed) + 2_901
    cfg.evaluation.save_plot = True

    if seed is not None:
        cfg.training.seed = int(seed)
        cfg.evaluation.seed = int(seed) + 2_901
        cfg.logging.mfm.seed = int(seed) + 2_901
    if early_stopping_patience is not None:
        cfg.optimization.early_stopping.patience = int(early_stopping_patience)
    cfg.optimization.early_stopping.check_freq = 100
    cfg.optimization.early_stopping.min_delta = 0.0
    cfg.optimization.early_stopping.warmup_steps = 0
    cfg.optimization.early_stopping.metric = "validation_loss"
    cfg.optimization.early_stopping.mode = "min"
    if int(cfg.optimization.early_stopping.patience) < 0:
        raise ValueError("early_stopping_patience must be non-negative.")

    cfg.logging.scalar_freq = 50
    cfg.logging.progress_freq = 50
    cfg.logging.save_freq = 5_000
    cfg.logging.wandb_name = (
        f"{cfg.problem.dataset_name}_pca100_holdout_day"
        f"{cfg.problem.heldout_timepoint}_{variant_name}"
    )
    cfg.logging.output_name = cfg.logging.wandb_name
    cfg.logging.comparison_mode = variant_name

    override_tags = []
    if seed is not None:
        override_tags.append(f"seed{int(seed)}")
    if learning_rate is not None:
        override_tags.append(f"lr{float(learning_rate):g}")
    if diffusion_scale is not None:
        override_tags.append(f"diff{float(diffusion_scale):g}")
    if gamma_scale is not None:
        override_tags.append(f"gamma{float(gamma_scale):g}")
    if constraint_weight is not None:
        override_tags.append(f"cw{float(constraint_weight):g}")
    if entropy_weight is not None:
        override_tags.append(f"ew{float(entropy_weight):g}")
    if override_tags:
        suffix = "_".join(override_tags).replace(".", "p").replace("-", "m")
        cfg.logging.wandb_name = f"{cfg.logging.wandb_name}_{suffix}"
        cfg.logging.output_name = cfg.logging.wandb_name

    return cfg

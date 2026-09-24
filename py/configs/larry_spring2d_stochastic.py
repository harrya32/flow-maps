"""Strong stochastic flow maps for LARRY in the published SPRING space.

The data split, two-dimensional representation, cell-type classifiers, and
lineage graph come from :mod:`configs.larry_spring2d`.  This configuration is
consumed by the isolated SSFM launcher and does not alter deterministic runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import ml_collections

from common import larry
from . import larry_pca50


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
    (
        "bio_prior_minibatch_ot_interpolant_ssfm",
        "ot_endpoint_interpolant",
        False,
        True,
    ),
    (
        "bio_prior_minibatch_ot_interpolant_constrained_ssfm",
        "ot_endpoint_interpolant",
        True,
        True,
    ),
)


def get_hparam_sweep_spec(slurm_id: int) -> dict:
    """Declare the grid dimensions relevant to one stochastic variant."""
    name, pair_mode, constrained, minibatch_ot = VARIANTS[
        int(slurm_id) % len(VARIANTS)
    ]
    return {
        "variant_name": name,
        "learning_rate": True,
        "diffusion_scale": True,
        "constraint_weight": bool(constrained),
        "entropy_weight": bool(constrained),
        "minibatch_ot": bool(minibatch_ot),
        "interpolant_filter": pair_mode == "ot_endpoint_interpolant",
        "launcher": "larry_stochastic.py",
        "objective_sampler": "ssfm",
    }


def _checkpoint_exists(path_value: str) -> bool:
    path = Path(path_value).expanduser()
    npz = path if path.suffix == ".npz" else path.with_suffix(".npz")
    return npz.is_file()


def get_config(
    slurm_id: int,
    dataset_location: str = "",
    output_folder: str = "",
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
    ot_minibatch_size: Optional[int] = None,
    interpolant_check_times: Optional[int] = None,
    clone_samples_per_source: Optional[int] = None,
    clone_noise_draws: Optional[int] = None,
    clone_min_source_cells: Optional[int] = None,
    clone_min_target_cells: Optional[int] = None,
    larry_representation: str = larry.SPRING2D_REPRESENTATION,
    larry_clone_labelled_only: bool = False,
    larry_n_pcs: Optional[int] = None,
) -> ml_collections.ConfigDict:
    """Return one of nine LARRY SSFM experiments in one representation."""
    variant_name, pair_mode, constrained, minibatch_ot = VARIANTS[
        int(slurm_id) % len(VARIANTS)
    ]
    rollout_constrained = "rollout_constrained" in variant_name
    ot_size = 32 if ot_minibatch_size is None else int(ot_minibatch_size)
    representation = larry.canonical_representation(larry_representation)

    # ID 0 is used only as a neutral source of LARRY data and classifier
    # metadata. The objective and pair modes are replaced below.
    cfg = larry_pca50.get_config(
        0,
        dataset_location=dataset_location,
        output_folder=output_folder,
        classifier_path=classifier_path,
        full_data_classifier_path=full_data_classifier_path,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        ot_minibatch_size=ot_size,
        larry_representation=representation,
        larry_clone_labelled_only=larry_clone_labelled_only,
        larry_n_pcs=larry_n_pcs,
    )

    cfg.problem.maizels_pair_mode = pair_mode
    cfg.problem.pair_mode = pair_mode
    # Existing stochastic bio-prior variants retain endpoint-only filtering.
    # The explicitly named interpolant variants form a matched ablation that
    # additionally checks 50 points on each deterministic coupling segment.
    if interpolant_check_times is not None and pair_mode != "ot_endpoint_interpolant":
        raise ValueError(
            "interpolant_check_times is relevant only to an "
            "ot_endpoint_interpolant variant."
        )
    cfg.problem.n_interpolant_check_times = (
        int(50 if interpolant_check_times is None else interpolant_check_times)
        if pair_mode == "ot_endpoint_interpolant"
        else 0
    )
    if (
        pair_mode == "ot_endpoint_interpolant"
        and cfg.problem.n_interpolant_check_times <= 0
    ):
        raise ValueError("interpolant_check_times must be positive.")
    if minibatch_ot:
        cfg.problem.maizels_ot_coupling = "minibatch_ot"
    cfg.problem.ot_minibatch_size = ot_size
    cfg.problem.n = int(500_000 if n_pairs is None else n_pairs)
    if cfg.problem.n <= 0:
        raise ValueError("n_pairs must be positive.")
    if cfg.problem.ot_minibatch_size <= 0:
        raise ValueError("ot_minibatch_size must be positive.")
    cfg.problem.interp_uses_labels = True
    cfg.problem.pair_time_bounds_in_label = True

    # The stochastic lineage loss must use the D2/D6-only classifier. The
    # all-days classifier remains isolated for evaluation.
    cfg.problem.classifier_path = cfg.problem.training_classifier_path
    cfg.problem.flow_training_requires_classifier = bool(constrained)
    cfg.problem.training_classifier_available = _checkpoint_exists(
        cfg.problem.training_classifier_path
    )
    cfg.problem.full_data_classifier_available = _checkpoint_exists(
        cfg.problem.full_data_classifier_path
    )
    if (
        constrained
        and not cfg.problem.training_classifier_available
        and not bool(getattr(cfg.problem, "larry_auto_prepare_artifacts", False))
    ):
        representation_label = (
            "SPRING2D" if representation == larry.SPRING2D_REPRESENTATION else "PCA50"
        )
        raise FileNotFoundError(
            f"A constrained LARRY {representation_label} SSFM requires the "
            "D2/D6 classifier "
            f"checkpoint at {cfg.problem.training_classifier_path}. Run "
            "`python scripts/train_larry_celltype_classifiers.py "
            f"--representation {representation}` first."
        )

    cfg.optimization.bs = int(128 if batch_size is None else batch_size)
    cfg.optimization.total_steps = int(50_000 if total_steps is None else total_steps)
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
    cfg.constraints.type = "larry_ssfm_lineage_path"
    cfg.constraints.path_mode = (
        "stochastic_rollout_endpoint_nll"
        if rollout_constrained
        else "direct_offdiagonal_endpoint_nll"
    )
    cfg.constraints.constraint_batch_size = max(1, min(64, cfg.optimization.bs // 4))
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
        0.0 if entropy_weight is None else entropy_weight
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
    # Exact population EMD uses dense 1,024-by-1,024 costs, matching the
    # deterministic LARRY memory bound. Full SSFMs use composed maps; the
    # local-only variant is evaluated with Euler--Maruyama rollouts.
    cfg.evaluation.max_source_points = 1_024
    cfg.evaluation.max_target_points = 1_024
    cfg.evaluation.flowmap_n_steps = 50
    cfg.evaluation.euler_maruyama_n_steps = 50
    cfg.evaluation.lineage_max_source_points = 1_024
    cfg.evaluation.lineage_n_steps = 50
    cfg.evaluation.seed = int(cfg.training.seed) + 2_901
    cfg.evaluation.save_plot = True
    # Compare the raw optimizer weights with their matching EMA snapshot at
    # the validation-selected best step. Other stochastic datasets retain the
    # existing EMA-only final evaluation unless they opt into this flag.
    cfg.evaluation.evaluate_instantaneous_and_ema = False

    # Clone-conditioned evaluation draws an equal number of independent
    # stochastic trajectories from every D2 cell, while retaining every
    # observed target cousin. D4 is the held-out test; D6 is an endpoint
    # diagnostic.
    cfg.evaluation.clone_wasserstein_enabled = True
    cfg.evaluation.clone_source_time = "D2"
    cfg.evaluation.clone_target_times = ["D4", "D6"]
    cfg.evaluation.clone_samples_per_source = int(
        32 if clone_samples_per_source is None else clone_samples_per_source
    )
    cfg.evaluation.clone_n_noise_draws = int(
        3 if clone_noise_draws is None else clone_noise_draws
    )
    cfg.evaluation.clone_flowmap_n_steps = 50
    cfg.evaluation.clone_batch_size = 2_048
    cfg.evaluation.clone_min_source_cells = int(
        1 if clone_min_source_cells is None else clone_min_source_cells
    )
    cfg.evaluation.clone_min_target_cells = int(
        10 if clone_min_target_cells is None else clone_min_target_cells
    )
    cfg.evaluation.clone_max_target_cells = 0
    cfg.evaluation.clone_seed = int(cfg.training.seed) + 3_901
    if cfg.evaluation.clone_samples_per_source <= 0:
        raise ValueError("clone_samples_per_source must be positive.")
    if cfg.evaluation.clone_n_noise_draws <= 0:
        raise ValueError("clone_noise_draws must be positive.")
    if cfg.evaluation.clone_min_source_cells <= 0:
        raise ValueError("clone_min_source_cells must be positive.")
    if cfg.evaluation.clone_min_target_cells <= 0:
        raise ValueError("clone_min_target_cells must be positive.")

    # Keep the inherited metadata informative even though stochastic metrics
    # are produced by cfg.evaluation rather than deterministic logging hooks.
    cfg.logging.larry.clone_sampler = (
        "euler_maruyama" if rollout_constrained else "stochastic_composed_flowmap"
    )
    cfg.logging.larry.clone_flowmap_n_steps = 50
    cfg.logging.larry.clone_samples_per_source = cfg.evaluation.clone_samples_per_source
    cfg.logging.larry.clone_n_noise_draws = cfg.evaluation.clone_n_noise_draws

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
    subset_tag = (
        "_clone_labelled"
        if bool(getattr(cfg.problem, "larry_clone_labelled_only", False))
        else ""
    )
    representation_tag = (
        "spring2d"
        if representation == larry.SPRING2D_REPRESENTATION
        else f"pca{int(cfg.problem.n_pcs)}"
    )
    cfg.logging.wandb_name = (
        f"larry_{representation_tag}{subset_tag}_holdout_d4_{variant_name}"
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

import pytest

from configs import maizels_pca50
from scripts import sweep_maizels_hparams as sweep


def test_unconstrained_variant_sweeps_only_learning_rate():
    spec = maizels_pca50.get_hparam_sweep_spec(3)
    grid = sweep.build_grid(
        spec,
        learning_rates=(3e-4, 1e-3),
        constraint_weights=(100.0, 1000.0),
        entropy_weights=(0.0, 0.1),
    )

    assert len(grid) == 2
    assert {row["learning_rate"] for row in grid} == {3e-4, 1e-3}
    assert all(row["constraint_weight"] is None for row in grid)
    assert all(row["entropy_weight"] is None for row in grid)


def test_constrained_nll_variant_sweeps_all_relevant_dimensions():
    spec = maizels_pca50.get_hparam_sweep_spec(4)
    grid = sweep.build_grid(
        spec,
        learning_rates=(3e-4, 1e-3),
        constraint_weights=(100.0, 1000.0),
        entropy_weights=(0.0, 0.1),
    )

    assert len(grid) == 8
    assert {row["constraint_weight"] for row in grid} == {100.0, 1000.0}
    assert {row["entropy_weight"] for row in grid} == {0.0, 0.1}


def test_maizels_config_applies_only_relevant_overrides():
    constrained = maizels_pca50.get_config(
        4,
        learning_rate=3e-4,
        constraint_weight=350.0,
        entropy_weight=0.1,
        seed=2,
    )
    assert constrained.optimization.learning_rate == 3e-4
    assert constrained.constraints.weight == 350.0
    assert constrained.constraints.loss_point_entropy_weight == 0.1
    assert constrained.training.seed == 2

    with pytest.raises(ValueError, match="not relevant"):
        maizels_pca50.get_config(3, constraint_weight=350.0)
    with pytest.raises(ValueError, match="relevant only"):
        maizels_pca50.get_config(3, entropy_weight=0.1)


def test_seed_summary_uses_mean_validation_emd_across_completed_seeds():
    setting = {
        "setting_id": "setting-a",
        "slurm_id": 3,
        "variant_name": "bio_prior_flow_map",
        "maizels_schedule": "d3_d3p8_d8",
        "maizels_time_mode": "real_time",
        "hparam_val_times": "D3.4,D6",
        "learning_rate": 1e-3,
        "constraint_weight": "",
        "entropy_weight": "",
    }
    metric = "final_eval/direct_mean_emd_hparam_val_times"
    rows = [
        {
            "setting_id": "setting-a",
            "status": "complete",
            "seed": seed,
            metric: value,
        }
        for seed, value in zip((0, 1, 2), (1.0, 2.0, 3.0))
    ]

    summary = sweep.summarize_settings(rows, [setting], (0, 1, 2), metric)[0]

    assert summary["status"] == "complete"
    assert summary["objective_mean"] == 2.0
    assert summary["objective_std"] == 1.0
    assert summary["n_seeds_completed"] == 3

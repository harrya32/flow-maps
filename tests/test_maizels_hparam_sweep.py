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
    )
    assert constrained.optimization.learning_rate == 3e-4
    assert constrained.constraints.weight == 350.0
    assert constrained.constraints.loss_point_entropy_weight == 0.1

    with pytest.raises(ValueError, match="not relevant"):
        maizels_pca50.get_config(3, constraint_weight=350.0)
    with pytest.raises(ValueError, match="relevant only"):
        maizels_pca50.get_config(3, entropy_weight=0.1)

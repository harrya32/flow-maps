from types import SimpleNamespace

import numpy as np

from common import maizels
from configs import maizels_pca50


def _cfg(*, pair_mode="none", n_pairs=12):
    return SimpleNamespace(
        problem=SimpleNamespace(
            n=n_pairs,
            retained_timepoints=["D3", "D3.8", "D8"],
            timepoint_order=list(maizels.TIMEPOINTS),
            timepoint_values=[
                0.0,
                0.04,
                0.08,
                0.12,
                0.16,
                0.2,
                0.4,
                0.6,
                0.8,
                1.0,
            ],
            source_time="D3",
            target_time="D8",
            pair_time_bounds_in_label=True,
            maizels_pair_mode=pair_mode,
            maizels_ot_coupling="minibatch_ot",
            lineage_transition_mode="descendant",
            ot_minibatch_size=4,
            ot_minibatch_max_resamples=3,
            ot_minibatch_infeasible_fallback="partial",
        ),
        training=SimpleNamespace(seed=7),
    )


def _pool(value, *, train_n=6, holdout_n=3):
    x = np.full((train_n + holdout_n, 2), value, dtype=np.float32)
    types = np.full(train_n + holdout_n, "NMP", dtype=object)
    return {
        "x": x,
        "types": types,
        "train_x": x[:train_n],
        "train_types": types[:train_n],
        "holdout_x": x[train_n:],
        "holdout_types": types[train_n:],
        "train_idx": np.arange(train_n),
        "holdout_idx": np.arange(train_n, train_n + holdout_n),
    }


def test_maizels_clock_and_retained_interval_lookup():
    cfg = _cfg()

    assert maizels.normalized_time("D3.8", cfg) == 0.16
    assert maizels.retained_interval_for_timepoint(cfg, "D3.4") == ("D3", "D3.8")
    assert maizels.retained_interval_for_timepoint(cfg, "D6") == ("D3.8", "D8")


def test_three_timepoint_config_supports_real_and_equal_clocks():
    real_cfg = maizels_pca50.get_config(
        0,
        maizels_schedule="d3_d3p8_d8",
        maizels_time_mode="real_time",
    )
    equal_cfg = maizels_pca50.get_config(
        0,
        maizels_schedule="d3_d3p8_d8",
        maizels_time_mode="equal_time",
    )

    assert list(real_cfg.problem.retained_timepoints) == ["D3", "D3.8", "D8"]
    assert list(real_cfg.problem.evaluation_timepoints) == [
        "D3.2",
        "D3.4",
        "D3.6",
        "D4",
        "D5",
        "D6",
        "D7",
    ]
    assert np.isclose(real_cfg.problem.timepoint_values[4], 0.16)
    assert np.isclose(equal_cfg.problem.timepoint_values[4], 0.5)
    assert real_cfg.problem.interp_type == "time_rescaled_linear"
    assert real_cfg.problem.pair_time_bounds_in_label
    assert "maizels_classifier_d3_d3p8_d8" in real_cfg.problem.classifier_path
    assert real_cfg.logging.maizels.full_data_classifier_path.endswith(
        "celltype_classifier_pca50.pt"
    )
    assert (
        real_cfg.logging.maizels.full_data_classifier_path
        != real_cfg.problem.classifier_path
    )
    assert real_cfg.logging.maizels.trajectory_diagnostics_enabled
    assert real_cfg.logging.output_name == "maizels_pca50_vanilla_flow_matching"
    assert equal_cfg.logging.output_name == real_cfg.logging.output_name


def test_default_config_preserves_endpoint_training():
    cfg = maizels_pca50.get_config(0)

    assert cfg.problem.maizels_schedule == "d3_d8"
    assert cfg.problem.maizels_time_mode == "real_time"
    assert list(cfg.problem.retained_timepoints) == ["D3", "D8"]
    assert cfg.problem.interp_type == "linear"
    assert not cfg.problem.pair_time_bounds_in_label
    assert (
        cfg.logging.maizels.full_data_classifier_path
        == cfg.problem.classifier_path
    )


def test_training_pool_contains_only_adjacent_retained_intervals(monkeypatch):
    cfg = _cfg(n_pairs=12)
    pools = {"D3": _pool(3.0), "D3.8": _pool(3.8), "D8": _pool(8.0)}
    monkeypatch.setattr(
        maizels,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )

    paired, stats = maizels.make_pair_pool(cfg)

    assert paired["label"].shape == (12, 4)
    bounds, counts = np.unique(paired["label"][:, 2:4], axis=0, return_counts=True)
    np.testing.assert_allclose(bounds, [[0.0, 0.16], [0.16, 1.0]])
    np.testing.assert_array_equal(counts, [6, 6])
    assert set(stats["intervals"]) == {"D3_to_D3p8", "D3p8_to_D8"}


def test_retained_days_are_split_and_endpoint_holdouts_are_unchanged(
    monkeypatch,
):
    cfg = _cfg()
    cfg.problem.maizels_holdout_fraction = 0.25
    cfg.problem.maizels_holdout_n = 0
    cfg.problem.maizels_holdout_seed = 701
    rows_per_timepoint = 8
    timepoints = np.repeat(np.asarray(maizels.TIMEPOINTS, dtype=object), rows_per_timepoint)
    data = {
        "x": np.arange(timepoints.size * 2, dtype=np.float32).reshape(-1, 2),
        "timepoints": timepoints,
        "cell_types": np.full(timepoints.size, "NMP", dtype=object),
    }
    monkeypatch.setattr(maizels, "resolve_dataset_path", lambda _: "unused.csv")
    monkeypatch.setattr(maizels, "load_pca50_dataset", lambda _: data)

    pools = maizels.timepoint_pool_splits(cfg, dataset_location="unused.csv")
    expected_d3 = maizels._split_train_holdout_indices(
        rows_per_timepoint,
        holdout_fraction=0.25,
        holdout_n=0,
        seed=701 + 11,
    )
    expected_d8 = maizels._split_train_holdout_indices(
        rows_per_timepoint,
        holdout_fraction=0.25,
        holdout_n=0,
        seed=701 + 29,
    )

    np.testing.assert_array_equal(pools["D3"]["train_idx"], expected_d3[0])
    np.testing.assert_array_equal(pools["D3"]["holdout_idx"], expected_d3[1])
    np.testing.assert_array_equal(pools["D8"]["train_idx"], expected_d8[0])
    np.testing.assert_array_equal(pools["D8"]["holdout_idx"], expected_d8[1])
    expected_d3p8 = maizels._split_train_holdout_indices(
        rows_per_timepoint,
        holdout_fraction=0.25,
        holdout_n=0,
        seed=701 + 101 * (maizels.TIMEPOINTS.index("D3.8") + 1),
    )
    np.testing.assert_array_equal(pools["D3.8"]["train_idx"], expected_d3p8[0])
    np.testing.assert_array_equal(
        pools["D3.8"]["holdout_idx"], expected_d3p8[1]
    )


def test_validation_pool_matches_training_intervals_on_heldout_cells(monkeypatch):
    cfg = _cfg(n_pairs=12)
    pools = {"D3": _pool(3.0), "D3.8": _pool(3.8), "D8": _pool(8.0)}
    monkeypatch.setattr(
        maizels,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )

    paired, stats = maizels.make_validation_pair_pool(cfg, 8, pair_mode="none")

    assert paired["label"].shape == (8, 4)
    bounds, counts = np.unique(paired["label"][:, 2:4], axis=0, return_counts=True)
    np.testing.assert_allclose(bounds, [[0.0, 0.16], [0.16, 1.0]])
    np.testing.assert_array_equal(counts, [4, 4])
    assert stats["split"] == "heldout"


def test_global_diagnostic_pairs_include_full_trajectory_bounds(monkeypatch):
    cfg = _cfg(n_pairs=12)
    source = _pool(3.0)
    target = _pool(8.0)
    splits = maizels._endpoint_split_dict(source, target)
    monkeypatch.setattr(
        maizels,
        "endpoint_pool_splits",
        lambda cfg, dataset_location=None: splits,
    )

    paired, _ = maizels.make_heldout_pair_pool(cfg, 8, pair_mode="none")

    assert paired["label"].shape == (8, 4)
    np.testing.assert_allclose(paired["label"][:, 2:4], [[0.0, 1.0]] * 8)


def test_dynamic_minibatch_ot_is_balanced_by_interval():
    cfg = _cfg(pair_mode="ot_endpoint", n_pairs=16)
    rng = np.random.default_rng(9)
    class_id = maizels.class_to_id_map()["NMP"]
    bounds = np.repeat(
        np.asarray([[0.0, 0.16], [0.16, 1.0]], dtype=np.float32),
        8,
        axis=0,
    )
    candidates = {
        "x0": rng.normal(size=(16, 3)).astype(np.float32),
        "x1": rng.normal(size=(16, 3)).astype(np.float32),
        "label": np.concatenate(
            [np.full((16, 2), class_id, dtype=np.float32), bounds], axis=1
        ),
    }

    paired, stats = maizels.couple_minibatch_ot_pair_pool(
        cfg,
        candidates,
        8,
        seed=19,
    )

    unique_bounds, counts = np.unique(
        paired["label"][:, 2:4], axis=0, return_counts=True
    )
    np.testing.assert_allclose(unique_bounds, [[0.0, 0.16], [0.16, 1.0]])
    np.testing.assert_array_equal(counts, [4, 4])
    assert stats["coupling"] == "dynamic_minibatch_ot"

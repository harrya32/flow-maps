from types import SimpleNamespace

import numpy as np

from common import cite_multi, maizels


def _cfg(pair_mode="ot_endpoint", coupling="minibatch_ot", batch_size=4):
    return SimpleNamespace(
        problem=SimpleNamespace(
            maizels_pair_mode=pair_mode,
            maizels_ot_coupling=coupling,
            ot_minibatch_size=batch_size,
            ot_minibatch_max_resamples=3,
            ot_minibatch_infeasible_fallback="partial",
            lineage_transition_mode="descendant",
        ),
        training=SimpleNamespace(seed=7),
    )


def test_maizels_minibatch_ot_is_optional():
    cfg = _cfg()
    assert maizels.uses_minibatch_ot(cfg)

    cfg.problem.maizels_ot_coupling = "global_ot"
    assert not maizels.uses_minibatch_ot(cfg)

    cfg.problem.maizels_ot_coupling = "minibatch_ot"
    cfg.problem.maizels_pair_mode = "endpoint_interpolant"
    assert not maizels.uses_minibatch_ot(cfg)


def test_minibatch_ot_cost_is_raw_squared_euclidean():
    source = np.asarray([[0.0, 0.0], [1.0, 10.0]])
    target = np.asarray([[3.0, 4.0], [1.0, 12.0]])

    cost = cite_multi._squared_euclidean_cost(source, target)

    np.testing.assert_allclose(cost, [[25.0, 145.0], [40.0, 4.0]])


def test_maizels_masked_minibatch_ot_returns_fresh_batch():
    cfg = _cfg()
    class_to_id = maizels.class_to_id_map(maizels.CLASS_NAMES)
    rng = np.random.default_rng(4)
    candidates = {
        "x0": rng.normal(size=(12, 3)).astype(np.float32),
        "x1": rng.normal(size=(12, 3)).astype(np.float32),
        "label": np.column_stack(
            [
                np.full(12, class_to_id["NMP"]),
                np.full(12, class_to_id["Mesoderm"]),
            ]
        ).astype(np.int32),
    }

    paired, stats = maizels.couple_minibatch_ot_pair_pool(
        cfg,
        candidates,
        4,
        seed=19,
    )

    assert paired["x0"].shape == (4, 3)
    assert paired["x1"].shape == (4, 3)
    assert paired["label"].shape == (4, 2)
    assert stats["coupling"] == "dynamic_minibatch_ot"
    assert stats["ot_minibatch_size"] == 4
    assert stats["ot_cost"] == "raw_sqeuclidean"


def test_training_pool_stays_independent_until_batch_coupling(monkeypatch):
    cfg = _cfg(pair_mode="ot_plain", batch_size=8)
    cfg.problem.n = 20
    rng = np.random.default_rng(9)
    source_x = rng.normal(size=(7, 3)).astype(np.float32)
    target_x = rng.normal(size=(9, 3)).astype(np.float32)
    class_names = np.asarray(maizels.CLASS_NAMES, dtype=object)
    source_types = class_names[np.arange(source_x.shape[0]) % len(class_names)]
    target_types = class_names[np.arange(target_x.shape[0]) % len(class_names)]
    monkeypatch.setattr(
        maizels,
        "endpoint_pool_splits",
        lambda cfg, dataset_location=None: {
            "source_train_x": source_x,
            "source_train_types": source_types,
            "target_train_x": target_x,
            "target_train_types": target_types,
            "source_n": source_x.shape[0],
            "target_n": target_x.shape[0],
            "source_train_n": source_x.shape[0],
            "source_holdout_n": 0,
            "target_train_n": target_x.shape[0],
            "target_holdout_n": 0,
        },
    )

    paired, stats = maizels.make_pair_pool(cfg)

    assert paired["x0"].shape == (20, 3)
    assert stats["pair_pool_mode"] == "independent_candidates"
    assert stats["coupling"] == "dynamic_minibatch_ot"
    assert stats["pair_mode"] == "ot_plain"


def test_compact_maizels_pool_keeps_original_cells(monkeypatch):
    cfg = _cfg(pair_mode="ot_plain", batch_size=4)
    cfg.problem.n = 500_000
    source_x = np.arange(15, dtype=np.float32).reshape(5, 3)
    target_x = np.arange(21, dtype=np.float32).reshape(7, 3)
    source_types = np.full(5, "NMP", dtype=object)
    target_types = np.full(7, "Mesoderm", dtype=object)
    monkeypatch.setattr(
        maizels,
        "endpoint_pool_splits",
        lambda cfg, dataset_location=None: {
            "source_x": source_x,
            "source_types": source_types,
            "target_x": target_x,
            "target_types": target_types,
            "source_train_x": source_x,
            "source_train_types": source_types,
            "target_train_x": target_x,
            "target_train_types": target_types,
            "source_holdout_x": source_x[:0],
            "source_holdout_types": source_types[:0],
            "target_holdout_x": target_x[:0],
            "target_holdout_types": target_types[:0],
        },
    )

    pools, stats = maizels.make_minibatch_ot_training_pools(cfg)

    assert "x0" not in pools and "x1" not in pools
    assert sum(item["x"].shape[0] for item in pools["timepoints"].values()) == 12
    assert pools["nominal_n"] == 500_000
    assert stats["pair_pool_mode"] == "direct_timepoint_pools"
    assert stats["stored_endpoint_cells"] == 12
    assert stats["expanded_pair_rows_avoided"] == 500_000


def test_direct_maizels_pool_sampling_is_seeded_and_uses_replacement(monkeypatch):
    cfg = _cfg(pair_mode="ot_plain", batch_size=4)
    class_id = maizels.class_to_id_map()["NMP"]
    pools = {
        "timepoints": {
            "D3": {
                "x": np.arange(6, dtype=np.float32).reshape(2, 3),
                "type_ids": np.full(2, class_id, dtype=np.int32),
            },
            "D8": {
                "x": np.arange(9, dtype=np.float32).reshape(3, 3),
                "type_ids": np.full(3, class_id, dtype=np.int32),
            },
        },
        "intervals": (
            {
                "source_time": "D3",
                "target_time": "D8",
                "t_start": 0.0,
                "t_end": 1.0,
                "nominal_pairs": 500_000,
            },
        ),
        "include_time_bounds": False,
    }
    cost_shapes = []

    def fake_ot(cost, n_samples, rng):
        cost_shapes.append(cost.shape)
        indices = np.arange(n_samples, dtype=np.int64)
        return indices, indices, {"ot_solver_mode": "test"}

    monkeypatch.setattr(cite_multi, "_sample_dense_ot_plan", fake_ot)
    first, first_stats = maizels.couple_minibatch_ot_timepoint_pools(
        cfg, pools, 4, seed=19
    )
    second, _ = maizels.couple_minibatch_ot_timepoint_pools(
        cfg, pools, 4, seed=19
    )

    assert cost_shapes == [(4, 4), (4, 4)]
    for key in ("x0", "x1", "label"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["label"].shape == (4, 2)
    assert first_stats["pair_pool_mode"] == "direct_timepoint_pools"


def test_compact_cite_pool_balances_retained_intervals(monkeypatch):
    cfg = SimpleNamespace(
        problem=SimpleNamespace(
            n=500_000,
            pair_mode="ot_plain",
            ot_minibatch_size=4,
            ot_minibatch_max_resamples=3,
            heldout_timepoint="4",
            dataset_name="cite",
        ),
        training=SimpleNamespace(seed=7),
    )

    def pool(value, n):
        x = np.full((n, 3), value, dtype=np.float32)
        types = np.full(n, "HSC", dtype=object)
        return {
            "x": x,
            "types": types,
            "train_x": x,
            "train_types": types,
            "holdout_x": x[:0],
            "holdout_types": types[:0],
        }

    raw_pools = {
        "2": pool(2.0, 3),
        "3": pool(3.0, 4),
        "4": pool(4.0, 5),
        "7": pool(7.0, 6),
    }
    monkeypatch.setattr(
        cite_multi,
        "_timepoint_splits",
        lambda cfg, dataset_location=None: raw_pools,
    )
    pools, stats = cite_multi.make_minibatch_ot_training_pools(cfg)

    assert set(pools["timepoints"]) == {"2", "3", "7"}
    assert stats["stored_endpoint_cells"] == 13
    assert [item["nominal_pairs"] for item in pools["intervals"]] == [250_000] * 2

    def fake_ot(cost, n_samples, rng):
        indices = np.arange(n_samples, dtype=np.int64)
        return indices, indices, {"ot_solver_mode": "test"}

    monkeypatch.setattr(cite_multi, "_sample_dense_ot_plan", fake_ot)
    paired, batch_stats = cite_multi.couple_minibatch_ot_timepoint_pools(
        cfg, pools, 8, seed=29
    )

    bounds, counts = np.unique(
        paired["label"][:, 2:4], axis=0, return_counts=True
    )
    np.testing.assert_allclose(bounds, [[0.0, 1.0 / 3.0], [1.0 / 3.0, 1.0]])
    np.testing.assert_array_equal(counts, [4, 4])
    assert batch_stats["pair_pool_mode"] == "direct_timepoint_pools"

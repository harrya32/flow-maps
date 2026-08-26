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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import common.logging as logging_common
from common import larry
from configs import larry_pca50, larry_spring2d


@pytest.fixture(autouse=True)
def no_installed_larry_classifiers(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: False,
    )


def test_larry_has_independent_and_filtered_flow_map_variants():
    cfg = larry_pca50.get_config(0)

    assert cfg.problem.target == "larry_pca50"
    assert cfg.problem.base == "larry_day2"
    assert cfg.problem.source_time == "D2"
    assert cfg.problem.target_time == "D6"
    assert cfg.problem.heldout_timepoint == "D4"
    assert list(cfg.problem.retained_timepoints) == ["D2", "D6"]
    assert list(cfg.problem.evaluation_timepoints) == ["D4"]
    assert cfg.problem.maizels_pair_mode == "none"
    assert cfg.optimization.diag_fraction == 0.75
    assert not cfg.constraints.enabled
    assert cfg.logging.comparison_mode == "independent_flow_map"
    assert larry_pca50.ACTIVE_VARIANTS == (
        "independent_flow_map",
        "bio_prior_flow_map",
        "bio_prior_constrained_flow_map",
        "ot_flow_map",
        "bio_prior_ot_flow_map",
        "bio_prior_ot_constrained_flow_map",
    )

    with pytest.raises(ValueError, match="Slurm ids 0--5"):
        larry_pca50.get_config(6)


def test_larry_filtered_variant_uses_d2_d6_classifier(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )
    cfg = larry_pca50.get_config(1)

    assert cfg.logging.comparison_mode == "bio_prior_flow_map"
    assert cfg.problem.maizels_pair_mode == "endpoint_interpolant"
    assert cfg.problem.flow_training_requires_classifier
    assert cfg.problem.training_classifier_available
    assert Path(cfg.problem.training_classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca50_train_days_d2_d6.pt"
    )
    assert Path(cfg.problem.full_data_classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca50_all_days.pt"
    )
    assert cfg.problem.classifier_path == cfg.problem.training_classifier_path
    assert cfg.problem.n_interpolant_check_times == 50
    assert cfg.problem.classifier_prob_threshold == 0.0
    assert cfg.problem.classifier_margin_threshold == 0.0
    assert cfg.optimization.diag_fraction == 0.75
    assert not cfg.constraints.enabled


def test_clone_labelled_setting_selects_separate_cached_artifacts(tmp_path):
    cfg = larry_pca50.get_config(
        0,
        dataset_location=str(tmp_path),
        larry_clone_labelled_only=True,
    )

    assert cfg.problem.larry_clone_labelled_only
    assert cfg.problem.larry_auto_prepare_artifacts
    assert Path(cfg.problem.dataset_location).name == (
        "stateFate_inVitro_clone_labelled_hvg2000_pca50.h5ad"
    )
    assert list(cfg.problem.pca_fit_timepoints) == ["D2", "D4", "D6"]
    assert Path(cfg.problem.training_classifier_path).name == (
        "celltype_classifier_larry_clone_labelled_hvg2000_pca50_"
        "train_days_d2_d6.pt"
    )
    assert Path(cfg.problem.full_data_classifier_path).name == (
        "celltype_classifier_larry_clone_labelled_hvg2000_pca50_all_days.pt"
    )
    assert "clone_labelled" in cfg.logging.output_name


def test_larry_pca_dimension_selects_matching_data_model_and_cache_names(tmp_path):
    cfg = larry_pca50.get_config(
        0,
        dataset_location=str(tmp_path),
        larry_n_pcs=20,
    )

    assert cfg.problem.n_pcs == 20
    assert cfg.problem.d == 20
    assert tuple(cfg.network.input_dims) == (20,)
    assert cfg.network.output_dim == 20
    assert cfg.problem.larry_auto_prepare_artifacts
    assert Path(cfg.problem.dataset_location).name == (
        "stateFate_inVitro_hvg2000_pca20.h5ad"
    )
    assert Path(cfg.problem.training_classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca20_train_days_d2_d6.pt"
    )
    assert Path(cfg.problem.full_data_classifier_path).name == (
        "celltype_classifier_larry_hvg2000_pca20_all_days.pt"
    )
    assert "pca20" in cfg.logging.output_name


def test_larry_pca_dimension_must_be_positive():
    with pytest.raises(ValueError, match="larry_n_pcs must be positive"):
        larry_pca50.get_config(0, larry_n_pcs=0)


def test_clone_labelled_setting_is_not_available_for_spring():
    with pytest.raises(ValueError, match="PCA"):
        larry_pca50.get_config(
            0,
            larry_representation=larry.SPRING2D_REPRESENTATION,
            larry_clone_labelled_only=True,
        )
    with pytest.raises(ValueError, match="only for PCA"):
        larry_spring2d.get_config(0, larry_n_pcs=20)


@pytest.mark.parametrize(
    (
        "slurm_id",
        "variant_name",
        "pair_mode",
        "constrained",
        "minibatch_ot",
        "requires_classifier",
    ),
    [
        (
            2,
            "bio_prior_constrained_flow_map",
            "endpoint_interpolant",
            True,
            False,
            True,
        ),
        (3, "ot_flow_map", "ot_plain", False, True, False),
        (
            4,
            "bio_prior_ot_flow_map",
            "ot_endpoint_interpolant",
            False,
            True,
            True,
        ),
        (
            5,
            "bio_prior_ot_constrained_flow_map",
            "ot_endpoint_interpolant",
            True,
            True,
            True,
        ),
    ],
)
def test_larry_additional_flow_map_variants(
    monkeypatch,
    slurm_id,
    variant_name,
    pair_mode,
    constrained,
    minibatch_ot,
    requires_classifier,
):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )

    cfg = larry_pca50.get_config(slurm_id)

    assert cfg.logging.comparison_mode == variant_name
    assert cfg.problem.maizels_pair_mode == pair_mode
    assert cfg.constraints.enabled is constrained
    assert cfg.constraints.type == "maizels_lineage_path"
    assert cfg.constraints.path_mode == "loss_points_nll"
    assert cfg.problem.flow_training_requires_classifier is requires_classifier
    assert cfg.problem.maizels_ot_coupling == "minibatch_ot"
    assert larry.uses_minibatch_ot(cfg) is minibatch_ot


def test_larry_constraint_sweep_spec_only_exposes_constraint_hparams_when_used():
    unconstrained = larry_pca50.get_hparam_sweep_spec(3)
    constrained = larry_pca50.get_hparam_sweep_spec(5)

    assert not unconstrained["constraint_weight"]
    assert not unconstrained["entropy_weight"]
    assert constrained["constraint_weight"]
    assert constrained["entropy_weight"]


def test_larry_ot_minibatch_size_can_be_smaller_than_optimizer_batch(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )

    cfg = larry_pca50.get_config(4, ot_minibatch_size=32)

    assert cfg.optimization.bs == 128
    assert cfg.problem.ot_minibatch_size == 32

    with pytest.raises(ValueError, match="ot_minibatch_size must be positive"):
        larry_pca50.get_config(4, ot_minibatch_size=0)


def test_larry_lineage_allows_progenitor_differentiation_and_committed_self_only():
    invalid = larry.lineage_invalid_transition_matrix(larry.CLASS_NAMES)
    class_id = {name: index for index, name in enumerate(larry.CLASS_NAMES)}
    progenitor = class_id["Undifferentiated"]

    assert np.all(invalid[progenitor] == 0.0)
    for cell_type in larry.CLASS_NAMES:
        if cell_type == "Undifferentiated":
            continue
        row = invalid[class_id[cell_type]]
        assert row[class_id[cell_type]] == 0.0
        assert int(np.count_nonzero(row == 0.0)) == 1


def test_larry_filtered_pair_builder_receives_lineage_graph(monkeypatch):
    cfg = larry_pca50.get_config(0)
    source = np.zeros((3, 50), dtype=np.float32)
    target = np.ones((4, 50), dtype=np.float32)
    source_types = np.full(3, "Undifferentiated", dtype=object)
    target_types = np.full(4, "Neutrophil", dtype=object)
    seen = {}

    def fake_builder(*args, **kwargs):
        seen.update(kwargs)
        n_pairs = int(kwargs["n_pairs"])
        return {
            "x0": np.repeat(source[:1], n_pairs, axis=0),
            "x1": np.repeat(target[:1], n_pairs, axis=0),
            "label": np.zeros((n_pairs, 2), dtype=np.int32),
        }, {"accepted_pairs": n_pairs}

    monkeypatch.setattr(
        larry.maizels,
        "_make_pair_pool_from_endpoint_arrays",
        fake_builder,
    )
    paired, stats = larry._pair_from_arrays(
        cfg,
        source,
        source_types,
        target,
        target_types,
        n_pairs=7,
        pair_mode="endpoint_interpolant",
        seed=5,
    )

    assert paired["x0"].shape == (7, 50)
    assert stats["accepted_pairs"] == 7
    assert seen["pair_mode"] == "endpoint_interpolant"
    assert tuple(seen["class_names"]) == larry.CLASS_NAMES
    assert tuple(seen["transition_edges"]) == larry.TRANSITION_EDGES


def test_larry_filtered_minibatch_ot_checks_interpolants(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )
    cfg = larry_pca50.get_config(4)
    cfg.problem.ot_minibatch_size = 4
    cfg.problem.ot_minibatch_max_resamples = 2
    class_id = {name: index for index, name in enumerate(larry.CLASS_NAMES)}
    pools = {
        "timepoints": {
            "D2": {
                "x": np.arange(24, dtype=np.float32).reshape(8, 3),
                "type_ids": np.full(8, class_id["Undifferentiated"], dtype=np.int32),
            },
            "D6": {
                "x": np.arange(24, 48, dtype=np.float32).reshape(8, 3),
                "type_ids": np.full(8, class_id["Neutrophil"], dtype=np.int32),
            },
        },
        "intervals": (
            {
                "source_time": "D2",
                "target_time": "D6",
                "t_start": 0.0,
                "t_end": 1.0,
                "nominal_pairs": 500_000,
            },
        ),
    }
    seen = {}

    def accept_interpolants(**kwargs):
        seen.update(kwargs)
        count = kwargs["source_x"].shape[0]
        return {"valid": np.ones(count, dtype=bool)}

    monkeypatch.setattr(
        larry.maizels,
        "_check_candidate_interpolants",
        accept_interpolants,
    )
    paired, stats = larry.couple_minibatch_ot_timepoint_pools(
        cfg,
        pools,
        4,
        seed=17,
    )

    assert paired["x0"].shape == (4, 3)
    assert paired["x1"].shape == (4, 3)
    assert paired["label"].shape == (4, 2)
    assert stats["pair_mode"] == "ot_endpoint_interpolant"
    assert stats["coupling"] == "minibatch_ot"
    assert stats["interpolant_checked"] == 4
    assert stats["interpolant_rejected"] == 0
    assert tuple(seen["transition_edges"]) == larry.TRANSITION_EDGES


def test_larry_minibatch_ot_keeps_compact_endpoint_pools(monkeypatch):
    cfg = larry_pca50.get_config(3)
    cfg.problem.n = 500_000
    source = np.zeros((7, 50), dtype=np.float32)
    target = np.ones((9, 50), dtype=np.float32)
    source_types = np.full(7, "Undifferentiated", dtype=object)
    target_types = np.full(9, "Neutrophil", dtype=object)
    monkeypatch.setattr(
        larry,
        "endpoint_pool_splits",
        lambda cfg, dataset_location=None: {
            "source_train_x": source,
            "source_train_types": source_types,
            "target_train_x": target,
            "target_train_types": target_types,
        },
    )

    pools, stats = larry.make_minibatch_ot_training_pools(cfg)

    assert "x0" not in pools and "x1" not in pools
    assert pools["nominal_n"] == 500_000
    assert pools["dimension"] == 50
    assert stats["pair_pool_mode"] == "direct_timepoint_pools"
    assert stats["coupling"] == "dynamic_minibatch_ot"
    assert stats["stored_endpoint_cells"] == 16


def test_larry_default_processed_path_and_evaluation_protocol():
    cfg = larry_pca50.get_config(0)

    assert Path(cfg.problem.dataset_location) == (
        Path.home() / "Desktop" / "flow-maps-data" / larry.DEFAULT_FILENAME
    )
    assert cfg.problem.pca_fit_timepoints == ["D2", "D4", "D6"]
    assert cfg.logging.maizels.distribution_eval_timepoints == ["D4"]
    assert cfg.logging.maizels.distribution_eval_source_pool == "all"
    assert cfg.logging.maizels.distribution_eval_source_max_points == 1024
    assert cfg.logging.maizels.distribution_eval_points_per_time == 1024
    assert cfg.logging.maizels.distribution_eval_samplers == ["flowmap"]
    assert cfg.logging.maizels.distribution_eval_flowmap_n_steps == 50
    assert cfg.logging.maizels.distribution_eval_metrics == ["emd"]
    assert cfg.logging.maizels.trajectory_eval_samplers == ["flowmap"]
    assert cfg.logging.maizels.check_n_times == 50
    assert not cfg.logging.maizels.violation_metrics_available
    assert "No LARRY" in cfg.logging.maizels.violation_metrics_unavailable_reason
    assert cfg.logging.larry.clone_source_time == "D2"
    assert cfg.logging.larry.clone_target_time == "D4"
    assert cfg.logging.larry.clone_sampler == "flowmap"
    assert cfg.logging.larry.clone_flowmap_n_steps == 50
    assert cfg.logging.larry.clone_max_cells_per_population == 0


def test_larry_spring_config_mirrors_protocol_in_two_dimensions(monkeypatch):
    monkeypatch.setattr(
        larry_pca50,
        "_runtime_checkpoint_exists",
        lambda path: True,
    )

    cfg = larry_spring2d.get_config(4, ot_minibatch_size=32)

    assert cfg.problem.target == "larry_spring2d"
    assert cfg.problem.larry_representation == "spring2d"
    assert cfg.problem.larry_representation_key == "X_spring"
    assert cfg.problem.d == 2
    assert cfg.network.input_dims == (2,)
    assert cfg.network.output_dim == 2
    assert cfg.optimization.bs == 128
    assert cfg.problem.ot_minibatch_size == 32
    assert Path(cfg.problem.dataset_location).name == larry.DEFAULT_SPRING_FILENAME
    assert Path(cfg.problem.training_classifier_path).name == (
        "celltype_classifier_larry_spring2d_train_days_d2_d6.pt"
    )
    assert Path(cfg.problem.full_data_classifier_path).name == (
        "celltype_classifier_larry_spring2d_all_days.pt"
    )
    assert cfg.logging.maizels.distribution_eval_timepoints == ["D4"]
    assert cfg.logging.larry.clone_target_time == "D4"


def test_larry_pair_pool_uses_d2_and_d6_only(monkeypatch):
    cfg = larry_pca50.get_config(0)
    cfg.problem.n = 12

    def pool(value):
        x = np.full((5, 50), value, dtype=np.float32)
        types = np.full(5, "Undifferentiated", dtype=object)
        return {
            "x": x,
            "types": types,
            "train_x": x[:4],
            "train_types": types[:4],
            "holdout_x": x[4:],
            "holdout_types": types[4:],
            "train_idx": np.arange(4, dtype=np.int64),
            "holdout_idx": np.asarray([4], dtype=np.int64),
        }

    monkeypatch.setattr(
        larry,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: {
            "D2": pool(2.0),
            "D4": pool(4.0),
            "D6": pool(6.0),
        },
    )
    paired, stats = larry.make_pair_pool(cfg)

    assert paired["x0"].shape == (12, 50)
    assert paired["x1"].shape == (12, 50)
    assert np.all(paired["x0"] == 2.0)
    assert np.all(paired["x1"] == 6.0)
    assert stats["heldout_timepoint"] == "D4"


def test_missing_classifier_is_reported_instead_of_scored(capsys):
    cfg = larry_pca50.get_config(0)
    metrics = logging_common._compute_maizels_population_trajectory_metrics(
        cfg,
        SimpleNamespace(),
        {},
    )

    assert metrics == {
        "lineage_eval/violation_metrics_available": 0.0,
        "lineage_eval/violation_metrics_skipped_no_classifier": 1.0,
    }
    assert "No LARRY PCA50" in capsys.readouterr().out


def test_classifier_metrics_use_only_composed_flowmap_paths(monkeypatch):
    cfg = larry_pca50.get_config(0)
    cfg.logging.maizels.violation_metrics_available = True
    cfg.problem.classifier_path = "fake-classifier.pt"
    cfg.logging.maizels.full_data_classifier_path = "fake-classifier.pt"
    source = np.zeros((2, 5), dtype=np.float32)
    source_types = np.full(2, "Undifferentiated", dtype=object)
    seen = {}

    monkeypatch.setattr(
        logging_common,
        "_maizels_trajectory_eval_population",
        lambda *args, **kwargs: (source, source_types, source, source_types),
    )

    def composed_paths(apply_fn, params, x0, labels, n_steps):
        seen["n_steps"] = n_steps
        return np.repeat(np.asarray(x0)[:, None, :], n_steps + 1, axis=1)

    monkeypatch.setattr(logging_common, "_multi_step_paths", composed_paths)
    monkeypatch.setattr(
        logging_common,
        "_one_step_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct paths must not be evaluated")
        ),
    )
    monkeypatch.setattr(
        logging_common,
        "_euler_paths",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Euler paths must not be evaluated")
        ),
    )

    def classify(paths, **kwargs):
        seen["classifier_path_shape"] = paths.shape
        return {"valid": np.asarray([True, False])}

    monkeypatch.setattr(larry, "check_paths_with_classifier", classify)
    metrics = logging_common._compute_maizels_population_trajectory_metrics(
        cfg,
        SimpleNamespace(apply_fn=None),
        {},
    )

    assert seen["n_steps"] == 50
    assert seen["classifier_path_shape"] == (2, 50, 5)
    assert metrics["maizels/model_flowmap_invalid_trajectory_pct"] == 50.0
    assert not any("model_direct" in key for key in metrics)
    assert not any("model_euler" in key for key in metrics)


def test_d4_emd_uses_capped_composed_flowmap_even_when_forced(monkeypatch):
    cfg = larry_pca50.get_config(0)
    cfg.logging.maizels.distribution_eval_source_max_points = 2
    cfg.logging.maizels.distribution_eval_points_per_time = 2
    d2 = np.arange(15, dtype=np.float32).reshape(3, 5)
    d4 = np.arange(10, dtype=np.float32).reshape(2, 5)
    d6 = np.arange(20, dtype=np.float32).reshape(4, 5)
    data = {
        "x": np.concatenate([d2, d4, d6]),
        "timepoints": np.asarray(["D2"] * 3 + ["D4"] * 2 + ["D6"] * 4),
        "time_values": np.asarray([2.0] * 3 + [4.0] * 2 + [6.0] * 4),
    }
    types2 = np.full(3, "Undifferentiated", dtype=object)
    types6 = np.full(4, "Neutrophil", dtype=object)
    endpoint_splits = {
        "source_x": d2,
        "source_types": types2,
        "target_x": d6,
        "target_types": types6,
        "source_train_x": d2[:2],
        "source_train_types": types2[:2],
        "target_train_x": d6[:3],
        "target_train_types": types6[:3],
        "source_holdout_x": d2[2:],
        "source_holdout_types": types2[2:],
        "target_holdout_x": d6[3:],
        "target_holdout_types": types6[3:],
        "source_train_n": 2,
        "source_holdout_n": 1,
        "target_train_n": 3,
        "target_holdout_n": 1,
    }
    seen = {}
    monkeypatch.setattr(larry, "all_timepoint_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        larry,
        "endpoint_pool_splits",
        lambda *args, **kwargs: endpoint_splits,
    )

    def record_pushforward(apply_fn, params, source, labels, **kwargs):
        seen["source"] = np.asarray(source).copy()
        seen["times"] = (kwargs["start_time"], kwargs["end_time"])
        seen["n_steps"] = kwargs["n_steps"]
        assert labels is None
        return np.asarray(source)

    def record_emd(predicted, actual):
        seen["emd_shapes"] = (predicted.shape, actual.shape)
        return 7.0

    monkeypatch.setattr(
        logging_common,
        "_flowmap_terminal_between",
        record_pushforward,
    )
    monkeypatch.setattr(logging_common, "_mfm_exact_emd", record_emd)
    monkeypatch.setattr(logging_common.wandb, "log", lambda payload: None)

    metrics = logging_common._log_maizels_distribution_eval(
        cfg,
        SimpleNamespace(apply_fn=None),
        {},
        force=True,
    )

    assert seen["source"].shape == (2, 5)
    assert set(map(tuple, seen["source"])).issubset(set(map(tuple, d2)))
    assert seen["times"] == (0.0, 0.5)
    assert seen["n_steps"] == 50
    assert seen["emd_shapes"] == ((2, 5), (2, 5))
    assert metrics["distribution_eval/D4_flowmap_emd"] == 7.0
    assert set(key for key in metrics if key.endswith("_emd")) == {
        "distribution_eval/D4_flowmap_emd"
    }


def test_clone_wasserstein_uses_shared_d2_d4_clones_and_all_members(monkeypatch):
    cfg = larry_pca50.get_config(0)
    data = {
        "x": np.asarray([[0.0], [2.0], [10.0], [1.0], [12.0], [14.0], [99.0]]),
        "timepoints": np.asarray(
            ["D2", "D2", "D2", "D4", "D4", "D4", "D4"], dtype=object
        ),
        "clone_ids": np.asarray([0, 0, 1, 0, 1, 1, -1], dtype=np.int64),
    }
    seen = {}

    monkeypatch.setattr(larry, "all_timepoint_data", lambda *args, **kwargs: data)

    def identity_pushforward(apply_fn, params, source, **kwargs):
        seen["source"] = np.asarray(source).copy()
        seen["times"] = (kwargs["start_time"], kwargs["end_time"])
        seen["n_steps"] = kwargs["n_steps"]
        return np.asarray(source)

    monkeypatch.setattr(
        logging_common,
        "_composed_flowmap_population_in_batches",
        identity_pushforward,
    )
    monkeypatch.setattr(
        logging_common,
        "_mfm_exact_emd",
        lambda predicted, actual: abs(float(np.mean(predicted) - np.mean(actual))),
    )
    monkeypatch.setattr(logging_common.wandb, "log", lambda payload: None)

    metrics = logging_common._log_larry_clone_wasserstein_eval(
        cfg,
        SimpleNamespace(apply_fn=None),
        {},
        force=True,
    )

    assert seen["source"].shape[0] == 3
    assert seen["times"] == (0.0, 0.5)
    assert seen["n_steps"] == 50
    assert metrics["clone_eval/d2_to_d4_eligible_clone_count"] == 2
    assert metrics["clone_eval/d2_to_d4_source_cell_count"] == 3
    assert metrics["clone_eval/d2_to_d4_target_cell_count"] == 3
    assert metrics["clone_eval/d2_to_d4_wasserstein_macro"] == pytest.approx(1.5)
    assert metrics["clone_eval/d2_to_d4_wasserstein_source_weighted"] == pytest.approx(
        1.0
    )
    assert metrics["clone_eval/d2_to_d4_wasserstein_target_weighted"] == pytest.approx(
        2.0
    )

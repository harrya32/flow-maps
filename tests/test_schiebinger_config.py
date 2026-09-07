from pathlib import Path

import numpy as np
import pytest

from common import maizels, schiebinger
from configs import schiebinger_lsd


def test_custom_training_times_include_endpoints_and_define_eval_days():
    cfg = schiebinger_lsd.get_config(
        0,
        schiebinger_train_times="0, 3, 8.25, 18",
    )

    assert list(cfg.problem.retained_timepoints) == ["0", "3", "8.25", "18"]
    assert "3" not in cfg.problem.evaluation_timepoints
    assert "8.25" not in cfg.problem.evaluation_timepoints
    assert list(cfg.problem.evaluation_timepoints) == [
        timepoint
        for timepoint in schiebinger.TIMEPOINTS
        if timepoint not in {"0", "3", "8.25", "18"}
    ]
    assert cfg.problem.pair_time_bounds_in_label
    assert cfg.problem.interp_type == "time_rescaled_linear"


def test_internal_training_times_define_a_local_experiment_window():
    cfg = schiebinger_lsd.get_config(0, schiebinger_train_times="10,11,12")

    assert cfg.problem.source_time == "10"
    assert cfg.problem.target_time == "12"
    assert list(cfg.problem.retained_timepoints) == ["10", "11", "12"]
    assert list(cfg.problem.evaluation_timepoints) == ["10.5", "11.5"]
    assert list(cfg.problem.timepoint_order) == [
        "10", "10.5", "11", "11.5", "12"
    ]
    assert np.allclose(cfg.problem.timepoint_values, [0, 0.25, 0.5, 0.75, 1])
    assert schiebinger.experiment_timepoints(cfg) == (
        "10", "10.5", "11", "11.5", "12"
    )
    assert schiebinger.evaluation_timepoints(cfg) == ("10.5", "11.5")


def test_default_dataset_path_and_future_classifier_path():
    cfg = schiebinger_lsd.get_config(0)

    assert Path(cfg.problem.dataset_location) == (
        Path.home() / "Desktop" / "schiebinger" / schiebinger.DEFAULT_FILENAME
    )
    assert list(cfg.problem.retained_timepoints) == ["0", "18"]
    assert Path(cfg.problem.training_classifier_path).name == (
        "celltype_classifier_schiebinger_hvg1479_pca5_train_days_0_18.pt"
    )
    assert Path(cfg.problem.classifier_path).name == (
        "celltype_classifier_schiebinger_hvg1479_pca5_all_days.pt"
    )
    assert Path(cfg.problem.classifier_path).parent.name == "schiebinger_classifiers"
    assert cfg.logging.maizels.enabled
    assert cfg.logging.maizels.distribution_eval_interval_local
    assert cfg.logging.maizels.distribution_eval_source_max_points == 0
    assert cfg.logging.maizels.distribution_eval_points_per_time == 0
    assert not cfg.problem.flow_training_requires_classifier


def test_arbitrary_pca_dimension_changes_data_model_and_classifier_paths():
    cfg = schiebinger_lsd.get_config(
        2,
        schiebinger_n_pcs=12,
        schiebinger_train_times="0,3,18",
    )

    assert cfg.problem.n_pcs == 12
    assert cfg.problem.d == 12
    assert tuple(cfg.network.input_dims) == (12,)
    assert cfg.network.output_dim == 12
    assert Path(cfg.problem.dataset_location).name == (
        "schiebinger_serum_serum_hvg1479_pca12.h5ad"
    )
    assert "hvg1479_pca12" in Path(cfg.problem.classifier_path).name
    assert "hvg1479_pca12" in Path(
        cfg.logging.maizels.full_data_classifier_path
    ).name


def test_schiebinger_graph_and_unconstrained_types_are_explicit():
    cfg = schiebinger_lsd.get_config(4)

    assert tuple(tuple(edge) for edge in cfg.problem.lineage_transition_edges) == (
        ("MEF/other", "MET"),
        ("MET", "IPS"),
        ("MEF/other", "Stromal"),
    )
    assert set(cfg.problem.lineage_unconstrained_class_names) == {
        "Epithelial",
        "Trophoblast",
        "Neural",
    }
    assert cfg.problem.maizels_pair_mode == "endpoint_interpolant"
    assert cfg.constraints.enabled
    assert cfg.problem.flow_training_requires_classifier
    assert cfg.problem.classifier_path == cfg.problem.training_classifier_path
    assert (
        cfg.problem.classifier_path
        != cfg.logging.maizels.full_data_classifier_path
    )


def test_unconstrained_types_do_not_bridge_forbidden_constrained_transitions():
    reachable = maizels.build_transition_reachable(
        "descendant",
        edges=schiebinger.TRANSITION_EDGES,
        class_names=schiebinger.CLASS_NAMES,
        unconstrained_class_names=schiebinger.UNCONSTRAINED_CLASS_NAMES,
    )

    # Explicit and transitively implied transitions are allowed.
    assert "MET" in reachable["MEF/other"]
    assert "IPS" in reachable["MEF/other"]
    assert "Stromal" in reachable["MEF/other"]

    # Any transition touching a deliberately unconstrained class is allowed.
    for cell_type in schiebinger.CLASS_NAMES:
        assert cell_type in reachable["Neural"]
        assert "Neural" in reachable[cell_type]

    # Wildcards are added after closure, so they do not create this path.
    assert "Stromal" not in reachable["MET"]
    assert "MEF/other" not in reachable["IPS"]


def test_prior_free_and_plain_ot_variants_do_not_enable_lineage_constraints():
    vanilla = schiebinger_lsd.get_config(0)
    plain_ot = schiebinger_lsd.get_config(7)

    assert vanilla.problem.maizels_pair_mode == "none"
    assert not vanilla.constraints.enabled
    assert plain_ot.problem.maizels_pair_mode == "ot_plain"
    assert not plain_ot.constraints.enabled


def test_pair_pool_uses_only_adjacent_selected_training_times(monkeypatch):
    cfg = schiebinger_lsd.get_config(
        0,
        schiebinger_train_times="0,9,18",
    )
    cfg.problem.n = 6

    def pool(value):
        x = np.full((4, 5), value, dtype=np.float32)
        types = np.full(4, "MEF/other", dtype=object)
        return {
            "x": x,
            "types": types,
            "train_x": x[:3],
            "train_types": types[:3],
            "holdout_x": x[3:],
            "holdout_types": types[3:],
        }

    monkeypatch.setattr(
        schiebinger,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: {
            "0": pool(0.0),
            "9": pool(9.0),
            "18": pool(18.0),
        },
    )
    paired, stats = schiebinger.make_pair_pool(cfg)

    assert paired["x0"].shape == (6, 5)
    assert paired["label"].shape == (6, 4)
    assert set(map(tuple, paired["label"][:, 2:4])) == {
        (0.0, 0.5),
        (0.5, 1.0),
    }
    assert set(stats["intervals"]) == {"0_to_9", "9_to_18"}
    assert all(item["sampled_pairs"] == 3 for item in stats["intervals"].values())

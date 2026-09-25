from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common import cite_multi
from common import logging as logging_common
from configs import cite_multi_pca100


def test_time_modes_use_global_four_day_clock_and_are_named_explicitly():
    equal_cfg = cite_multi_pca100.get_config(0, cite_multi_time_mode="equal_time")
    real_cfg = cite_multi_pca100.get_config(0, cite_multi_time_mode="real_time")

    assert equal_cfg.problem.cite_multi_time_mode == "equal_time"
    assert equal_cfg.problem.timepoint_values == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    )
    assert real_cfg.problem.cite_multi_time_mode == "real_time"
    assert real_cfg.problem.timepoint_values == pytest.approx([0.0, 0.2, 0.4, 1.0])
    assert "equal_time" in equal_cfg.logging.wandb_name
    assert "real_time" in real_cfg.logging.wandb_name
    assert cite_multi.normalized_time("4", real_cfg) == pytest.approx(0.4)
    assert equal_cfg.logging.maizels.observed_distribution_eval_enabled

    paired = {
        "x0": np.zeros((2, 1), dtype=np.float32),
        "x1": np.ones((2, 1), dtype=np.float32),
        "label": np.zeros((2, 2), dtype=np.float32),
    }
    timed = cite_multi._add_time_bounds(real_cfg, paired, "3", "7")
    np.testing.assert_allclose(timed["label"][:, 2:], [[0.2, 1.0], [0.2, 1.0]])


def test_invalid_time_mode_is_rejected():
    with pytest.raises(ValueError, match="time mode"):
        cite_multi_pca100.get_config(0, cite_multi_time_mode="calendar")


def test_plain_ot_flow_matching_variant_accepts_seed_and_learning_rate():
    cfg = cite_multi_pca100.get_config(
        9,
        dataset_name="cite",
        heldout_day="4",
        learning_rate=1e-3,
        seed=3,
    )

    assert cfg.logging.comparison_mode == "ot_flow_matching"
    assert cfg.problem.maizels_pair_mode == "ot_plain"
    assert cfg.optimization.diag_fraction == 1.0
    assert cfg.optimization.learning_rate == pytest.approx(1e-3)
    assert cfg.training.seed == 3
    assert "seed3_lr0p001" in cfg.logging.output_name


def test_observed_eval_uses_euler_for_cite_flow_matching(monkeypatch):
    cfg = cite_multi_pca100.get_config(0, heldout_day="4")
    pools = {
        "2": {"x": np.zeros((2, 2), dtype=np.float32)},
        "3": {"x": np.full((3, 2), 10.0, dtype=np.float32)},
        "7": {"x": np.full((4, 2), 20.0, dtype=np.float32)},
    }
    monkeypatch.setattr(
        cite_multi,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    seen = []

    def fake_euler(apply_fn, params, xs, labels, **kwargs):
        seen.append((float(np.mean(xs)), kwargs["n_steps"]))
        return np.asarray(xs) + 1.0

    monkeypatch.setattr(logging_common, "_euler_terminal_between", fake_euler)
    monkeypatch.setattr(
        logging_common,
        "_flowmap_terminal_between",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("flow-matching evaluation used a flow-map sampler")
        ),
    )
    monkeypatch.setattr(
        logging_common,
        "_mfm_exact_emd",
        lambda prediction, target: float(np.mean(prediction)),
    )

    metrics = logging_common._compute_observed_marginal_emd_metrics(
        cfg,
        SimpleNamespace(apply_fn=lambda *args, **kwargs: None),
        {},
    )

    assert seen == [(0.0, 100), (10.0, 100), (1.0, 100)]
    assert metrics["final_eval/observed_day2_to_day3_euler_emd"] == 1.0
    assert metrics["final_eval/observed_day3_to_day7_euler_emd"] == 11.0
    assert metrics["final_eval/day2_to_day7_rollout_euler_emd"] == 2.0


def test_training_and_evaluation_use_separate_dataset_specific_classifiers():
    for dataset_name in ("cite", "multi"):
        for heldout_day in ("3", "4"):
            cfg = cite_multi_pca100.get_config(
                0,
                dataset_name=dataset_name,
                heldout_day=heldout_day,
            )

            classifier_dir = f"{dataset_name}-classifiers"
            assert Path(cfg.problem.classifier_path).parent.name == classifier_dir
            assert cfg.problem.classifier_path.endswith(
                f"celltype_classifier_{dataset_name}_pca100_"
                f"except_day{heldout_day}.pt"
            )
            assert (
                Path(cfg.logging.maizels.full_data_classifier_path).parent.name
                == classifier_dir
            )
            assert cfg.logging.maizels.full_data_classifier_path.endswith(
                f"celltype_classifier_{dataset_name}_pca100_all_days.pt"
            )
            assert (
                cfg.logging.maizels.full_data_classifier_path
                != cfg.problem.classifier_path
            )


def test_training_and_evaluation_classifier_overrides_are_independent(tmp_path):
    training_path = tmp_path / "training.pt"
    evaluation_path = tmp_path / "evaluation.pt"
    cfg = cite_multi_pca100.get_config(
        0,
        dataset_name="cite",
        heldout_day="3",
        classifier_path=str(training_path),
        full_data_classifier_path=str(evaluation_path),
    )

    assert cfg.problem.classifier_path == str(training_path.resolve())
    assert cfg.logging.maizels.full_data_classifier_path == str(
        evaluation_path.resolve()
    )

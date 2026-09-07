from types import SimpleNamespace

import numpy as np

import common.logging as logging_common
from common import maizels
from configs import maizels_pca50


def _pool(value):
    x = np.full((2, 2), value, dtype=np.float32)
    types = np.full(2, "NMP", dtype=object)
    return {
        "x": x,
        "types": types,
        "train_x": x,
        "train_types": types,
        "holdout_x": x,
        "holdout_types": types,
    }


def test_distribution_eval_reports_all_validation_and_test_emd_means(monkeypatch):
    cfg = maizels_pca50.get_config(
        1,
        maizels_schedule="d3_d3p8_d8",
        hparam_val_times="D3.4",
    )
    cfg.logging.maizels.distribution_eval_timepoints = ["D3.4", "D6"]
    cfg.logging.maizels.distribution_eval_points_per_time = 2
    cfg.logging.maizels.distribution_eval_source_max_points = 2

    data = {
        "x": np.concatenate(
            [
                np.ones((2, 2), dtype=np.float32),
                np.full((2, 2), 3.0, dtype=np.float32),
            ],
            axis=0,
        ),
        "timepoints": np.asarray(["D3.4", "D3.4", "D6", "D6"]),
        "time_values": np.asarray([3.4, 3.4, 6.0, 6.0], dtype=np.float32),
    }
    pools = {
        "D3": _pool(0.0),
        "D3.8": _pool(0.0),
        "D8": _pool(0.0),
    }

    monkeypatch.setattr(
        maizels,
        "all_timepoint_data",
        lambda dataset_location=None: data,
    )
    monkeypatch.setattr(
        maizels,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    monkeypatch.setattr(
        logging_common,
        "_flow_map_batch",
        lambda apply_fn, params, s, t, xs, labels: np.asarray(xs),
    )
    monkeypatch.setattr(
        logging_common,
        "_flowmap_terminal_between",
        lambda apply_fn, params, xs, labels, **kwargs: np.asarray(xs),
    )
    monkeypatch.setattr(
        logging_common,
        "_euler_terminal_between",
        lambda apply_fn, params, xs, labels, **kwargs: np.asarray(xs),
    )
    monkeypatch.setattr(logging_common, "_rbf_mmd2_np", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        logging_common,
        "_mfm_exact_emd",
        lambda predicted, actual: float(np.mean(actual)),
    )
    monkeypatch.setattr(logging_common.wandb, "log", lambda payload: None)

    metrics = logging_common._log_maizels_distribution_eval(
        cfg,
        SimpleNamespace(apply_fn=lambda *args, **kwargs: None),
        {},
        force=True,
    )

    assert metrics["distribution_eval/direct_emd_mean"] == 2.0
    assert metrics["distribution_eval/direct_mean_emd_hparam_val_times"] == 1.0
    assert metrics["distribution_eval/direct_mean_emd_test_times"] == 3.0


def test_final_eval_promotes_validation_and_test_emd_metrics(monkeypatch):
    cfg = maizels_pca50.get_config(1, maizels_schedule="d3_d3p8_d8")
    distribution_metrics = {
        "distribution_eval/direct_emd_mean": 2.0,
        "distribution_eval/direct_mean_emd_hparam_val_times": 1.0,
        "distribution_eval/direct_mean_emd_test_times": 3.0,
    }
    monkeypatch.setattr(
        logging_common,
        "_log_maizels_distribution_eval",
        lambda *args, **kwargs: distribution_metrics,
    )
    monkeypatch.setattr(
        logging_common,
        "_compute_maizels_population_trajectory_metrics",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(logging_common.wandb, "log", lambda payload: None)
    monkeypatch.setattr(logging_common.wandb, "run", None)

    metrics = logging_common.log_maizels_final_evaluation(
        cfg,
        SimpleNamespace(),
        {},
        best_step=10,
    )

    assert metrics["final_eval/direct_mean_emd"] == 2.0
    assert metrics["final_eval/direct_mean_emd_hparam_val_times"] == 1.0
    assert metrics["final_eval/direct_mean_emd_test_times"] == 3.0

from types import SimpleNamespace

import numpy as np

import common.logging as logging_common
from common import schiebinger
from configs import schiebinger_lsd


def test_interval_local_distribution_eval_supports_schiebinger(monkeypatch):
    cfg = schiebinger_lsd.get_config(
        1,
        schiebinger_train_times="0,3,18",
    )
    cfg.logging.maizels.distribution_eval_timepoints = ["0.5"]
    cfg.logging.maizels.distribution_eval_points_per_time = 2
    cfg.logging.maizels.distribution_eval_source_max_points = 2
    cfg.logging.maizels.distribution_eval_max_timepoints = 0

    data = {
        "x": np.arange(30, dtype=np.float32).reshape(6, 5),
        "timepoints": np.asarray(["0", "0", "0.5", "0.5", "3", "18"]),
        "time_values": np.asarray([0, 0, 0.5, 0.5, 3, 18], dtype=np.float32),
    }
    source = np.zeros((2, 5), dtype=np.float32)
    target = np.ones((2, 5), dtype=np.float32)
    pools = {
        "0": {"x": source, "train_x": source, "holdout_x": source},
        "3": {"x": target, "train_x": target, "holdout_x": target},
    }
    endpoint_counts = {
        "source_holdout_n": 2,
        "target_holdout_n": 2,
        "source_train_n": 2,
        "target_train_n": 2,
    }

    monkeypatch.setattr(
        schiebinger,
        "all_timepoint_data",
        lambda dataset_location=None, cfg=None: data,
    )
    monkeypatch.setattr(
        schiebinger,
        "timepoint_pool_splits",
        lambda cfg, dataset_location=None: pools,
    )
    monkeypatch.setattr(
        schiebinger,
        "endpoint_pool_splits",
        lambda cfg, dataset_location=None: endpoint_counts,
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
    monkeypatch.setattr(logging_common, "_mfm_exact_emd", lambda *args: 0.0)
    monkeypatch.setattr(logging_common.wandb, "log", lambda payload: None)

    metrics = logging_common._log_maizels_distribution_eval(
        cfg,
        SimpleNamespace(apply_fn=lambda *args, **kwargs: None),
        {},
        force=True,
    )

    assert metrics["distribution_eval/0p5_direct_emd"] == 0.0
    assert metrics["distribution_eval/direct_emd_mean"] == 0.0

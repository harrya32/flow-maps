"""Maizels MFM evaluation must retain unequal, complete cell populations."""

import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MFM_ROOT = REPO_ROOT / "metric-flow-matching"
if str(MFM_ROOT) not in sys.path:
    sys.path.insert(0, str(MFM_ROOT))

from mfm.flow_matchers import maizels_eval  # noqa: E402
from mfm.train.parsers import parse_args  # noqa: E402
from scripts import mfm_maizels_sweep_utils, sweep_maizels_mfm_hparams  # noqa: E402


class ZeroField(torch.nn.Module):
    def forward(self, time, state):
        return torch.zeros_like(state)


def population(size, offset):
    return np.column_stack(
        [offset + np.arange(size) * 0.0001, np.zeros(size)]
    ).astype(np.float32)


def run_evaluations(cap=None):
    with mock.patch.object(sys, "argv", ["mfm.train.main"]):
        args = parse_args()
    args.seed_current = 0
    args.maizels_eval_euler_steps = 2
    if cap is not None:
        args.maizels_eval_points_per_time = cap

    retained = ["D3", "D3.8", "D8"]
    pools = {
        day: {"x": population(n, offset), "holdout_x": population(h, offset)}
        for day, n, h, offset in (
            ("D3", 1103, 5, 0.0),
            ("D3.8", 1301, 7, 0.5),
            ("D8", 901, 4, 1.0),
        )
    }
    omitted = {
        "D3.4": population(1207, 1.0),
        "D4": population(1409, 3.0),
        "D6": population(2, 9.0),
    }
    data = {
        "x": np.concatenate(list(omitted.values())),
        "timepoints": np.concatenate(
            [np.full(len(x), day) for day, x in omitted.items()]
        ),
        "time_values": np.concatenate(
            [np.full(len(x), float(day[1:])) for day, x in omitted.items()]
        ),
    }
    datamodule = SimpleNamespace(
        cfg=SimpleNamespace(problem=SimpleNamespace(retained_timepoints=retained)),
        retained_timepoints=retained,
        timepoint_splits=pools,
        all_timepoint_data=data,
        eval_pairs={
            "x0": pools["D3"]["holdout_x"][:2],
            "x1": pools["D8"]["holdout_x"][:2],
            "label": np.zeros((2, 2), dtype=np.int32),
        },
        classifier_path="classifier.pt",
    )
    flow = ZeroField()
    module = SimpleNamespace(flow_net=flow, device=torch.device("cpu"))
    trainer = SimpleNamespace(
        global_step=args.maizels_eval_every_n_steps,
        is_global_zero=True,
        checkpoint_callback=SimpleNamespace(
            best_model_score=torch.tensor(0.25), best_model_path="best.ckpt"
        ),
    )
    callback = maizels_eval.MaizelsEvaluationCallback(args, datamodule)
    populations = []
    logged = []
    summary = {}

    def record_emd(prediction, actual):
        populations.append((prediction.copy(), actual.copy()))
        return float(np.mean(actual[:, 0]))

    with (
        mock.patch.object(maizels_eval.wasserstein, "exact_emd", record_emd),
        mock.patch.object(maizels_eval, "rbf_mmd2", return_value=0.0) as mmd,
        mock.patch.object(
            maizels_eval.maizels,
            "check_paths_with_classifier",
            return_value={"valid": np.ones(2, dtype=bool)},
        ),
        mock.patch.object(callback, "_distribution_figure", return_value=None),
        mock.patch.object(callback, "_trajectory_figure", return_value=None),
        mock.patch.object(maizels_eval.wandb, "Image", return_value="plot"),
        mock.patch.object(maizels_eval.wandb, "log", side_effect=logged.append),
        mock.patch.object(
            maizels_eval.wandb, "run", SimpleNamespace(summary=summary)
        ),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        callback.on_train_batch_end(trainer, module, None, None, 0)
        callback.on_test_start(trainer, module)

    return SimpleNamespace(
        args=args,
        pools=pools,
        omitted=omitted,
        populations=populations,
        mmd_calls=mmd.call_args_list,
        logged=logged,
        summary=summary,
        flow=flow,
    )


class MaizelsEvaluationTests(unittest.TestCase):
    def test_periodic_and_final_metrics_use_complete_unequal_populations(self):
        result = run_evaluations()
        self.assertEqual(result.args.maizels_eval_points_per_time, 0)
        expected = [
            (result.pools["D3"]["x"], result.omitted["D3.4"]),
            (result.pools["D3.8"]["x"], result.omitted["D4"]),
            (result.pools["D3.8"]["x"], result.omitted["D6"]),
            (result.pools["D3"]["holdout_x"], result.pools["D3.8"]["holdout_x"]),
            (result.pools["D3"]["holdout_x"], result.pools["D3.8"]["holdout_x"]),
            (result.pools["D3.8"]["holdout_x"], result.pools["D8"]["holdout_x"]),
            (result.pools["D3.8"]["holdout_x"], result.pools["D8"]["holdout_x"]),
        ]
        self.assertEqual(len(result.populations), 2 * len(expected))
        for actual_pair, expected_pair in zip(result.populations, expected * 2):
            for actual, original in zip(actual_pair, expected_pair):
                np.testing.assert_array_equal(actual, original)
        for call, expected_pair in zip(result.mmd_calls, expected[:3] * 2):
            for actual, original in zip(call.args[:2], expected_pair):
                np.testing.assert_array_equal(actual, original)

        test_mean = float(np.mean(result.omitted["D4"][:, 0]))
        hparam_mean = float(
            np.mean(
                [float(np.mean(result.omitted[day][:, 0])) for day in ("D3.4", "D6")]
            )
        )
        self.assertEqual(
            result.logged[0]["distribution_eval/euler_mean_emd_test_times"],
            test_mean,
        )
        self.assertEqual(
            result.logged[1]["final_eval/euler_mean_emd_test_times"], test_mean
        )
        self.assertAlmostEqual(
            result.summary["final_eval/euler_mean_emd_hparam_val_times"],
            hparam_mean,
        )
        self.assertTrue(result.flow.training)

    def test_optional_cap_samples_each_population_independently(self):
        result = run_evaluations(cap=3)
        for stage_start in (0, 7):
            prediction, actual = result.populations[stage_start + 2]
            self.assertEqual(prediction.shape[0], 3)
            self.assertEqual(actual.shape[0], 2)
            self.assertEqual(np.unique(prediction[:, 0]).size, 3)
            self.assertTrue(
                np.isin(prediction[:, 0], result.pools["D3.8"]["x"][:, 0]).all()
            )
            np.testing.assert_array_equal(actual, result.omitted["D6"])

    def test_sweep_accepts_the_all_cell_override(self):
        args = sweep_maizels_mfm_hparams.parse_args(
            ["--dataset-path", "unused.csv.gz", "--eval-points-per-time", "0"]
        )
        mfm_maizels_sweep_utils.validate_common_args(args)
        self.assertEqual(
            mfm_maizels_sweep_utils.common_runtime_overrides(args)[
                "maizels_eval_points_per_time"
            ],
            0,
        )
        args.eval_points_per_time = -1
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            mfm_maizels_sweep_utils.validate_common_args(args)


if __name__ == "__main__":
    unittest.main()

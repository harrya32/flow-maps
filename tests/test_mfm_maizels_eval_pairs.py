"""All-cell MFM evaluation must create non-empty classifier trajectories."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "metric-flow-matching"))

from mfm.dataloaders.maizels_data import MaizelsEndpointDataModule
from mfm.flow_matchers import maizels_eval
from mfm.train.parsers import parse_args

maizels = maizels_eval.maizels


class ZeroField(torch.nn.Module):
    def forward(self, time, state):
        return torch.zeros_like(state)


class NMPClassifier:
    def __init__(self):
        self.batch_sizes = []

    def predict_logits(self, x):
        self.batch_sizes.append(len(x))
        logits = np.zeros((len(x), len(maizels.CLASS_NAMES)), dtype=np.float32)
        logits[:, maizels.CLASS_NAMES.index("NMP")] = 5.0
        return logits


def make_datamodule(schedule, cap, output_dir):
    with mock.patch.object(sys, "argv", ["mfm.train.main"]):
        args = parse_args()
    args.seed_current = 0
    args.working_dir = str(output_dir)
    args.maizels_dataset_path = "fixture.csv"
    args.maizels_classifier_path = "fixture.pt"
    args.maizels_schedule = schedule
    args.maizels_pair_mode = "none"
    args.maizels_eval_points_per_time = cap
    args.maizels_eval_euler_steps = 2
    args.maizels_classifier_batch_size = 3
    args.maizels_n_pairs = 12
    args.maizels_validation_pairs = 8
    args.batch_size = 2
    args.final_metrics_path = str(output_dir / "final_metrics.json")

    pools = {}
    for index, day in enumerate(maizels.TIMEPOINTS):
        count = 12 + index % 3
        x = np.column_stack([
            index + np.arange(count) * 0.01,
            np.zeros(count),
        ]).astype(np.float32)
        holdout_n = 5 if day == "D3" else 4
        holdout_idx = np.arange(holdout_n)
        train_idx = np.arange(holdout_n, count)
        types = np.full(count, "NMP")
        pools[day] = {
            "x": x, "types": types,
            "train_x": x[train_idx], "train_types": types[train_idx],
            "holdout_x": x[holdout_idx], "holdout_types": types[holdout_idx],
            "train_idx": train_idx, "holdout_idx": holdout_idx,
        }
    data = {
        "x": np.concatenate([pool["x"] for pool in pools.values()]),
        "timepoints": np.concatenate([
            np.full(len(pool["x"]), day) for day, pool in pools.items()
        ]),
        "time_values": np.concatenate([
            np.full(len(pool["x"]), maizels.parse_timepoint(day))
            for day, pool in pools.items()
        ]),
    }
    splits = maizels._endpoint_split_dict(pools["D3"], pools["D8"])
    with (
        mock.patch.object(maizels, "resolve_dataset_path", return_value=Path("fixture.csv")),
        mock.patch.object(maizels, "resolve_classifier_path", return_value=Path("fixture.pt")),
        mock.patch.object(maizels, "all_timepoint_data", return_value=data),
        mock.patch.object(maizels, "timepoint_pool_splits", return_value=pools),
        mock.patch.object(maizels, "endpoint_pool_splits", return_value=splits),
    ):
        datamodule = MaizelsEndpointDataModule(args)
    return args, datamodule


class EvaluationPairTests(unittest.TestCase):
    def test_uncapped_periodic_and_final_evaluation_classify_real_paths(self):
        for schedule in ("d3_d8", "d3_d3p8_d8"):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                args, data = make_datamodule(schedule, 0, output)
                classifier = NMPClassifier()
                callback = maizels_eval.MaizelsEvaluationCallback(args, data)
                flow = ZeroField()
                module = SimpleNamespace(flow_net=flow, device=torch.device("cpu"))
                trainer = SimpleNamespace(
                    global_step=args.maizels_eval_every_n_steps,
                    is_global_zero=True,
                    checkpoint_callback=SimpleNamespace(
                        best_model_score=torch.tensor(0.25), best_model_path="best.ckpt",
                    ),
                )
                logged = []
                summary = {}
                # Keep pair generation, integration, classifier inference and EMD
                # real; replace only artifact I/O and W&B/figure rendering.
                with (
                    mock.patch.object(maizels, "load_classifier", return_value=(
                        classifier, maizels.CLASS_NAMES,
                        np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32),
                    )),
                    mock.patch.object(callback, "_distribution_figure", return_value=None),
                    mock.patch.object(callback, "_trajectory_figure", return_value=None),
                    mock.patch.object(maizels_eval.wandb, "Image", return_value="plot"),
                    mock.patch.object(maizels_eval.wandb, "log", side_effect=logged.append),
                    mock.patch.object(maizels_eval.wandb, "run", SimpleNamespace(summary=summary)),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    callback.on_train_batch_end(trainer, module, None, None, 0)
                    callback.on_test_start(trainer, module)

                self.assertEqual(len(data.eval_pairs["x0"]), 4)
                self.assertEqual(data.eval_pair_stats["split"], "heldout")
                self.assertEqual(data.eval_pair_stats["accepted_pairs"], 4)
                self.assertTrue(all(size > 0 for size in classifier.batch_sizes))
                self.assertEqual(sum(classifier.batch_sizes), 2 * 4 * 2)
                self.assertEqual(len(logged), 2)
                self.assertTrue(flow.training)
                for record in logged:
                    self.assertEqual(record["maizels/model_euler_invalid_trajectory_pct"], 0.0)
                metrics = json.loads((output / "final_metrics.json").read_text())
                self.assertEqual(metrics["final_eval/euler_invalid_trajectory_pct"], 0.0)
                self.assertTrue(np.isfinite(metrics["final_eval/euler_mean_emd_test_times"]))
                self.assertEqual(metrics["final_eval/euler_mean_emd_test_times"], summary[
                    "final_eval/euler_mean_emd_test_times"
                ])

    def test_positive_caps_preserve_diagnostic_pair_counts_and_split_selection(self):
        for schedule in ("d3_d8", "d3_d3p8_d8"):
            for cap in (3, 10):
                with self.subTest(schedule=schedule, cap=cap), tempfile.TemporaryDirectory() as tmp:
                    _, data = make_datamodule(schedule, cap, Path(tmp))
                    expected_n = min(cap, 4) if schedule == "d3_d3p8_d8" else cap
                    expected_split = "train" if schedule == "d3_d8" and cap > 4 else "heldout"
                    self.assertEqual(len(data.eval_pairs["x0"]), expected_n)
                    self.assertEqual(len(data.eval_pairs["x1"]), expected_n)
                    self.assertEqual(data.eval_pairs["label"].shape, (expected_n, 2))
                    self.assertEqual(data.eval_pair_stats["split"], expected_split)


if __name__ == "__main__":
    unittest.main()

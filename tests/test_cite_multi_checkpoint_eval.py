"""CITE/Multi interval evaluation preserves all cells and the training fold."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "py"))

from common import cite_multi, cite_multi_checkpoint_eval as evaluation
from launchers import eval_cite_multi


def fixture_data(large=False):
    sizes = (1109, 1188, 1147, 1217) if large else (3, 5, 4, 6)
    populations = {
        day: np.full((n, 100), index * 0.25, dtype=np.float32)
        for index, (day, n) in enumerate(zip(cite_multi.TIMEPOINTS, sizes))
    }
    return {
        "x": np.concatenate(list(populations.values())),
        "timepoints": np.concatenate(
            [np.full(len(x), day) for day, x in populations.items()]
        ),
    }, populations


def write_dataset(root, dataset_name, data=None):
    import anndata as ad
    import pandas as pd

    data = fixture_data()[0] if data is None else data
    path = root / cite_multi.DATASET_FILES[dataset_name]
    obs = pd.DataFrame(
        {
            "day": pd.Categorical(data["timepoints"], categories=cite_multi.TIMEPOINTS),
            "cell_type": pd.Categorical(["HSC"] * len(data["x"])),
        },
        index=[f"cell_{index}" for index in range(len(data["x"]))],
    )
    adata = ad.AnnData(np.zeros((len(obs), 1), dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = data["x"]
    adata.write_h5ad(path)
    return path


class MarginalProtocolTests(unittest.TestCase):
    def test_both_folds_and_clocks_restart_from_real_full_populations(self):
        data, populations = fixture_data(large=True)
        for omitted in ("3", "4"):
            for clock in cite_multi.TIME_MODES:
                with self.subTest(omitted=omitted, clock=clock):
                    cfg = SimpleNamespace(
                        problem=SimpleNamespace(
                            heldout_timepoint=omitted,
                            cite_multi_time_mode=clock,
                        )
                    )
                    marginals = evaluation.build_marginals(cfg, data)
                    calls = []

                    def pushforward(source, **kwargs):
                        calls.append((source.copy(), kwargs))
                        return source + 0.4 * (
                            kwargs["end_time"] - kwargs["start_time"]
                        )

                    rows = evaluation.evaluate_marginals(
                        marginals,
                        pushforward,
                        context={
                            "method": "test",
                            "seed": 4,
                            "checkpoint": "test.pkl",
                            "sampler": "flowmap",
                            "n_steps_per_interval": 8,
                        },
                        score=lambda prediction, target: np.linalg.norm(
                            prediction[0].astype(float) - target[0]
                        ),
                    )
                    by_day = {row["timepoint"]: row for row in rows}
                    self.assertEqual(
                        set(by_day), set(cite_multi.TIMEPOINTS) | {"test_times"}
                    )
                    self.assertEqual({row["protocol"] for row in rows}, {"interval"})
                    self.assertEqual(marginals.test_times, (omitted,))
                    self.assertEqual(marginals.hparam_val_times, ())
                    self.assertEqual(len(calls), 3)
                    for index, day in enumerate(cite_multi.TIMEPOINTS[1:]):
                        previous = max(
                            (
                                left
                                for left in marginals.retained
                                if float(left) < float(day)
                            ),
                            key=float,
                        )
                        row = by_day[day]
                        self.assertEqual(row["source_timepoint"], previous)
                        self.assertEqual(row["source_n"], len(populations[previous]))
                        self.assertEqual(row["target_n"], len(populations[day]))
                        np.testing.assert_array_equal(
                            calls[index][0], populations[previous]
                        )
                        self.assertAlmostEqual(
                            calls[index][1]["start_time"], marginals.times[previous]
                        )
                        self.assertAlmostEqual(
                            calls[index][1]["end_time"], marginals.times[day]
                        )
                    self.assertEqual(by_day["test_times"]["n_timepoints"], 1)
                    self.assertEqual(by_day["test_times"]["w1"], by_day[omitted]["w1"])
                    records = evaluation.metric_records(rows)
                    self.assertIn(
                        "final_eval/flowmap_mean_emd_test_times", records[0]["metrics"]
                    )

    def test_missing_population_invalid_clock_and_nonfinite_data_fail(self):
        data, _ = fixture_data()
        cfg = SimpleNamespace(problem=SimpleNamespace(heldout_timepoint="4"))
        keep = data["timepoints"] != "3"
        with self.assertRaisesRegex(ValueError, "Missing.*3"):
            evaluation.build_marginals(
                cfg, {key: value[keep] for key, value in data.items()}
            )
        invalid = {**data, "x": data["x"].copy()}
        invalid["x"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.build_marginals(cfg, invalid)
        cfg.problem.timepoint_order = cite_multi.TIMEPOINTS
        cfg.problem.timepoint_values = (0.0, 0.8, 0.2, 1.0)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            evaluation.build_marginals(cfg, data)

    def test_saved_torch_fold_and_clock_are_used_and_ambiguous_folds_need_flag(self):
        for backend, config, omitted in (
            ("otpfm", {"dataset_name": "multi", "heldout_day": "3"}, "3"),
            (
                "mfm",
                {
                    "data_type": "scrna",
                    "data_name": "multi",
                    "t_exclude_current": 2,
                    "t_exclude": [1, 2],
                },
                "4",
            ),
            ("mfm", {"data_name": "multi", "t_exclude": [1]}, "3"),
        ):
            cfg = evaluation.torch_config(config, backend)
            self.assertEqual(cfg.problem.dataset_name, "multi")
            self.assertEqual(cfg.problem.heldout_timepoint, omitted)
        ambiguous = {"data_name": "cite", "t_exclude": [1, 2]}
        with self.assertRaisesRegex(ValueError, "--heldout_day"):
            evaluation.torch_config(ambiguous, "mfm")
        self.assertEqual(
            evaluation.torch_config(
                ambiguous, "mfm", omitted_day="4"
            ).problem.heldout_timepoint,
            "4",
        )
        saved = {
            "dataset_name": "cite",
            "heldout_day": "4",
            "time_map": {"2": 0.0, "3": 0.12, "4": 0.48, "7": 1.0},
        }
        cfg = evaluation.torch_config(saved, "otpfm")
        self.assertAlmostEqual(cite_multi.normalized_time("3", cfg), 0.12)
        cfg = evaluation.torch_config(saved, "otpfm", time_mode="real_time")
        self.assertAlmostEqual(cite_multi.normalized_time("3", cfg), 0.2)
        with self.assertRaisesRegex(ValueError, "saved training fold"):
            evaluation.torch_config(saved, "otpfm", omitted_day="3")
        with self.assertRaisesRegex(ValueError, "saved training dataset"):
            evaluation.torch_config(saved, "otpfm", dataset_name="multi")

    def test_parser_validates_steps_backend_flags_and_native_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            path.write_bytes(b"test")
            base = ["--backend", "otpfm", "--checkpoint", f"otpfm:4:{path}"]
            self.assertEqual(
                eval_cite_multi.parse_args(
                    base + ["--n_steps_per_interval", "1", "2", "2"]
                ).n_steps_per_interval,
                [1, 2],
            )
            for extra in (
                ["--n_steps_per_interval", "0"],
                ["--batch_size", "0"],
                ["--ema_fac", "0.99"],
            ):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                    SystemExit
                ):
                    eval_cite_multi.parse_args(base + extra)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                eval_cite_multi.parse_args(
                    ["--backend", "flowmaps", "--checkpoint", f"map:4:{path}"]
                )


if __name__ == "__main__":
    unittest.main()

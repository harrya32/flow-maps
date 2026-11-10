"""Cell-MNN grids, strict config roundtrips, direct-EMD selection, and resume."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cellmnn_baseline  # Configure upstream paths and writable caches first.
import numpy as np
import torch

from cellmnn_baseline.train import experiment_config, parse_args, run_experiment
from common import maizels
from scripts import mfm_maizels_sweep_utils as shared
from scripts import sweep_maizels_cellmnn_hparams as sweep


class CellMNNHparamSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_grid_has_27_settings_and_54_seed_runs_with_fixed_gamma(self):
        args = sweep.parse_args(["--dataset-location", "unused.csv.gz"])
        grid = sweep.parameter_grid(args)
        self.assertEqual(len(grid), 27)
        self.assertEqual(shared.parse_seed_grid(args.seeds), (0, 1))
        self.assertEqual(len(grid) * len(shared.parse_seed_grid(args.seeds)), 54)
        self.assertEqual({row["lr"] for row in grid}, {3e-4, 1e-3, 3e-3})
        self.assertEqual({row["lambda_kinetic"] for row in grid}, {0.01, 0.1, 1})
        self.assertEqual({row["mmd_sigma"] for row in grid}, {0.3, 1, 3})
        self.assertEqual(len({tuple(row.items()) for row in grid}), 27)
        self.assertTrue(all("gamma" not in row for row in grid))
        self.assertEqual(sweep.resolved_base_config(args)["gamma"], 0.1)
        self.assertEqual(
            sweep.OBJECTIVE_METRIC, "final_eval/direct_mean_emd_hparam_val_times"
        )

    def test_invalid_grids_are_rejected_and_zero_kinetic_weight_is_allowed(self):
        for option, value in (
            ("--learning-rates", "0"),
            ("--learning-rates", "nan"),
            ("--learning-rates", "0.001,0.001"),
            ("--kinetic-weights", "-1"),
            ("--kinetic-weights", "inf"),
            ("--mmd-sigmas", "0"),
            ("--mmd-sigmas", "-1"),
            ("--mmd-sigmas", ""),
        ):
            with self.subTest(option=option, value=value), self.assertRaises(
                (ValueError, argparse.ArgumentTypeError)
            ):
                sweep.parameter_grid(
                    sweep.parse_args(
                        ["--dataset-location", "unused.csv.gz", f"{option}={value}"]
                    )
                )
        grid = sweep.parameter_grid(
            sweep.parse_args(
                [
                    "--dataset-location", "unused.csv.gz",
                    "--learning-rates", "0.001",
                    "--kinetic-weights", "0",
                    "--mmd-sigmas", "1",
                ]
            )
        )
        self.assertEqual(grid, [{"lr": 0.001, "lambda_kinetic": 0.0, "mmd_sigma": 1.0}])

    def test_presets_budget_overrides_and_derived_configs_roundtrip(self):
        defaults = sweep.resolved_base_config(
            sweep.parse_args(["--dataset-location", "unused.csv.gz"])
        )
        self.assertEqual(
            (defaults["preset"], defaults["pca_dim"], defaults["batch_size"]),
            ("cite50", 50, 1028),
        )
        self.assertEqual((defaults["width"], defaults["depth"]), (1024, 3))
        self.assertEqual(
            (
                defaults["lr"], defaults["lambda_kinetic"],
                defaults["mmd_sigma"], defaults["gamma"],
            ),
            (3e-4, 0.1, 0.3, 0.1),
        )
        self.assertEqual((defaults["epochs"], defaults["time_limit"]), (1000, 240))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.json"
            path.write_text(json.dumps({
                "preset": "script", "lr": 0.9, "lambda_kinetic": 9,
                "mmd_sigma": 9, "gamma": 0.7, "width": 12, "depth": 1,
                "kinetic_dt": 0.2,
            }))
            args = sweep.parse_args([
                "--dataset-location", "unused.csv.gz", "--config-path", str(path),
                "--gamma", "0.1", "--epochs", "7", "--time-limit", "30",
                "--train-batches", "2", "--val-batches", "3",
                "--check-val-every-n-epoch", "1", "--patience", "4",
            ])
            base = sweep.resolved_base_config(args)
            self.assertEqual((base["preset"], base["batch_size"]), ("script", 200))
            self.assertEqual((base["epochs"], base["time_limit"], base["gamma"]), (7, 30, 0.1))
            hparams = {"lr": 0.001, "lambda_kinetic": 0.01, "mmd_sigma": 0.3}
            config = sweep.derived_config(base, hparams, seed=1)
            self.assertNotIn("pca_dim", config)
            self.assertFalse(any(key.startswith("sweep_") for key in config))
            path.write_text(json.dumps(config))
            restored = experiment_config(parse_args([
                "--dataset_name", "maizels", "--config_path", str(path),
            ]))
            self.assertEqual(restored, {**base, **hparams, "seed": 1})

    def test_invalid_fixed_parameters_and_evaluation_options_fail_before_launch(self):
        for options in (
            ["--gamma", "1.1"], ["--gamma", "nan"],
            ["--seeds", str(2**32)], ["--epochs", "0"],
            ["--eval-every-epochs", "-1"], ["--eval-max-points", "-1"],
            ["--eval-batch-size", "0"],
        ):
            with self.subTest(options=options), patch.object(sweep.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    sweep.main(["--dataset-location", "unused.csv.gz", "--dry-run", *options])
                run.assert_not_called()

    def test_dry_run_prints_the_full_grid_without_writing_or_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), patch.object(sweep.subprocess, "run") as run:
                result = sweep.main([
                    "--dataset-location", str(Path(temporary) / "missing.csv.gz"),
                    "--output-dir", str(output), "--dry-run",
                ])
                run.assert_not_called()
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            text = stream.getvalue()
            self.assertIn("27 settings x 2 seeds [0, 1] = 54 runs", text)
            self.assertIn("gamma=0.1 (fixed)", text)
            self.assertIn("sampler=direct", text)
            self.assertIn("--eval_every_epochs 0", text)
            self.assertEqual(text.count("--run_name "), 54)

    def test_trials_rank_direct_validation_emd_and_resume_valid_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "fixture.csv.gz"
            dataset.touch()
            output = root / "outputs"
            argv = [
                "--dataset-location", str(dataset), "--output-dir", str(output),
                "--learning-rates", "0.0003,0.001", "--kinetic-weights", "0.1",
                "--mmd-sigmas", "1",
            ]

            def train(command, **kwargs):
                config_path = Path(command[command.index("--config_path") + 1])
                config = json.loads(config_path.read_text())
                parsed = parse_args(command[3:])
                self.assertEqual(parsed.run_name, config_path.parent.name)
                self.assertEqual(parsed.seeds, [config["seed"]])
                self.assertEqual(experiment_config(parsed), {**config, "pca_dim": 50})
                self.assertEqual(config["gamma"], 0.1)
                better = config["lr"] == 0.001
                metrics = {
                    sweep.OBJECTIVE_METRIC: (1 if better else 3) + config["seed"],
                    # Opposing diagnostics must not influence the winner.
                    "final_eval/direct_mean_emd_test_times": 100 if better else 0,
                    "final_eval/flowmap_mean_emd_hparam_val_times": 100 if better else 0,
                    "final_eval/full_data_classifier/direct_invalid_trajectory_pct": 20,
                }
                (config_path.parent / "final_metrics.json").write_text(json.dumps(metrics))
                return SimpleNamespace(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(run.call_count, 4)
            rows = shared.read_rows(output / "results.csv")
            self.assertEqual(len({row["run_id"] for row in rows}), 4)
            self.assertTrue(all(row["status"] == "complete" for row in rows))
            self.assertTrue(all(row["gamma"] == "0.1" for row in rows))
            best = json.loads((output / "best_setting.json").read_text())
            self.assertEqual(best["selected_hparams"]["lr"], 0.001)
            self.assertEqual(best["objective_mean"], 1.5)
            self.assertEqual(best["n_seeds_completed"], 2)
            self.assertAlmostEqual(best["objective_std"], 2**-0.5)
            best_config = experiment_config(parse_args([
                "--dataset_name", "maizels", "--config_path", str(output / "best_config.json"),
            ]))
            self.assertEqual((best_config["lr"], best_config["gamma"]), (0.001, 0.1))
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(sweep.subprocess, "run") as run,
            ):
                self.assertEqual(sweep.main(argv), 0)
                run.assert_not_called()
            # An old complete CSV row cannot hide a missing/corrupt objective.
            Path(rows[0]["metrics_json"]).write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(run.call_count, 1)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main([*argv, "--rerun-completed"]), 0)
                self.assertEqual(run.call_count, 4)
            # Changing a fixed parameter invalidates the resume IDs.
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main([*argv, "--epochs", "2"]), 0)
                self.assertEqual(run.call_count, 4)
            self.assertEqual(len(shared.read_rows(output / "results.csv")), 8)

    def test_partial_seed_pairs_cannot_win_and_failures_are_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "fixture.csv.gz"
            dataset.touch()
            output = root / "outputs"
            argv = [
                "--dataset-location", str(dataset), "--output-dir", str(output),
                "--learning-rates", "0.001", "--kinetic-weights", "0.1", "--mmd-sigmas", "1",
            ]

            def train(command, *, fail_second=False, **kwargs):
                config_path = Path(command[command.index("--config_path") + 1])
                config = json.loads(config_path.read_text())
                if fail_second and config["seed"] == 1:
                    return SimpleNamespace(returncode=7)
                (config_path.parent / "final_metrics.json").write_text(
                    json.dumps({sweep.OBJECTIVE_METRIC: 1 + config["seed"]})
                )
                return SimpleNamespace(returncode=0)

            def failing_train(command, **kwargs):
                return train(command, fail_second=True, **kwargs)

            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=failing_train
            ):
                self.assertEqual(sweep.main(argv), 1)
            summary = shared.read_rows(output / "summary.csv")[0]
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["objective_rank"], "")
            self.assertFalse((output / "best_config.json").exists())
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(run.call_count, 1)
            self.assertTrue((output / "best_config.json").is_file())
            # A failed forced rerun must not leave the previous best config.
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", return_value=SimpleNamespace(returncode=7)
            ):
                self.assertEqual(sweep.main([*argv, "--rerun-completed", "--fail-fast"]), 7)
            self.assertFalse((output / "best_config.json").exists())
            self.assertFalse((output / "best_setting.json").exists())

    def test_missing_or_nonfinite_objectives_are_not_complete_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            for metrics in ([], {}, {sweep.OBJECTIVE_METRIC: True},
                            {sweep.OBJECTIVE_METRIC: float("nan")},
                            {sweep.OBJECTIVE_METRIC: float("inf")}):
                with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                    path.write_text(json.dumps(metrics))
                    sweep._read_metrics(path)
            dataset = Path(temporary) / "fixture.csv.gz"
            dataset.touch()
            output = Path(temporary) / "outputs"
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ):
                self.assertEqual(sweep.main([
                    "--dataset-location", str(dataset), "--output-dir", str(output),
                    "--learning-rates", "0.001", "--kinetic-weights", "0.1",
                    "--mmd-sigmas", "1", "--seeds", "0",
                ]), 1)
            self.assertEqual(shared.read_rows(output / "results.csv")[0]["status"], "failed")

    def test_named_smoke_run_saves_checkpoint_metrics_and_wandb_name(self):
        rng = np.random.default_rng(12)
        days = np.repeat(maizels.TIMEPOINTS, 10)
        raw = {
            "x": rng.normal(size=(len(days), 50)).astype(np.float32),
            "timepoints": days,
            "cell_types": np.full(len(days), maizels.CLASS_NAMES[0], dtype=object),
        }
        metrics = {
            "cellmnn/test_EMD": 1.0,
            "final_eval/direct_invalid_trajectory_pct": 0.0,
            sweep.OBJECTIVE_METRIC: 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args([
                "--dataset_name", "maizels", "--run_name", "setting_seed0",
                "--working_dir", temporary, "--smoke_test", "--batch_size", "4",
                "--width", "8", "--depth", "1", "--device", "cpu", "--eval_every_epochs", "0",
            ])
            config = experiment_config(args)
            with (
                patch.object(maizels, "load_pca50_dataset", return_value=raw),
                patch("cellmnn_baseline.train.classifier_paths", return_value={}),
                patch("cellmnn_baseline.train.evaluate", return_value=(metrics, {})) as evaluate,
                patch("cellmnn_baseline.train.wandb.require"),
                patch("cellmnn_baseline.train.wandb.init") as init,
                patch("cellmnn_baseline.train.wandb.log"),
            ):
                result = run_experiment(args, config, None, 0)
            evaluate.assert_called_once()
            self.assertTrue(evaluate.call_args.kwargs["include_observed"])
            self.assertEqual(result["final_eval/best_step"], 1)
            self.assertEqual(init.call_args.kwargs["name"], "setting_seed0")
            output = Path(temporary) / "setting_seed0"
            for filename in ("experiment.json", "losses.csv", "best.pt", "final_metrics.json"):
                self.assertTrue((output / filename).is_file(), filename)
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["run_name"], "setting_seed0")
            self.assertEqual(metadata["primary_sampler"], "direct")

    def test_run_names_cannot_escape_the_output_directory(self):
        for name in (".", "..", "nested/trial", "nested\\trial"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--run_name", name])


if __name__ == "__main__":
    unittest.main()

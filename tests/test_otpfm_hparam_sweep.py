"""Conditional kernel grids, resumable trial selection, and final-only evaluation."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts import sweep_maizels_otpfm_hparams as sweep
from scripts import mfm_maizels_sweep_utils as shared
from otpfm_baseline.train import experiment_config, parse_args, run_experiment
from common import maizels
from experiments.train import create_model, create_potentials


class OTPFMHparamSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_grid_has_48_settings_and_96_seed_runs(self):
        args = sweep.parse_args(["--dataset-location", "unused.csv.gz"])
        grid = sweep.parameter_grid(args)
        self.assertEqual(len(grid), 48)
        self.assertEqual(len(grid) * len(shared.parse_seed_grid(args.seeds)), 96)
        self.assertEqual({row["lr"] for row in grid}, {3e-4, 1e-3, 3e-3})
        self.assertEqual({row["strength"] for row in grid}, {30, 100, 300, 1000})
        delta = [row for row in grid if row["lambda_type"] == "delta"]
        gaussian = [row for row in grid if row["lambda_type"] == "gaussian"]
        self.assertEqual(len(delta), 12)
        self.assertEqual(len(gaussian), 36)
        self.assertEqual({row["lambda_width"] for row in delta}, {0.0})
        self.assertEqual({row["lambda_width"] for row in gaussian}, {0.05, 0.1, 0.2})
        self.assertEqual(len({tuple(row.items()) for row in grid}), 48)

    def test_delta_does_not_expand_or_validate_the_unused_width_grid(self):
        args = sweep.parse_args(
            [
                "--dataset-location",
                "unused.csv.gz",
                "--kernels",
                "delta",
                "--gaussian-widths",
                "",
                "--learning-rates",
                "0.001",
                "--strengths",
                "300",
            ]
        )
        self.assertEqual(
            sweep.parameter_grid(args),
            [
                {
                    "lr": 0.001,
                    "strength": 300.0,
                    "lambda_type": "delta",
                    "lambda_width": 0.0,
                }
            ],
        )
        args.kernels, args.gaussian_widths = "gaussian", "0.05,0.2"
        self.assertEqual(len(sweep.parameter_grid(args)), 2)

    def test_invalid_active_grids_are_rejected(self):
        for option, value in (
            ("--learning-rates", "0"),
            ("--learning-rates", "nan"),
            ("--strengths", "-1"),
            ("--strengths", "inf"),
            ("--gaussian-widths", "0"),
            ("--gaussian-widths", "0.1,0.1"),
            ("--kernels", "triangle"),
            ("--kernels", "delta,delta"),
        ):
            with self.subTest(option=option, value=value), self.assertRaises(
                (ValueError, argparse.ArgumentTypeError)
            ):
                sweep.parameter_grid(
                    sweep.parse_args(
                        ["--dataset-location", "unused.csv.gz", f"{option}={value}"]
                    )
                )
        with self.assertRaisesRegex(ValueError, "retained D3.8"):
            sweep.run_sweep(
                sweep.parse_args(
                    [
                        "--dataset-location",
                        "unused.csv.gz",
                        "--maizels-schedule",
                        "d3_d8",
                    ]
                )
            )

    def test_presets_and_alias_cleanup_make_each_grid_value_effective(self):
        for options, potential, coupled, epochs in (
            ([], "w2inf", False, 80),
            (["--ot-coupling"], "w2inf", True, 200),
            (["--potential", "w2"], "w2", False, 200),
        ):
            args = sweep.parse_args(["--dataset-location", "unused.csv.gz", *options])
            base = sweep.resolved_base_config(args)
            self.assertEqual(
                (base["potential"], base["ot_coupling"], base["epochs"]),
                (potential, coupled, epochs),
            )
            self.assertEqual(
                (base["hidden_dim"], base["num_hidden_layers"], base["pca_dim"]),
                (768, 10, 50),
            )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.json"
            path.write_text(json.dumps({"strengths": [999], "widths": [9], "width": 2}))
            args = sweep.parse_args(
                ["--dataset-location", "unused.csv.gz", "--config-path", str(path)]
            )
            base = sweep.resolved_base_config(args)
            self.assertTrue(
                all(key not in base for key in ("strengths", "widths", "width"))
            )
            hparams = {
                "lr": 0.001,
                "strength": 30,
                "lambda_type": "gaussian",
                "lambda_width": 0.05,
            }
            config = sweep.derived_config(base, hparams, "s", "r", 1, "objective")
            config["tks"] = [0.16]
            potential = next(iter(create_potentials(config).values()))
            self.assertEqual((potential.strength, potential.width), (30, 0.05))
            self.assertEqual(config["seed"], 1)

    def test_dry_run_prints_values_and_counts_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                result = sweep.main(
                    [
                        "--dataset-location",
                        str(Path(temporary) / "missing.csv.gz"),
                        "--output-dir",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            text = stream.getvalue()
            self.assertIn("48 settings x 2 seeds [0, 1] = 96 runs", text)
            self.assertIn("12 delta (width ignored), 36 Gaussian", text)
            self.assertIn("widths [0.05, 0.1, 0.2]", text)
            self.assertIn("--eval_every_epochs 0", text)
            self.assertIn("--seeds 1", text)

    def test_trials_resume_and_rank_by_validation_day_emd_across_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "fixture.csv.gz"
            dataset.touch()
            argv = [
                "--dataset-location",
                str(dataset),
                "--output-dir",
                str(root / "outputs"),
                "--learning-rates",
                "0.001",
                "--strengths",
                "30,100",
                "--kernels",
                "delta",
            ]
            objective = "final_eval/flowmap_mean_emd_hparam_val_times"

            def train(command, **kwargs):
                config_path = Path(command[command.index("--config_path") + 1])
                config = json.loads(config_path.read_text())
                self.assertEqual(
                    int(command[command.index("--seeds") + 1]), config["seed"]
                )
                self.assertEqual(
                    command[command.index("--run_name") + 1], config_path.parent.name
                )
                value = (3 if config["strength"] == 30 else 1) + config["seed"]
                metrics = {
                    objective: value,
                    "final_eval/flowmap_mean_emd_test_times": (
                        0 if config["strength"] == 30 else 10
                    ),
                    "final_eval/full_data_classifier/flowmap_invalid_trajectory_pct": 20,
                }
                (config_path.parent / "final_metrics.json").write_text(
                    json.dumps(metrics)
                )
                return SimpleNamespace(returncode=0)

            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(run.call_count, 4)
            rows = shared.read_rows(root / "outputs" / "results.csv")
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["status"] == "complete" for row in rows))
            best = json.loads((root / "outputs" / "best_setting.json").read_text())
            self.assertEqual(best["selected_hparams"]["strength"], 100)
            self.assertEqual(best["objective_mean"], 1.5)
            self.assertEqual(best["n_seeds_completed"], 2)
            self.assertEqual(
                json.loads((root / "outputs" / "best_config.json").read_text())[
                    "strength"
                ],
                100,
            )
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run"
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                run.assert_not_called()
            # A stale completed CSV row cannot hide a missing/corrupt objective.
            Path(rows[0]["metrics_json"]).write_text("{}")
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", side_effect=train
            ) as run:
                self.assertEqual(sweep.main(argv), 0)
                self.assertEqual(run.call_count, 1)

    def test_failed_trials_do_not_produce_an_eligible_best_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "fixture.csv.gz"
            dataset.touch()
            argv = [
                "--dataset-location",
                str(dataset),
                "--output-dir",
                str(root / "outputs"),
                "--learning-rates",
                "0.001",
                "--strengths",
                "300",
                "--kernels",
                "delta",
                "--fail-fast",
            ]
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                sweep.subprocess, "run", return_value=SimpleNamespace(returncode=7)
            ):
                self.assertEqual(sweep.main(argv), 7)
            self.assertFalse((root / "outputs" / "best_config.json").exists())
            self.assertEqual(
                shared.read_rows(root / "outputs" / "results.csv")[0]["status"],
                "failed",
            )

    def test_zero_interval_evaluates_once_after_restoring_best_validation_model(self):
        rng = np.random.default_rng(12)
        days = np.repeat(maizels.TIMEPOINTS, 10)
        raw = {
            "x": rng.normal(size=(len(days), 50)).astype(np.float32),
            "timepoints": days,
            "cell_types": np.full(len(days), maizels.CLASS_NAMES[0], dtype=object),
        }
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(
                [
                    "--dataset_name",
                    "maizels",
                    "--run_name",
                    "best_only",
                    "--working_dir",
                    temporary,
                    "--smoke_test",
                    "--batch_size",
                    "4",
                    "--eval_every_epochs",
                    "0",
                    "--wandb_mode",
                    "disabled",
                    "--device",
                    "cpu",
                ]
            )
            config = experiment_config(args)
            config.update(hidden_dim=8, num_hidden_layers=1, dropout=0.0, epochs=2)

            def factory(*factory_args):
                model = create_model(*factory_args)
                training_forward = model.forward_with_loss
                validation = iter((3.0, 1.0, 2.0))

                def forward(xs, *positional, **keywords):
                    if not model.training:
                        return torch.tensor(next(validation))
                    return training_forward(xs, *positional, **keywords)

                model.forward_with_loss = forward
                return model

            metrics = {
                "distribution_eval/flowmap_emd_mean": 1.0,
                "maizels/full_data_classifier/model_flowmap_invalid_trajectory_pct": 0.0,
                "final_eval/flowmap_mean_emd_hparam_val_times": 1.0,
            }
            with patch.object(maizels, "load_pca50_dataset", return_value=raw), patch(
                "otpfm_baseline.train.create_model", side_effect=factory
            ), patch("otpfm_baseline.train.classifier_paths", return_value={}), patch(
                "otpfm_baseline.train.evaluate", return_value=(metrics, {})
            ) as evaluate, patch(
                "otpfm_baseline.train.wandb.require"
            ):
                result = run_experiment(args, config, None, 42)
            evaluate.assert_called_once()
            self.assertTrue(evaluate.call_args.kwargs["include_observed"])
            self.assertEqual(result["final_eval/best_step"], 1)
            self.assertEqual(result["final_eval/best_validation_loss"], 1.0)
            output = Path(temporary) / "best_only"
            self.assertEqual(
                pd.read_csv(output / "losses.csv")["val_loss"].tolist(), [3.0, 1.0, 2.0]
            )
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["eval_every_epochs"], 0)
            self.assertEqual(metadata["evaluation_schedule"], "best_checkpoint_only")
            self.assertTrue((output / "models" / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()

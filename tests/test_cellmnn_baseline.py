"""Cell-MNN decoder parity, fair holdouts, and benchmark integration."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cellmnn_baseline  # Configure upstream paths and writable caches first.
import matplotlib.pyplot as plt
import numpy as np
import torch

from cellmnn_baseline.data import PC1Scaler, PopulationDataset, prepare_data
from cellmnn_baseline.evaluation import evaluate
from cellmnn_baseline.model import BenchmarkCellMNN, CellMNNSampler
from cellmnn_baseline.train import (
    BenchmarkTrainer,
    experiment_config,
    parse_args,
    upstream_defaults,
)
from common import cite_multi, maizels
from lib.model import CellMNN


def fixture(dataset):
    backend = maizels if dataset == "maizels" else cite_multi
    dim = 50 if dataset == "maizels" else 100
    rng = np.random.default_rng(13)
    arrays = [
        rng.normal(float(day.removeprefix("D")), 0.2, (40, dim))
        for day in backend.TIMEPOINTS
    ]
    days = np.repeat(backend.TIMEPOINTS, 40)
    return {
        "x": np.concatenate(arrays).astype(np.float32),
        "timepoints": days,
        "time_values": np.array([maizels.parse_timepoint(day) for day in days]),
        "cell_types": np.full(len(days), backend.CLASS_NAMES[0]),
        "obs_names": np.arange(len(days)).astype(str),
    }


def prepared(
    dataset="cite", *, heldout="3", time_mode="equal_time", schedule="d3_d3p8_d8"
):
    args = parse_args(
        [
            "--dataset_name",
            dataset,
            "--cite_multi_time_mode",
            time_mode,
            "--maizels_schedule",
            schedule,
            "--batch_size",
            "8",
            "--train_batches",
            "1",
            "--val_batches",
            "2",
        ]
    )
    config = experiment_config(args)
    backend = maizels if dataset == "maizels" else cite_multi
    loader = "load_pca50_dataset" if dataset == "maizels" else "load_dataset"
    with mock.patch.object(backend, loader, return_value=fixture(dataset)):
        data = prepare_data(args, config, heldout if dataset != "maizels" else None, 42)
    return args, config, data


def model_for(data, config, *, dim=None):
    return BenchmarkCellMNN(
        latent_dim=dim or config["pca_dim"],
        lr=config["lr"],
        retained_times=[data.model_time_map[day] for day in data.retained],
        depth=1,
        width=12,
        gamma=0.1,
    )


class DiagonalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.latent_dim = 3
        self.register_buffer("rates", torch.tensor([0.2, -0.1, 0.3]))

    def forward(self, x, t, times):
        pred = x * torch.exp(self.rates * (times - t))
        return pred, None, None, None, self.rates * pred


class RecordingSampler:
    def __init__(self):
        self.calls = []

    def sample(
        self,
        source,
        *,
        start_time=0.0,
        end_time=1.0,
        n_steps=50,
        sampler="direct",
        paths=False,
    ):
        x = np.asarray(source, np.float32)
        self.calls.append((start_time, end_time, sampler, x.copy(), paths))
        if paths:
            return (
                x[:, None]
                + np.linspace(0, end_time - start_time, n_steps + 1)[None, :, None]
            )
        return x + end_time - start_time


class DirectOnlySampler(RecordingSampler):
    def sample(self, source, **kwargs):
        if kwargs.get("sampler", "direct") != "direct":
            raise AssertionError("Cell-MNN evaluation requested a non-native sampler.")
        return super().sample(source, **kwargs)


class CellMNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(14)
        np.random.seed(14)

    def test_tuned_defaults_apply_to_all_dimensionality_presets(self):
        defaults = upstream_defaults()
        tuned = {
            "lr": 3e-4,
            "batch_size": 1028,
            "lambda_kinetic": 0.1,
            "gamma": 0.1,
            "width": 1024,
            "depth": 3,
            "mmd_sigma": 0.3,
        }
        for name in ("cite", "multi", "maizels"):
            preset = "cite50" if name == "maizels" else "cite100"
            for requested in ("auto", preset):
                with self.subTest(dataset=name, preset=requested):
                    config = experiment_config(parse_args([
                        "--dataset_name", name, "--preset", requested,
                    ]))
                    self.assertEqual(config["preset"], preset)
                    self.assertEqual(
                        config["pca_dim"], 50 if name == "maizels" else 100
                    )
                    for key, value in tuned.items():
                        self.assertEqual(config[key], value, key)
                    for key in defaults.keys() - tuned.keys():
                        self.assertEqual(config[key], defaults[key], key)
        script = experiment_config(parse_args(["--preset", "script"]))
        for key, value in defaults.items():
            self.assertEqual(script[key], value, key)

    def test_cli_and_json_override_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps({
                "preset": "cite50", "epochs": 30, "lr": 0.002,
                "width": 64, "depth": 2, "mmd_sigma": 2.0,
                "lambda_kinetic": 0.2, "gamma": 0.7,
            }))
            for preset in ("auto", "script"):
                with self.subTest(preset=preset):
                    config = experiment_config(parse_args([
                        "--config_path", str(path), "--preset", preset,
                        "--epochs", "7", "--width", "12", "--gamma", "0.1",
                        "--no-normalize",
                    ]))
                    self.assertEqual(
                        config["preset"], "cite50" if preset == "auto" else "script"
                    )
                    self.assertEqual(config["lr"], 0.002)
                    self.assertEqual(config["epochs"], 7)
                    self.assertEqual(config["width"], 12)
                    self.assertEqual(config["depth"], 2)
                    self.assertEqual(config["mmd_sigma"], 2.0)
                    self.assertEqual(config["lambda_kinetic"], 0.2)
                    self.assertEqual(config["gamma"], 0.1)
                    self.assertEqual(
                        config["batch_size"], 1028 if preset == "auto" else 200
                    )
                    self.assertFalse(config["normalize"])

    def test_cite_folds_clocks_and_normalization_exclude_validation_and_omitted_cells(
        self,
    ):
        for dataset in ("cite", "multi"):
            for fold in ("3", "4"):
                for clock in ("equal_time", "real_time"):
                    with self.subTest(dataset=dataset, fold=fold, clock=clock):
                        args, config, data = prepared(
                            dataset, heldout=fold, time_mode=clock
                        )
                        expected = [day for day in cite_multi.TIMEPOINTS if day != fold]
                        self.assertEqual(list(data.retained), expected)
                        self.assertEqual(
                            data.time_map, cite_multi.normalized_time_map(clock)
                        )
                        self.assertEqual(data.cfg.problem.maizels_holdout_seed, 701)
                        for values, day in zip(data.train_dataset.X_filtered, expected):
                            np.testing.assert_allclose(
                                values,
                                data.timepoint_splits[day]["train_x"]
                                / data.scaler.scale_,
                            )
                        changed = fixture(dataset)
                        for day in cite_multi.TIMEPOINTS:
                            row_ids = np.where(changed["timepoints"] == day)[0]
                            alter = (
                                row_ids
                                if day == fold
                                else row_ids[data.timepoint_splits[day]["holdout_idx"]]
                            )
                            changed["x"][alter] += 10000
                        with mock.patch.object(
                            cite_multi, "load_dataset", return_value=changed
                        ):
                            second = prepare_data(args, config, fold, 42)
                        self.assertAlmostEqual(data.scaler.scale_, second.scaler.scale_)
                        np.testing.assert_array_equal(
                            next(iter(data.train_dataset))[3][0, :, 0],
                            np.array(
                                [data.model_time_map[day] for day in expected],
                                np.float32,
                            ),
                        )

    def test_maizels_schedules_reuse_only_retained_pools(self):
        for schedule, expected in (
            ("d3_d8", ("D3", "D8")),
            ("d3_d3p8_d8", ("D3", "D3.8", "D8")),
        ):
            _, _, data = prepared("maizels", schedule=schedule)
            self.assertEqual(data.retained, expected)
            self.assertEqual(len(data.train_dataset.X_filtered), len(expected))
            self.assertEqual(data.time_origin, 3)
            self.assertEqual(data.time_scale, 5)
            self.assertEqual(len(list(data.train_loader)), 1)
            self.assertEqual(len(list(data.val_loader)), 2)
            batch = next(iter(data.train_loader))
            self.assertEqual(tuple(batch[2].shape), (8, len(expected), 50))
            self.assertTrue((batch[1] < data.model_time_map["D8"]).all())

    def test_fast_decoder_predictions_velocities_and_gradients_match_upstream(self):
        for constant_dims in (0, 1):
            kwargs = dict(
                latent_dim=5, lr=2e-4, depth=2, width=12, num_const_dims=constant_dims
            )
            original = CellMNN(days_w_data=[2.0, 3.0, 7.0], skip_day=3.0, **kwargs)
            adapted = BenchmarkCellMNN(retained_times=[2.0, 3.0, 7.0], **kwargs)
            adapted.mlp.load_state_dict(original.mlp.state_dict())
            x = torch.randn(4, 1, 5)
            t = torch.tensor([2.0, 3.0, 2.0, 3.0]).reshape(4, 1, 1)
            times = torch.tensor([2.0, 3.0, 4.6, 7.0])[None, :, None].expand(4, -1, -1)
            left, right = original(x, t, times), adapted(x, t, times)
            for index in (0, 4):
                torch.testing.assert_close(
                    left[index], right[index], rtol=2e-5, atol=2e-6
                )
            for result in (left, right):
                (result[0].square().mean() + 0.3 * result[4].square().mean()).backward()
            for a, b in zip(original.parameters(), adapted.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=2e-5)

    def test_maizels_equal_clock_supervises_the_retained_middle_day(self):
        args, config, _ = prepared("maizels")
        args.maizels_time_mode = "equal_time"
        with mock.patch.object(
            maizels, "load_pca50_dataset", return_value=fixture("maizels")
        ):
            data = prepare_data(args, config, None, 42)
        self.assertEqual(data.time_map["D3.8"], 0.5)
        self.assertEqual(data.model_time_map["D3.8"], 5.5)
        model = model_for(data, config)
        terms = model.loss_terms(next(iter(data.train_loader)))
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_complete_loss_matches_upstream_training_objective(self):
        arrays = [np.random.randn(20, 5).astype(np.float32) for _ in range(3)]
        batch = next(iter(PopulationDataset(arrays, [2.0, 3.0, 7.0], 8, batches=1)))
        original = CellMNN(
            latent_dim=5,
            lr=2e-4,
            days_w_data=[2.0, 3.0, 7.0],
            skip_day=3.0,
            train_on_skip_day=True,
            gamma=0.1,
            depth=1,
            width=12,
        )
        adapted = BenchmarkCellMNN(
            latent_dim=5,
            lr=2e-4,
            retained_times=[2.0, 3.0, 7.0],
            gamma=0.1,
            depth=1,
            width=12,
        )
        adapted.mlp.load_state_dict(original.mlp.state_dict())
        original.log = lambda *args, **kwargs: None
        expected = original.training_step(batch, 0)
        actual = adapted.loss_terms(batch)["loss"]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
        expected.backward()
        actual.backward()
        for reference, candidate in zip(original.parameters(), adapted.parameters()):
            torch.testing.assert_close(
                reference.grad, candidate.grad, rtol=3e-4, atol=2e-5
            )
        self.assertTrue(
            all(
                torch.isfinite(p.grad).all()
                for p in adapted.parameters()
                if p.grad is not None
            )
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_loss_preserves_cpu_values_and_gradients(self):
        kwargs = dict(
            latent_dim=5,
            lr=2e-4,
            retained_times=[2.0, 3.0, 7.0],
            gamma=0.1,
            depth=2,
            width=12,
        )
        reference = BenchmarkCellMNN(**kwargs)
        candidate = BenchmarkCellMNN(**kwargs).to("mps")
        candidate.load_state_dict(reference.state_dict())
        arrays = [np.random.randn(20, 5).astype(np.float32) for _ in range(3)]
        batch = next(iter(PopulationDataset(arrays, [2.0, 3.0, 7.0], 16, batches=1)))
        expected = reference.loss_terms(batch)
        actual = candidate.loss_terms(tuple(value.to("mps") for value in batch))
        for name in expected:
            self.assertEqual(actual[name].device.type, "mps")
            torch.testing.assert_close(
                expected[name], actual[name].cpu(), rtol=3e-4, atol=2e-5
            )
        expected["loss"].backward()
        actual["loss"].backward()
        for cpu_parameter, mps_parameter in zip(
            reference.parameters(), candidate.parameters()
        ):
            torch.testing.assert_close(
                cpu_parameter.grad, mps_parameter.grad.cpu(), rtol=5e-4, atol=3e-5
            )

    def test_equal_clock_supervision_is_exact_and_retains_kinetic_grid(self):
        times = [2.0, 2.0 + 5.0 / 3.0, 7.0]
        model = BenchmarkCellMNN(
            latent_dim=5, lr=2e-4, retained_times=times, width=12, depth=1
        )
        torch.testing.assert_close(
            model.loss_times[model.population_indices], torch.tensor(times)
        )
        self.assertEqual(len(model.kinetic_indices), 51)
        arrays = [np.random.randn(20, 5).astype(np.float32) for _ in times]
        terms = model.loss_terms(
            next(iter(PopulationDataset(arrays, times, 8, batches=1)))
        )
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))

    def test_sampling_raw_units_nonzero_start_and_explicit_step_count(self):
        model = DiagonalModel()
        sampler = CellMNNSampler(
            model,
            SimpleNamespace(scale_=4.0),
            time_origin=2.0,
            time_scale=5.0,
            batch_size=2,
        )
        source = np.array([[1, 2, 3], [3, 4, 5], [2, 5, 1]], np.float32)
        expected = source * np.exp(model.rates.numpy())
        with mock.patch.object(model, "forward", wraps=model.forward) as forward:
            paths = sampler.sample(
                source,
                start_time=0.2,
                end_time=0.4,
                n_steps=50,
                sampler="direct",
                paths=True,
            )
            self.assertEqual(paths.shape, (3, 51, 3))
            np.testing.assert_array_equal(paths[:, 0], source)
            np.testing.assert_allclose(paths[:, -1], expected, rtol=1e-6, atol=1e-5)
            self.assertEqual(forward.call_count, 2)  # One operator per inference chunk.
        for mode in ("flowmap", "euler"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "only direct"):
                sampler.sample(source, sampler=mode)
        self.assertTrue(model.training)
        with self.assertRaises(ValueError):
            sampler.sample(source, start_time=0.4, end_time=0.2)

    def test_validation_is_repeatable_and_does_not_advance_training_rng(self):
        args, config, data = prepared()
        model = model_for(data, config)
        with tempfile.TemporaryDirectory() as temp:
            trainer = BenchmarkTrainer(model, data, config, args, {}, temp)
            state, torch_state = np.random.get_state(), torch.random.get_rng_state()
            expected_np, expected_torch = np.random.rand(5), torch.rand(5)
            np.random.set_state(state)
            torch.random.set_rng_state(torch_state)
            first, second = trainer.validation_loss(), trainer.validation_loss()
        self.assertEqual(first, second)
        np.testing.assert_array_equal(expected_np, np.random.rand(5))
        torch.testing.assert_close(expected_torch, torch.rand(5), rtol=0, atol=0)

    def test_best_checkpoint_uses_val_loss_and_restores_weights_without_rewinding_step(
        self,
    ):
        args, config, data = prepared()
        config.update(epochs=2, check_val_every_n_epoch=1)
        args.eval_every_epochs = 0
        model = model_for(data, config)
        parameter = next(model.parameters())
        with tempfile.TemporaryDirectory() as temp:
            trainer = BenchmarkTrainer(model, data, config, args, {}, temp)

            def train_epoch(*args, **kwargs):
                trainer.global_step += 1
                with torch.no_grad():
                    parameter.fill_(trainer.global_step)
                return {"loss": 0.5}

            trainer._epoch_loss = train_epoch
            trainer.validation_loss = mock.Mock(
                side_effect=[{"loss": 1.0}, {"loss": 2.0}]
            )
            trainer.evaluate = lambda **kwargs: {
                "restored": parameter.detach().mean().item()
            }
            with mock.patch("cellmnn_baseline.train.wandb.log"):
                result = trainer.train()
            self.assertEqual(result["restored"], 1.0)
            self.assertEqual(trainer.best_step, 1)
            self.assertEqual(trainer.global_step, 2)
            with (Path(temp) / "losses.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_maizels_metrics_use_local_sources_and_heldout_paths_for_both_classifiers(
        self,
    ):
        _, _, data = prepared("maizels")
        sampler = DirectOnlySampler()
        path_calls = []

        def check_paths(paths, source_types, **kwargs):
            path_calls.append(paths.shape)
            return {"valid": np.arange(len(paths)) % 2 == 0}

        def emd(x, y):
            return float(np.linalg.norm(x.mean(0) - y.mean(0)))

        with mock.patch(
            "cellmnn_baseline.evaluation.maizels.check_paths_with_classifier",
            side_effect=check_paths,
        ), mock.patch(
            "cellmnn_baseline.evaluation.wasserstein.exact_emd", side_effect=emd
        ):
            metrics, figures = evaluate(
                data,
                sampler,
                {"observed_days": Path("observed.pt"), "all_days": Path("all.pt")},
                seed=42,
                max_points=4,
                make_plot=False,
                include_observed=True,
            )
        self.assertFalse(figures)
        self.assertEqual(path_calls, [(4, 50, 50)] * 2)
        self.assertEqual(metrics["final_eval/direct_invalid_trajectory_pct"], 50.0)
        self.assertEqual(
            metrics["final_eval/full_data_classifier/direct_invalid_trajectory_pct"],
            50.0,
        )
        self.assertIn("final_eval/observed_D3p8_to_D8_direct_emd", metrics)
        self.assertIn("final_eval/direct_mean_emd_test_times", metrics)
        self.assertNotIn("distribution_eval/D3p8_direct_emd", metrics)
        self.assertTrue(
            any(
                abs(start - 0.16) < 1e-6 and abs(end - 0.2) < 1e-6
                for start, end, *_ in sampler.calls
            )
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(all(call[2] == "direct" for call in sampler.calls))
        self.assertFalse(any(
            "flowmap" in key or "euler" in key or "rollout" in key for key in metrics
        ))

    def test_cite_reuses_benchmark_evaluation_and_native_local_prediction(self):
        _, _, data = prepared(heldout="4")
        sampler = RecordingSampler()
        with mock.patch(
            "cellmnn_baseline.evaluation.evaluate_cite_multi",
            return_value=({"final_eval/direct_mean_emd": 2.0}, None),
        ) as shared, mock.patch(
            "cellmnn_baseline.evaluation.wasserstein.exact_emd", return_value=1.0
        ):
            metrics, _ = evaluate(
                data,
                sampler,
                {},
                seed=42,
                max_points=4,
                make_plot=False,
                include_observed=True,
            )
        shared.assert_called_once_with(
            data, sampler, {}, seed=42, max_points=4, make_plot=False,
            samplers=("direct",), primary_sampler="direct",
        )
        self.assertEqual(sampler.calls[0][0], data.time_map["3"])
        self.assertEqual(sampler.calls[0][1], data.time_map["4"])
        self.assertEqual(metrics["test_EMD"], 1.0)
        self.assertEqual(metrics["final_eval/direct_mean_emd"], 2.0)
        self.assertIn("final_eval/observed_day2_to_day3_direct_emd", metrics)
        self.assertFalse(any(
            "flowmap" in key or "euler" in key or "rollout" in key for key in metrics
        ))

    def test_cite_and_multi_evaluation_use_only_direct_predictions_and_paths(self):
        for dataset in ("cite", "multi"):
            with self.subTest(dataset=dataset):
                _, _, data = prepared(dataset, heldout="4")
                sampler = DirectOnlySampler()
                with mock.patch(
                    "cellmnn_baseline.evaluation.wasserstein.exact_emd", return_value=1.0
                ), mock.patch(
                    "otpfm_baseline.evaluation.score_paths_with_all_days_classifier",
                    return_value={"valid": np.array([True, False, True, False])},
                ):
                    metrics, figures = evaluate(
                        data, sampler,
                        {"observed_days": Path("a"), "all_days": Path("b")},
                        seed=42, max_points=4, include_observed=True,
                    )
                self.assertEqual(metrics["cellmnn/test_EMD"], 1.0)
                self.assertEqual(metrics["test_EMD"], 1.0)
                self.assertEqual(metrics["final_eval/direct_invalid_trajectory_pct"], 50.0)
                self.assertTrue(all(call[2] == "direct" for call in sampler.calls))
                self.assertFalse(any(
                    "flowmap" in key or "euler" in key or "rollout" in key for key in metrics
                ))
                self.assertNotIn("mfm/test_EMD", metrics)
                self.assertIn("lineage_trajectories", figures)
                # Every prediction starts from a measured population, including
                # observed intervals; no carried predictions are re-encoded.
                for start, _, _, source, _ in sampler.calls:
                    day = next(day for day in data.retained if data.time_map[day] == start)
                    pool = data.timepoint_splits[day]["x"]
                    self.assertTrue(all(np.any(np.all(pool == row, axis=1)) for row in source))
                for figure in figures.values():
                    plt.close(figure)

    def test_nonfinite_direct_sampling_still_raises(self):
        model = DiagonalModel()
        model.rates.fill_(1000)
        sampler = CellMNNSampler(model, None, time_origin=0, time_scale=1)
        with self.assertRaisesRegex(FloatingPointError, "Non-finite Cell-MNN direct"):
            sampler.sample(np.ones((2, 3), np.float32))

    def test_periodic_and_final_evaluation_request_only_direct_sampling(self):
        args, config, data = prepared(heldout="4")
        config.update(epochs=2, check_val_every_n_epoch=1)
        args.eval_every_epochs, args.eval_max_points = 1, 4
        with tempfile.TemporaryDirectory() as temporary:
            trainer = BenchmarkTrainer(
                model_for(data, config), data, config, args,
                {"observed_days": Path("a"), "all_days": Path("b")}, temporary,
            )
            trainer.validation_loss = mock.Mock(side_effect=[{"loss": 2.0}, {"loss": 1.0}])
            samplers = []

            def sampler_factory(*args, **kwargs):
                sampler = DirectOnlySampler()
                samplers.append(sampler)
                return sampler

            with mock.patch(
                "cellmnn_baseline.train.CellMNNSampler", side_effect=sampler_factory
            ), mock.patch(
                "cellmnn_baseline.evaluation.wasserstein.exact_emd", return_value=1.0
            ), mock.patch(
                "otpfm_baseline.evaluation.score_paths_with_all_days_classifier",
                return_value={"valid": np.ones(4, dtype=bool)},
            ), mock.patch("cellmnn_baseline.train.wandb.log"):
                result = trainer.train()
            self.assertEqual(len(samplers), 3)  # Two periodic evaluations, then best weights.
            self.assertTrue(all(call[2] == "direct" for s in samplers for call in s.calls))
            self.assertEqual(result["final_eval/best_epoch"], 2)
            self.assertEqual(result["cellmnn/test_EMD"], 1.0)
            self.assertFalse(any(
                "flowmap" in key or "euler" in key or "rollout" in key for key in result
            ))
            self.assertTrue((Path(temporary) / "final_metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()

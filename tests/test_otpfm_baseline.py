"""OTP-FM benchmark integration checks runnable in mfm_env with unittest."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from otpfm_baseline import UPSTREAM_ROOT
from otpfm_baseline.data import prepare_data
from otpfm_baseline.evaluation import evaluate
from otpfm_baseline.model import OTPFMSampler
from otpfm_baseline.train import experiment_config, parse_args, run_experiment
from common import cite_multi
from experiments.citeseq.data import compute_ot_alignments
from experiments.train import create_model, create_potentials, load_config
from otpfm.potentials import W2Potential


def fixture_data():
    days = np.repeat(np.asarray(cite_multi.TIMEPOINTS, dtype=object), 10)
    x = np.repeat((np.arange(40, dtype=np.float32) + 1)[:, None], 100, axis=1)
    x[days == "4"] += 10000
    return {
        "x": x,
        "timepoints": days,
        "cell_types": np.asarray(["HSC", "BP"] * 20, dtype=object),
    }


class TimeDependentVelocity(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.d = dim
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def v_func(self, x, t1, t2):
        self.calls.append((float(t1[0]), float(t2[0])))
        return torch.ones_like(x) * (1 + (t2 - t1)[:, None])

    v_func_ema = v_func


class OTPFMIntegrationTests(unittest.TestCase):
    def test_defaults_are_the_upstream_cite_w2inf_configuration(self):
        actual = experiment_config(parse_args([]))
        expected = load_config("citeseq", "W2Inf")
        for key, value in expected.items():
            if key not in ("pca_dim", "solver_type"):
                self.assertEqual(actual[key], value, key)
        self.assertEqual((actual["epochs"], actual["lr_schedule"]), (80, None))
        self.assertEqual((actual["hidden_dim"], actual["num_hidden_layers"]), (768, 10))
        self.assertEqual(actual["pca_dim"], 100)
        self.assertEqual(actual["solver_type"], "direct")

    def test_each_variant_inherits_its_upstream_cite_defaults(self):
        cases = (
            (["--ot_coupling"], "W2", "w2inf", True),
            (["--potential", "w2"], None, "w2", False),
            (["--potential", "w2", "--ot_coupling"], None, "w2", True),
            (
                ["--config_path", str(UPSTREAM_ROOT / "configs/citeseq/W2.json")],
                "W2",
                "w2inf",
                True,
            ),
        )
        for argv, preset, potential, coupled in cases:
            with self.subTest(argv=argv):
                actual = experiment_config(parse_args(argv))
                expected = load_config("citeseq", preset)
                for key, value in expected.items():
                    if key not in (
                        "pca_dim",
                        "solver_type",
                        "potential",
                        "ot_coupling",
                    ):
                        self.assertEqual(actual[key], value, key)
                self.assertEqual(
                    (actual["epochs"], actual["lr_schedule"]), (200, "cosine")
                )
                self.assertEqual(actual["potential"], potential)
                self.assertEqual(actual["ot_coupling"], coupled)
                self.assertEqual(actual["upstream_config_preset"], preset or "defaults")

    def test_coupling_and_potential_options_override_json_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "config.json"
            override.write_text(json.dumps({"potential": "w2", "ot_coupling": True}))
            config = experiment_config(parse_args(["--config_path", str(override)]))
            self.assertEqual(config["potential"], "w2")
            self.assertTrue(config["ot_coupling"])
            self.assertEqual(config["solver_type"], "anderson_safe")
            self.assertEqual((config["epochs"], config["lr_schedule"]), (200, "cosine"))
            config = experiment_config(
                parse_args(
                    [
                        "--config_path",
                        str(override),
                        "--potential",
                        "W2Inf",
                        "--no-ot_coupling",
                        "--ot_method",
                        "sinkhorn",
                        "--ot_reg",
                        "0.05",
                    ]
                )
            )
            self.assertEqual(config["potential"], "w2inf")
            self.assertFalse(config["ot_coupling"])
            self.assertEqual(config["solver_type"], "direct")
            self.assertEqual(
                (config["ot_method"], config["ot_reg"]), ("sinkhorn", 0.05)
            )
            self.assertEqual((config["epochs"], config["lr_schedule"]), (80, None))
            override.write_text(
                json.dumps(
                    {
                        "potential": "w2inf",
                        "ot_coupling": True,
                        "epochs": 37,
                        "lr_schedule": None,
                    }
                )
            )
            config = experiment_config(parse_args(["--config_path", str(override)]))
            self.assertEqual(config["upstream_config_preset"], "W2")
            self.assertEqual((config["epochs"], config["lr_schedule"]), (37, None))
            config = experiment_config(
                parse_args(
                    [
                        "--config_path",
                        str(override),
                        "--epochs",
                        "9",
                    ]
                )
            )
            self.assertEqual((config["epochs"], config["lr_schedule"]), (9, None))
            override.write_text(
                json.dumps({"potential": "w2", "solver_type": "direct"})
            )
            with self.assertRaisesRegex(ValueError, "iterative solver"):
                experiment_config(parse_args(["--config_path", str(override)]))

    def test_both_clocks_and_folds_exclude_omitted_and_validation_cells(self):
        raw = fixture_data()
        config = experiment_config(parse_args([]))
        for day in ("3", "4"):
            for mode in cite_multi.TIME_MODES:
                with self.subTest(day=day, mode=mode), patch.object(
                    cite_multi, "load_dataset", return_value=raw
                ):
                    data = prepare_data(
                        "cite", day, mode, "/private/tmp/fixture_cite.h5ad", config
                    )
                self.assertNotIn(day, data.retained)
                self.assertEqual(data.time_map, cite_multi.normalized_time_map(mode))
                self.assertFalse(data.train_dataset.use_ot)
                self.assertEqual(
                    data.train_dataset.train_times,
                    [cite_multi.TIMEPOINTS.index(d) for d in data.retained],
                )
                train = np.concatenate(
                    [data.timepoint_splits[d]["train_x"] for d in data.retained]
                )
                np.testing.assert_allclose(
                    data.scaler.mean_, train.mean(axis=0, keepdims=True)
                )
                for retained_day in data.retained:
                    pool = data.timepoint_splits[retained_day]
                    self.assertFalse(set(pool["train_idx"]) & set(pool["holdout_idx"]))
                    array = data.train_dataset.data_by_time[
                        cite_multi.TIMEPOINTS.index(retained_day)
                    ].numpy()
                    np.testing.assert_allclose(
                        data.scaler.inverse_transform(array),
                        pool["train_x"],
                        rtol=1e-5,
                        atol=1e-3,
                    )

    def test_ot_chains_use_separate_retained_train_and_validation_populations(self):
        raw = fixture_data()
        config = experiment_config(parse_args(["--ot_coupling"]))
        for dataset_name in ("cite", "multi"):
            for day in ("3", "4"):
                with self.subTest(dataset=dataset_name, day=day), patch.object(
                    cite_multi, "load_dataset", return_value=raw
                ), patch(
                    "otpfm_baseline.data.compute_ot_alignments",
                    wraps=compute_ot_alignments,
                ) as solve:
                    data = prepare_data(
                        dataset_name,
                        day,
                        "equal_time",
                        "/private/tmp/fixture.h5ad",
                        config,
                    )
                self.assertEqual(solve.call_count, 2)
                for call, split, dataset in zip(
                    solve.call_args_list,
                    ("train_x", "holdout_x"),
                    (data.train_dataset, data.val_dataset),
                ):
                    pcs, labels, train_times = call.args
                    expected = np.concatenate(
                        [data.timepoint_splits[d][split] for d in data.retained]
                    )
                    np.testing.assert_allclose(
                        pcs, data.scaler.transform(expected).astype(np.float32)
                    )
                    self.assertNotIn(cite_multi.TIMEPOINTS.index(day), labels)
                    self.assertEqual(train_times, dataset.train_times)
                    self.assertEqual(call.kwargs, {"method": "emd", "reg": 0.01})
                    self.assertTrue(dataset.use_ot)
                    self.assertEqual(
                        len(dataset), len(data.timepoint_splits["2"][split])
                    )
                    chains = dataset.ot_chains.copy()
                    for k, edge in enumerate(zip(train_times[:-1], train_times[1:])):
                        np.testing.assert_array_equal(
                            chains[:, k + 1], dataset.ot_alignments[edge][chains[:, k]]
                        )
                    dataset.reshuffle()
                    np.testing.assert_array_equal(dataset.ot_chains, chains)

    def test_ot_smoke_caps_alignment_populations_and_passes_sinkhorn_settings(self):
        raw = fixture_data()
        config = experiment_config(
            parse_args(
                [
                    "--ot_coupling",
                    "--ot_method",
                    "sinkhorn",
                    "--ot_reg",
                    "0.05",
                    "--batch_size",
                    "4",
                ]
            )
        )

        def align(pcs, labels, times, **kwargs):
            return {
                (src, tgt): np.arange(np.count_nonzero(labels == src))
                % np.count_nonzero(labels == tgt)
                for src, tgt in zip(times[:-1], times[1:])
            }

        with patch.object(cite_multi, "load_dataset", return_value=raw), patch(
            "otpfm_baseline.data.compute_ot_alignments", side_effect=align
        ) as solve:
            data = prepare_data(
                "cite",
                "4",
                "equal_time",
                "/private/tmp/fixture.h5ad",
                config,
                smoke_test=True,
            )
        self.assertEqual(len(data.train_loader), 1)
        self.assertEqual(len(data.val_loader), 1)
        for call in solve.call_args_list:
            _, labels, times = call.args
            self.assertTrue(all(np.count_nonzero(labels == t) <= 4 for t in times))
            self.assertEqual(call.kwargs, {"method": "sinkhorn", "reg": 0.05})
        with patch.object(cite_multi, "load_dataset", return_value=raw), patch(
            "otpfm_baseline.data.compute_ot_alignments", return_value=None
        ), self.assertRaisesRegex(ImportError, "POT"):
            prepare_data("cite", "4", "equal_time", "/private/tmp/fixture.h5ad", config)

    def test_explicit_sampling_preserves_units_and_counts_steps_per_horizon(self):
        model = TimeDependentVelocity(3)
        scaler = SimpleNamespace(mean_=np.full((1, 3), 5, dtype=np.float32), scale_=3.0)
        sampler = OTPFMSampler(model, scaler, batch_size=2)
        source = np.full((2, 3), 8, dtype=np.float32)
        direct = sampler.sample(source, end_time=0.6, n_steps=50, sampler="direct")
        self.assertEqual(len(model.calls), 1)
        np.testing.assert_allclose(direct, 8 + 3 * 0.6 * 1.6, atol=1e-5)
        model.calls.clear()
        flowmap = sampler.sample(
            source, end_time=0.6, n_steps=2, sampler="flowmap", paths=True
        )
        self.assertEqual(flowmap.shape, (2, 3, 3))
        self.assertEqual(len(model.calls), 2)
        np.testing.assert_allclose(flowmap[:, -1], 8 + 3 * 0.6 * 1.3, atol=1e-5)
        model.calls.clear()
        euler = sampler.sample(source, end_time=0.6, n_steps=50, sampler="euler")
        self.assertEqual(len(model.calls), 50)
        self.assertTrue(all(left == right for left, right in model.calls))
        np.testing.assert_allclose(euler, 8 + 3 * 0.6, atol=1e-4)

    def test_upstream_loss_backward_ema_and_fixed_point_work(self):
        config = experiment_config(parse_args([]))
        config.update(
            hidden_dim=8, num_hidden_layers=1, dropout=0.0, lossfn="mse", tks=[0.4]
        )
        model = create_model(config, 3, create_potentials(config), "cpu")
        self.assertIsNotNone(model._I_minus_A_inv)
        xs = torch.randn(4, 3, 3)
        before = [p.detach().clone() for p in model.flownet_ema.parameters()]
        optimizer = torch.optim.Adam(model.flownet.parameters(), lr=config["lr"])
        loss = model.forward_with_loss(xs, otp_alpha=1.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.flownet.parameters()
            )
        )
        optimizer.step()
        model.update_ema()
        for initial, current, ema in zip(
            before, model.flownet.parameters(), model.flownet_ema.parameters()
        ):
            torch.testing.assert_close(ema, initial * 0.99 + current.detach() * 0.01)
        _, _, residual = model.solve_X_tks(
            xs[:, 0],
            xs[:, -1] - xs[:, 0],
            xs[:, 1].unsqueeze(0),
            list(model.potentials.values()),
            [0.4],
            1.0,
        )
        self.assertLess(float(residual.max()), 1e-4)

    def test_w2_potential_recomputes_ot_and_is_invariant_to_target_permutation(self):
        config = experiment_config(parse_args(["--potential", "w2"]))
        config.update(
            hidden_dim=8, num_hidden_layers=1, dropout=0.0, lossfn="mse", tks=[0.4]
        )
        torch.manual_seed(42)
        model = create_model(config, 3, create_potentials(config), "cpu")
        model.eval()
        self.assertIsInstance(next(iter(model.potentials.values())), W2Potential)
        self.assertEqual(model.solver_type, "anderson_safe")
        self.assertIsNone(model._I_minus_A_inv)
        source = torch.tensor(
            [[-3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        )
        targets = torch.tensor(
            [[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-4.0, 0.0, 0.0]]
        )

        def solve(target):
            return model.solve_X_tks(
                source,
                torch.zeros_like(source),
                target[None],
                list(model.potentials.values()),
                [0.4],
                1.0,
            )

        for original, permuted in zip(solve(targets), solve(targets.flip(0))):
            torch.testing.assert_close(original, permuted)
            self.assertTrue(torch.isfinite(original).all())
        with patch.object(
            W2Potential, "grad_gk", autospec=True, side_effect=W2Potential.grad_gk
        ) as gradient:
            loss = model.forward_with_loss(torch.randn(4, 3, 3), otp_alpha=1.0)
            loss.backward()
        self.assertGreater(gradient.call_count, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.flownet.parameters()
            )
        )

    def test_evaluation_keeps_population_sources_and_shared_metric_names(self):
        pools = {}
        for day, value in (("2", 2), ("3", 10), ("4", 20), ("7", 30)):
            pools[day] = {
                "x": np.full((3, 100), value, dtype=np.float32),
                "holdout_x": np.full((2, 100), value, dtype=np.float32),
                "holdout_types": np.asarray(["HSC", "BP"]),
            }
        data = SimpleNamespace(
            timepoint_splits=pools,
            heldout_day="4",
            time_map=cite_multi.normalized_time_map(),
        )
        calls = []

        class IdentitySampler:
            def sample(self, source, **kwargs):
                calls.append((float(source.mean()), kwargs))
                if kwargs.get("paths"):
                    return np.repeat(source[:, None], kwargs["n_steps"] + 1, axis=1)
                return source

        def score(paths, source_types, **kwargs):
            self.assertEqual(paths.shape[1], 50)
            np.testing.assert_array_equal(source_types, ["HSC", "BP"])
            return {"valid": np.asarray([True, False])}

        with patch(
            "otpfm_baseline.evaluation.wasserstein.exact_emd",
            side_effect=lambda x, y: float(x.mean()),
        ), patch("otpfm_baseline.evaluation.rbf_mmd2", return_value=0.125), patch(
            "otpfm_baseline.evaluation.score_paths_with_all_days_classifier",
            side_effect=score,
        ):
            metrics, figure = evaluate(
                data,
                IdentitySampler(),
                {"observed_days": "a", "all_days": "b"},
                make_plot=False,
            )
        self.assertIsNone(figure)
        self.assertEqual(metrics["distribution_eval/4_flowmap_emd"], 2)
        self.assertEqual(metrics["final_eval/flowmap_mean_emd"], 2)
        self.assertEqual(metrics["mfm/test_EMD"], 10)
        self.assertEqual(metrics["mfm/test_EMD_flowmap"], 10)
        self.assertTrue(
            all(kwargs["n_steps"] == 100 for mean, kwargs in calls if mean == 10)
        )
        self.assertEqual(
            metrics[
                "maizels/full_data_classifier/model_flowmap_invalid_trajectory_pct"
            ],
            50,
        )
        self.assertEqual(
            metrics["final_eval/full_data_classifier/flowmap_invalid_trajectory_pct"],
            50,
        )
        self.assertEqual(metrics["final_eval/source_count"], 2)

    def test_runner_saves_validation_selected_model_and_final_metrics(self):
        raw = fixture_data()

        def score(paths, source_types, **kwargs):
            return {"valid": np.ones(len(paths), dtype=bool)}

        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(
                [
                    "--heldout_day",
                    "4",
                    "--smoke_test",
                    "--wandb_mode",
                    "disabled",
                    "--working_dir",
                    temporary,
                    "--batch_size",
                    "4",
                    "--eval_batch_size",
                    "4",
                ]
            )
            config = experiment_config(args)
            config.update(hidden_dim=8, num_hidden_layers=1, dropout=0.0, epochs=2)

            def model_with_validation_sequence(*factory_args):
                model = create_model(*factory_args)
                training_forward = model.forward_with_loss
                validation_losses = iter((3.0, 1.0, 2.0))

                def forward(xs, *loss_args, **loss_kwargs):
                    if not model.training:
                        return torch.tensor(next(validation_losses))
                    return training_forward(xs, *loss_args, **loss_kwargs)

                model.forward_with_loss = forward
                return model

            fake_paths = {
                key: Path(temporary) / f"{key}.npz"
                for key in ("observed_days", "all_days")
            }
            with patch.object(cite_multi, "load_dataset", return_value=raw), patch(
                "otpfm_baseline.train.create_model",
                side_effect=model_with_validation_sequence,
            ), patch(
                "otpfm_baseline.train.classifier_paths", return_value=fake_paths
            ), patch(
                "otpfm_baseline.evaluation.score_paths_with_all_days_classifier",
                side_effect=score,
            ), patch(
                "otpfm_baseline.train.wandb.require"
            ) as require_backend:
                metrics = run_experiment(args, config, "4", 42)
            require_backend.assert_called_once_with("core")
            output = next(Path(temporary).iterdir())
            self.assertTrue((output / "models" / "best.pt").is_file())
            saved = json.loads((output / "final_metrics.json").read_text())
            self.assertEqual(saved, metrics)
            self.assertEqual(saved["final_eval/best_step"], 1)
            self.assertEqual(saved["final_eval/best_validation_loss"], 1.0)
            self.assertEqual(
                saved["final_eval/full_data_classifier/flowmap_invalid_trajectory_pct"],
                0,
            )
            losses = pd.read_csv(output / "losses.csv")
            self.assertEqual(int(losses["train_loss"].count()), 2)
            self.assertEqual(losses["val_loss"].tolist(), [3.0, 1.0, 2.0])
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["tks"], [1 / 3])
            self.assertEqual(metadata["retained_days"], ["2", "3", "7"])

    def test_runner_combines_w2_potential_with_precomputed_ot_and_distinct_metadata(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            args = parse_args(
                [
                    "--potential",
                    "w2",
                    "--ot_coupling",
                    "--heldout_day",
                    "4",
                    "--smoke_test",
                    "--wandb_mode",
                    "disabled",
                    "--working_dir",
                    temporary,
                    "--batch_size",
                    "4",
                    "--eval_batch_size",
                    "4",
                ]
            )
            config = experiment_config(args)
            config.update(hidden_dim=8, num_hidden_layers=1, dropout=0.0)
            classifiers = {
                key: Path(temporary) / f"{key}.npz"
                for key in ("observed_days", "all_days")
            }
            with patch.object(
                cite_multi, "load_dataset", return_value=fixture_data()
            ), patch(
                "otpfm_baseline.train.classifier_paths", return_value=classifiers
            ), patch(
                "otpfm_baseline.evaluation.score_paths_with_all_days_classifier",
                side_effect=lambda paths, *a, **kw: {
                    "valid": np.ones(len(paths), dtype=bool)
                },
            ), patch(
                "otpfm_baseline.train.wandb.require"
            ):
                metrics = run_experiment(args, config, "4", 42)
            output = next(Path(temporary).iterdir())
            self.assertIn("otpfm_w2_ot_emd_seed42", output.name)
            metadata = json.loads((output / "experiment.json").read_text())
            self.assertEqual(metadata["potential_class"], "W2Potential")
            self.assertEqual(metadata["coupling_type"], "precomputed_ot")
            self.assertEqual(metadata["solver_type"], "anderson_safe")
            self.assertEqual(
                metadata["tuple_train_cells_per_day"], {"2": 4, "3": 4, "7": 4}
            )
            self.assertEqual(
                json.loads((output / "final_metrics.json").read_text()), metrics
            )
            self.assertTrue((output / "models/best.pt").is_file())


if __name__ == "__main__":
    unittest.main()

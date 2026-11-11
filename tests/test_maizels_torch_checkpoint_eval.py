"""Pretrained OTP-FM/MFM evaluation restores weights and scores full marginals."""

from __future__ import annotations

from argparse import Namespace
import contextlib
import csv
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
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT, REPO_ROOT / "py", REPO_ROOT / "metric-flow-matching"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from common import maizels, maizels_torch_checkpoint
from launchers import eval_maizels_torch
from mfm.flow_matchers.ema import EMA
from mfm.networks.flow_networks.mlp import VelocityNet
from otpfm_baseline.model import OTPFMSampler
from experiments.train import create_model, create_potentials, load_config


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fixture_data():
    populations = {
        day: np.tile([index * 2.0, index * 3.0], (2 + index % 3, 1)).astype(np.float32)
        for index, day in enumerate(maizels.TIMEPOINTS)
    }
    data = {
        "x": np.concatenate(list(populations.values())),
        "timepoints": np.concatenate([np.full(len(x), day) for day, x in populations.items()]),
    }
    return data, populations


def save_otpfm(root):
    run = root / "otpfm_run"
    path = run / "models" / "best.pt"
    path.parent.mkdir(parents=True)
    config = {
        **load_config("citeseq", "w2inf"),
        "dataset_name": "maizels", "pca_dim": 2,
        "maizels_schedule": "d3_d3p8_d8", "maizels_time_mode": "real_time",
        "hidden_dim": 8, "num_hidden_layers": 1,
        "x_emb_dim": 4, "t_emb_dim": 4, "dropout": 0.0,
        "tks": [0.16], "strength": 1.0, "evaluation_weights": "ema",
        "normalization_mean": [[5.0, -2.0]], "normalization_scale": 3.0,
    }
    model = create_model(config, 2, create_potentials(config), "cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for network, value in ((model.flownet, 1.0), (model.flownet_ema, 2.0)):
            output = [module for module in network.modules() if isinstance(module, torch.nn.Linear)][-1]
            output.bias.fill_(value)
    torch.save({"model_state_dict": model.state_dict(), "global_step": 17}, path)
    (run / "experiment.json").write_text(json.dumps(config))
    return path, config, model


def save_mfm(root, ema=False, embedded=False, companion=True):
    run = root / "mfm_run"
    path = run / "checkpoints" / "maizels" / "run_id" / "flow_model" / "best.ckpt"
    path.parent.mkdir(parents=True)
    config = {
        "data_type": "maizels", "dim": 2, "mfm": True,
        "hidden_dims_flow": [4], "activation_flow": "selu",
        "maizels_schedule": "d3_d3p8_d8", "maizels_time_mode": "real_time",
        "ema_decay": 0.99 if ema else None,
        "maizels_hparam_val_times": ["D3.4", "D6"],
    }
    model = VelocityNet(dim=2, hidden_dims=[4], activation="selu", batch_norm=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.model[-1].bias.fill_(1.0)
    if ema:
        model = EMA(model, decay=0.99)
        with torch.no_grad():
            model.shadow_params[-1].fill_(3.0)
    payload = {
        "state_dict": {
            **{f"flow_net.{key}": value for key, value in model.state_dict().items()},
            "flow_matcher.geopath_net.unneeded_weight": torch.ones(7),
        },
        "global_step": 23,
    }
    if embedded:
        payload["hyper_parameters"] = {"args": Namespace(**config)}
    torch.save(payload, path)
    config_path = run / "sweep_config.yaml"
    if companion:
        config_path.write_text(yaml.safe_dump(config))
    return path, config_path, config


class CheckpointLoaderTests(unittest.TestCase):
    def test_otpfm_restores_normalization_and_selects_ema_or_instantaneous(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config, original = save_otpfm(Path(tmp))
            source = np.asarray([[7.0, -5.0], [9.0, 3.0]], dtype=np.float32)
            scaler = SimpleNamespace(mean_=np.asarray(config["normalization_mean"]), scale_=3.0)
            for ema, velocity in ((None, 2.0), (False, 1.0)):
                loaded = maizels_torch_checkpoint.load_checkpoint(
                    "otpfm", path, device=torch.device("cpu"), batch_size=1, ema=ema,
                )
                prediction = loaded.sampler.sample(
                    source, start_time=0.16, end_time=1.0, n_steps=4, sampler="flowmap",
                )
                reference = OTPFMSampler(original, scaler, batch_size=1, ema=ema is not False).sample(
                    source, start_time=0.16, end_time=1.0, n_steps=4, sampler="flowmap",
                )
                np.testing.assert_allclose(prediction, reference, rtol=1e-6)
                np.testing.assert_allclose(prediction, source + velocity * 3.0 * 0.84, rtol=1e-6)
                np.testing.assert_array_equal(loaded.sampler.center.cpu(), config["normalization_mean"])
                self.assertEqual(loaded.sampler.scale, 3.0)
                self.assertEqual(loaded.metadata["global_step"], 17)

    def test_otpfm_requires_saved_normalization_when_training_used_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config, _ = save_otpfm(Path(tmp))
            del config["normalization_mean"]
            (path.parent.parent / "experiment.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "normalization is missing"):
                maizels_torch_checkpoint.load_checkpoint("otpfm", path, device=torch.device("cpu"))

    def test_mfm_loads_flow_weights_without_geopath_and_restores_ema(self):
        source = np.arange(14, dtype=np.float32).reshape(7, 2)
        for embedded in (False, True):
            with self.subTest(embedded=embedded), tempfile.TemporaryDirectory() as tmp:
                path, _, _ = save_mfm(Path(tmp), ema=True, embedded=embedded, companion=not embedded)
                for ema, velocity in ((None, 3.0), (False, 1.0)):
                    loaded = maizels_torch_checkpoint.load_checkpoint(
                        "mfm", path, device=torch.device("cpu"), batch_size=3, ema=ema,
                    )
                    prediction = loaded.sampler.sample(
                        source, start_time=0.16, end_time=1.0, n_steps=4,
                    )
                    np.testing.assert_allclose(prediction, source + velocity * 0.84, rtol=1e-6)
                    self.assertEqual(loaded.metadata["parameters"], "EMA" if ema is None else "instantaneous")
                    self.assertEqual(loaded.dimension, 2)

    def test_mfm_missing_config_and_wrong_network_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config_path, config = save_mfm(Path(tmp), companion=False)
            with self.assertRaisesRegex(FileNotFoundError, "--config_path"):
                maizels_torch_checkpoint.load_checkpoint("mfm", path, device=torch.device("cpu"))
            config["hidden_dims_flow"] = [8]
            config_path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "does not match checkpoint architecture"):
                maizels_torch_checkpoint.load_checkpoint(
                    "mfm", path, device=torch.device("cpu"), config_path=config_path,
                )

    def test_mfm_rejects_geopath_and_sf2m_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, config = save_mfm(Path(tmp))
            for state, message in (
                ({"geopath_net.weight": torch.ones(2)}, "flow_model checkpoint"),
                ({"score_net.weight": torch.ones(2)}, "SF2M checkpoints"),
            ):
                torch.save({"state_dict": state, "config": config}, path)
                with self.assertRaisesRegex(ValueError, message):
                    maizels_torch_checkpoint.load_checkpoint("mfm", path, device=torch.device("cpu"))

    def test_mfm_euler_step_count_clock_and_full_partial_batches(self):
        model = VelocityNet(dim=2, hidden_dims=[], activation="selu", batch_norm=False)
        with torch.no_grad():
            model.model[0].weight.copy_(torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
            model.model[0].bias.zero_()
        model.eval()
        source = np.arange(2210, dtype=np.float32).reshape(1105, 2) * 0.001
        sampler = maizels_torch_checkpoint.MFMSampler(model, 2, torch.device("cpu"), 128)
        for steps in (1, 2, 8):
            prediction = sampler.sample(source, start_time=0.16, end_time=1.0, n_steps=steps)
            # For v(t,x)=x+t, explicit Euler gives this closed-form recurrence.
            expected = (1.0 + 0.84 / steps) ** steps * (source + 1.16) - 2.0
            np.testing.assert_allclose(prediction, expected, rtol=3e-6, atol=1e-6)
            self.assertEqual(prediction.shape, source.shape)
        with self.assertRaisesRegex(ValueError, "Euler sampling"):
            sampler.sample(source, start_time=0.0, end_time=1.0, n_steps=2, sampler="flowmap")


class CheckpointCLITests(unittest.TestCase):
    def test_both_backends_evaluate_every_interval_and_multiple_steps_without_training(self):
        data, populations = fixture_data()
        for backend in ("otpfm", "mfm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if backend == "otpfm":
                    path, _, _ = save_otpfm(root)
                else:
                    path, _, _ = save_mfm(root)
                out = root / "eval"
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(maizels, "all_timepoint_data", return_value=data))
                    stack.enter_context(mock.patch.object(maizels, "timepoint_pool_splits", side_effect=AssertionError("Training data rebuilt")))
                    loader = stack.enter_context(mock.patch.object(maizels_torch_checkpoint, "load_checkpoint", wraps=maizels_torch_checkpoint.load_checkpoint))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    eval_maizels_torch.main([
                        "--backend", backend, "--checkpoint", f"{backend}:0:{path}",
                        "--dataset_location", "dataset.csv", "--device", "cpu",
                        "--batch_size", "2", "--n_steps_per_interval", "1", "4",
                        "--out_dir", str(out),
                    ])
                self.assertEqual(loader.call_count, 1)
                rows = read_csv(out / "per_checkpoint.csv")
                self.assertEqual(len(rows), 22)
                self.assertEqual({row["protocol"] for row in rows}, {"interval"})
                sampler = "flowmap" if backend == "otpfm" else "euler"
                self.assertEqual({row["sampler"] for row in rows}, {sampler})
                self.assertEqual({row["n_steps_per_interval"] for row in rows}, {"1", "4"})
                metadata = json.loads((out / "metadata.json").read_text())
                self.assertEqual(set(metadata["protocols"]), {"interval"})
                self.assertEqual(metadata["test_times"], ["D3.2", "D3.6", "D4", "D5", "D7"])
                for steps in ("1", "4"):
                    step_rows = {row["timepoint"]: row for row in rows if row["n_steps_per_interval"] == steps}
                    self.assertEqual(set(step_rows), set(maizels.TIMEPOINTS) | {"test_times"})
                    values = []
                    for day in maizels.TIMEPOINTS:
                        row = step_rows[day]
                        left = "D3" if maizels.parse_timepoint(day) <= 3.8 else "D3.8"
                        self.assertEqual(int(row["source_n"]), len(populations[left]))
                        self.assertEqual(int(row["target_n"]), len(populations[day]))
                        if day == "D3":
                            expected = 0.0
                        else:
                            dt = metadata["normalized_times"][day] - metadata["normalized_times"][left]
                            velocity = 6.0 if backend == "otpfm" else 1.0
                            difference = populations[day][0] - populations[left][0] - velocity * dt
                            expected = np.linalg.norm(difference)
                        self.assertAlmostEqual(float(row["w1"]), expected, places=4)
                        if day in metadata["test_times"]:
                            values.append(float(row["w1"]))
                    self.assertAlmostEqual(float(step_rows["test_times"]["w1"]), np.mean(values))
                records = json.loads((out / "metrics.json").read_text())
                self.assertEqual(len(records), 2)
                for record in records:
                    self.assertIn(f"final_eval/{sampler}_mean_emd_test_times", record["metrics"])
                    self.assertTrue(all(name.startswith(f"final_eval/{sampler}_") for name in record["metrics"]))

    def test_saved_clock_and_endpoint_defaults_are_preserved(self):
        for backend, config, expected_retained, midpoint in (
            ("mfm", {"data_type": "maizels"}, ("D3", "D8"), 0.16),
            ("otpfm", {"dataset_name": "maizels", "maizels_time_mode": "equal_time"}, ("D3", "D3.8", "D8"), 0.5),
        ):
            cfg = maizels_torch_checkpoint.maizels_config(config, backend)
            self.assertEqual(cfg.problem.retained_timepoints, expected_retained)
            self.assertAlmostEqual(maizels.normalized_time("D3.8", cfg), midpoint)
        cfg = maizels_torch_checkpoint.maizels_config({"maizels_time_mode": "real_time"}, "mfm", schedule="d3_d3p8_d8", time_mode="equal_time")
        self.assertAlmostEqual(maizels.normalized_time("D3.8", cfg), 0.5)

    def test_parser_rejects_invalid_step_counts_and_deduplicates_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = save_mfm(Path(tmp))
            base = ["--backend", "mfm", "--checkpoint", f"mfm:0:{path}"]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                eval_maizels_torch.parse_args(base + ["--n_steps_per_interval", "1", "0"])
            args = eval_maizels_torch.parse_args(base + ["--n_steps_per_interval", "1", "2", "2"])
            self.assertEqual(args.n_steps_per_interval, [1, 2])


if __name__ == "__main__":
    unittest.main()

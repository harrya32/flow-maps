"""SF2M checkpoint sampling matches training and averages noise within seeds."""

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

from common import maizels, maizels_torch_checkpoint as checkpoint
from launchers import eval_maizels_torch
from mfm.flow_matchers.ema import EMA
from mfm.networks.flow_networks.mlp import VelocityNet


def save_sf2m(root, *, seed=0, dim=2, ema=False, embedded=False):
    run = root.resolve() / f"seed_{seed}"
    path = run / "checkpoints" / "maizels" / "run_id" / "flow_model" / "best.ckpt"
    path.parent.mkdir(parents=True)
    ema_heads = ema if isinstance(ema, tuple) else (ema, ema)
    config = {
        "data_type": "maizels",
        "dim": dim,
        "mfm": False,
        "sf2m": True,
        "sf2m_sigma": 0.2,
        "sf2m_score_weight": 1.0,
        "hidden_dims_flow": [4],
        "activation_flow": "selu",
        "maizels_schedule": "d3_d3p8_d8",
        "maizels_time_mode": "real_time",
        "maizels_hparam_val_times": ["D3.4", "D6"],
        "ema_decay": 0.99 if any(ema_heads) else None,
    }
    state = {}
    for prefix, raw_bias, shadow_bias, use_ema in (
        ("flow_net.", 1.0 + seed, 3.0 + seed, ema_heads[0]),
        ("score_net.", 2.0 + seed, 5.0 + seed, ema_heads[1]),
    ):
        model = VelocityNet(
            dim=dim, hidden_dims=[4], activation="selu", batch_norm=False
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.model[-1].bias.fill_(raw_bias)
        if use_ema:
            model = EMA(model, decay=0.99)
            with torch.no_grad():
                model.shadow_params[-1].fill_(shadow_bias)
        state.update({prefix + key: value for key, value in model.state_dict().items()})
    payload = {"state_dict": state, "global_step": 23}
    if embedded:
        payload["hyper_parameters"] = {"args": Namespace(**config)}
    else:
        (run / "sweep_config.yaml").write_text(yaml.safe_dump(config))
    torch.save(payload, path)
    return path, config


def sweep_row(path, *, seed=0, status="complete", method="sf2m", metrics=None):
    run = path.parents[4]
    metrics_path = run / "final_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"final_eval/best_checkpoint_path": str(path)}
            if metrics is None
            else metrics
        )
    )
    return {
        "method": method,
        "seed": seed,
        "status": status,
        "run_output_folder": str(run),
        "metrics_json": str(metrics_path),
    }


def write_sweep(root, rows):
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class SF2MCheckpointTests(unittest.TestCase):
    def test_loader_restores_both_ema_heads_and_saved_configuration(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded), tempfile.TemporaryDirectory() as tmp:
                path, _ = save_sf2m(Path(tmp), ema=True, embedded=embedded)
                for ema, flow_value, score_value in (
                    (None, 3.0, 5.0),
                    (False, 1.0, 2.0),
                ):
                    loaded = checkpoint.load_checkpoint(
                        "sf2m", path, device=torch.device("cpu"), ema=ema
                    )
                    state = torch.ones(3, 2)
                    time = torch.full((3, 1), 0.16)
                    torch.testing.assert_close(
                        loaded.sampler.velocity(time, state),
                        torch.full_like(state, flow_value),
                    )
                    torch.testing.assert_close(
                        loaded.sampler.score(time, state),
                        torch.full_like(state, score_value),
                    )
                    self.assertEqual(
                        loaded.metadata["parameters"],
                        "EMA" if ema is None else "instantaneous",
                    )
                    self.assertEqual(loaded.metadata["sf2m_sigma"], 0.2)
                    self.assertEqual(loaded.metadata["global_step"], 23)
                    self.assertEqual(loaded.dimension, 2)
                    cfg = checkpoint.maizels_config(loaded.config, "sf2m")
                    self.assertEqual(
                        cfg.problem.retained_timepoints, ("D3", "D3.8", "D8")
                    )
                    self.assertAlmostEqual(maizels.normalized_time("D3.8", cfg), 0.16)

    def test_sampler_matches_training_clock_diffusion_and_score_with_partial_batches(
        self,
    ):
        from mfm.flow_matchers.flow_net_train import SF2MFlowNetTrainTrajectory

        fields = []
        for time_weight, state_weight in ((0.5, 0.25), (0.7, -0.3)):
            model = VelocityNet(
                dim=2, hidden_dims=[], activation="selu", batch_norm=False
            )
            with torch.no_grad():
                model.model[0].weight.copy_(
                    torch.tensor(
                        [
                            [time_weight, state_weight, 0.0],
                            [time_weight, 0.0, state_weight],
                        ]
                    )
                )
                model.model[0].bias.zero_()
            fields.append(model.eval())
        reference = SimpleNamespace(
            flow_net=fields[0],
            score_net=fields[1],
            sf2m_sigma=0.2,
            seed_current=7,
        )
        source = np.arange(2210, dtype=np.float32).reshape(1105, 2) * 0.001
        original_source = source.copy()
        rng_before = torch.get_rng_state()
        for steps in (1, 2, 8):
            expected = SF2MFlowNetTrainTrajectory.evaluation_trajectory(
                reference,
                torch.from_numpy(source),
                start_time=0.16,
                end_time=1.0,
                n_steps=steps,
                seed=2701,
            )[-1].numpy()
            for batch_size in (37, 128, 2048):
                sampler = checkpoint.SF2MSampler(
                    *fields, 2, torch.device("cpu"), batch_size, 0.2
                )
                prediction = sampler.sample(
                    source, start_time=0.16, end_time=1.0, n_steps=steps
                )
                np.testing.assert_allclose(prediction, expected, rtol=2e-6, atol=1e-6)
                np.testing.assert_array_equal(source, original_source)
                self.assertEqual(prediction.shape, (1105, 2))
        repeated = sampler.sample(source, start_time=0.16, end_time=1.0, n_steps=8)
        np.testing.assert_array_equal(prediction, repeated)
        other_draw = sampler.sample(
            source, start_time=0.16, end_time=1.0, n_steps=8, seed=2702
        )
        self.assertFalse(np.array_equal(prediction, other_draw))
        torch.testing.assert_close(torch.get_rng_state(), rng_before)
        unchanged = sampler.sample(source, start_time=0.16, end_time=0.16, n_steps=1)
        np.testing.assert_array_equal(unchanged, source)
        self.assertFalse(np.shares_memory(unchanged, source))

    def test_missing_fields_wrong_backend_and_missing_diffusion_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = save_sf2m(Path(tmp))
            payload = torch.load(path, weights_only=False)
            cases = [
                ({**config, "sf2m_sigma": 0.0}, payload["state_dict"], "sf2m", "sigma"),
                (
                    {
                        key: value
                        for key, value in config.items()
                        if key != "sf2m_sigma"
                    },
                    payload["state_dict"],
                    "sf2m",
                    "Saved sf2m_sigma",
                ),
                (
                    {**config, "mfm": True},
                    payload["state_dict"],
                    "sf2m",
                    "requires an SF2M checkpoint",
                ),
                (config, payload["state_dict"], "mfm", "--backend sf2m"),
                (
                    {**config, "hidden_dims_flow": [8]},
                    payload["state_dict"],
                    "sf2m",
                    "does not match checkpoint architecture",
                ),
            ]
            for prefix in ("flow_net.", "score_net."):
                cases.append(
                    (
                        config,
                        {
                            key: value
                            for key, value in payload["state_dict"].items()
                            if not key.startswith(prefix)
                        },
                        "sf2m",
                        prefix.rstrip("."),
                    )
                )
            for saved_config, state, backend, message in cases:
                with self.subTest(message=message):
                    torch.save({"config": saved_config, "state_dict": state}, path)
                    with self.assertRaisesRegex(ValueError, message):
                        checkpoint.load_checkpoint(
                            backend, path, device=torch.device("cpu")
                        )

    def test_mixed_ema_heads_require_explicit_instantaneous_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = save_sf2m(Path(tmp), ema=(True, False))
            with self.assertRaisesRegex(ValueError, "inconsistent EMA"):
                checkpoint.load_checkpoint("sf2m", path, device=torch.device("cpu"))
            loaded = checkpoint.load_checkpoint(
                "sf2m", path, device=torch.device("cpu"), ema=False
            )
            self.assertEqual(loaded.metadata["parameters"], "instantaneous")
            with self.assertRaisesRegex(ValueError, "score_net checkpoint has no EMA"):
                checkpoint.load_checkpoint(
                    "sf2m", path, device=torch.device("cpu"), ema=True
                )

    def test_sampler_rejects_invalid_inputs_and_nonfinite_predictions(self):
        class InvalidField(torch.nn.Module):
            def forward(self, time, state):
                return torch.full_like(state, float("nan"))

        sampler = checkpoint.SF2MSampler(
            InvalidField(), InvalidField(), 2, torch.device("cpu"), 128, 0.2
        )
        source = np.zeros((3, 2), dtype=np.float32)
        base = {"start_time": 0.0, "end_time": 1.0, "n_steps": 1}
        for changes in (
            {"sampler": "euler"},
            {"n_steps": 0},
            {"start_time": 0.8, "end_time": 0.2},
            {"start_time": -0.1},
            {"end_time": 1.1},
            {"end_time": float("nan")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                sampler.sample(source, **{**base, **changes})
        for invalid in (
            np.empty((0, 2)),
            np.zeros((3, 3)),
            np.full((3, 2), float("inf")),
        ):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                sampler.sample(invalid, **base)
        with self.assertRaisesRegex(FloatingPointError, "sampling step 1"):
            sampler.sample(source, **base)


class SF2MSweepDiscoveryTests(unittest.TestCase):
    def test_discovery_selects_recorded_best_and_completed_requested_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [save_sf2m(root / "setting", seed=seed)[0] for seed in (1, 0, 2)]
            rows = [
                sweep_row(path, seed=seed, status="failed" if seed == 2 else "complete")
                for path, seed in zip(paths, (1, 0, 2))
            ]
            (paths[0].parent / "other_epoch.ckpt").touch()
            rows.append(
                {**rows[0], "method": "mfm", "seed": 9, "metrics_json": "missing.json"}
            )
            write_sweep(root, rows)
            self.assertEqual(
                checkpoint.checkpoints_from_sweep(root, "sf2m"),
                [("sf2m", 0, paths[1]), ("sf2m", 1, paths[0])],
            )
            self.assertEqual(
                checkpoint.checkpoints_from_sweep(root, "sf2m", [1]),
                [("sf2m", 1, paths[0])],
            )
            with self.assertRaisesRegex(ValueError, "seeds \\[2\\]"):
                checkpoint.checkpoints_from_sweep(root, "sf2m", [2])

    def test_missing_recorded_checkpoint_and_ambiguous_configurations_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = save_sf2m(root)
            row = sweep_row(path)
            write_sweep(root, [row, row])
            with self.assertRaisesRegex(ValueError, "Multiple completed"):
                checkpoint.checkpoints_from_sweep(root, "sf2m")
            row = sweep_row(
                path,
                metrics={
                    "final_eval/best_checkpoint_path": str(path.parent / "missing.ckpt")
                },
            )
            write_sweep(root, [row])
            with self.assertRaisesRegex(
                FileNotFoundError, "Recorded best checkpoint is missing"
            ):
                checkpoint.checkpoints_from_sweep(root, "sf2m")

    def test_fallback_requires_one_flow_checkpoint_and_excludes_last_and_geopath(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = save_sf2m(root)
            row = sweep_row(path, metrics={})
            write_sweep(root, [row])
            (path.parent / "last.ckpt").touch()
            geopath = path.parent.parent / "geopath_model" / "best.ckpt"
            geopath.parent.mkdir()
            geopath.touch()
            self.assertEqual(
                checkpoint.checkpoints_from_sweep(root, "sf2m"), [("sf2m", 0, path)]
            )
            (path.parent / "other_epoch.ckpt").touch()
            with self.assertRaisesRegex(ValueError, "Cannot identify one best"):
                checkpoint.checkpoints_from_sweep(root, "sf2m")


class SF2MCheckpointCLITests(unittest.TestCase):
    def test_sweep_scores_all_pca50_marginals_and_averages_draws_before_seeds(self):
        populations = {
            day: np.tile(np.arange(50) * index * 0.1, (2 + index % 3, 1)).astype(
                np.float32
            )
            for index, day in enumerate(maizels.TIMEPOINTS)
        }
        data = {
            "x": np.concatenate(list(populations.values())),
            "timepoints": np.concatenate(
                [np.full(len(x), day) for day, x in populations.items()]
            ),
        }
        source_before = data["x"].copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                save_sf2m(root / "setting", seed=seed, dim=50)[0] for seed in (0, 1)
            ]
            write_sweep(
                root, [sweep_row(path, seed=seed) for path, seed in zip(paths, (0, 1))]
            )
            out = root / "eval"
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(maizels, "all_timepoint_data", return_value=data)
                )
                stack.enter_context(
                    mock.patch.object(
                        maizels,
                        "timepoint_pool_splits",
                        side_effect=AssertionError("Training data rebuilt"),
                    )
                )
                loader = stack.enter_context(
                    mock.patch.object(
                        checkpoint, "load_checkpoint", wraps=checkpoint.load_checkpoint
                    )
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                eval_maizels_torch.main(
                    [
                        "--backend",
                        "sf2m",
                        "--sweep_root",
                        str(root),
                        "--seeds",
                        "0",
                        "1",
                        "--device",
                        "cpu",
                        "--batch_size",
                        "2",
                        "--dataset_location",
                        "dataset.csv",
                        "--n_steps_per_interval",
                        "1",
                        "4",
                        "--noise_draws",
                        "3",
                        "--out_dir",
                        str(out),
                    ]
                )
            self.assertEqual(loader.call_count, 2)
            np.testing.assert_array_equal(data["x"], source_before)
            per_draw = read_csv(out / "per_draw.csv")
            per_checkpoint = read_csv(out / "per_checkpoint.csv")
            summaries = read_csv(out / "summary.csv")
            self.assertEqual(
                (len(per_draw), len(per_checkpoint), len(summaries)), (132, 44, 22)
            )
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["noise_draws"], 3)
            self.assertEqual(metadata["sampler"], "euler_maruyama")
            self.assertEqual(set(metadata["protocols"]), {"interval"})
            self.assertEqual(metadata["test_times"], ["D3.2", "D3.6", "D4", "D5", "D7"])
            for row in per_draw:
                self.assertEqual(row["sampler"], "euler_maruyama")
                self.assertEqual(row["protocol"], "interval")
                day = row["timepoint"]
                if day == "test_times":
                    values = [
                        float(other["w1"])
                        for other in per_draw
                        if (
                            other["seed"] == row["seed"]
                            and other["noise_draw"] == row["noise_draw"]
                            and other["n_steps_per_interval"]
                            == row["n_steps_per_interval"]
                            and other["timepoint"] in metadata["test_times"]
                        )
                    ]
                    self.assertAlmostEqual(float(row["w1"]), np.mean(values))
                else:
                    left = "D3" if maizels.parse_timepoint(day) <= 3.8 else "D3.8"
                    self.assertEqual(row["source_timepoint"], left)
                    self.assertEqual(int(row["source_n"]), len(populations[left]))
                    self.assertEqual(int(row["target_n"]), len(populations[day]))
            for row in per_checkpoint:
                draws = [
                    float(other["w1"])
                    for other in per_draw
                    if all(
                        other[key] == row[key]
                        for key in ("seed", "n_steps_per_interval", "timepoint")
                    )
                ]
                self.assertEqual(int(row["n_draws"]), 3)
                self.assertAlmostEqual(float(row["w1_mean"]), np.mean(draws))
            for row in summaries:
                values = [
                    float(other["w1_mean"])
                    for other in per_checkpoint
                    if all(
                        other[key] == row[key]
                        for key in ("n_steps_per_interval", "timepoint")
                    )
                ]
                self.assertEqual(int(row["n_checkpoints"]), 2)
                self.assertAlmostEqual(float(row["w1_mean"]), np.mean(values))
                self.assertAlmostEqual(
                    float(row["w1_se"]), np.std(values, ddof=1) / np.sqrt(2)
                )
            records = json.loads((out / "metrics.json").read_text())
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertIn(
                    "final_eval/euler_maruyama_mean_emd_test_times", record["metrics"]
                )
                self.assertEqual(len(record["metrics"]), 11)

    def test_parser_selects_sweep_seeds_and_rejects_invalid_noise_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = save_sf2m(root)
            write_sweep(root, [sweep_row(path)])
            base = ["--backend", "sf2m", "--sweep_root", str(root)]
            args = eval_maizels_torch.parse_args(
                base + ["--seeds", "0", "--n_steps_per_interval", "1", "2", "2"]
            )
            self.assertEqual(args.checkpoint, [("sf2m", 0, path)])
            self.assertEqual(args.n_steps_per_interval, [1, 2])
            for options in (
                ["--noise_draws", "0"],
                ["--evaluation_seed", "-1"],
                ["--seeds", "0", "0"],
                ["--seeds", "-1"],
            ):
                with self.subTest(options=options), contextlib.redirect_stderr(
                    io.StringIO()
                ), self.assertRaises(SystemExit):
                    eval_maizels_torch.parse_args(base + options)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                eval_maizels_torch.parse_args(
                    [
                        "--backend",
                        "sf2m",
                        "--checkpoint",
                        f"sf2m:0:{path}",
                        "--seeds",
                        "0",
                    ]
                )


if __name__ == "__main__":
    unittest.main()

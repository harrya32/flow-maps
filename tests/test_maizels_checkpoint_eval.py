"""Regression coverage for all-marginal checkpoint evaluation and sampler flags."""

from __future__ import annotations

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

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

PY_ROOT = Path(__file__).resolve().parents[1] / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import maizels, maizels_checkpoint_eval, maizels_stochastic_eval
from common.stochastic_flow_map import StrongStochasticFlowMapMLP
from configs import maizels_stochastic
from launchers import eval_maizels_observed_w1 as deterministic
from launchers import eval_maizels_stochastic_observed_w1 as stochastic


TEST_TIMES = ("D3.2", "D3.6", "D4", "D5", "D7")


def data_and_config(large=False):
    cfg = SimpleNamespace(problem=SimpleNamespace(
        retained_timepoints=("D3", "D3.8", "D8"),
        hparam_val_times=("D3.4", "D6"),
    ))
    populations = {
        day: np.full(((1103 + index * 19) if large else (2 + index % 3), 2),
                     index * 2.0, dtype=np.float32)
        for index, day in enumerate(maizels.TIMEPOINTS)
    }
    data = {
        "x": np.concatenate(list(populations.values())),
        "timepoints": np.concatenate([
            np.full(len(x), day) for day, x in populations.items()
        ]),
    }
    return data, cfg, populations


def context(seed=0):
    return {
        "method": "ot", "seed": seed, "checkpoint": f"seed{seed}.msgpack",
        "sampler": "flowmap", "n_steps_per_interval": 7,
    }


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class MarginalProtocolTests(unittest.TestCase):
    def test_every_marginal_full_populations_and_rollout_only_advances_at_observations(self):
        data, cfg, populations = data_and_config(large=True)
        marginals = maizels_checkpoint_eval.build_marginals(cfg, data)
        calls = []
        scored_sizes = []

        def pushforward(source, **kwargs):
            calls.append((source.copy(), kwargs))
            return source + 10.0 + kwargs["end_time"] - kwargs["start_time"]

        def score(prediction, target):
            scored_sizes.append((len(prediction), len(target)))
            return abs(float(np.mean(prediction)) - float(np.mean(target)))

        rows = maizels_checkpoint_eval.evaluate_marginals(
            marginals, pushforward, context=context(), score=score
        )
        self.assertEqual(marginals.test_times, TEST_TIMES)
        self.assertEqual(len(rows), 22)
        for protocol in ("interval", "rollout"):
            day_rows = {r["timepoint"]: r for r in rows if r["protocol"] == protocol}
            self.assertEqual(set(day_rows), set(maizels.TIMEPOINTS) | {"test_times"})
            for day in maizels.TIMEPOINTS:
                row = day_rows[day]
                self.assertEqual(row["target_n"], len(populations[day]))
                left = "D3" if maizels.parse_timepoint(day) <= 3.8 else "D3.8"
                expected_source = "D3" if protocol == "rollout" else left
                self.assertEqual(row["source_n"], len(populations[expected_source]))
            aggregate = day_rows["test_times"]
            self.assertEqual(aggregate["n_timepoints"], 5)
            self.assertAlmostEqual(
                aggregate["w1"], np.mean([day_rows[day]["w1"] for day in TEST_TIMES])
            )
        self.assertIn((len(populations["D3"]), len(populations["D3.8"])), scored_sizes)
        self.assertIn((len(populations["D3.8"]), len(populations["D7"])), scored_sizes)
        carried_sources = [source for source, kw in calls if kw["protocol"] == "rollout"]
        self.assertEqual(len(carried_sources), 5)
        for source in carried_sources:
            np.testing.assert_allclose(source, populations["D3"] + 10.16)
        records = maizels_checkpoint_eval.metric_records(rows)
        self.assertEqual(len(records), 1)
        self.assertIn("final_eval/flowmap_mean_emd_test_times", records[0]["metrics"])

    def test_validation_override_and_missing_marginal(self):
        data, cfg, _ = data_and_config()
        marginals = maizels_checkpoint_eval.build_marginals(cfg, data, "3.2,d7,")
        self.assertEqual(marginals.hparam_val_times, ("D3.2", "D7"))
        self.assertNotIn("D3.2", marginals.test_times)
        self.assertIn("D3.4", marginals.test_times)
        self.assertEqual(len(maizels_checkpoint_eval.build_marginals(cfg, data, "none").test_times), 7)
        with self.assertRaisesRegex(ValueError, "retained observation"):
            maizels_checkpoint_eval.build_marginals(cfg, data, "D3.8")
        keep = data["timepoints"] != "D5"
        with self.assertRaisesRegex(ValueError, "Missing.*D5"):
            maizels_checkpoint_eval.build_marginals(
                cfg, {key: value[keep] for key, value in data.items()}
            )

    def test_endpoint_schedule_and_equal_time_clock_use_the_saved_intervals(self):
        data, _, _ = data_and_config()
        for schedule, clock, middle_time, n_test in (
            ("d3_d8", "real_time", 0.16, 6),
            ("d3_d3p8_d8", "equal_time", 0.5, 5),
        ):
            with self.subTest(schedule=schedule, clock=clock):
                cfg = maizels_stochastic.get_config(
                    5, maizels_schedule=schedule, maizels_time_mode=clock,
                    hparam_val_times=("D3.4", "D6"),
                )
                marginals = maizels_checkpoint_eval.build_marginals(cfg, data)
                self.assertAlmostEqual(marginals.times["D3.8"], middle_time)
                self.assertEqual(len(marginals.test_times), n_test)
                calls = []

                def sample(source, **kwargs):
                    calls.append(kwargs)
                    return source.copy()

                rows = maizels_checkpoint_eval.evaluate_marginals(
                    marginals, sample, context=context(), score=lambda x, y: 0.0
                )
                self.assertEqual(len(rows), 22)
                d4_call = next(call for call in calls
                               if call["target_index"] == maizels.TIMEPOINTS.index("D4")
                               and call["protocol"] == "interval")
                self.assertAlmostEqual(d4_call["start_time"], 0.0 if n_test == 6 else 0.5)
                self.assertAlmostEqual(d4_call["end_time"], marginals.times["D4"])


class ToyDeterministicMap:
    def __init__(self, euler):
        self.euler = euler

    def apply(self, params, left, *args, **kwargs):
        if self.euler:
            assert kwargs.get("method") == "calc_b"
            return args[0]
        assert "method" not in kwargs
        right, point = args
        return point * (1.0 + 2.0 * (right - left))


class DeterministicSamplerTests(unittest.TestCase):
    def test_exact_step_count_and_euler_field_with_partial_batches(self):
        source = np.arange(1, 15, dtype=np.float32).reshape(7, 2)
        for euler in (False, True):
            for n_steps in (2, 5):
                with self.subTest(euler=euler, n_steps=n_steps):
                    prediction = deterministic._pushforward(
                        ToyDeterministicMap(euler), {}, source,
                        start_time=0.16, end_time=1.0, n_steps=n_steps,
                        batch_size=4, euler=euler,
                    )
                    rate = 1.0 if euler else 2.0
                    expected = source * (1.0 + rate * 0.84 / n_steps) ** n_steps
                    np.testing.assert_allclose(prediction, expected, rtol=1e-6)
                    self.assertEqual(prediction.shape, source.shape)

    def test_cli_scores_every_day_and_reports_only_selected_sampler(self):
        data, _, _ = data_and_config()
        for euler in (False, True):
            with self.subTest(euler=euler), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "best.msgpack"
                checkpoint.write_bytes(b"test")
                out = Path(tmp) / "eval"
                argv = [
                    "eval", "--checkpoint", f"ot:0:{checkpoint}",
                    "--n_steps_per_interval", "3", "--batch_size", "4",
                    "--hparam_val_times", "D3.4,D6", "--out_dir", str(out),
                ] + (["--euler"] if euler else [])
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(sys, "argv", argv))
                    stack.enter_context(mock.patch.object(maizels, "all_timepoint_data", return_value=data))
                    stack.enter_context(mock.patch.object(deterministic, "_adaptive_rescale"))
                    stack.enter_context(mock.patch.object(deterministic, "_restore", return_value=(ToyDeterministicMap(euler), {})))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    deterministic.main()
                rows = read_csv(out / "per_checkpoint.csv")
                sampler = "euler" if euler else "flowmap"
                self.assertEqual({r["sampler"] for r in rows}, {sampler})
                self.assertEqual({r["n_steps_per_interval"] for r in rows}, {"3"})
                self.assertEqual({r["timepoint"] for r in rows}, set(maizels.TIMEPOINTS) | {"test_times"})
                record = json.loads((out / "metrics.json").read_text())[0]
                self.assertIn(f"final_eval/{sampler}_mean_emd_test_times", record["metrics"])
                self.assertTrue(all(f"/{sampler}_" in name for name in record["metrics"]))


def save_stochastic_checkpoint(path, seed=0, local_fraction=0.75, embedded=True):
    cfg = maizels_stochastic.get_config(5, hparam_val_times=("D3.4", "D6"))
    cfg.problem.d = 2
    cfg.ssfm.hidden_dim = 8
    cfg.ssfm.n_hidden = 1
    cfg.ssfm.n_coefficients = 1
    cfg.ssfm.uncertainty_hidden_dim = 4
    cfg.ssfm.local_fraction = local_fraction
    cfg.evaluation.flowmap_n_steps = 3
    cfg.evaluation.euler_maruyama_n_steps = 5
    model = StrongStochasticFlowMapMLP(
        data_dim=2, n_coefficients=1, hidden_dim=8, n_hidden=1,
        uncertainty_hidden_dim=4,
    )
    variables = model.init(
        jax.random.PRNGKey(0), jnp.zeros(1), jnp.ones(1),
        jnp.zeros((1, 2)), jnp.zeros((1, 1, 2)),
    )
    params = jax.tree_util.tree_map(jnp.zeros_like, variables["params"])
    params["drift"]["output"]["bias"] = jnp.full((2,), float(seed))
    payload = {
        "ema_params": params, "pc_scale": np.ones(2, dtype=np.float32),
        "best_step": 11, "best_validation_loss": 0.25,
    }
    if embedded:
        payload["config"] = cfg.to_dict()
    path.with_name("config.json").write_text(json.dumps(cfg.to_dict()))
    path.write_bytes(flax.serialization.to_bytes(payload))


class StochasticCheckpointTests(unittest.TestCase):
    def test_loads_companion_config_and_ema_parameters_for_old_and_new_checkpoints(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "best.msgpack"
                save_stochastic_checkpoint(path, seed=2, embedded=embedded)
                model, params, cfg, _ = stochastic._load_checkpoint(path)
                prediction, _ = model.apply(
                    {"params": params}, jnp.zeros(1), jnp.ones(1),
                    jnp.zeros((1, 2)), jnp.zeros((1, 1, 2)),
                )
                np.testing.assert_allclose(prediction, [[2.0, 2.0]])
                self.assertEqual(cfg.problem.d, 2)
                self.assertEqual(tuple(cfg.problem.retained_timepoints), ("D3", "D3.8", "D8"))

    def run_cli(self, checkpoints, data, out, extra):
        argv = ["eval", "--dataset_location", "dataset.csv", "--out_dir", str(out)]
        for seed, checkpoint in enumerate(checkpoints):
            argv.extend(["--checkpoint", f"ot:{seed}:{checkpoint}"])
        argv.extend(extra)
        euler = "--euler" in extra
        selected = "sample_euler_maruyama_pushforward" if euler else "sample_composed_pushforward"
        unselected = "sample_composed_pushforward" if euler else "sample_euler_maruyama_pushforward"

        def sample(model, params, source, start, end, key, **kwargs):
            seed = float(params["drift"]["output"]["bias"][0])
            noise = int(np.asarray(key, dtype=np.uint64).sum() % 97) * 0.001
            return source + (end - start) * (1.0 + seed) + noise

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", argv))
            stack.enter_context(mock.patch.object(maizels, "all_timepoint_data", return_value=data))
            sampler = stack.enter_context(mock.patch.object(maizels_stochastic_eval, selected, side_effect=sample))
            forbidden = stack.enter_context(mock.patch.object(maizels_stochastic_eval, unselected, side_effect=AssertionError("Unselected sampler called")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stochastic.main()
        forbidden.assert_not_called()
        return sampler

    def test_composed_cli_test_mean_and_uncertainty_are_across_seed_means(self):
        data, _, _ = data_and_config()
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f"seed{seed}.msgpack" for seed in (0, 1)]
            for seed, path in enumerate(paths):
                save_stochastic_checkpoint(path, seed=seed)
            out = Path(tmp) / "eval"
            sampler = self.run_cli(paths, data, out, ["--noise_draws", "2", "--n_steps_per_interval", "7"])
            self.assertTrue(sampler.call_args_list)
            self.assertEqual({call.kwargs["n_steps"] for call in sampler.call_args_list}, {7})
            raw = read_csv(out / "per_draw.csv")
            self.assertEqual({r["sampler"] for r in raw}, {"flowmap"})
            self.assertEqual({r["timepoint"] for r in raw}, set(maizels.TIMEPOINTS) | {"test_times"})
            for seed in ("0", "1"):
                for draw in ("0", "1"):
                    day_rows = {r["timepoint"]: r for r in raw
                                if r["seed"] == seed and r["noise_draw"] == draw and r["protocol"] == "interval"}
                    expected = np.mean([float(day_rows[day]["w1"]) for day in TEST_TIMES])
                    self.assertAlmostEqual(float(day_rows["test_times"]["w1"]), expected)
            means = [
                np.mean([float(r["w1"]) for r in raw if r["seed"] == str(seed)
                         and r["timepoint"] == "test_times" and r["protocol"] == "interval"])
                for seed in (0, 1)
            ]
            summary = next(r for r in read_csv(out / "summary.csv")
                           if r["protocol"] == "interval" and r["timepoint"] == "test_times")
            self.assertEqual(summary["n"], "2")
            self.assertAlmostEqual(float(summary["w1_mean"]), np.mean(means))
            self.assertAlmostEqual(float(summary["w1_se"]), np.std(means, ddof=1) / np.sqrt(2))
            records = json.loads((out / "metrics.json").read_text())
            self.assertEqual(len(records), 2)
            for record in records:
                self.assertAlmostEqual(record["metrics"]["final_eval/flowmap_mean_emd_test_times"], means[record["seed"]])
                self.assertTrue(all("/flowmap_" in name for name in record["metrics"]))

    def test_euler_flag_uses_selected_checkpoint_default_and_explicit_steps(self):
        data, _, _ = data_and_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.msgpack"
            save_stochastic_checkpoint(path, local_fraction=1.0)
            for requested, expected in ((0, 5), (9, 9)):
                out = Path(tmp) / f"eval{requested}"
                sampler = self.run_cli([path], data, out, [
                    "--euler", "--noise_draws", "1", "--n_steps_per_interval", str(requested),
                ])
                self.assertEqual({call.kwargs["n_steps"] for call in sampler.call_args_list}, {expected})
                record = json.loads((out / "metrics.json").read_text())[0]
                self.assertIn("final_eval/euler_maruyama_mean_emd_test_times", record["metrics"])
                self.assertTrue(all("/euler_maruyama_" in name for name in record["metrics"]))
            with self.assertRaisesRegex(ValueError, "select --euler"):
                self.run_cli([path], data, Path(tmp) / "invalid", [])


if __name__ == "__main__":
    unittest.main()

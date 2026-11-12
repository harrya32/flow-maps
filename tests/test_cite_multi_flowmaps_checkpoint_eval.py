"""Native CITE/Multi evaluation restores real Flax checkpoints and PCA scaling."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for directory in (REPO_ROOT / "py", REPO_ROOT / "tests"):
    sys.path.insert(0, str(directory))

from common import (
    cite_multi,
    cite_multi_flowmaps_checkpoint as loader,
    datasets,
    state_utils,
    wasserstein,
)
from configs import cite_multi_pca100
from launchers import eval_cite_multi
from test_cite_multi_checkpoint_eval import fixture_data, write_dataset


def save_checkpoint(root, dataset_name="cite", omitted="4", clock="equal_time"):
    cfg = cite_multi_pca100.get_config(
        7,
        str(root),
        dataset_name=dataset_name,
        heldout_day=omitted,
        cite_multi_time_mode=clock,
        seed=4,
    )
    cfg.network.n_neurons = 4
    cfg.network.n_hidden = 0
    cfg.network.rescale = 2.5
    cfg.training.ema_facs = [0.9]
    cfg.problem.n = 21
    state, _, _, _ = state_utils.setup_training_state(
        cfg, jnp.zeros(100), jax.random.PRNGKey(4)
    )
    params = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    params["params"]["flow_map"]["phi_mlp"]["Dense_1"]["bias"] = jnp.ones(100)
    ema = jax.tree_util.tree_map(lambda x: x * 2.0, params)
    state = state.replace(params=params, ema_params={0.9: ema}, step=jnp.asarray(17))
    path = root / "best.pkl"
    path.write_bytes(flax.serialization.to_bytes(state))
    config_path = root / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()))
    return path, config_path, cfg


class NativeCheckpointTests(unittest.TestCase):
    def test_test_only_scores_only_the_omitted_day_for_each_step_count(self):
        _, populations = fixture_data()
        for omitted in ("3", "4"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_dataset(root, "cite")
                path, config_path, _ = save_checkpoint(root, omitted=omitted)
                out = root / "eval"
                with mock.patch.object(
                    wasserstein, "exact_emd", wraps=wasserstein.exact_emd
                ) as scoring, contextlib.redirect_stdout(io.StringIO()):
                    eval_cite_multi.main(
                        [
                            "--backend",
                            "flowmaps",
                            "--checkpoint",
                            f"map:4:{path}",
                            "--config_path",
                            str(config_path),
                            "--dataset_location",
                            str(root),
                            "--rescale",
                            "2.5",
                            "--device",
                            "cpu",
                            "--test-only",
                            "--n_steps_per_interval",
                            "1",
                            "4",
                            "--out_dir",
                            str(out),
                        ]
                    )
                self.assertEqual(scoring.call_count, 2)
                with (out / "per_checkpoint.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 4)
                self.assertEqual(
                    {row["timepoint"] for row in rows}, {omitted, "test_times"}
                )
                previous = "2" if omitted == "3" else "3"
                for row in rows:
                    if row["timepoint"] == omitted:
                        self.assertEqual(row["source_timepoint"], previous)
                        self.assertEqual(
                            int(row["source_n"]), len(populations[previous])
                        )
                metadata = json.loads((out / "metadata.json").read_text())
                self.assertTrue(metadata["test_only"])
                self.assertEqual(metadata["evaluated_timepoints"], [omitted])

    def test_all_folds_clocks_and_samplers_restore_once_and_score_all_marginals(self):
        _, populations = fixture_data()
        for name in ("cite", "multi"):
            for omitted in ("3", "4"):
                for clock in cite_multi.TIME_MODES:
                    with self.subTest(
                        name=name, omitted=omitted, clock=clock
                    ), tempfile.TemporaryDirectory() as tmp:
                        root = Path(tmp)
                        write_dataset(root, name)
                        path, config_path, _ = save_checkpoint(
                            root, name, omitted, clock
                        )
                        for euler in (False, True):
                            out = root / ("euler" if euler else "flowmap")
                            with contextlib.ExitStack() as stack:
                                wrapped = stack.enter_context(
                                    mock.patch.object(
                                        loader,
                                        "load_checkpoint",
                                        wraps=loader.load_checkpoint,
                                    )
                                )
                                stack.enter_context(
                                    mock.patch.object(
                                        cite_multi,
                                        "make_pair_pool",
                                        side_effect=AssertionError(
                                            "Scale override rebuilt pairs"
                                        ),
                                    )
                                )
                                stack.enter_context(
                                    contextlib.redirect_stdout(io.StringIO())
                                )
                                eval_cite_multi.main(
                                    [
                                        "--backend",
                                        "flowmaps",
                                        "--checkpoint",
                                        f"ot_map:4:{path}",
                                        "--config_path",
                                        str(config_path),
                                        "--dataset_location",
                                        str(root),
                                        "--rescale",
                                        "2.5",
                                        "--device",
                                        "cpu",
                                        "--batch_size",
                                        "2",
                                        "--n_steps_per_interval",
                                        "1",
                                        "4",
                                        "--out_dir",
                                        str(out),
                                    ]
                                    + (["--euler", "--ema_fac", "0.9"] if euler else [])
                                )
                            self.assertEqual(wrapped.call_count, 1)
                            with (out / "per_checkpoint.csv").open() as handle:
                                rows = list(csv.DictReader(handle))
                            sampler = "euler" if euler else "flowmap"
                            self.assertEqual(len(rows), 10)
                            self.assertEqual(
                                {row["sampler"] for row in rows}, {sampler}
                            )
                            self.assertEqual(
                                {row["protocol"] for row in rows}, {"interval"}
                            )
                            metadata = json.loads((out / "metadata.json").read_text())
                            self.assertEqual(metadata["test_times"], [omitted])
                            self.assertEqual(metadata["checkpoints"][0]["rescale"], 2.5)
                            self.assertEqual(
                                metadata["checkpoints"][0]["global_step"], 17
                            )
                            for steps in ("1", "4"):
                                by_day = {
                                    row["timepoint"]: row
                                    for row in rows
                                    if row["n_steps_per_interval"] == steps
                                }
                                for day in cite_multi.TIMEPOINTS:
                                    previous = (
                                        max(
                                            (
                                                left
                                                for left in metadata[
                                                    "retained_timepoints"
                                                ]
                                                if float(left) < float(day)
                                            ),
                                            key=float,
                                        )
                                        if day != "2"
                                        else "2"
                                    )
                                    dt = (
                                        metadata["normalized_times"][day]
                                        - metadata["normalized_times"][previous]
                                    )
                                    velocity = 5.0 if euler else 2.5
                                    expected = np.linalg.norm(
                                        populations[day][0].astype(float)
                                        - populations[previous][0]
                                        - velocity * dt
                                    )
                                    self.assertAlmostEqual(
                                        float(by_day[day]["w1"]), expected, places=4
                                    )
                                    self.assertEqual(
                                        int(by_day[day]["source_n"]),
                                        len(populations[previous]),
                                    )
                                    self.assertEqual(
                                        int(by_day[day]["target_n"]),
                                        len(populations[day]),
                                    )
                                self.assertEqual(
                                    by_day["test_times"]["w1"], by_day[omitted]["w1"]
                                )
                            records = json.loads((out / "metrics.json").read_text())
                            self.assertEqual(len(records), 2)
                            self.assertTrue(
                                all(
                                    f"final_eval/{sampler}_mean_emd_test_times"
                                    in record["metrics"]
                                    for record in records
                                )
                            )

    def test_reconstructed_scale_matches_training_for_compact_and_sampled_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_dataset(root, "cite")
            for preset in (1, 7):
                cfg = cite_multi_pca100.get_config(preset, str(root), seed=4)
                cfg.problem.n = 21
                if preset == 7:
                    pools, _ = cite_multi.make_minibatch_ot_training_pools(cfg)
                    expected = datasets._compact_endpoint_pool_std(pools)
                else:
                    pairs, _ = cite_multi.make_pair_pool(cfg)
                    expected = np.std(np.concatenate([pairs["x0"], pairs["x1"]]))
                with contextlib.redirect_stdout(io.StringIO()):
                    scale, _ = loader.training_rescale(cfg)
                self.assertAlmostEqual(scale, float(expected), places=12)

    def test_native_factory_uses_checkpoint_seed_and_flow_matching_preset(self):
        cfg, _ = loader.training_config(
            seed=4,
            dataset_name="multi",
            omitted_day="3",
            slurm_id=0,
            time_mode="real_time",
        )
        self.assertEqual(cfg.training.seed, 4)
        self.assertEqual(cfg.optimization.diag_fraction, 1.0)
        self.assertEqual(cfg.problem.d, 100)
        self.assertEqual(cfg.problem.timepoint_values, [0.0, 0.2, 0.4, 1.0])

    def test_missing_ema_and_wrong_architecture_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, cfg = save_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "EMA decay.*absent"):
                loader.load_checkpoint(
                    path, cfg, device=jax.devices("cpu")[0], ema_fac=0.99, rescale=2.5
                )
            cfg.network.n_neurons = 8
            with self.assertRaisesRegex(ValueError, "checkpoint architecture"):
                loader.load_checkpoint(
                    path, cfg, device=jax.devices("cpu")[0], rescale=2.5
                )


if __name__ == "__main__":
    unittest.main()

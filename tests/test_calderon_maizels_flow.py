"""Check flow holdouts, native validation, and interval-local evaluation."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "tests"))

from common import calderon, calderon_logging
from configs import calderon_pca100, calderon_filtered_pca100, calderon_constrained_pca100, calderon_constrained_w2000_pca100, maizels_pca50
from launchers.calderon_ot_flow_map import configure, make_validation_evaluator, parse_args
from test_calderon_three_time_flow import fixture


def toy_map(params, s, t, x=None, **kwargs):
    if x is None:
        return .01 * t + 1. + s
    return x + (t - s) * (.01 * x + 1. + s)


def manual_compose(x, start, end, steps):
    result = np.asarray(x, dtype=np.float32).copy()
    times = np.linspace(start, end, steps + 1, dtype=np.float32)
    for s, t in zip(times[:-1], times[1:]):
        result = result + (t - s) * (.01 * result + 1. + s)
    return result


class CalderonMaizelsProtocolTests(unittest.TestCase):
    def test_requested_defaults_match_corresponding_maizels_variants(self):
        for module, variant in ((calderon_pca100, 7), (calderon_filtered_pca100, 5), (calderon_constrained_pca100, 6),
                                (calderon_constrained_w2000_pca100, 6)):
            cfg = module.get_config()
            reference = maizels_pca50.get_config(variant, maizels_schedule="d3_d3p8_d8", seed=0)
            self.assertEqual(cfg.optimization.to_dict(), reference.optimization.to_dict())
            for field in ("scalar_freq", "progress_freq", "visual_freq", "save_freq", "plot_bs", "traj_plot_bs",
                          "line_plot_bs", "line_plot_n_times", "multi_step_line_steps", "euler_line_steps", "visual_ema_factor"):
                self.assertEqual(cfg.logging[field], reference.logging[field])
            self.assertIsNone(cfg.evaluation.ema_factor)
            self.assertEqual(cfg.evaluation.flowmap_n_steps, 50)
            self.assertEqual(cfg.evaluation.source_max_points, 512)
            self.assertEqual(cfg.evaluation.marginal_max_points, 2048)
            self.assertEqual(cfg.logging.maizels.distribution_eval_source_max_points, 2048)
            self.assertEqual(cfg.logging.maizels.distribution_eval_points_per_time, 2048)
            self.assertEqual(cfg.logging.maizels.distribution_eval_samplers, ["flowmap", "euler"])
            self.assertEqual(cfg.logging.maizels.trajectory_eval_samplers, ["flowmap", "euler"])
            self.assertEqual(cfg.logging.maizels.distribution_eval_metrics, ["emd"])
            self.assertFalse(cfg.logging.direct_plots_enabled)
            self.assertFalse(cfg.logging.constraint_metrics_enabled)
            if variant != 7:
                self.assertEqual(cfg.problem.interpolant_filter_backend, "jax")
            self.assertEqual(cfg.evaluation.validation_batch_size, 1024)
            self.assertEqual(cfg.evaluation.validation_batches, 1)
            self.assertEqual(cfg.evaluation.validation_objective, "same_as_training")
            self.assertEqual(cfg.problem.ot_minibatch_infeasible_fallback, reference.problem.ot_minibatch_infeasible_fallback)
            self.assertEqual(cfg.classifier.prob_threshold, 0)
            self.assertEqual(cfg.classifier.margin_threshold, 0)

    def test_training_logging_skips_the_extra_constraint_forward_pass(self):
        import jax
        import jax.numpy as jnp
        from common import logging
        cfg = calderon_constrained_pca100.get_config()
        cfg.training.ndevices = 1
        cfg.logging.visual_freq = 0
        state = SimpleNamespace(step=jnp.asarray(1))
        statics = SimpleNamespace(schedule=lambda step: jnp.asarray(.0003))
        with patch("common.logging.compute_constraint_metrics") as additional, \
             patch("common.logging.compute_endpoint_matching_metrics", return_value={}), \
             patch("common.logging.compute_mfm_distribution_metrics", return_value={}), \
             patch("common.logging.wandb.log"):
            _, metrics = logging.log_metrics(cfg, statics, state, jnp.ones(2), jnp.asarray(7.), (),
                                             jax.random.PRNGKey(0), .1, compute_validation=False)
        additional.assert_not_called()
        self.assertEqual(float(metrics["loss"]), 7.)
        self.assertFalse(any(name.startswith("constraint/") for name in metrics))

    def test_flow_holdouts_follow_maizels_and_do_not_change_with_model_seed(self):
        from common.maizels import _split_train_holdout_indices
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            cfg.training.seed = 19
            recreated = calderon.load_endpoint_experiment(cfg)
            for window, offset in (("4-6", 11), ("8-10", 303), ("12-14", 29)):
                train, heldout = _split_train_holdout_indices(30, holdout_fraction=.1, holdout_n=0, seed=701 + offset)
                np.testing.assert_array_equal(data.splits[window]["train"], train)
                np.testing.assert_array_equal(data.splits[window]["validation"], heldout)
                np.testing.assert_array_equal(recreated.splits[window]["validation"], heldout)

    def test_fixed_validation_pairs_keep_filtering_and_use_only_holdout_cells(self):
        import jax
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
            cfg.problem.n_interpolant_check_times = 50
            cfg.problem.classifier_prob_threshold = 0.
            cfg.problem.classifier_margin_threshold = 0.
            cfg.problem.classifier_batch_size = 8192
            Path(cfg.classifier.allowed_pairs_path).write_text("source,target\n" + "".join(f"{a},{b}\n" for a in "ABC" for b in "ABC"))
            calderon_logging.register(cfg, data)
            inspected = []
            def checker(source_x, source_type_ids, target_x, target_type_ids):
                inspected.append(len(source_x))
                return {"valid": np.ones(len(source_x), dtype=bool)}
            def dataset(copy, population, split):
                self.assertEqual(copy.problem.maizels_pair_mode, "ot_endpoint_interpolant")
                self.assertEqual(split, "validation")
                return calderon.MultiMarginalOTDataset(copy, population, split, interpolant_checker=checker)
            with patch("common.calderon.training_dataset", side_effect=dataset):
                pairs, _ = calderon_logging.make_validation_pair_pool(cfg, 1024, pair_mode="same_as_training", seed=2701)
                repeated, _ = calderon_logging.make_validation_pair_pool(cfg, 1024, pair_mode="same_as_training", seed=2701)
            self.assertTrue(inspected)
            self.assertEqual(pairs["x0"].shape, (1024, 100))
            np.testing.assert_array_equal(pairs["x0"], repeated["x0"])
            for field in ("x0", "x1"):
                allowed_rows = np.concatenate([data.pool(window, "validation")["x"] for window in data.training_times])
                self.assertTrue(np.isin(pairs[field][:, 0], allowed_rows[:, 0]).all())

    def test_validation_evaluates_training_constraint_with_instantaneous_params(self):
        import jax.numpy as jnp
        from ml_collections import ConfigDict
        for extra, weight in (([], 1000.), (["--constraint-weight", "100"], 100.)):
            with self.subTest(weight=weight), tempfile.TemporaryDirectory() as directory:
                argv = ["calderon_ot_flow_map.py", "--config", "configs.calderon_constrained_pca100", *extra]
                with patch.object(sys, "argv", argv):
                    configured = configure(parse_args())
                cfg, _, _ = fixture(Path(directory))
                cfg.constraints = ConfigDict(configured.constraints.to_dict())
                data = calderon.load_endpoint_experiment(cfg)
                seen = []
                def loss(params, teacher, *args):
                    seen.append(args[0].shape)
                    self.assertTrue(cfg.constraints.enabled)
                    return params["marker"] + teacher["marker"] + cfg.constraints.weight
                state = SimpleNamespace(params={"marker": jnp.array(7.)},
                    ema_params={.999: {"marker": jnp.array(100.)}}, step=jnp.array(1))
                evaluator = make_validation_evaluator(cfg, SimpleNamespace(loss=loss), data)
                self.assertEqual(evaluator(state), weight + 14.)
                self.assertEqual(seen, [(1024, 100)])

    def test_constraint_weight_cli_preserves_defaults_and_rejects_invalid_overrides(self):
        for module, expected in (("configs.calderon_constrained_pca100", 1000.),
                                 ("configs.calderon_constrained_w2000_pca100", 2000.)):
            argv = ["calderon_ot_flow_map.py", "--config", module]
            for extra, weight in (([], expected), (["--constraint-weight", "100"], 100.),
                                  (["--constraint-weight", "0"], 0.)):
                with self.subTest(module=module, weight=weight), patch.object(sys, "argv", argv + extra):
                    cfg = configure(parse_args())
                    self.assertEqual(cfg.constraints.weight, weight)
                    self.assertEqual(cfg.to_dict()["constraints"]["weight"], weight)
        for value in ("-1", "nan", "inf"):
            argv = ["calderon_ot_flow_map.py", "--config", "configs.calderon_constrained_pca100",
                    "--constraint-weight", value]
            with self.subTest(value=value), patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "finite nonnegative"):
                    configure(parse_args())
        with patch.object(sys, "argv", ["calderon_ot_flow_map.py", "--constraint-weight", "100"]):
            with self.assertRaisesRegex(ValueError, "requires a constrained config"):
                configure(parse_args())
        with patch.object(sys, "argv", ["calderon_ot_flow_map.py", "--stage", "evaluate",
                                       "--output-dir", "/private/tmp/calderon_saved_weight", "--constraint-weight", "100"]):
            with self.assertRaisesRegex(ValueError, "training/data overrides"):
                configure(parse_args())

    def test_uncapped_w1_resets_to_observed_sources_and_chains_retained_intervals(self):
        from common import logging
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            calderon_logging.register(cfg, data)
            cfg.evaluation.marginal_max_points = 0
            cfg.logging.maizels.distribution_eval_source_max_points = 0
            cfg.logging.maizels.distribution_eval_points_per_time = 0
            cfg.logging.maizels.distribution_eval_samplers = ["flowmap"]
            cfg.logging.maizels.distribution_eval_metrics = ["emd"]
            state = SimpleNamespace(apply_fn=toy_map)
            populations = []
            with patch("common.logging.wandb.log"):
                logging._log_maizels_distribution_eval(cfg, state, {}, force=True, population_callback=lambda **p: populations.append(p))
                logging._compute_observed_marginal_emd_metrics(cfg, state, {}, population_callback=lambda **p: populations.append(p))
            self.assertEqual(len(populations), 5)
            self.assertEqual([p["source_window"] for p in populations[:2]], ["4-6", "8-10"])
            self.assertEqual([p["n_steps"] for p in populations], [50, 50, 50, 50, 100])
            for population in populations:
                self.assertEqual(len(population["source_x"]), 30)
                self.assertEqual(len(population["reference_x"]), 30)
                if population["kind"] != "rollout":
                    expected = manual_compose(population["source_x"], population["start_time"], population["end_time"], 50)
                    np.testing.assert_allclose(population["prediction"], expected, atol=2e-4, rtol=2e-6)
            final = populations[-1]
            expected = manual_compose(manual_compose(final["source_x"], 0., .5, 50), .5, 1., 50)
            np.testing.assert_allclose(final["prediction"], expected, atol=2e-4, rtol=2e-6)
            self.assertGreater(abs(float(final["prediction"][0, 0] - populations[-2]["prediction"][0, 0])), 100.)

    def test_reduced_evaluation_skips_removed_metrics_and_caps_both_populations(self):
        from common import logging
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            calderon_logging.register(cfg, data)
            cfg.logging.maizels.distribution_eval_source_max_points = 7
            cfg.logging.maizels.distribution_eval_points_per_time = 7
            state = SimpleNamespace(apply_fn=toy_map)
            populations, scores = [], []
            def score(prediction, reference):
                self.assertEqual((len(prediction), len(reference)), (7, 7))
                scores.append((prediction, reference))
                return float(np.linalg.norm(prediction.mean(axis=0) - reference.mean(axis=0)))
            def classify(paths, **kwargs):
                return {"valid": np.ones(len(paths), dtype=bool)}
            with patch("common.logging.wandb.log"), \
                 patch("common.logging._rbf_mmd2_np", side_effect=AssertionError("MMD must not run")), \
                 patch("common.logging._flow_map_population_in_batches", side_effect=AssertionError("Direct evaluation must not run")), \
                 patch("common.logging._one_step_paths", side_effect=AssertionError("Direct trajectories must not run")), \
                 patch("common.logging._mfm_exact_emd", side_effect=score), \
                 patch("common.calderon_logging.check_paths_with_classifier", side_effect=classify):
                distribution = logging._log_maizels_distribution_eval(
                    cfg, state, {}, force=True, population_callback=lambda **p: populations.append(p))
                observed = logging._compute_observed_marginal_emd_metrics(cfg, state, {})
                trajectories = logging._compute_maizels_population_trajectory_metrics(cfg, state, {})
            self.assertEqual(len(scores), 7)
            self.assertEqual(len(populations), 2)
            self.assertEqual([p["source_window"] for p in populations], ["4-6", "8-10"])
            for population in populations:
                self.assertEqual(len(np.unique(population["source_x"][:, 0])), 7)
                self.assertEqual(len(np.unique(population["reference_x"][:, 0])), 7)
            for key in {**distribution, **observed, **trajectories}:
                self.assertFalse(any(removed in key for removed in ("mmd", "direct", "linear")), key)
            self.assertIn("distribution_eval/6-8_euler_emd", distribution)
            self.assertIn("maizels/model_euler_invalid_trajectory_pct", trajectories)

    def test_saved_run_reduction_preserves_training_config_and_honors_cap_override(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg, _, _ = fixture(folder)
            cfg.evaluation.marginal_max_points = 0
            cfg.logging.maizels.distribution_eval_source_max_points = 0
            cfg.logging.maizels.distribution_eval_points_per_time = 0
            for field in ("distribution_eval_samplers", "distribution_eval_metrics", "trajectory_eval_samplers"):
                del cfg.logging.maizels[field]
            cfg.logging.direct_plots_enabled = True
            saved = json.dumps(cfg.to_dict())
            (folder / "config.json").write_text(saved)
            argv = ["calderon_ot_flow_map.py", "--stage", "evaluate", "--output-dir", str(folder), "--reduced-evaluation"]
            for extra, cap in (([], 2048), (["--marginal-cap", "1024"], 1024)):
                with patch.object(sys, "argv", argv + extra):
                    loaded = configure(parse_args())
                self.assertEqual(loaded.evaluation.marginal_max_points, cap)
                self.assertEqual(loaded.logging.maizels.distribution_eval_source_max_points, cap)
                self.assertEqual(loaded.logging.maizels.distribution_eval_points_per_time, cap)
                self.assertEqual(loaded.logging.maizels.distribution_eval_samplers, ["flowmap", "euler"])
                self.assertEqual(loaded.evaluation.source_max_points, 512)
                self.assertEqual((folder / "config.json").read_text(), saved)

    def test_blockwise_mmd_includes_every_pair_and_matches_dense_kernel_means(self):
        from common import logging
        rng = np.random.default_rng(19)
        x, y = rng.normal(size=(23, 5)), rng.normal(size=(31, 5))
        bandwidths = np.array([.5, 1., 2.])
        actual = logging._rbf_kernel_means_blockwise(x, y, bandwidths, block_size=7)
        distances = np.sum((x[:, None] - y[None]) ** 2, axis=-1)
        expected = np.array([np.exp(-distances / (2 * bw * bw)).mean() for bw in bandwidths])
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_disk_backed_w1_cost_preserves_the_float32_euclidean_metric(self):
        from common.wasserstein import _emd_cost_matrix
        rng = np.random.default_rng(31)
        x, y = rng.normal(size=(23, 7)).astype(np.float32), rng.normal(size=(31, 7)).astype(np.float32)
        with _emd_cost_matrix(x, y, max_in_memory_pairs=10000) as dense:
            expected = dense.copy()
        with _emd_cost_matrix(x, y, max_in_memory_pairs=0, row_block_size=5) as disk:
            self.assertIsInstance(disk, np.memmap)
            np.testing.assert_allclose(disk, expected, atol=2e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

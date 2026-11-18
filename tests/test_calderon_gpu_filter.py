"""Check fused filtering against the reference scan and its near-tie fallback."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from common.calderon import interpolant_validity
from common.calderon_filter import JAXInterpolantFilter, get_jax_interpolant_filter


class ToyEnsemble:
    def __init__(self, tie=False):
        import jax
        import jax.numpy as jnp
        self.class_names = ("A", "B", "C")
        self.mean = np.zeros(100, dtype=np.float64)
        self.scale = np.ones(100, dtype=np.float64)
        self.batch_size = 512
        self.spec = {"members": [{"temperature": 1.2}, {"temperature": .8}], "temperature": 1.1}
        self.reference_rows = 0
        self.tie = tie

        def logits(x):
            pair = jnp.rint(x[:, 1]).astype(jnp.int32)
            labels = jnp.where(x[:, 0] > .2, 1, 0)
            labels = jnp.where((pair == 1) & (x[:, 0] > .6), 2, labels)
            labels = jnp.where(pair == 3, jnp.where(x[:, 0] > .6, 0, 1), labels)
            values = 8 * jax.nn.one_hot(labels, 3)
            if tie:
                values = jnp.where((pair == 2)[:, None], jnp.array([3., 3., -4.]), values)
            return jnp.stack([values, values * .8])
        self._logits = jax.jit(logits)

    def predict_proba(self, x):
        import jax
        from scipy.special import softmax
        self.reference_rows += len(x)
        normalized = np.asarray(x, dtype=np.float32).copy()
        normalized -= self.mean
        normalized /= self.scale
        logits = np.asarray(jax.device_get(self._logits(normalized)))
        member = [softmax(value.astype(np.float64) / spec["temperature"], axis=-1).astype(np.float32)
                  for value, spec in zip(logits, self.spec["members"])]
        probabilities = softmax(np.log(np.maximum(np.mean(member, axis=0), 1e-30)).astype(np.float64)
                                / self.spec["temperature"], axis=-1).astype(np.float32)
        if self.tie:
            # Represent a CPU/GPU rounding difference at an ambiguous tie.
            ambiguous = np.rint(x[:, 1]).astype(int) == 2
            probabilities[ambiguous] = [0.499999, 0.500001, 0.]
        return probabilities


def endpoints():
    source = np.zeros((5, 100), dtype=np.float32)
    source[:, 1] = np.arange(5)
    target = source.copy()
    target[:, 0] = 1
    return source, np.array([0, 0, 0, 1, 0]), target, np.array([1, 2, 2, 0, 0])


class CalderonGPUFilterTests(unittest.TestCase):
    def test_full_grid_and_observed_final_label_match_reference(self):
        model = ToyEnsemble()
        allowed = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool)
        checker = JAXInterpolantFilter(model, allowed, 50)
        source, starts, target, ends = endpoints()
        times = np.linspace(0., 1., 52, dtype=np.float32)[1:-1][None, :, None]
        paths = source[:, None] * (1 - times) + target[:, None] * times
        expected = interpolant_validity(starts, ends, model.predict_proba(paths.reshape(-1, 100)).reshape(5, 50, 3), allowed)
        model.reference_rows = 0
        result = checker(source, starts, target, ends)
        for name in ("valid", "confident", "rejected_at_checkpoint", "rejected_at_final"):
            np.testing.assert_array_equal(result[name], expected[name])
        np.testing.assert_array_equal(result["valid"], [True, True, True, False, False])
        self.assertEqual(result["n_points"], 250)
        self.assertEqual(result["n_confident"], 250)
        self.assertEqual(result["reference_fallback_pairs"], 0)
        self.assertEqual(model.reference_rows, 0)

    def test_only_ambiguous_paths_use_reference_predictions(self):
        model = ToyEnsemble(tie=True)
        allowed = np.eye(3, dtype=bool)
        allowed[0, 1] = True
        checker = JAXInterpolantFilter(model, allowed, 50)
        source, starts, target, ends = endpoints()
        ends[2] = 0
        result = checker(source, starts, target, ends)
        self.assertEqual(result["reference_fallback_pairs"], 1)
        self.assertEqual(result["reference_fallback_points"], 50)
        self.assertEqual(model.reference_rows, 50)
        self.assertFalse(result["valid"][2])
        self.assertTrue(result["rejected_at_final"][2])

    def test_confidence_threshold_scan_matches_reference(self):
        model = ToyEnsemble(tie=True)
        allowed = np.eye(3, dtype=bool)
        allowed[0, 1] = True
        source, starts, target, ends = endpoints()
        checker = JAXInterpolantFilter(model, allowed, 7, .7, .2)
        times = np.linspace(0., 1., 9, dtype=np.float32)[1:-1][None, :, None]
        paths = source[:, None] * (1 - times) + target[:, None] * times
        expected = interpolant_validity(starts, ends, model.predict_proba(paths.reshape(-1, 100)).reshape(5, 7, 3), allowed, .7, .2)
        actual = checker(source, starts, target, ends)
        for name in ("valid", "confident", "rejected_at_checkpoint", "rejected_at_final"):
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_intervals_share_compiled_filter_only_for_same_relation(self):
        model = ToyEnsemble()
        allowed = np.eye(3, dtype=bool)
        first = get_jax_interpolant_filter(model, allowed, 50)
        self.assertIs(first, get_jax_interpolant_filter(model, allowed.copy(), 50))
        other = allowed.copy()
        other[0, 1] = True
        self.assertIsNot(first, get_jax_interpolant_filter(model, other, 50))

    def test_invalid_endpoint_inputs_are_rejected(self):
        checker = JAXInterpolantFilter(ToyEnsemble(), np.eye(3, dtype=bool), 50)
        source, starts, target, ends = endpoints()
        source[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite PCA100"):
            checker(source, starts, target, ends)

    def test_batched_intervals_preserve_rng_ot_refinements_and_pairs(self):
        import jax
        from common import calderon
        from test_calderon_three_time_flow import fixture
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
            cfg.problem.n_interpolant_check_times = 50
            cfg.problem.classifier_batch_size = 512
            cfg.problem.classifier_prob_threshold = 0.
            cfg.problem.classifier_margin_threshold = 0.
            cfg.problem.interpolant_filter_backend = "jax"
            outputs, statistics = [], []
            model = ToyEnsemble()
            for batched in (False, True):
                cfg.problem.batch_interpolant_checks_across_intervals = batched
                with patch("common.calderon_classifier.FrozenTissueEnsemble", return_value=model):
                    dataset = calderon.MultiMarginalOTDataset(cfg, data, "train")
                result = dataset.sample_device_batch(16, jax.random.PRNGKey(13))
                outputs.append({name: np.asarray(value) for name, value in result.items()})
                statistics.append(dataset.last_pair_stats)
            for name in outputs[0]:
                np.testing.assert_array_equal(outputs[0][name], outputs[1][name])
            self.assertEqual(statistics[0], statistics[1])
            self.assertGreater(statistics[0]["interpolant_rejected"], 0)
            self.assertGreater(dataset.last_filter_dispatch_stats["dispatches"], 0)
            self.assertIs(dataset.datasets[0].jax_interpolant_filter, dataset.datasets[1].jax_interpolant_filter)

    def test_batched_filter_failure_releases_workers_and_restores_checkers(self):
        import jax
        from common import calderon
        from test_calderon_three_time_flow import fixture
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            cfg.problem.maizels_pair_mode = "ot_endpoint_interpolant"
            cfg.problem.n_interpolant_check_times = 50
            cfg.problem.classifier_batch_size = 512
            cfg.problem.classifier_prob_threshold = 0.
            cfg.problem.classifier_margin_threshold = 0.
            cfg.problem.interpolant_filter_backend = "jax"
            cfg.problem.batch_interpolant_checks_across_intervals = True
            with patch("common.calderon_classifier.FrozenTissueEnsemble", return_value=ToyEnsemble()):
                dataset = calderon.MultiMarginalOTDataset(cfg, data, "train")
            original = dataset.datasets[0].jax_interpolant_filter
            with patch.object(original, "_filter_batch", side_effect=RuntimeError("Deliberate GPU failure")):
                with self.assertRaisesRegex(RuntimeError, "Deliberate GPU failure"):
                    dataset.sample_device_batch(16, jax.random.PRNGKey(13))
            self.assertTrue(all(interval.jax_interpolant_filter is original for interval in dataset.datasets))


if __name__ == "__main__":
    unittest.main()

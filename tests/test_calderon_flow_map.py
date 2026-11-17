"""Checks for population isolation, OT pairing and tissue path semantics."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

from common import calderon
from configs.calderon_pca100 import get_config
from configs.calderon_filtered_pca100 import get_config as filtered_config
from configs.calderon_constrained_pca100 import get_config as constrained_config
from configs.maizels_pca50 import get_config as maizels_config
from launchers.calderon_ot_flow_map import rollout_checkpoints, validate_config


def one_hot_paths(ids, n_classes=3):
    return np.eye(n_classes)[np.asarray(ids)]


def endpoint_fixture_config(factory=get_config, **kwargs):
    """Exercise the legacy endpoint protocol with an explicit endpoint config."""
    cfg = factory(**kwargs)
    del cfg.problem.retained_timepoints
    del cfg.problem.evaluation_timepoints
    cfg.problem.reference_time = "8-10"
    cfg.problem.reference_hours = 9.
    cfg.problem.cohort_branch_prefix = "Ubi_"
    cfg.problem.pair_time_bounds_in_label = False
    cfg.problem.interp_type = "linear"
    cfg.problem.interp_uses_labels = bool(cfg.constraints.enabled)
    return cfg


class CalderonPathTests(unittest.TestCase):
    def setUp(self):
        # A can reach B and C; B and C are sibling terminal states.
        self.allowed = np.eye(3, dtype=bool)
        self.allowed[0, 1:] = True

    def test_whole_path_detects_sibling_switch_and_backtracking(self):
        probabilities = one_hot_paths([[1, 1], [1, 2], [1, 0]])
        result = calderon.score_tissue_paths([0, 0, 0], probabilities, self.allowed)
        np.testing.assert_array_equal(result["first_rejection"], [-1, 1, 1])
        self.assertAlmostEqual(result["metrics"]["violation_pct_all_sources"], 200 / 3)
        # All endpoints are reachable from A; endpoint-only scoring misses this.
        self.assertEqual(result["metrics"]["endpoint_violation_pct_assessed"], 0)

    def test_uncertain_checkpoint_is_skipped_without_changing_current_tissue(self):
        probabilities = np.array([[[.025, .95, .025], [.1, .35, .55], [.025, .95, .025]]])
        raw = calderon.score_tissue_paths([0], probabilities, self.allowed)
        filtered = calderon.score_tissue_paths([0], probabilities, self.allowed, .9)
        self.assertEqual(raw["first_rejection"][0], 1)
        self.assertEqual(filtered["status"][0], "no_detected_violation")
        np.testing.assert_array_equal(filtered["confident"], [[True, False, True]])
        self.assertEqual(filtered["metrics"]["full_checkpoint_coverage_pct"], 0)

    def test_no_confident_checkpoint_is_unassessed_with_null_rate(self):
        result = calderon.score_tissue_paths([0, 1], np.full((2, 4, 3), 1 / 3), self.allowed, .9)
        self.assertTrue((result["status"] == "unassessed").all())
        self.assertEqual(result["metrics"]["n_unassessed"], 2)
        self.assertEqual(result["metrics"]["trajectory_coverage_pct"], 0)
        self.assertIsNone(result["metrics"]["violation_pct_assessed"])
        self.assertIsNone(result["metrics"]["violation_pct_fully_assessed"])

    def test_relation_is_loaded_without_transitive_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowed.csv"
            path.write_text("source,target\nA,A\nB,B\nC,C\nA,B\nB,C\n")
            allowed = calderon.load_allowed_pairs(path, ("A", "B", "C"))
        self.assertFalse(allowed[0, 2])
        result = calderon.score_tissue_paths([0], one_hot_paths([[1, 2]]), allowed)
        self.assertEqual(result["metrics"]["violation_pct_all_sources"], 0)
        self.assertEqual(result["metrics"]["endpoint_violation_pct_assessed"], 100)

    def test_unknown_label_uses_its_defined_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowed.csv"
            path.write_text("source,target\nUnknown,Unknown\nBrain,Brain\n")
            allowed = calderon.load_allowed_pairs(path, ("Unknown", "Brain"))
        result = calderon.score_tissue_paths([0], one_hot_paths([[1]], 2), allowed)
        self.assertEqual(result["metrics"]["violation_pct_all_sources"], 100)

    def test_observed_sequence_matches_maizels_path_checker(self):
        from common import maizels
        classes = ("A", "B", "C")
        probabilities = np.random.default_rng(9).dirichlet([1, 2, 3], size=(20, 4))
        source = np.arange(20) % 3
        result = calderon.score_tissue_paths(source, probabilities, self.allowed, .55, .1)
        reachable = {label: {classes[j] for j in np.flatnonzero(self.allowed[i])} for i, label in enumerate(classes)}
        expected = maizels.path_validity_from_predictions(
            source, np.asarray(classes)[result["predicted_ids"]], result["confidence"], result["margin"],
            classes, reachable, .55, .1,
        )
        np.testing.assert_array_equal(result["first_rejection"] < 0, expected["valid"])
        np.testing.assert_array_equal(result["confident"], expected["confident"])

    def test_training_interpolants_include_the_observed_final_label(self):
        probabilities = one_hot_paths([[1] * 50])
        result = calderon.interpolant_validity([0], [2], probabilities, self.allowed)
        self.assertFalse(result["valid"][0])
        self.assertTrue(result["rejected_at_final"][0])
        self.assertFalse(result["rejected_at_checkpoint"][0])

    def test_training_interpolant_scan_matches_native_maizels(self):
        from common import maizels
        classes = ("A", "B", "C")
        probabilities = np.random.default_rng(14).dirichlet([1, 2, 3], size=(30, 50))
        source, target = np.arange(30) % 3, (np.arange(30) + 1) % 3
        result = calderon.interpolant_validity(source, target, probabilities, self.allowed, .55, .1)
        predicted = probabilities.argmax(-1)
        top = probabilities.max(-1)
        margin = top - np.partition(probabilities, -2, axis=-1)[..., -2]
        reachable = {label: {classes[j] for j in np.flatnonzero(self.allowed[i])} for i, label in enumerate(classes)}
        expected = maizels.path_validity_from_predictions(source, np.asarray(classes)[predicted], top, margin,
                                                        classes, reachable, .55, .1, final_type_ids=target)
        for key in ("valid", "confident", "rejected_at_checkpoint", "rejected_at_final"):
            np.testing.assert_array_equal(result[key], expected[key])


class CalderonPopulationTests(unittest.TestCase):
    def test_endpoint_splits_are_disjoint_complete_and_reproducible(self):
        splits = calderon.split_endpoint_indices(101, .1, .1, 701)
        recreated = calderon.split_endpoint_indices(101, .1, .1, 701)
        self.assertEqual([len(splits[key]) for key in ("train", "validation", "test")], [79, 11, 11])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(splits.values()))), np.arange(101))
        for key in splits:
            np.testing.assert_array_equal(splits[key], recreated[key])

    def test_empty_train_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            calderon.split_endpoint_indices(2, .1, .1, 0)

    def test_loader_preserves_pca_rows_and_excludes_other_components(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "data.h5ad"
            windows = np.array(["4-6"] * 12 + ["8-10"] * 9 + ["12-14"] * 12)
            branches = np.array(["Ubi_A"] * 10 + ["Yolk_A"] * 2 + ["Ubi_AA"] * 8 + ["Blastoderm_A"]
                                + ["Ubi_AAA"] * 11 + ["Yolk_A"])
            labels = np.array(["A", "B", "C"] * 11)
            pca = np.arange(3300, dtype=np.float32).reshape(33, 100)
            strings = h5py.string_dtype("utf-8")
            with h5py.File(path, "w") as handle:
                obs = handle.create_group("obs")
                values = {"day": windows, "branch_id": branches, "broad_tissue": labels,
                          "classifier_split": np.where(windows == "8-10", "test", "train")}
                for name, column in values.items():
                    categories = sorted(set(column))
                    group = obs.create_group(name)
                    group.create_dataset("categories", data=np.asarray(categories, dtype=object), dtype=strings)
                    group.create_dataset("codes", data=[categories.index(value) for value in column])
                obs.create_dataset("cell_id", data=np.asarray([f"cell_{i}" for i in range(33)], dtype=object), dtype=strings)
                handle.create_dataset("obsm/X_pca", data=pca)
                preprocessing = handle.create_group("uns/preprocessing")
                for name, value in (("variant", "heldout_8_10"), ("heldout_window", "8-10")):
                    preprocessing.create_dataset(name, data=value, dtype=strings)
                preprocessing.create_dataset("n_pcs", data=100)
                preprocessing.create_dataset("n_hvgs", data=3000)
            checkpoint = folder / "model.pt"
            checkpoint.write_bytes(b"Only metadata loading is exercised here.")
            manifest = folder / "model.json"
            manifest.write_text(json.dumps({"target": "broad_tissue", "variant": "heldout_8_10", "n_pcs": 100,
                "class_names": ["A", "B", "C"], "scaler_mean": [0.] * 100, "scaler_scale": [1.] * 100,
                "members": [{"checkpoint": "model.pt"}], "selection_sha256": "fixture", "source_h5ad": str(path)}))
            pairs = folder / "allowed.csv"
            pairs.write_text("source,target\nA,A\nB,B\nC,C\nA,B\nA,C\n")
            cfg = endpoint_fixture_config(dataset_location=str(path), classifier_path=str(manifest))
            cfg.classifier.allowed_pairs_path = str(pairs)
            data = calderon.load_endpoint_experiment(cfg)
            self.assertEqual([len(data.marginals[window].x) for window in ("4-6", "8-10", "12-14")], [10, 8, 11])
            for marginal in data.marginals.values():
                np.testing.assert_array_equal(marginal.x, pca[marginal.metadata.pca_row.to_numpy()])
            table = data.split_table()
            self.assertTrue(table.branch_id.str.startswith("Ubi_").all())
            self.assertTrue((table.loc[table.day == "8-10", "flow_split"] == "heldout_reference").all())
            self.assertEqual(set(table.loc[table.flow_split == "train", "day"]), {"4-6", "12-14"})
            self.assertEqual(len(table), table.cell_id.nunique())
            different = folder / "different_pca.h5ad"
            different.write_bytes(path.read_bytes())
            with h5py.File(different, "r+") as handle:
                handle["obsm/X_pca"][0, 0] += 1
            cfg.problem.dataset_location = str(different)
            with self.assertRaisesRegex(ValueError, "PCA export differs"):
                calderon.load_endpoint_experiment(cfg)


class CalderonNativeFlowTests(unittest.TestCase):
    def test_constrained_config_matches_id6_without_conditioning_the_network(self):
        cfg = constrained_config()
        reference = maizels_config(6, hparam_val_times=(), maizels_ot_coupling="minibatch_ot")
        self.assertEqual(cfg.constraints.to_dict(), reference.constraints.to_dict())
        self.assertTrue(cfg.problem.interp_uses_labels)
        self.assertFalse(cfg.training.conditional)
        self.assertEqual(cfg.evaluation.validation_pair_mode, "ot_plain")
        self.assertEqual(cfg.evaluation.validation_objective, "plain_ot_lsd_without_constraint")
        validate_config(cfg)

    def test_loss_labels_do_not_change_filtered_ot_pairs(self):
        import jax
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "allowed.csv"
            pairs.write_text("source,target\nA,A\nA,B\nB,A\nB,B\n")
            x = np.stack([np.arange(4), np.zeros(4)], axis=1).astype(np.float32)
            source = {"x": x, "type_ids": np.arange(4, dtype=np.int32) % 2}
            target = {"x": x + 10, "type_ids": 1 - np.arange(4, dtype=np.int32) % 2}
            def checker(source_x, source_type_ids, target_x, target_type_ids):
                return {"valid": np.ones(len(source_x), dtype=bool)}
            batches = []
            for factory in (filtered_config, constrained_config):
                cfg = endpoint_fixture_config(factory)
                cfg.classifier.allowed_pairs_path = str(pairs)
                cfg.problem.ot_minibatch_size = 4
                ds = calderon.EndpointOTDataset(cfg, source, target, ("A", "B"), interpolant_checker=checker)
                batches.append(ds.sample_device_batch(4, jax.random.PRNGKey(13)))
            for key in ("x0", "x1"):
                np.testing.assert_array_equal(batches[0][key], batches[1][key])
            labels = np.asarray(batches[1]["label"])
            np.testing.assert_array_equal(labels[:, 0], np.asarray(batches[1]["x0"])[:, 0].astype(int) % 2)
            np.testing.assert_array_equal(labels[:, 1], 1 - (np.asarray(batches[1]["x1"])[:, 0].astype(int) - 10) % 2)
            self.assertNotIn("label", batches[0])

    def test_constraint_uses_the_supplied_relation_without_closure(self):
        from unittest.mock import patch
        from common import calderon_lineage
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowed.csv"
            path.write_text("source,target\nA,A\nB,B\nC,C\nA,B\nB,C\n")
            cfg = constrained_config()
            cfg.classifier.allowed_pairs_path = str(path)
            with patch.object(calderon_lineage, "DifferentiableTissueEnsemble", return_value=SimpleNamespace(class_names=("A", "B", "C"))):
                classifier = calderon_lineage.setup_lineage_classifier(cfg)
            self.assertEqual(float(classifier["invalid_transition"][0, 2]), 1.)
            self.assertEqual(float(classifier["invalid_transition"][0, 1]), 0.)
            np.testing.assert_array_equal(classifier["canonical_to_classifier"], [0, 1, 2])

    def test_early_rejection_preserves_interpolant_and_final_label_decisions(self):
        class Decoder:
            def predict_proba(self, x):
                pair = np.rint(x[:, 1]).astype(int)
                ids = np.where(x[:, 0] > .2, 1, 0)
                ids[(pair == 1) & (x[:, 0] > .6)] = 2
                ids[pair == 3] = np.where(x[pair == 3, 0] > .6, 0, 1)
                return np.eye(3, dtype=np.float32)[ids]
        source = np.zeros((4, 100), dtype=np.float32)
        source[:, 1] = np.arange(4)
        target = source.copy()
        target[:, 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "allowed.csv"
            path.write_text("source,target\nA,A\nB,B\nC,C\nA,B\nA,C\n")
            outputs, counts = [], []
            for chunk in (50, 10):
                cfg = constrained_config()
                cfg.problem.interpolant_check_chunk_size = chunk
                cfg.classifier.allowed_pairs_path = str(path)
                ds = calderon.EndpointOTDataset(cfg, {}, {}, ("A", "B", "C"), interpolant_checker=lambda **kwargs: None)
                ds.classifier = Decoder()
                outputs.append(ds.check_interpolants(source, np.array([0, 0, 0, 1]), target, np.array([1, 2, 2, 0])))
                counts.append(ds.interpolant_classifier_points)
            for key in ("valid", "rejected_at_checkpoint", "rejected_at_final"):
                np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
            np.testing.assert_array_equal(outputs[1]["valid"], [True, False, False, False])
            self.assertTrue(outputs[1]["confident"][0].all())
            self.assertLess(counts[1], counts[0])

    def test_filtered_config_matches_id5_and_preserves_shared_validation(self):
        cfg = filtered_config()
        reference = maizels_config(5, hparam_val_times=(), maizels_ot_coupling="minibatch_ot")
        for name in ("maizels_pair_mode", "n_interpolant_check_times", "classifier_prob_threshold", "classifier_margin_threshold",
                     "ot_minibatch_infeasible_fallback"):
            self.assertEqual(cfg.problem[name], reference.problem[name])
        self.assertEqual(cfg.evaluation.validation_pair_mode, "ot_plain")
        self.assertEqual(len(cfg.evaluation.checkpoint_times), 50)
        self.assertFalse(cfg.constraints.enabled)
        self.assertEqual(cfg.network.n_neurons, get_config().network.n_neurons)
        validate_config(cfg)

    def test_config_retains_maizels_ot_flow_map_recipe(self):
        cfg = get_config()
        base = maizels_config(7, hparam_val_times=(), maizels_ot_coupling="minibatch_ot")
        for name in ("loss_type", "stopgrad_type", "psd_type", "ema_facs"):
            self.assertEqual(cfg.training[name], base.training[name])
        for name in ("diag_fraction", "learning_rate", "bs", "clip"):
            self.assertEqual(cfg.optimization[name], base.optimization[name])
        self.assertEqual(cfg.network.n_hidden, base.network.n_hidden)
        self.assertEqual(cfg.network.n_neurons, base.network.n_neurons)
        self.assertEqual(cfg.network.output_dim, 100)
        self.assertFalse(cfg.training.conditional)
        self.assertFalse(cfg.constraints.enabled)
        validate_config(cfg)

    def test_plain_ot_pairing_is_independent_of_tissue_labels(self):
        import jax
        cfg = endpoint_fixture_config()
        cfg.problem.ot_minibatch_size = 4
        source = {"x": np.array([[0, 0], [2, 3], [4, 1], [5, 7]], dtype=np.float32),
                  "type_ids": np.array([0, 0, 0, 0], dtype=np.int32)}
        target = {"x": source["x"] + 1, "type_ids": np.array([1, 1, 1, 1], dtype=np.int32)}
        first = calderon.EndpointOTDataset(cfg, source, target, ("A", "B", "C"))
        second = calderon.EndpointOTDataset(cfg, {**source, "type_ids": np.array([2, 1, 0, 2], dtype=np.int32)},
                                             {**target, "type_ids": np.array([0, 2, 1, 0], dtype=np.int32)}, ("A", "B", "C"))
        key = jax.random.PRNGKey(19)
        a, b = first.sample_device_batch(4, key), second.sample_device_batch(4, key)
        self.assertEqual(set(a), {"x0", "x1"})
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertEqual(first.last_pair_stats["endpoint_rejected"], 0)

    def test_filtered_ot_reassigns_after_invalid_interpolants(self):
        import jax
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "allowed.csv"
            pairs.write_text("source,target\nA,A\nB,B\n")
            cfg = filtered_config()
            cfg.classifier.allowed_pairs_path = str(pairs)
            cfg.problem.ot_minibatch_size = 4
            x = np.stack([np.arange(4), np.zeros(4)], axis=1).astype(np.float32)
            pool = {"x": x, "type_ids": np.zeros(4, dtype=np.int32)}
            def checker(source_x, source_type_ids, target_x, target_type_ids):
                return {"valid": np.linalg.norm(source_x - target_x, axis=1) > .5}
            ds = calderon.EndpointOTDataset(cfg, pool, pool, ("A", "B"), interpolant_checker=checker)
            paired = ds.sample_device_batch(4, jax.random.PRNGKey(22))
            self.assertTrue((np.linalg.norm(paired["x0"] - paired["x1"], axis=1) > .5).all())
            self.assertGreaterEqual(ds.last_pair_stats["ot_refinements"], 2)
            self.assertGreater(ds.last_pair_stats["interpolant_rejected"], 0)
            self.assertEqual(ds.pair_filter_summary()["sampled_pairs"], 4)

    def test_explicit_ot_mask_does_not_receive_another_closure(self):
        from common import cite_multi
        cfg = get_config()
        cfg.problem.ot_minibatch_size = 4
        cfg.problem.ot_minibatch_max_resamples = 2
        allowed = np.eye(3, dtype=bool)
        allowed[0, 1] = allowed[1, 2] = True
        with self.assertRaisesRegex(RuntimeError, "valid candidate block"):
            cite_multi._make_minibatch_ot_pairs_from_arrays(
                cfg, np.zeros((4, 2)), np.zeros(4, dtype=np.int32), np.ones((4, 2)), np.full(4, 2, dtype=np.int32),
                n_pairs=4, rng=np.random.default_rng(5), pair_mode="ot_endpoint", class_names=("A", "B", "C"),
                transition_edges=(("A", "B"), ("B", "C")), allowed_class_mask=allowed,
            )

    def test_composed_rollout_keeps_absolute_time_and_partial_batches(self):
        import jax.numpy as jnp
        def translation(params, s, t, x, **kwargs):
            return x + (t - s) * jnp.array([1., -2.])
        source = np.arange(10, dtype=np.float32).reshape(5, 2)
        evaluation = SimpleNamespace(flowmap_n_steps=8, flow_batch_size=4, checkpoint_times=[.25, .5, .75, 1.])
        paths = rollout_checkpoints(translation, {}, source, evaluation)
        expected = source[:, None] + np.array(evaluation.checkpoint_times)[None, :, None] * np.array([1., -2.])
        np.testing.assert_allclose(paths, expected, atol=1e-6)

    def test_fifty_checks_keep_the_same_integrated_path(self):
        import jax.numpy as jnp
        def drift(params, s, t, x, **kwargs):
            return x + (t - s) * (s * x + jnp.array([1., -2.]))
        source = np.arange(10, dtype=np.float32).reshape(5, 2)
        dense = SimpleNamespace(flowmap_n_steps=100, flow_batch_size=4,
                                checkpoint_times=(np.arange(1, 51) / 50).tolist())
        sparse = SimpleNamespace(flowmap_n_steps=100, flow_batch_size=4, checkpoint_times=[.25, .5, .75, 1.])
        dense_paths = rollout_checkpoints(drift, {}, source, dense)
        sparse_paths = rollout_checkpoints(drift, {}, source, sparse)
        self.assertEqual(dense_paths.shape, (5, 50, 2))
        np.testing.assert_array_equal(dense_paths[:, [24, 49]], sparse_paths[:, [1, 3]])
        expected = source.copy()
        for index in range(100):
            expected += .01 * ((index / 100) * expected + np.array([1., -2.], dtype=np.float32))
        np.testing.assert_allclose(dense_paths[:, -1], expected, atol=1e-5)

    def test_dense_checkpoint_grid_and_invalid_grids(self):
        cfg = get_config()
        cfg.evaluation.checkpoint_times = (np.arange(1, 51) / 50).tolist()
        validate_config(cfg)
        for times in ([.25, .75, 1.], [0., .5, 1.], [.5, .5, 1.], [.5, .9], [.5, float("nan"), 1.], [.015, .5, 1.]):
            cfg.evaluation.checkpoint_times = list(times)
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate_config(cfg)


if __name__ == "__main__":
    unittest.main()

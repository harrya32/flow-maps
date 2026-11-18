"""Three-window flow isolation, interval clocks and distinct classifier roles."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT))

from common import calderon
from configs.calderon_pca100 import get_config
from configs.calderon_filtered_pca100 import get_config as filtered_config
from configs.calderon_constrained_pca100 import get_config as constrained_config
from configs.calderon_constrained_w2000_pca100 import get_config as constrained2000_config
from launchers.calderon_ot_flow_map import check_classifier_python, validate_config
from scripts.evaluate_calderon_tissue_paths import validate_input_vocabulary
from scripts.audit_calderon_endpoint_support import audit


def fixture(folder):
    windows = ("2-4", "4-6", "6-8", "8-10", "10-12", "12-14")
    records, coordinates = [], []
    for wi, window in enumerate(windows):
        for j in range(30):
            tissue = "Blastoderm" if wi == 0 else ("A", "B", "C")[j % 3]
            fold = ("train" if j < 24 else "validation" if j < 27 else "test")
            if window in ("6-8", "10-12"):
                fold = "intermediate_holdout"
            records.append({"cell_id": f"cell_{wi}_{j}", "day": window,
                            "branch_id": "Yolk_A" if tissue == "C" else "Ubi_A",
                            "broad_tissue": tissue, "classifier_split": fold,
                            "source_pca_row": len(records)})
            x = np.arange(100, dtype=np.float32) * .01
            x[0] = wi * 100 + j
            coordinates.append(x)
    frame = pd.DataFrame(records)
    pca = np.stack(coordinates)
    dataset = folder / "all_times.h5ad"
    strings = h5py.string_dtype("utf-8")
    with h5py.File(dataset, "w") as handle:
        obs = handle.create_group("obs")
        for name in ("cell_id", "day", "branch_id", "broad_tissue"):
            obs.create_dataset(name, data=frame[name].to_numpy(dtype=object), dtype=strings)
        obs.create_dataset("classifier_split", data=np.asarray(["train"] * len(frame), dtype=object), dtype=strings)
        handle.create_dataset("obsm/X_pca", data=pca)
        preprocessing = handle.create_group("uns/preprocessing")
        for name, value in (("variant", "all_times"), ("hvg_fit", "all_cells"), ("pca_fit", "all_cells")):
            preprocessing.create_dataset(name, data=value, dtype=strings)
        preprocessing.create_dataset("n_pcs", data=100)
        preprocessing.create_dataset("n_hvgs", data=3000)
    selected = frame.loc[frame.day.isin(("4-6", "8-10", "12-14"))]
    cache = folder / "selected.h5ad"
    with h5py.File(cache, "w") as handle:
        obs = handle.create_group("obs")
        for name in ("cell_id", "classifier_split"):
            obs.create_dataset(name, data=selected[name].to_numpy(dtype=object), dtype=strings)
    (folder / "model.pt").write_bytes(b"Metadata-only classifier fixture.")
    common = {"target": "broad_tissue", "n_pcs": 100, "scaler_mean": [0.] * 100,
              "scaler_scale": [1.] * 100, "members": [{"checkpoint": "model.pt", "temperature": 1.}],
              "temperature": 1., "selection_sha256": "fixture"}
    training = folder / "training.json"
    training.write_text(json.dumps({**common, "variant": "symmetric_4_8_12", "class_names": ["A", "B", "C"],
        "training_windows": ["4-6", "8-10", "12-14"], "intermediate_holdout_windows": ["6-8", "10-12"],
        "source_h5ad": str(cache), "source_h5ad_sha256": calderon.sha256_file(cache),
        "pca_representation_source_sha256": calderon.sha256_file(dataset)}))
    evaluation = folder / "evaluation.json"
    evaluation.write_text(json.dumps({**common, "variant": "all_times",
        "class_names": ["C", "Blastoderm", "A", "B"], "source_h5ad": str(dataset),
        "source_h5ad_sha256": calderon.sha256_file(dataset)}))
    table = folder / "splits.csv"
    frame.loc[frame.day.ne("2-4")].to_csv(table, index=False)
    allowed = folder / "allowed.csv"
    allowed.write_text("source,target\nA,A\nB,B\nC,C\nBlastoderm,Blastoderm\nA,B\nB,C\nBlastoderm,A\n")
    cfg = get_config(dataset_location=str(dataset), classifier_path=str(training),
                     evaluation_classifier_path=str(evaluation))
    cfg.problem.split_manifest_path = str(table)
    cfg.classifier.allowed_pairs_path = str(allowed)
    cfg.problem.ot_minibatch_size = 8
    cfg.training.ndevices = 1
    return cfg, frame, pca


class CalderonThreeTimeFlowTests(unittest.TestCase):
    def test_all_variants_share_selected_schedule_and_classifier_roles(self):
        for factory in (get_config, filtered_config, constrained_config, constrained2000_config):
            cfg = factory()
            cfg.training.ndevices = 1
            self.assertEqual(list(cfg.problem.retained_timepoints), ["4-6", "8-10", "12-14"])
            self.assertEqual(list(cfg.problem.evaluation_timepoints), ["6-8", "10-12"])
            self.assertEqual(cfg.problem.dataset_location, str(calderon.DEFAULT_DATASET))
            self.assertEqual(cfg.classifier.manifest_path, str(calderon.DEFAULT_TRAINING_MANIFEST))
            self.assertEqual(cfg.classifier.evaluation_manifest_path, str(calderon.DEFAULT_EVALUATION_MANIFEST))
            self.assertEqual(cfg.classifier.jax_export_path, str(calderon.DEFAULT_TRAINING_EXPORT))
            self.assertEqual(len(cfg.evaluation.checkpoint_times), 50)
            self.assertFalse(cfg.training.conditional)
            self.assertTrue(cfg.problem.pair_time_bounds_in_label)
            self.assertNotIn("calderon_ubi_", cfg.logging.output_folder)
            validate_config(cfg)
        self.assertEqual(constrained2000_config().constraints.weight, 2000.)

    def test_loader_uses_independent_holdouts_and_excludes_outside_times(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, expected, pca = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            table = data.split_table()
            self.assertEqual(len(table), 150)
            self.assertIn("Yolk_A", set(table.branch_id))
            self.assertEqual(set(table.loc[table.flow_split.eq("train"), "day"]), {"4-6", "8-10", "12-14"})
            middle = table.day.isin(("6-8", "10-12"))
            self.assertTrue(table.loc[middle, "flow_split"].eq("heldout_reference").all())
            self.assertTrue((table.loc[~middle, "flow_split"] != table.loc[~middle, "classifier_split"]).any())
            for window in data.training_times:
                self.assertEqual(set(data.splits[window]), {"train", "validation"})
                self.assertEqual([len(data.splits[window][name]) for name in ("train", "validation")], [27, 3])
                combined = np.concatenate(list(data.splits[window].values()))
                np.testing.assert_array_equal(np.sort(combined), np.arange(30))
            recreated = calderon.load_endpoint_experiment(cfg)
            for window in data.training_times:
                np.testing.assert_array_equal(data.splits[window]["validation"], recreated.splits[window]["validation"])
            for marginal in data.marginals.values():
                np.testing.assert_array_equal(marginal.x, pca[marginal.metadata.pca_row.to_numpy()])
            self.assertEqual(data.provenance["evaluation_classifier_manifest"], cfg.classifier.evaluation_manifest_path)
            self.assertNotEqual(data.provenance["classifier_manifest"], data.provenance["evaluation_classifier_manifest"])
            populations = list(data.evaluation_populations())
            self.assertEqual([window for _, window, _, _ in populations], ["6-8", "8-10", "10-12", "12-14"])
            self.assertEqual([len(indices) for _, _, indices, _ in populations], [30, 30, 30, 30])

    def test_intermediate_training_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            table = pd.read_csv(cfg.problem.split_manifest_path)
            table.loc[table.day.eq("6-8"), "classifier_split"] = "train"
            table.to_csv(cfg.problem.split_manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "Intermediate cells"):
                calderon.load_endpoint_experiment(cfg)

    def test_folds_cannot_diverge_from_the_fitted_classifier(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            table = pd.read_csv(cfg.problem.split_manifest_path)
            indices = table.index[table.day.eq("4-6") & table.classifier_split.eq("validation")]
            table.loc[indices, "classifier_split"] = "train"
            table.to_csv(cfg.problem.split_manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "must match.*classifier"):
                calderon.load_endpoint_experiment(cfg)

    def test_pca_modification_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            with h5py.File(cfg.problem.dataset_location, "r+") as handle:
                handle["obsm/X_pca"][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "same unchanged all-times PCA"):
                calderon.load_endpoint_experiment(cfg)

    def test_adjacent_pairs_carry_absolute_times_and_use_the_middle_twice(self):
        import jax
        import jax.numpy as jnp
        from common.interpolant import TimeRescaledLinearInterpolant
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            data = calderon.load_endpoint_experiment(cfg)
            ds = calderon.MultiMarginalOTDataset(cfg, data, "train")
            batch = ds.sample_device_batch(16, jax.random.PRNGKey(7))
            labels = np.asarray(batch["label"])
            bounds, counts = np.unique(labels[:, 2:4], axis=0, return_counts=True)
            np.testing.assert_array_equal(bounds, [[0, .5], [.5, 1]])
            np.testing.assert_array_equal(counts, [8, 8])
            early = labels[:, 2] == 0
            x0, x1 = np.asarray(batch["x0"]), np.asarray(batch["x1"])
            self.assertTrue(((x0[early, 0] >= 100) & (x0[early, 0] < 200)).all())
            self.assertTrue(((x1[early, 0] >= 300) & (x1[early, 0] < 400)).all())
            self.assertTrue(((x0[~early, 0] >= 300) & (x0[~early, 0] < 400)).all())
            self.assertTrue(((x1[~early, 0] >= 500) & (x1[~early, 0] < 600)).all())
            interp = TimeRescaledLinearInterpolant()
            for i in (np.flatnonzero(early)[0], np.flatnonzero(~early)[0]):
                label = jnp.asarray(labels[i])
                a, b = batch["x0"][i], batch["x1"][i]
                np.testing.assert_array_equal(interp.calc_It(label[2], a, b, label), a)
                np.testing.assert_array_equal(interp.calc_It(label[3], a, b, label), b)
                np.testing.assert_allclose(interp.calc_It_dot(label[2], a, b, label), 2 * (b - a))

    def test_training_graph_restriction_adds_no_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, _, _ = fixture(Path(directory))
            allowed = calderon.load_allowed_pairs(cfg.classifier.allowed_pairs_path, ("A", "B", "C"), allow_extra_classes=True)
            self.assertTrue(allowed[0, 1] and allowed[1, 2])
            self.assertFalse(allowed[0, 2])

    def test_support_audit_checks_adjacent_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg, frame, _ = fixture(folder)
            frame["flow_split"] = frame.classifier_split
            for window, tissue in (("4-6", "A"), ("8-10", "B"), ("12-14", "C")):
                frame.loc[frame.day.eq(window), "broad_tissue"] = tissue
            frame.to_csv(folder / "cell_splits.csv.gz", index=False)
            (folder / "config.json").write_text(json.dumps(cfg.to_dict()))
            result = audit(folder, n_blocks=3)
            self.assertEqual([(row["source_window"], row["target_window"]) for row in result["intervals"]],
                             [("4-6", "8-10"), ("8-10", "12-14")])
            self.assertEqual([row["balanced_maximum_supported_mass"] for row in result["intervals"]], [1., 1.])

    def test_violation_worker_receives_only_the_evaluation_manifest(self):
        cfg = filtered_config()
        with patch("launchers.calderon_ot_flow_map.subprocess.run") as run:
            check_classifier_python(cfg)
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--manifest") + 1], cfg.classifier.evaluation_manifest_path)
        self.assertNotIn(cfg.classifier.manifest_path, args)

    def test_oracle_vocabulary_accepts_extra_and_reordered_classes(self):
        data = {"class_names": np.array(["A", "B", "C"]), "source_tissues": np.array(["B"]),
                "reference_tissues": np.array(["C"]), "target_test_tissues": np.array(["A"])}
        validate_input_vocabulary(data, ("C", "Blastoderm", "A", "B"))
        with self.assertRaisesRegex(ValueError, "cover all saved"):
            validate_input_vocabulary(data, ("A", "B"))

    def test_fifty_step_w1_queries_need_no_quarter_snapshot_grid(self):
        cfg = get_config()
        cfg.training.ndevices = 1
        cfg.evaluation.flowmap_n_steps = 50
        validate_config(cfg)
        cfg.evaluation.protocol = "legacy"
        cfg.evaluation.validation_pair_mode = "ot_plain"
        cfg.evaluation.validation_objective = "plain_ot_lsd_without_constraint"
        cfg.evaluation.ema_factor = .999
        with self.assertRaisesRegex(ValueError, "W1 marginals"):
            validate_config(cfg)

    def test_saved_endpoint_config_retains_its_reference_classifier(self):
        cfg = SimpleNamespace(classifier={"manifest_path": "old.json"})
        cfg.classifier = SimpleNamespace(manifest_path="old.json", get=lambda key, fallback: fallback)
        self.assertEqual(calderon.evaluation_manifest_path(cfg), "old.json")


if __name__ == "__main__":
    unittest.main()

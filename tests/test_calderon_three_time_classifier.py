"""Guard cell holdouts and complete temporal holdouts for the constraint model."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from scripts.prepare_calderon_three_time_data import (
    TRAINING_WINDOWS, INTERMEDIATE_WINDOWS, SPLITS, split_selected_times,
)
from scripts.train_calderon_three_time_classifier import evaluation_indices, evaluate


def example():
    rows = []
    for window in ("0-2", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16"):
        for branch, tissue, count in (("a", "A", 100), ("b", "B", 200), ("c", "B", 300)):
            rows.extend({"day": window, "branch_id": branch, "broad_tissue": tissue}
                        for _ in range(count))
    obs = pd.DataFrame(rows)
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))], name="cell_id")
    return obs


class ThreeTimeClassifierTests(unittest.TestCase):
    def test_selected_time_split_preserves_proportions_without_temporal_leakage(self):
        obs = example()
        assignment, selected = split_selected_times(obs, seed=0)
        repeated, repeated_rows = split_selected_times(obs, seed=0)
        np.testing.assert_array_equal(assignment, repeated)
        np.testing.assert_array_equal(selected, repeated_rows)
        self.assertEqual({name: int((assignment == name).sum()) for name in SPLITS},
                         {"train": 1440, "validation": 180, "test": 180})
        self.assertTrue(np.all(assignment[obs.day.isin(INTERMEDIATE_WINDOWS)] == "intermediate_holdout"))
        self.assertTrue(np.all(assignment[obs.day.isin(("0-2", "14-16"))] == "outside_interval"))
        for name in SPLITS:
            portion = obs.loc[assignment == name]
            self.assertEqual(set(portion.day), set(TRAINING_WINDOWS))
            for window in TRAINING_WINDOWS:
                whole = obs.loc[obs.day.eq(window)]
                part = portion.loc[portion.day.eq(window)]
                np.testing.assert_allclose(part.broad_tissue.value_counts(normalize=True).sort_index(),
                                           whole.broad_tissue.value_counts(normalize=True).sort_index(), atol=1e-12)
        table = obs.loc[assignment != "outside_interval"].reset_index()
        table["source_pca_row"] = np.flatnonzero(assignment != "outside_interval")
        table["classifier_split"] = assignment[assignment != "outside_interval"]
        populations = evaluation_indices(table)
        self.assertEqual(len(populations["intermediate_pooled"]), 1200)
        self.assertEqual(len(populations["intermediate_6_8"]), 600)
        self.assertEqual(len(populations["intermediate_10_12"]), 600)
        self.assertEqual(len(populations["selected_time_test"]), 180)

    def test_rejects_unseen_intermediate_tissue(self):
        obs = example()
        obs.loc[obs.day.eq("6-8"), "broad_tissue"] = "only_intermediate"
        with self.assertRaisesRegex(ValueError, "do not cover intermediate tissues"):
            split_selected_times(obs)

    def test_rejects_temporal_holdout_assigned_to_fitting(self):
        obs = example()
        assignment, _ = split_selected_times(obs)
        table = obs.loc[assignment != "outside_interval"].reset_index()
        table["source_pca_row"] = np.flatnonzero(assignment != "outside_interval")
        table["classifier_split"] = assignment[assignment != "outside_interval"]
        table.loc[table.day.eq("6-8").idxmax(), "classifier_split"] = "train"
        with self.assertRaisesRegex(ValueError, "entered a selected-time training split"):
            evaluation_indices(table)

    def test_requires_frozen_selection_before_evaluation(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Freeze selected-time validation choices"):
                evaluate(SimpleNamespace(output_dir=Path(directory)))


if __name__ == "__main__":
    unittest.main()

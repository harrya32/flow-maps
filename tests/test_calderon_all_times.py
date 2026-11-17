"""All-cell feature fitting and proportional, label-disjoint evaluation."""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from scripts.prepare_calderon_all_times_data import split_diagnostics
from scripts.prepare_calderon_lineage_data import fit_scaled_pca, split_cells


class AllTimeAtlasTests(unittest.TestCase):
    def test_branch_stratification_preserves_both_label_proportions(self):
        rows = []
        for branch, tissue, count in (("a", "x", 1000), ("b", "x", 500), ("c", "y", 200), ("d", "z", 100)):
            rows.extend({"branch_id": branch, "broad_tissue": tissue, "day": f"{(i%9)*2}-{(i%9)*2+2}"} for i in range(count))
        obs = pd.DataFrame(rows)
        split = split_cells(obs, "random", 0)
        proportions, audit = split_diagnostics(obs, split)
        self.assertEqual(audit["split_sizes"], {"train": 1440, "validation": 180, "test": 180})
        for target in ("branch_id", "broad_tissue"):
            frame = proportions.loc[proportions.target.eq(target)]
            np.testing.assert_allclose(frame.validation_pct, frame.dataset_pct, atol=1e-12)
            np.testing.assert_allclose(frame.test_pct, frame.dataset_pct, atol=1e-12)
        self.assertTrue(audit["all_classes_and_time_windows_present_in_every_split"])
        self.assertTrue(np.any((obs.day == "8-10") & (split == "train")))
        self.assertTrue(np.any((obs.day == "8-10") & (split == "test")))

    def test_all_cell_pca_matches_dense_fit_and_uses_all_profiles(self):
        rng = np.random.default_rng(44)
        expression = np.log1p(rng.poisson(np.linspace(.1, 4, 25), size=(300, 25))).astype(np.float32)
        mask = np.ones(len(expression), dtype=bool)
        scores, mean, scale, components, variance = fit_scaled_pca(sparse.csr_matrix(expression), mask, 7)
        reference = PCA(n_components=7, svd_solver="full").fit((expression.astype(np.float64)-mean)/scale)
        np.testing.assert_allclose(variance, reference.explained_variance_, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(components.T@components, reference.components_.T@reference.components_, atol=1e-9)
        np.testing.assert_allclose(scores, ((expression-mean)/scale)@components.T, rtol=1e-4, atol=2e-6)
        modified = expression.copy()
        modified[-30:, :5] += 7
        _, changed_mean, _, _, _ = fit_scaled_pca(sparse.csr_matrix(modified), mask, 7)
        self.assertGreater(np.max(abs(changed_mean-mean)), .5)

    def test_split_audit_rejects_inconsistent_branch_decoder(self):
        obs = pd.DataFrame({"branch_id": ["a"]*6, "broad_tissue": ["x"]*3+["y"]*3, "day": ["8-10"]*6})
        with self.assertRaisesRegex(ValueError, "exactly one tissue"):
            split_diagnostics(obs, np.array(["train", "validation", "test"]*2))


if __name__ == "__main__":
    unittest.main()

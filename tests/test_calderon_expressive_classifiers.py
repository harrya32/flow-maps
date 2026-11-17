import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_calderon_expressive_classifiers import (
    ExpressiveClassifier, fit_temperature, greedy_ensemble, hierarchy_js,
    load_checkpoint_model, quick_metrics, scaled_probabilities,
)


class ExpressiveClassifierTests(unittest.TestCase):
    def test_hierarchy_loss_respects_branch_to_tissue_marginals(self):
        branch = torch.tensor([[0.2, -0.3, 0.7], [0.4, 0.9, -0.2]], requires_grad=True)
        mapping = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        marginal = torch.softmax(branch, dim=1) @ mapping
        matching_tissue_logits = marginal.detach().log().requires_grad_()
        matched = hierarchy_js(branch, matching_tissue_logits, mapping)
        self.assertLess(abs(matched.item()), 1e-7)
        mismatch = torch.tensor([[-3., 3.], [3., -3.]], requires_grad=True)
        loss = hierarchy_js(branch, mismatch, mapping)
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(branch.grad).all())
        self.assertTrue(torch.isfinite(mismatch.grad).all())

    def test_both_network_families_reload_two_heads(self):
        torch.manual_seed(9)
        x = torch.randn(8, 100)
        for family in ("wide", "residual"):
            model = ExpressiveClassifier(family, .1, {"branch_id": 5, "broad_tissue": 3}).eval()
            with torch.no_grad():
                original = model(x)
            metadata = dict(recipe_config=dict(family=family, dropout=.1),
                            class_sizes={"branch_id": 5, "broad_tissue": 3})
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "checkpoint.pt"
                torch.save(dict(model_state_dict=model.state_dict(), metadata=metadata), path)
                reloaded, _ = load_checkpoint_model(path)
                with torch.no_grad():
                    restored = reloaded(x)
                for target, size in (("branch_id", 5), ("broad_tissue", 3)):
                    self.assertEqual(restored[target].shape, (8, size))
                    torch.testing.assert_close(original[target], restored[target])

    def test_macro_f1_uses_only_present_true_classes(self):
        truth = np.array([0, 0, 1, 1, 1])
        logits = np.array([[4, 0, 0], [0, 3, 0], [0, 4, 0], [0, 0, 5], [0, 4, 0]], dtype=np.float32)
        accuracy, macro_f1, loss = quick_metrics(truth, logits)
        self.assertAlmostEqual(accuracy, .6)
        self.assertAlmostEqual(macro_f1, f1_score(truth, logits.argmax(1), labels=[0, 1], average="macro"))
        self.assertGreater(loss, 0)

    def test_temperature_reduces_overconfidence_without_changing_classes(self):
        truth = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        logits = np.array([[4, -1]] * 4 + [[-1, 4]] * 3 + [[4, -1]], dtype=np.float32)
        temperature = fit_temperature(logits, truth)
        probabilities = scaled_probabilities(logits, temperature)
        np.testing.assert_array_equal(probabilities.argmax(1), logits.argmax(1))
        np.testing.assert_allclose(probabilities.sum(1), 1, rtol=1e-6)
        self.assertLess(quick_metrics(truth, logits / temperature)[2], quick_metrics(truth, logits)[2])

    def test_greedy_ensemble_uses_complementary_validation_errors(self):
        truth = np.array([0, 1, 0, 1])
        probabilities = {
            "a": np.array([[.8, .2], [.2, .8], [.8, .2], [.55, .45]], dtype=np.float32),
            "b": np.array([[.8, .2], [.55, .45], [.8, .2], [.2, .8]], dtype=np.float32),
            "weak": np.tile([.99, .01], (4, 1)).astype(np.float32),
        }
        members, prediction, measured, _ = greedy_ensemble(probabilities, truth, np.array([2, 2]))
        self.assertEqual(set(members), {"a", "b"})
        np.testing.assert_array_equal(prediction.argmax(1), truth)
        self.assertEqual(measured["macro_f1"], 1)


if __name__ == "__main__":
    unittest.main()

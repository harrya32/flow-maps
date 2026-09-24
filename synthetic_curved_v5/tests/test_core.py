from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from synthetic_curved_v5.config import PACKAGE_ROOT, load_config
from synthetic_curved_v5.models import PairBank, endpoint_lineage_nll, valid_transition_matrix
from synthetic_curved_v5.process_study import build_profile
from synthetic_curved_v5.training import real_clift_horizon_max


class ConfigurationTests(unittest.TestCase):
    def test_dotted_override(self):
        config = load_config(
            PACKAGE_ROOT / "configs" / "smoke.json",
            ["dataset.true_diffusion.first_diffusion_peak=0.35", "seeds=[7]"],
        )
        self.assertEqual(config["dataset"]["true_diffusion"]["first_diffusion_peak"], 0.35)
        self.assertEqual(config["seeds"], [7])

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(KeyError):
            load_config(PACKAGE_ROOT / "configs" / "smoke.json", ["dataset.not_a_parameter=1"])

    def test_real_clift_horizon_curriculum(self):
        settings = load_config(PACKAGE_ROOT / "configs" / "smoke.json")["real_clift"]
        self.assertAlmostEqual(
            real_clift_horizon_max(settings, 0),
            settings["local_step_fraction"],
        )
        self.assertAlmostEqual(
            real_clift_horizon_max(settings, settings["steps"]),
            settings["max_horizon_fraction"],
        )


class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from synthetic_curved_v5 import dataset, diffusion
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error))
        cls.dataset_module = dataset
        cls.diffusion_module = diffusion
        cls.config = load_config(PACKAGE_ROOT / "configs" / "smoke.json")

    def test_paper_endpoint_geometry(self):
        means, derivatives = self.dataset_module.means_and_derivatives(
            np.asarray([0.0, 2.0]), self.config["dataset"]
        )
        self.assertEqual(means.shape, (2, 3, 2))
        np.testing.assert_allclose(means[0, :, 0], 0.2, atol=1e-12)
        np.testing.assert_allclose(means[1, :, 0], 2.2, atol=1e-12)
        np.testing.assert_allclose(means[1, :, 1], [-0.41, 0.22, 0.92], atol=1e-12)
        self.assertTrue(np.isfinite(derivatives).all())

    def test_standard_deviation_schedule_telescopes(self):
        standard_deviation, _ = self.dataset_module.std_and_derivative(
            np.asarray([0.0, 2.0]), self.config["dataset"]
        )
        np.testing.assert_allclose(standard_deviation[0], np.asarray([0.29, 0.29]) * 0.2)
        np.testing.assert_allclose(standard_deviation[1], np.asarray([0.37, 0.45]) * 0.2)

    def test_optional_branch_specific_progress_offsets(self):
        dataset = copy.deepcopy(self.config["dataset"])
        baseline, _ = self.dataset_module.means_and_derivatives(
            np.asarray([0.0, 2.0]), dataset
        )
        dataset["ramps"]["x_shift"] = [1.0, 2.0]
        dataset["progress_coefficients"] = {
            "lower": {"x_shift": 1.0},
            "upper_down": {"x_shift": -2.0},
            "upper_up": {"x_shift": 0.5},
        }
        shifted, _ = self.dataset_module.means_and_derivatives(
            np.asarray([0.0, 2.0]), dataset
        )
        np.testing.assert_allclose(shifted[0, :, 0], baseline[0, :, 0])
        np.testing.assert_allclose(
            shifted[1, :, 0] - baseline[1, :, 0],
            dataset["coordinate_scale"] * np.asarray([1.0, -2.0, 0.5]),
        )

    def test_committed_swap_profile_reverses_child_progress_order(self):
        config = build_profile("committed_swap_moderate")
        means, _ = self.dataset_module.means_and_derivatives(
            np.asarray([1.0, 2.0]), config["dataset"]
        )
        np.testing.assert_allclose(means[0, :, 0], [1.0, 1.3, 1.7])
        np.testing.assert_allclose(means[1, :, 0], [2.0, 2.4, 2.15])
        self.assertLess(means[0, 1, 0], means[0, 2, 0])
        self.assertGreater(means[1, 1, 0], means[1, 2, 0])

    def test_integrated_variance_matches_quadrature(self):
        prior = self.config["dataset"]["true_diffusion"]
        grid = np.linspace(0.17, 1.81, 20001)
        instantaneous = self.diffusion_module.instantaneous_variance_numpy(grid, prior)
        trapezoid = getattr(np, "trapezoid", None)
        if trapezoid is None:
            trapezoid = np.trapz
        numerical = trapezoid(instantaneous, grid, axis=0)
        analytic = self.diffusion_module.integrated_variance_numpy(0.17, 1.81, prior)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=1e-10)

    def test_hard_labels_cover_five_regions(self):
        scale = self.config["dataset"]["coordinate_scale"]
        raw = np.asarray([[1.0, 0.0], [4.0, -1.0], [4.0, 1.0], [9.0, 1.0], [9.0, 4.0]])
        labels = self.dataset_module.hard_labels(raw * scale, self.config["dataset"])
        np.testing.assert_array_equal(labels, np.arange(5))

    def test_pair_bank_returns_labels_aligned_with_sampled_points(self):
        points = [
            np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            np.asarray([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32),
        ]
        labels = [np.asarray([1, 2]), np.asarray([3, 4])]
        plan = np.asarray([[0.5, 0.0], [0.0, 0.5]])
        bank = PairBank([0.0, 1.0], points, [plan], labels)
        source, target, source_labels, target_labels = bank.pair(
            0, 16, np.random.default_rng(7), torch.device("cpu"), return_labels=True
        )
        self.assertTrue(torch.equal(source[:, 0].long(), source_labels - 1))
        self.assertTrue(torch.equal(target[:, 0].long(), target_labels - 1))

    def test_endpoint_lineage_uses_classifier_distributions_at_both_ends(self):
        class IdentityModel:
            @staticmethod
            def physical(points):
                return points

        class FixedClassifier(torch.nn.Module):
            def forward(self, points):
                source = torch.tensor(
                    [0.02, 0.83, 0.08, 0.04, 0.03], dtype=points.dtype
                )
                target = torch.tensor(
                    [0.01, 0.72, 0.09, 0.11, 0.07], dtype=points.dtype
                )
                probabilities = torch.where(
                    (points[:, :1] < 0.5), source[None, :], target[None, :]
                )
                return probabilities.log()

        initial = torch.zeros((2, 2))
        final = torch.ones((2, 2))
        loss = endpoint_lineage_nll(
            IdentityModel(),
            FixedClassifier(),
            initial,
            final,
        )
        source = FixedClassifier()(initial).exp()[0]
        target = FixedClassifier()(final).exp()[0]
        allowed = valid_transition_matrix()
        expected = -(source[:, None] * target[None, :] * allowed).sum().log()
        self.assertAlmostEqual(float(loss), expected, places=6)


if __name__ == "__main__":
    unittest.main()

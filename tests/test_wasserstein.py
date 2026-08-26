import unittest

import numpy as np

from common import wasserstein


class ExactEmdTest(unittest.TestCase):
    def test_matches_uniform_one_dimensional_transport(self):
        x = np.asarray([[0.0], [2.0]], dtype=np.float32)
        y = np.asarray([[1.0], [5.0]], dtype=np.float32)

        self.assertAlmostEqual(wasserstein.exact_emd(x, y), 2.0)

    def test_uses_euclidean_not_squared_euclidean_cost(self):
        x = np.asarray([[0.0, 0.0]], dtype=np.float32)
        y = np.asarray([[3.0, 4.0]], dtype=np.float32)

        self.assertAlmostEqual(wasserstein.exact_emd(x, y), 5.0)

    def test_accepts_unequal_population_sizes(self):
        x = np.asarray([[0.0], [2.0]], dtype=np.float32)
        y = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32)

        self.assertAlmostEqual(wasserstein.exact_emd(x, y), 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()

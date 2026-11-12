"""Maizels flow-matching variants must use only diagonal training samples."""

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "py"))

from common.loss_args import _get_diag_offdiag_bs
from configs import maizels_pca50_constraint_sweep as sweep


class FlowMatchingConfigTests(unittest.TestCase):
    def test_flow_matching_batches_contain_no_off_diagonal_samples(self):
        for variant_id, variant in enumerate(sweep.VARIANTS):
            if "flow_matching" not in variant["mode"]:
                continue
            with self.subTest(variant=variant["mode"]):
                cfg = sweep.get_config(
                    variant_id, seed=4,
                    maizels_schedule="d3_d3p8_d8", maizels_time_mode="real_time",
                )
                self.assertEqual(cfg.training.seed, 4)
                self.assertEqual(
                    _get_diag_offdiag_bs(cfg, cfg.optimization.bs),
                    (cfg.optimization.bs, 0),
                )
                if variant["mode"] == "vanilla_flow_matching":
                    self.assertEqual(cfg.problem.maizels_pair_mode, "none")
                    self.assertFalse(cfg.constraints.enabled)
                elif variant["mode"] == "ot_flow_matching":
                    self.assertEqual(cfg.problem.maizels_pair_mode, "ot_plain")
                    self.assertFalse(cfg.constraints.enabled)

    def test_flow_map_batches_keep_their_off_diagonal_samples(self):
        for variant_id, variant in enumerate(sweep.VARIANTS):
            if "flow_matching" in variant["mode"]:
                continue
            with self.subTest(variant=variant["mode"]):
                cfg = sweep.get_config(variant_id)
                diag, offdiag = _get_diag_offdiag_bs(cfg, cfg.optimization.bs)
                self.assertEqual(diag, int(0.75 * cfg.optimization.bs))
                self.assertEqual(offdiag, cfg.optimization.bs - diag)
                self.assertGreater(offdiag, 0)


if __name__ == "__main__":
    unittest.main()

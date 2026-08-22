"""Endpoint-only Maizels PCA50 data module for metric flow matching."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import maizels  # noqa: E402


PAIR_MODES = (
    "none",
    "ot_plain",
    "endpoint_interpolant",
    "ot_endpoint_interpolant",
)
LEARNED_PATH_PRIOR_MODES = ("endpoint_interpolant", "ot_endpoint_interpolant")


def _prior_free_pair_mode(pair_mode: str) -> str:
    if pair_mode == "endpoint_interpolant":
        return "none"
    if pair_mode == "ot_endpoint_interpolant":
        return "ot_plain"
    return pair_mode


def _model_fingerprint(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous().numpy()
        hasher.update(str(value.shape).encode("utf-8"))
        hasher.update(value.tobytes())
    return hasher.hexdigest()


class _LearnedGeoPathBuilder:
    """Evaluate the frozen MFM conditional mean at requested interior times."""

    def __init__(self, geopath_net, device: torch.device, inference_batch_size: int):
        self.geopath_net = geopath_net.to(device)
        self.geopath_net.eval()
        self.device = device
        self.inference_batch_size = max(1, int(inference_batch_size))

    def __call__(self, source_x, target_x, taus):
        source_x = np.asarray(source_x, dtype=np.float32)
        target_x = np.asarray(target_x, dtype=np.float32)
        taus = np.asarray(taus, dtype=np.float32)
        paths = np.empty(
            (source_x.shape[0], taus.shape[0], source_x.shape[1]),
            dtype=np.float32,
        )
        pairs_per_batch = max(1, self.inference_batch_size // max(1, taus.shape[0]))

        with torch.inference_mode():
            for start in range(0, source_x.shape[0], pairs_per_batch):
                end = min(start + pairs_per_batch, source_x.shape[0])
                x0 = torch.from_numpy(source_x[start:end]).to(self.device)
                x1 = torch.from_numpy(target_x[start:end]).to(self.device)
                t = torch.from_numpy(taus).to(self.device)

                batch_n = end - start
                n_times = taus.shape[0]
                x0_flat = (
                    x0[:, None, :].expand(-1, n_times, -1).reshape(-1, x0.shape[1])
                )
                x1_flat = (
                    x1[:, None, :].expand(-1, n_times, -1).reshape(-1, x1.shape[1])
                )
                t_flat = t[None, :, None].expand(batch_n, -1, -1).reshape(-1, 1)
                correction = self.geopath_net(x0_flat, x1_flat, t_flat)
                gamma = 1.0 - t_flat.square() - (1.0 - t_flat).square()
                mu_t = (1.0 - t_flat) * x0_flat + t_flat * x1_flat + gamma * correction
                paths[start:end] = (
                    mu_t.reshape(batch_n, n_times, -1).detach().cpu().numpy()
                )
        return paths


def _default_classifier_path() -> str:
    repo_checkpoint = REPO_ROOT / "celltype_classifier_pca50.pt"
    if repo_checkpoint.exists():
        return str(repo_checkpoint)
    return maizels.DEFAULT_CLASSIFIER


def make_maizels_config(args) -> SimpleNamespace:
    """Build the small config surface required by the shared pair constructor."""
    pair_mode = str(args.maizels_pair_mode)
    if pair_mode not in PAIR_MODES:
        raise ValueError(
            f"maizels_pair_mode must be one of {PAIR_MODES}, got {pair_mode!r}."
        )

    dataset_location = str(args.maizels_dataset_path or args.data_path or "")
    classifier_path = str(args.maizels_classifier_path or _default_classifier_path())
    cache_dir = str(
        args.maizels_ot_cache_dir or (Path(args.working_dir) / ".maizels_ot_cache")
    )
    problem = SimpleNamespace(
        n=int(args.maizels_n_pairs),
        d=50,
        dataset_location=dataset_location,
        source_time="D3",
        target_time="D8",
        maizels_holdout_fraction=float(args.maizels_holdout_fraction),
        maizels_holdout_n=0,
        maizels_holdout_seed=int(args.maizels_holdout_seed),
        maizels_pair_mode=pair_mode,
        lineage_transition_mode=str(args.maizels_lineage_transition_mode),
        classifier_path=classifier_path,
        n_interpolant_check_times=int(args.maizels_interpolant_check_times),
        classifier_prob_threshold=float(args.maizels_classifier_prob_threshold),
        classifier_margin_threshold=float(args.maizels_classifier_margin_threshold),
        classifier_batch_size=int(args.maizels_classifier_batch_size),
        rejection_chunk_size=int(args.maizels_rejection_chunk_size),
        rejection_max_candidates=int(args.maizels_rejection_max_candidates),
        ot_candidate_chunk_size=int(args.maizels_ot_candidate_chunk_size),
        ot_mass_tolerance=1e-12,
        ot_drop_orphan_cells=True,
        ot_infeasible_fallback=str(args.maizels_ot_infeasible_fallback),
        ot_cache_enabled=True,
        ot_cache_dir=cache_dir,
        ot_cache_version="mfm_endpoint_v1",
        interpolant_path_kind="linear",
        interpolant_path_builder=None,
        ot_progress_enabled=bool(args.maizels_ot_progress_enabled),
        ot_verbose=bool(args.maizels_ot_verbose),
    )
    return SimpleNamespace(
        problem=problem,
        training=SimpleNamespace(seed=int(args.seed_current)),
        constraints=SimpleNamespace(lineage_transition_mode="same_as_problem"),
        logging=SimpleNamespace(output_folder=str(args.working_dir)),
    )


class MaizelsEndpointDataModule(pl.LightningDataModule):
    """D3 -> D8 paired training with every intermediate day reserved for evaluation."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_type = "maizels"
        self.batch_size = int(args.batch_size)
        self.whiten = False
        self.cfg = make_maizels_config(args)
        self.data_path = str(
            maizels.resolve_dataset_path(self.cfg.problem.dataset_location)
        )
        self.classifier_path = str(
            maizels.resolve_classifier_path(self.cfg.problem.classifier_path)
        )
        self.num_timesteps = 2
        self.times = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self.requested_pair_mode = str(self.cfg.problem.maizels_pair_mode)
        self.geopath_pair_mode = (
            _prior_free_pair_mode(self.requested_pair_mode)
            if bool(args.mfm)
            else self.requested_pair_mode
        )
        self._prepare_data()

    def _paired_loaders(self, paired, *, shuffle_once: bool, drop_last: bool):
        x0 = np.asarray(paired["x0"], dtype=np.float32)
        x1 = np.asarray(paired["x1"], dtype=np.float32)
        if shuffle_once:
            rng = np.random.default_rng(int(self.args.seed_current) + 811)
            order = rng.permutation(x0.shape[0])
            x0, x1 = x0[order], x1[order]
        return [
            DataLoader(
                torch.from_numpy(x0),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=drop_last,
            ),
            DataLoader(
                torch.from_numpy(x1),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=drop_last,
            ),
        ]

    def _prepare_data(self):
        self.splits = maizels.endpoint_pool_splits(
            self.cfg,
            dataset_location=self.cfg.problem.dataset_location,
        )
        self.all_timepoint_data = maizels.all_timepoint_data(
            self.cfg.problem.dataset_location
        )

        self._build_pair_loaders(self.geopath_pair_mode, stage="geopath")

        min_frame_size = min(
            int(self.splits["source_train_n"]),
            int(self.splits["target_train_n"]),
        )
        self.metric_samples_dataloaders = [
            DataLoader(
                torch.from_numpy(
                    np.asarray(self.splits["source_train_x"], dtype=np.float32)
                ),
                batch_size=min_frame_size,
                shuffle=True,
                drop_last=False,
            ),
            DataLoader(
                torch.from_numpy(
                    np.asarray(self.splits["target_train_x"], dtype=np.float32)
                ),
                batch_size=min_frame_size,
                shuffle=True,
                drop_last=False,
            ),
        ]

    def _build_pair_loaders(self, pair_mode: str, *, stage: str):
        self.cfg.problem.maizels_pair_mode = pair_mode
        train_pairs, pair_stats = maizels.make_pair_pool(
            self.cfg, dataset_location=self.cfg.problem.dataset_location
        )
        validation_n = max(
            int(self.args.maizels_validation_pairs),
            int(self.args.maizels_eval_points_per_time),
        )
        validation_pairs, validation_pair_stats = maizels.make_heldout_pair_pool(
            self.cfg,
            validation_n,
            dataset_location=self.cfg.problem.dataset_location,
            pair_mode=pair_mode,
            seed=int(self.args.seed_current) + 2701,
        )

        eval_n = int(self.args.maizels_eval_points_per_time)
        heldout_available = min(
            int(self.splits["source_holdout_n"]),
            int(self.splits["target_holdout_n"]),
        )
        eval_split = "heldout" if heldout_available >= eval_n else "train"
        self.eval_pairs, eval_pair_stats = maizels.make_endpoint_split_pair_pool(
            self.cfg,
            eval_n,
            split=eval_split,
            dataset_location=self.cfg.problem.dataset_location,
            pair_mode=pair_mode,
            seed=int(self.args.seed_current) + 1718,
        )

        for stats in (pair_stats, validation_pair_stats, eval_pair_stats):
            stats["pair_stage"] = stage
            stats["interpolant_path_kind"] = str(self.cfg.problem.interpolant_path_kind)
        self.pair_stats = pair_stats
        self.validation_pair_stats = validation_pair_stats
        self.eval_pair_stats = eval_pair_stats
        self.active_pair_mode = pair_mode

        self.train_dataloaders = self._paired_loaders(
            train_pairs, shuffle_once=True, drop_last=True
        )
        self.val_dataloaders = self._paired_loaders(
            validation_pairs, shuffle_once=False, drop_last=True
        )
        self.test_dataloaders = self._paired_loaders(
            validation_pairs, shuffle_once=False, drop_last=False
        )

    def apply_learned_geopath_prior(self, geopath_net, device: torch.device):
        """Rebuild flow-training pairs by checking the frozen learned geodesics."""
        self.geopath_pair_stats = self.pair_stats
        self.geopath_validation_pair_stats = self.validation_pair_stats
        self.geopath_eval_pair_stats = self.eval_pair_stats

        if self.requested_pair_mode not in LEARNED_PATH_PRIOR_MODES:
            return False

        builder = _LearnedGeoPathBuilder(
            geopath_net,
            device=device,
            inference_batch_size=int(self.args.maizels_classifier_batch_size),
        )
        self.cfg.problem.interpolant_path_kind = (
            "learned_geopath_sha256_" + _model_fingerprint(geopath_net)
        )
        self.cfg.problem.interpolant_path_builder = builder

        old_rejection_chunk = self.cfg.problem.rejection_chunk_size
        old_ot_chunk = self.cfg.problem.ot_candidate_chunk_size
        learned_chunk = max(1, int(self.args.maizels_geopath_filter_chunk_size))
        self.cfg.problem.rejection_chunk_size = min(old_rejection_chunk, learned_chunk)
        self.cfg.problem.ot_candidate_chunk_size = min(old_ot_chunk, learned_chunk)
        try:
            self._build_pair_loaders(self.requested_pair_mode, stage="flow")
        finally:
            self.cfg.problem.interpolant_path_builder = None
            self.cfg.problem.rejection_chunk_size = old_rejection_chunk
            self.cfg.problem.ot_candidate_chunk_size = old_ot_chunk
        return True

    def train_dataloader(self):
        combined = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined, mode="max_size_cycle")

    def val_dataloader(self):
        combined = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, mode="max_size")

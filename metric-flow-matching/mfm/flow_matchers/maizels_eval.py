"""Shared-style Maizels diagnostics for an MFM velocity field."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb


REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import maizels, wasserstein  # noqa: E402


def _sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.maximum(
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * (x @ y.T),
        0.0,
    )


def _median_bandwidth(x: np.ndarray, y: np.ndarray, rng) -> float:
    z = np.concatenate([x, y], axis=0)
    if z.shape[0] > 512:
        z = z[rng.choice(z.shape[0], size=512, replace=False)]
    distances = _sqdist(z, z)
    upper = distances[np.triu_indices(distances.shape[0], k=1)]
    upper = upper[upper > 1e-12]
    return 1.0 if upper.size == 0 else float(np.sqrt(np.median(upper)))


def rbf_mmd2(x: np.ndarray, y: np.ndarray, rng) -> float:
    bandwidth = _median_bandwidth(x, y, rng)
    bandwidths = bandwidth * np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    xx, yy, xy = _sqdist(x, x), _sqdist(y, y), _sqdist(x, y)
    values = []
    for bw in np.maximum(bandwidths, 1e-6):
        scale = 2.0 * bw * bw
        values.append(
            np.exp(-xx / scale).mean()
            + np.exp(-yy / scale).mean()
            - 2.0 * np.exp(-xy / scale).mean()
        )
    return max(float(np.mean(values)), 0.0)


def euler_rollout(
    flow_net,
    x0: torch.Tensor,
    end_time: float,
    n_steps: int,
    *,
    start_time: float = 0.0,
):
    n_steps = max(1, int(n_steps))
    if end_time < start_time:
        raise ValueError(
            f"Euler rollout end_time ({end_time}) precedes start_time ({start_time})."
        )
    dt = (float(end_time) - float(start_time)) / float(n_steps)
    x = x0
    trajectory = [x0]
    for step in range(n_steps):
        t = torch.full(
            (x.shape[0], 1),
            float(start_time) + step * dt,
            dtype=x.dtype,
            device=x.device,
        )
        x = x + dt * flow_net(t, x)
        trajectory.append(x)
    return x, torch.stack(trajectory, dim=1)


class MaizelsEvaluationCallback(pl.Callback):
    """Track omitted-day tests and held-out observed-marginal validation EMDs."""

    def __init__(self, args, datamodule):
        super().__init__()
        self.args = args
        self.datamodule = datamodule
        self.every_n_steps = int(args.maizels_eval_every_n_steps)
        self.last_evaluated_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every_n_steps > 0 and step > 0 and step % self.every_n_steps == 0:
            self._evaluate(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._evaluate(trainer, pl_module)

    def _heldout_pool(self, timepoint):
        if hasattr(self.datamodule, "timepoint_splits"):
            return np.asarray(
                self.datamodule.timepoint_splits[timepoint]["holdout_x"],
                dtype=np.float32,
            )

        retained = self.datamodule.retained_timepoints
        if timepoint == retained[0]:
            key = "source_holdout_x"
        elif timepoint == retained[-1]:
            key = "target_holdout_x"
        else:
            raise KeyError(f"No held-out Maizels pool for {timepoint}.")
        return np.asarray(self.datamodule.splits[key], dtype=np.float32)

    def _test_source_pool(self, timepoint):
        if hasattr(self.datamodule, "timepoint_splits"):
            return np.asarray(
                self.datamodule.timepoint_splits[timepoint]["x"],
                dtype=np.float32,
            )
        return np.asarray(self.datamodule.eval_pairs["x0"], dtype=np.float32)

    @staticmethod
    def _sample_rows(values, n, rng):
        values = np.asarray(values, dtype=np.float32)
        size = min(n, values.shape[0])
        indices = rng.choice(values.shape[0], size=size, replace=False)
        return values[indices]

    def _evaluate(self, trainer, pl_module):
        step = int(trainer.global_step)
        if not trainer.is_global_zero or step == self.last_evaluated_step:
            return
        self.last_evaluated_step = step

        flow_net = pl_module.flow_net
        was_training = flow_net.training
        flow_net.eval()
        device = pl_module.device
        x0 = torch.as_tensor(
            self.datamodule.eval_pairs["x0"],
            dtype=torch.float32,
            device=device,
        )
        labels = np.asarray(self.datamodule.eval_pairs["label"], dtype=np.int32)
        source_type_ids = labels[:, 0]
        n_steps = int(self.args.maizels_eval_euler_steps)
        eval_n = int(self.args.maizels_eval_points_per_time)
        seed = int(self.args.seed_current) + 1701
        rng = np.random.default_rng(seed)
        data = self.datamodule.all_timepoint_data
        cfg = self.datamodule.cfg
        retained = maizels.retained_timepoints(cfg)

        source_day = maizels.parse_timepoint(retained[0])
        target_day = maizels.parse_timepoint(retained[-1])
        timepoints = sorted(
            {
                str(tp)
                for tp, day in zip(data["timepoints"], data["time_values"])
                if source_day < float(day) < target_day and str(tp) not in retained
            },
            key=maizels.parse_timepoint,
        )

        metrics = {}
        mmd_values, emd_values = [], []
        plot_rows = []
        with torch.no_grad():
            for timepoint in timepoints:
                actual_all = np.asarray(
                    data["x"][data["timepoints"] == timepoint], dtype=np.float32
                )
                interval_source, _ = maizels.retained_interval_for_timepoint(
                    cfg, timepoint
                )
                source_all = self._test_source_pool(interval_source)
                n = min(eval_n, source_all.shape[0], actual_all.shape[0])
                source = self._sample_rows(source_all, n, rng)
                actual = self._sample_rows(actual_all, n, rng)
                start_time = maizels.normalized_time(interval_source, cfg)
                end_time = maizels.normalized_time(timepoint, cfg)
                prediction, _ = euler_rollout(
                    flow_net,
                    torch.as_tensor(source, dtype=torch.float32, device=device),
                    end_time,
                    n_steps,
                    start_time=start_time,
                )
                prediction = prediction.detach().cpu().numpy().astype(np.float32)

                mmd2 = rbf_mmd2(prediction, actual, rng)
                emd = wasserstein.exact_emd(prediction, actual)
                tag = str(timepoint).replace(".", "p").replace("/", "_")
                metrics[f"distribution_eval/{tag}_euler_rbf_mmd2"] = mmd2
                metrics[f"distribution_eval/{tag}_euler_emd"] = emd
                mmd_values.append(mmd2)
                emd_values.append(emd)
                plot_rows.append((timepoint, actual, prediction))

            validation_emd = []
            validation_relative_emd = []
            validation_rng = np.random.default_rng(seed + 101)
            for interval_source, interval_target in zip(retained[:-1], retained[1:]):
                source_all = self._heldout_pool(interval_source)
                target_all = self._heldout_pool(interval_target)
                n = min(eval_n, source_all.shape[0], target_all.shape[0])
                source = self._sample_rows(source_all, n, validation_rng)
                target = self._sample_rows(target_all, n, validation_rng)
                prediction, _ = euler_rollout(
                    flow_net,
                    torch.as_tensor(source, dtype=torch.float32, device=device),
                    maizels.normalized_time(interval_target, cfg),
                    n_steps,
                    start_time=maizels.normalized_time(interval_source, cfg),
                )
                prediction = prediction.detach().cpu().numpy().astype(np.float32)
                emd = wasserstein.exact_emd(prediction, target)
                baseline_emd = wasserstein.exact_emd(source, target)
                relative_emd = emd / max(baseline_emd, 1e-12)
                tag = str(interval_target).replace(".", "p").replace("/", "_")
                metrics[f"validation_distribution/{tag}_euler_emd"] = emd
                metrics[f"validation_distribution/{tag}_source_target_emd"] = (
                    baseline_emd
                )
                metrics[f"validation_distribution/{tag}_euler_relative_emd"] = (
                    relative_emd
                )
                validation_emd.append(emd)
                validation_relative_emd.append(relative_emd)

            _, paths = euler_rollout(
                flow_net,
                x0,
                maizels.normalized_time(retained[-1], cfg),
                n_steps,
                start_time=maizels.normalized_time(retained[0], cfg),
            )

        paths_np = paths[:, 1:, :].detach().cpu().numpy().astype(np.float32)
        validity = maizels.check_paths_with_classifier(
            paths=paths_np,
            start_type_ids=source_type_ids,
            classifier_path=self.datamodule.classifier_path,
            prob_threshold=float(self.args.maizels_classifier_prob_threshold),
            margin_threshold=float(self.args.maizels_classifier_margin_threshold),
            classifier_batch_size=int(self.args.maizels_classifier_batch_size),
            lineage_transition_mode=str(self.args.maizels_lineage_transition_mode),
        )
        valid = np.asarray(validity["valid"], dtype=bool)
        metrics["distribution_eval/euler_rbf_mmd2_mean"] = float(np.mean(mmd_values))
        metrics["distribution_eval/euler_emd_mean"] = float(np.mean(emd_values))
        metrics["validation_distribution/euler_emd_mean"] = float(
            np.mean(validation_emd)
        )
        metrics["validation_distribution/euler_relative_emd_mean"] = float(
            np.mean(validation_relative_emd)
        )
        metrics["maizels/model_euler_invalid_trajectory_pct"] = 100.0 * float(
            np.mean(~valid)
        )

        metrics["plots/maizels_intermediate_distributions"] = wandb.Image(
            self._distribution_figure(plot_rows)
        )
        metrics["plots/maizels_classifier_validity_paths"] = wandb.Image(
            self._trajectory_figure(
                paths.detach().cpu().numpy(),
                np.asarray(self.datamodule.eval_pairs["x0"]),
                np.asarray(self.datamodule.eval_pairs["x1"]),
                valid,
            )
        )
        wandb.log(metrics)
        plt.close("all")
        if was_training:
            flow_net.train()

    @staticmethod
    def _distribution_figure(rows):
        fig, axes = plt.subplots(
            2,
            len(rows),
            figsize=(3.2 * len(rows), 6.0),
            squeeze=False,
            constrained_layout=True,
        )
        all_xy = np.concatenate(
            [np.concatenate([actual[:, :2], pred[:, :2]]) for _, actual, pred in rows]
        )
        xlim = np.percentile(all_xy[:, 0], [1, 99])
        ylim = np.percentile(all_xy[:, 1], [1, 99])
        for col, (timepoint, actual, pred) in enumerate(rows):
            axes[0, col].scatter(actual[:, 0], actual[:, 1], s=2, alpha=0.35)
            axes[1, col].scatter(pred[:, 0], pred[:, 1], s=2, alpha=0.35, c="black")
            axes[0, col].set_title(f"Actual {timepoint}")
            axes[1, col].set_title(f"MFM {timepoint}")
            for ax in axes[:, col]:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.grid(alpha=0.15)
        return fig

    @staticmethod
    def _trajectory_figure(paths, x0, x1, valid):
        n = min(128, paths.shape[0])
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        for idx in range(n):
            color = "#2ca02c" if valid[idx] else "#d62728"
            ax.plot(paths[idx, :, 0], paths[idx, :, 1], color=color, alpha=0.3, lw=0.8)
        ax.scatter(x0[:n, 0], x0[:n, 1], s=8, c="black", label="D3")
        ax.scatter(x1[:n, 0], x1[:n, 1], s=8, c="#1f77b4", label="paired D8")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Held-out MFM Euler paths (green valid, red invalid)")
        ax.grid(alpha=0.15)
        ax.legend()
        return fig

"""Classifier-based CITE/Multi trajectory diagnostics for MFM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb


REPO_ROOT = Path(__file__).resolve().parents[3]
PY_ROOT = REPO_ROOT / "py"
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

from common import cite_multi, maizels  # noqa: E402
from mfm.flow_matchers.maizels_eval import euler_rollout  # noqa: E402


def default_all_days_classifier_path(dataset_name: str) -> Path:
    """Return the repository's evaluation-only classifier for a dataset."""
    dataset_name = cite_multi.canonical_dataset_name(dataset_name)
    return (
        REPO_ROOT
        / f"{dataset_name}-classifiers"
        / f"celltype_classifier_{dataset_name}_pca100_all_days.pt"
    )


def resolve_all_days_classifier_path(args) -> Path:
    configured = str(getattr(args, "cite_multi_eval_classifier_path", "") or "")
    path = (
        Path(configured).expanduser()
        if configured
        else default_all_days_classifier_path(args.data_name)
    ).resolve()
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    if not path.is_file() and not npz_path.is_file():
        raise FileNotFoundError(
            "CITE/Multi trajectory evaluation requires the all-days classifier; "
            f"neither {path} nor {npz_path} exists."
        )
    return path


def validate_cite_multi_evaluation_args(args) -> None:
    """Reject configurations whose model space does not match the classifier."""
    cite_multi.canonical_dataset_name(args.data_name)
    if int(args.dim) != 100:
        raise ValueError(
            "CITE/Multi classifier evaluation requires dim=100 because the "
            "all-days classifiers operate in PCA100 space."
        )
    if bool(args.whiten):
        raise ValueError(
            "CITE/Multi classifier evaluation requires whiten=false so model "
            "trajectories remain in the classifiers' PCA100 coordinate system."
        )
    if int(args.cite_multi_eval_check_times) <= 0:
        raise ValueError("cite_multi_eval_check_times must be positive.")
    if int(args.cite_multi_eval_source_max_points) < 0:
        raise ValueError("cite_multi_eval_source_max_points must be non-negative.")
    if int(args.cite_multi_eval_classifier_batch_size) <= 0:
        raise ValueError("cite_multi_eval_classifier_batch_size must be positive.")


def evaluation_population(
    datamodule,
    *,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select held-out day-2 sources once, without replacement."""
    if not hasattr(datamodule, "timepoint_splits"):
        raise RuntimeError(
            "CITE/Multi classifier evaluation needs cell-type-aware timepoint "
            "splits from TemporalDataModule."
        )
    try:
        source = datamodule.timepoint_splits[cite_multi.TIMEPOINTS[0]]
        target = datamodule.timepoint_splits[cite_multi.TIMEPOINTS[-1]]
    except KeyError as exc:
        raise KeyError(
            "CITE/Multi evaluation requires day-2 and day-7 populations."
        ) from exc

    source_x = np.asarray(source["holdout_x"], dtype=np.float32)
    source_types = np.asarray(source["holdout_types"], dtype=object)
    target_x = np.asarray(target["holdout_x"], dtype=np.float32)
    if source_x.shape[0] != source_types.shape[0]:
        raise ValueError("Held-out CITE/Multi source coordinates and labels disagree.")
    if source_x.shape[0] == 0:
        raise ValueError("The held-out CITE/Multi day-2 population is empty.")

    max_points = int(max_points)
    if 0 < max_points < source_x.shape[0]:
        indices = np.random.default_rng(seed).choice(
            source_x.shape[0], size=max_points, replace=False
        )
        source_x = source_x[indices]
        source_types = source_types[indices]
    return source_x, source_types, target_x


def score_paths_with_all_days_classifier(
    paths: np.ndarray,
    source_types: np.ndarray,
    *,
    classifier_path: str | Path,
    prob_threshold: float,
    margin_threshold: float,
    classifier_batch_size: int,
    lineage_transition_mode: str,
) -> Dict[str, Any]:
    """Apply the CITE/Multi lineage graph to all-days classifier predictions."""
    _, classifier_classes, _, _ = maizels.load_classifier(classifier_path)
    if set(classifier_classes) != set(cite_multi.CLASS_NAMES):
        raise ValueError(
            "The all-days classifier class names do not match the CITE/Multi "
            f"lineage classes: {classifier_classes}."
        )
    class_to_id = {name: idx for idx, name in enumerate(classifier_classes)}
    try:
        start_type_ids = np.asarray(
            [class_to_id[str(cell_type)] for cell_type in source_types],
            dtype=np.int32,
        )
    except KeyError as exc:
        raise ValueError(
            f"Unknown CITE/Multi source cell type: {exc.args[0]!r}."
        ) from exc

    return maizels.check_paths_with_classifier(
        paths=np.asarray(paths, dtype=np.float32),
        start_type_ids=start_type_ids,
        classifier_path=classifier_path,
        prob_threshold=float(prob_threshold),
        margin_threshold=float(margin_threshold),
        final_type_ids=None,
        classifier_batch_size=int(classifier_batch_size),
        lineage_transition_mode=lineage_transition_mode,
        transition_edges=cite_multi.TRANSITION_EDGES,
    )


def classifier_metric_values(
    validity: Dict[str, Any], source_count: int
) -> Dict[str, float | int]:
    """Build shared and explicit metric names from classifier validity output."""
    valid = np.asarray(validity["valid"], dtype=bool)
    valid_pct = 100.0 * float(np.mean(valid))
    invalid_pct = 100.0 * float(np.mean(~valid))
    confident_points = int(validity.get("n_confident", 0))
    total_points = int(validity.get("n_points", 0))
    confident_pct = 100.0 * confident_points / total_points if total_points > 0 else 0.0
    return {
        "cite_multi/eval_source_count": int(source_count),
        "cite_multi/full_data_classifier/model_euler_valid_trajectory_pct": valid_pct,
        "cite_multi/full_data_classifier/model_euler_invalid_trajectory_pct": invalid_pct,
        "cite_multi/full_data_classifier/confident_point_pct": confident_pct,
        # These are the names emitted by the shared JAX CITE/Multi evaluator,
        # retained here so MFM and flow-map runs can be plotted together.
        "maizels/eval_source_count": int(source_count),
        "maizels/full_data_classifier/model_euler_valid_trajectory_pct": valid_pct,
        "maizels/full_data_classifier/model_euler_invalid_trajectory_pct": invalid_pct,
    }


class CiteMultiEvaluationCallback(pl.Callback):
    """Evaluate MFM paths with CITE/Multi's all-days cell-type classifier."""

    def __init__(self, args, datamodule):
        super().__init__()
        validate_cite_multi_evaluation_args(args)
        self.args = args
        self.datamodule = datamodule
        self.dataset_name = cite_multi.canonical_dataset_name(args.data_name)
        self.classifier_path = resolve_all_days_classifier_path(args)
        self.every_n_steps = int(args.cite_multi_eval_every_n_steps)
        self.last_evaluated_step = -1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if self.every_n_steps > 0 and step > 0 and step % self.every_n_steps == 0:
            self._evaluate(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self._evaluate(trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        # The test stage restores the checkpoint selected by validation loss.
        self._evaluate(trainer, pl_module, force=True, final_best=True)

    def _evaluate(self, trainer, pl_module, *, force=False, final_best=False):
        step = int(trainer.global_step)
        if not trainer.is_global_zero or (
            not force and step == self.last_evaluated_step
        ):
            return
        self.last_evaluated_step = step

        seed = int(self.args.seed_current) + 1701
        source_x, source_types, target_x = evaluation_population(
            self.datamodule,
            max_points=int(self.args.cite_multi_eval_source_max_points),
            seed=seed,
        )

        flow_net = pl_module.flow_net
        was_training = flow_net.training
        flow_net.eval()
        try:
            with torch.no_grad():
                _, paths = euler_rollout(
                    flow_net,
                    torch.as_tensor(
                        source_x, dtype=torch.float32, device=pl_module.device
                    ),
                    end_time=1.0,
                    n_steps=int(self.args.cite_multi_eval_check_times),
                    start_time=0.0,
                )
            paths_np = paths[:, 1:, :].detach().cpu().numpy().astype(np.float32)
            validity = score_paths_with_all_days_classifier(
                paths_np,
                source_types,
                classifier_path=self.classifier_path,
                prob_threshold=float(self.args.cite_multi_eval_prob_threshold),
                margin_threshold=float(self.args.cite_multi_eval_margin_threshold),
                classifier_batch_size=int(
                    self.args.cite_multi_eval_classifier_batch_size
                ),
                lineage_transition_mode=str(
                    self.args.cite_multi_eval_lineage_transition_mode
                ),
            )
        finally:
            if was_training:
                flow_net.train()

        valid = np.asarray(validity["valid"], dtype=bool)
        metrics = classifier_metric_values(validity, source_x.shape[0])
        if final_best:
            valid_pct = metrics[
                "cite_multi/full_data_classifier/model_euler_valid_trajectory_pct"
            ]
            invalid_pct = metrics[
                "cite_multi/full_data_classifier/model_euler_invalid_trajectory_pct"
            ]
            metrics.update(
                {
                    "final_eval/euler_valid_trajectory_pct": valid_pct,
                    "final_eval/euler_invalid_trajectory_pct": invalid_pct,
                    "final_eval/full_data_classifier/euler_valid_trajectory_pct": valid_pct,
                    "final_eval/full_data_classifier/euler_invalid_trajectory_pct": invalid_pct,
                    "final_eval/best_step": step,
                }
            )
            checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
            best_score = getattr(checkpoint_callback, "best_model_score", None)
            if best_score is not None:
                if torch.is_tensor(best_score):
                    best_score = best_score.detach().cpu().item()
                metrics["final_eval/best_validation_loss"] = float(best_score)

        figure = self._trajectory_figure(
            paths.detach().cpu().numpy(), source_x, target_x, valid
        )
        metrics["plots/cite_multi_classifier_validity_paths"] = wandb.Image(figure)
        wandb.log(metrics)
        plt.close(figure)

        if final_best and wandb.run is not None and hasattr(wandb.run, "summary"):
            for key, value in metrics.items():
                if key.startswith("final_eval/"):
                    wandb.run.summary[key] = value
            wandb.run.summary["final_eval/classifier_path"] = str(self.classifier_path)
            best_path = getattr(
                getattr(trainer, "checkpoint_callback", None),
                "best_model_path",
                "",
            )
            if best_path:
                wandb.run.summary["final_eval/best_checkpoint_path"] = best_path
            print(
                f"Final best-model {self.dataset_name.upper()} evaluation: "
                "invalid_trajectory_pct="
                f"{metrics['final_eval/euler_invalid_trajectory_pct']:.8g}"
            )

    @staticmethod
    def _trajectory_figure(paths, source_x, target_x, valid):
        n = min(128, paths.shape[0])
        target_n = min(128, target_x.shape[0])
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        for idx in range(n):
            color = "#2ca02c" if valid[idx] else "#d62728"
            ax.plot(
                paths[idx, :, 0],
                paths[idx, :, 1],
                color=color,
                alpha=0.3,
                lw=0.8,
            )
        ax.scatter(source_x[:n, 0], source_x[:n, 1], s=8, c="black", label="Day 2")
        ax.scatter(
            target_x[:target_n, 0],
            target_x[:target_n, 1],
            s=8,
            c="#1f77b4",
            label="Day 7",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Held-out MFM Euler paths (green valid, red invalid)")
        ax.grid(alpha=0.15)
        ax.legend()
        return fig

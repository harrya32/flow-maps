"""Evaluation adapters for BranchSBM on the flow-maps benchmarks."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from mfm.flow_matchers.maizels_eval import (
    evaluation_rollout,
    evaluation_sampler_name,
    rbf_mmd2,
)

from common import wasserstein


class BranchSBMEvaluationMixin:
    """Turn BranchSBM's weighted branches into reproducible sampled paths."""

    evaluation_sampler = "branchsbm"

    def __init__(self, *args, terminal_branch_weights, **kwargs):
        super().__init__(*args, **kwargs)
        weights = torch.as_tensor(terminal_branch_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() != len(self.growth_nets):
            raise ValueError(
                "terminal_branch_weights must have one value per BranchSBM branch."
            )
        weights = weights / weights.sum()
        self.register_buffer("terminal_branch_weights", weights)
        self.evaluation_seed = int(getattr(self.args, "seed_current", 0)) + 3701

    @property
    def flow_net(self):
        # Shared evaluators use this handle only to toggle evaluation mode.
        # Returning the complete Lightning module toggles both velocity and
        # growth networks, while the actual rollout stays branch-aware below.
        return self

    @torch.no_grad()
    def evaluation_trajectory(
        self,
        x0: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
        n_steps: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        if abs(float(start_time)) > 1e-8:
            raise ValueError(
                "BranchSBM is an endpoint model; evaluation must start from "
                "the earliest observed marginal at model time 0."
            )
        if not 0.0 <= float(end_time) <= 1.0:
            raise ValueError("BranchSBM end_time must lie in [0, 1].")

        n_steps = max(1, int(n_steps))
        dt = float(end_time) / float(n_steps)
        branch_positions = [x0.clone() for _ in self.flow_nets]
        branch_weights = [
            torch.ones((x0.shape[0], 1), dtype=x0.dtype, device=x0.device)
        ]
        branch_weights.extend(
            torch.zeros((x0.shape[0], 1), dtype=x0.dtype, device=x0.device)
            for _ in range(len(self.flow_nets) - 1)
        )
        # First integrate all weighted branches only far enough to determine
        # their learned mass at the requested time.  Avoid retaining K complete
        # trajectories, which is prohibitively large for PCA100 populations.
        for step in range(n_steps):
            t = torch.full(
                (x0.shape[0], 1),
                float(step) * dt,
                dtype=x0.dtype,
                device=x0.device,
            )
            for branch_index, (flow_net, growth_net) in enumerate(
                zip(self.flow_nets, self.growth_nets)
            ):
                position = branch_positions[branch_index]
                branch_weights[branch_index] = (
                    branch_weights[branch_index] + dt * growth_net(t, position)
                )
                branch_positions[branch_index] = position + dt * flow_net(t, position)

        nonnegative_weights = torch.stack(branch_weights, dim=1).squeeze(-1).clamp_min(0)
        totals = nonnegative_weights.sum(dim=1, keepdim=True)
        fallback = self.terminal_branch_weights.to(
            device=x0.device,
            dtype=x0.dtype,
        ).expand(x0.shape[0], -1)
        probabilities = torch.where(
            totals > 1e-12,
            nonnegative_weights / totals.clamp_min(1e-12),
            fallback,
        )

        rng = np.random.default_rng(self.evaluation_seed if seed is None else seed)
        probabilities_np = probabilities.detach().cpu().numpy()
        uniforms = rng.random(x0.shape[0])
        branch_ids_np = np.sum(
            uniforms[:, None] > np.cumsum(probabilities_np, axis=1),
            axis=1,
        )
        branch_ids_np = np.minimum(branch_ids_np, len(self.flow_nets) - 1)
        branch_ids = torch.as_tensor(
            branch_ids_np,
            dtype=torch.long,
            device=x0.device,
        )

        # Re-integrate only the selected branch for each source.  This exactly
        # reproduces the selected position path while keeping memory O(TBD)
        # instead of O(KTBD).
        selected_position = x0.clone()
        selected_path = [selected_position.clone()]
        for step in range(n_steps):
            t = torch.full(
                (x0.shape[0], 1),
                float(step) * dt,
                dtype=x0.dtype,
                device=x0.device,
            )
            next_position = selected_position.clone()
            for branch_index, flow_net in enumerate(self.flow_nets):
                mask = branch_ids == branch_index
                if torch.any(mask):
                    next_position[mask] = selected_position[mask] + dt * flow_net(
                        t[mask], selected_position[mask]
                    )
            selected_position = next_position
            selected_path.append(selected_position.clone())
        return torch.stack(selected_path, dim=0)

    def test_step(self, batch, batch_idx):
        # All benchmark metrics are computed in on_test_start callbacks after
        # Lightning restores the validation-selected joint checkpoint.
        return None


class CiteMultiDistributionEvaluationCallback(pl.Callback):
    """Final omitted-day exact EMD and RBF MMD2 for CITE/Multi."""

    def __init__(self, args, datamodule):
        super().__init__()
        self.args = args
        self.datamodule = datamodule

    @staticmethod
    def _sample(values, max_points: int, rng: np.random.Generator):
        values = np.asarray(values, dtype=np.float32)
        if max_points <= 0 or values.shape[0] <= max_points:
            return values
        indices = rng.choice(values.shape[0], size=max_points, replace=False)
        return values[indices]

    def on_test_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        heldout_index = int(self.args.t_exclude_current)
        ordered_timepoints = tuple(self.datamodule.timepoint_splits)
        heldout_timepoint = ordered_timepoints[heldout_index]
        source_timepoint = ordered_timepoints[0]
        rng = np.random.default_rng(int(self.args.seed_current) + 2901)
        max_points = int(self.args.branchsbm_eval_points_per_time)
        source = self._sample(
            self.datamodule.timepoint_splits[source_timepoint]["x"],
            max_points,
            rng,
        )
        target = self._sample(
            self.datamodule.timepoint_splits[heldout_timepoint]["x"],
            max_points,
            rng,
        )

        was_training = pl_module.training
        pl_module.eval()
        try:
            prediction_chunks = []
            eval_batch_size = max(1, int(self.args.branchsbm_eval_batch_size))
            with torch.no_grad():
                for chunk_index, start in enumerate(
                    range(0, source.shape[0], eval_batch_size)
                ):
                    source_chunk = source[start : start + eval_batch_size]
                    prediction, _ = evaluation_rollout(
                        pl_module,
                        torch.as_tensor(
                            source_chunk,
                            dtype=torch.float32,
                            device=pl_module.device,
                        ),
                        end_time=float(
                            self.datamodule.base_datamodule.times[heldout_index]
                        ),
                        n_steps=int(self.args.branchsbm_eval_euler_steps),
                        start_time=0.0,
                        seed=int(self.args.seed_current) + 3901 + chunk_index,
                    )
                    prediction_chunks.append(
                        prediction.detach().cpu().numpy().astype(np.float32)
                    )
            prediction_np = np.concatenate(prediction_chunks, axis=0)
        finally:
            if was_training:
                pl_module.train()

        emd = wasserstein.exact_emd(prediction_np, target)
        mmd2 = rbf_mmd2(prediction_np, target, rng)
        sampler = evaluation_sampler_name(pl_module)
        tag = str(heldout_timepoint).replace(".", "p").replace("/", "_")
        metrics = {
            "test_EMD": emd,
            f"distribution_eval/{tag}_{sampler}_emd": emd,
            f"distribution_eval/{tag}_{sampler}_rbf_mmd2": mmd2,
            f"distribution_eval/{sampler}_emd_mean": emd,
            f"distribution_eval/{sampler}_rbf_mmd2_mean": mmd2,
            f"final_eval/{sampler}_mean_emd": emd,
            f"final_eval/{sampler}_mean_rbf_mmd2": mmd2,
            # Compatibility aliases used by the deterministic baseline plots.
            f"distribution_eval/{tag}_euler_emd": emd,
            f"distribution_eval/{tag}_euler_rbf_mmd2": mmd2,
            "distribution_eval/euler_emd_mean": emd,
            "distribution_eval/euler_rbf_mmd2_mean": mmd2,
            "final_eval/euler_mean_emd": emd,
            "final_eval/euler_mean_rbf_mmd2": mmd2,
        }
        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_score = getattr(checkpoint_callback, "best_model_score", None)
        if best_score is not None:
            if torch.is_tensor(best_score):
                best_score = best_score.detach().cpu().item()
            metrics["final_eval/best_validation_loss"] = float(best_score)
        wandb.log(metrics)
        if wandb.run is not None and hasattr(wandb.run, "summary"):
            for key, value in metrics.items():
                if key.startswith("final_eval/"):
                    wandb.run.summary[key] = value
        print(
            f"Final best-model {self.args.data_name.upper()} BranchSBM evaluation: "
            f"heldout_day={heldout_timepoint}, EMD={emd:.8g}, MMD2={mmd2:.8g}"
        )


def endpoint_maizels_evaluation_view(datamodule):
    """Expose endpoint rollout semantics to the shared Maizels evaluator.

    D3.8 remains visible to BranchSBM's learned metric, but it is neither an
    endpoint supervision target nor an omitted evaluation day.  Every reported
    omitted-day prediction therefore starts from D3, matching this baseline's
    D3-to-D8 flow definition.
    """

    base = datamodule.base_datamodule
    endpoints = (datamodule.source_timepoint, datamodule.terminal_timepoint)
    cfg = copy.deepcopy(base.cfg)
    cfg.problem.retained_timepoints = list(endpoints)
    cfg.problem.maizels_schedule = "d3_d8"

    excluded_metric_only_times = set(datamodule.retained_timepoints[1:-1])
    data = base.all_timepoint_data
    keep = ~np.isin(
        np.asarray(data["timepoints"], dtype=object),
        np.asarray(sorted(excluded_metric_only_times), dtype=object),
    )
    filtered_data = {}
    for key, value in data.items():
        array = np.asarray(value)
        filtered_data[key] = array[keep] if array.shape[:1] == keep.shape else value

    return SimpleNamespace(
        retained_timepoints=endpoints,
        cfg=cfg,
        all_timepoint_data=filtered_data,
        timepoint_splits=datamodule.timepoint_splits,
        eval_pairs=base.eval_pairs,
        classifier_path=base.classifier_path,
        splits=base.splits,
    )

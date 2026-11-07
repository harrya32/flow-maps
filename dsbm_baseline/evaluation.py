"""Shared benchmark evaluation entry points for the DSBM baseline."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import wandb

from common import wasserstein
from mfm.flow_matchers.cite_multi_eval import CiteMultiEvaluationCallback
from mfm.flow_matchers.maizels_eval import (
    MaizelsEvaluationCallback,
    evaluation_rollout,
    evaluation_sampler_name,
    rbf_mmd2,
)


def _sample_rows(values, max_points: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=np.float32)
    if max_points <= 0 or values.shape[0] <= max_points:
        return values
    indices = rng.choice(values.shape[0], size=max_points, replace=False)
    return values[indices]


def evaluate_cite_multi_distribution(
    args,
    datamodule,
    model,
    *,
    best_validation_emd: float,
) -> dict[str, float]:
    """Match MFM's omitted-day EMD/MMD protocol for CITE and Multi."""
    heldout_index = int(args.t_exclude_current)
    source_index = heldout_index - 1
    ordered_timepoints = tuple(datamodule.timepoint_splits)
    heldout_timepoint = ordered_timepoints[heldout_index]
    source_timepoint = ordered_timepoints[source_index]
    rng = np.random.default_rng(int(args.seed_current) + 2901)
    max_points = int(args.dsbm_eval_points_per_time)
    source = _sample_rows(
        datamodule.timepoint_splits[source_timepoint]["x"],
        max_points,
        rng,
    )
    target = _sample_rows(
        datamodule.timepoint_splits[heldout_timepoint]["x"],
        max_points,
        rng,
    )

    model.eval()
    predictions = []
    eval_batch_size = max(1, int(args.dsbm_eval_batch_size))
    with torch.no_grad():
        for chunk_index, start in enumerate(range(0, source.shape[0], eval_batch_size)):
            prediction, _ = evaluation_rollout(
                model,
                torch.as_tensor(
                    source[start : start + eval_batch_size],
                    dtype=torch.float32,
                    device=model.device,
                ),
                start_time=float(datamodule.times[source_index]),
                end_time=float(datamodule.times[heldout_index]),
                n_steps=int(args.dsbm_eval_euler_steps),
                seed=int(args.seed_current) + 3901 + chunk_index,
            )
            predictions.append(prediction.detach().cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)

    emd = wasserstein.exact_emd(prediction, target)
    mmd2 = rbf_mmd2(prediction, target, rng)
    sampler = evaluation_sampler_name(model)
    tag = str(heldout_timepoint).replace(".", "p").replace("/", "_")
    metrics = {
        "test_EMD": emd,
        "dsbm/test_EMD": emd,
        "dsbm/test_rbf_MMD2": mmd2,
        f"distribution_eval/{tag}_{sampler}_emd": emd,
        f"distribution_eval/{tag}_{sampler}_rbf_mmd2": mmd2,
        f"distribution_eval/{sampler}_emd_mean": emd,
        f"distribution_eval/{sampler}_rbf_mmd2_mean": mmd2,
        f"final_eval/{sampler}_mean_emd": emd,
        f"final_eval/{sampler}_mean_rbf_mmd2": mmd2,
        # Cross-method aliases retained by the existing benchmark plots.
        f"distribution_eval/{tag}_euler_emd": emd,
        f"distribution_eval/{tag}_euler_rbf_mmd2": mmd2,
        "distribution_eval/euler_emd_mean": emd,
        "distribution_eval/euler_rbf_mmd2_mean": mmd2,
        "final_eval/euler_mean_emd": emd,
        "final_eval/euler_mean_rbf_mmd2": mmd2,
        "final_eval/best_validation_loss": float(best_validation_emd),
        "final_eval/best_validation_emd": float(best_validation_emd),
    }
    wandb.log(metrics)
    if wandb.run is not None and hasattr(wandb.run, "summary"):
        for key, value in metrics.items():
            if key.startswith("final_eval/"):
                wandb.run.summary[key] = value
    print(
        f"Final best-model {args.data_name.upper()} DSBM evaluation: "
        f"heldout_day={heldout_timepoint}, EMD={emd:.8g}, MMD2={mmd2:.8g}"
    )
    return metrics


def run_final_evaluation(
    args,
    datamodule,
    model,
    *,
    global_step: int,
    best_validation_emd: float,
    checkpoint_path: str,
) -> None:
    """Run the same final distribution and lineage evaluators as MFM/SF2M."""
    trainer = SimpleNamespace(
        global_step=int(global_step),
        is_global_zero=True,
        checkpoint_callback=SimpleNamespace(
            best_model_score=torch.tensor(float(best_validation_emd)),
            best_model_path=str(checkpoint_path),
        ),
    )
    model.eval()
    if args.data_type == "maizels":
        MaizelsEvaluationCallback(args=args, datamodule=datamodule).on_test_start(
            trainer,
            model,
        )
        return

    evaluate_cite_multi_distribution(
        args,
        datamodule,
        model,
        best_validation_emd=best_validation_emd,
    )
    CiteMultiEvaluationCallback(args=args, datamodule=datamodule).on_test_start(
        trainer,
        model,
    )

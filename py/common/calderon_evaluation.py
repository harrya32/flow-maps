"""Saved Calderon results from the native Maizels evaluation protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import calderon


def evaluate(cfg, statics, state, data, output, checkpoint, evaluation_output=None):
    import jax
    from . import calderon_logging, dist_utils, logging
    from launchers.calderon_ot_flow_map import evaluation_params, rollout_checkpoints, write_json

    result_root = output if evaluation_output is None else Path(evaluation_output)
    evaluation_dir = result_root / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    calderon_logging.register(cfg, data)
    params = evaluation_params(cfg, state)
    source = data.marginals[data.source_time]
    available = data.splits[data.source_time]["validation"]
    selected = logging._population_indices_without_replacement(
        len(available), cfg.logging.maizels.trajectory_eval_source_max_points,
        np.random.default_rng(cfg.logging.maizels.trajectory_eval_seed))
    rows = available[selected]
    source_x, source_meta = source.x[rows], source.metadata.iloc[rows]
    trajectory_cfg = SimpleNamespace(flowmap_n_steps=len(cfg.evaluation.checkpoint_times),
        checkpoint_times=cfg.evaluation.checkpoint_times, flow_batch_size=cfg.evaluation.flow_batch_size)
    print(f"Evaluating {len(source_x)} held-out sources with {trajectory_cfg.flowmap_n_steps} full-window composed steps.", flush=True)
    paths = rollout_checkpoints(statics.net.apply, params, source_x, trajectory_cfg)
    distribution = {}

    def capture_population(**population):
        kind, window, source_window = population["kind"], population["window"], population["source_window"]
        key = (f"heldout_{window.replace('-', '_')}" if kind == "intermediate" else
               f"observed_{source_window.replace('-', '_')}_to_{window.replace('-', '_')}" if kind == "observed" else
               "source_to_final")
        path = evaluation_dir / f"{key}_populations.npz"
        np.savez_compressed(path, source_x=population["source_x"], generated_x=population["prediction"],
            reference_x=population["reference_x"], source_window=np.asarray(source_window), window=np.asarray(window),
            start_time=np.asarray(population["start_time"]), end_time=np.asarray(population["end_time"]),
            n_composed_steps=np.asarray(population["n_steps"]), w1=np.asarray(population["w1"]))
        distribution[key] = {"window": window, "source_window": source_window, "population": kind,
            "start_time": population["start_time"], "normalized_time": population["end_time"],
            "w1": population["w1"], "n_generated": len(population["prediction"]),
            "n_reference": len(population["reference_x"]), "n_available_source": len(data.marginals[source_window].x),
            "n_available_reference": len(data.marginals[window].x), "n_composed_steps": population["n_steps"],
            "steps_per_interval": cfg.evaluation.flowmap_n_steps, "sampler": "composed_flow_map",
            "source_pool": "all", "reference_pool": "all", "saved_populations": str(path.resolve())}

    summary_path = output / "training_summary.json"
    training = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    native_metrics = logging.log_maizels_final_evaluation(cfg, state, params,
        best_step=int(jax.device_get(dist_utils.safe_unreplicate(cfg, state).step)),
        best_metric=training.get("best_validation_loss"), population_callback=capture_population)
    native_metrics = {name: float(value) for name, value in native_metrics.items()}
    write_json(result_root / "maizels_evaluation_metrics.json", native_metrics)

    oracle = calderon_logging._model(calderon.evaluation_manifest_path(cfg), cfg.classifier.batch_size)
    data.provenance.update({"evaluation_classifier_export_sha256": oracle.export_sha256,
        "evaluation_classifier_forward_sha256": calderon.sha256_file(Path(__file__).with_name("calderon_classifier.py"))})
    probabilities = oracle.predict_proba(paths.reshape(-1, 100)).reshape(len(paths), paths.shape[1], -1)
    allowed = calderon.load_allowed_pairs(cfg.classifier.allowed_pairs_path, oracle.class_names)
    lookup = {name: i for i, name in enumerate(oracle.class_names)}
    source_labels = source_meta.broad_tissue.to_numpy(dtype=str)
    starts = np.array([lookup[name] for name in source_labels])
    raw = calderon.score_tissue_paths(starts, probabilities, allowed)
    hours = cfg.problem.source_hours + np.asarray(cfg.evaluation.checkpoint_times) * (cfg.problem.target_hours - cfg.problem.source_hours)
    tissue_metrics = {"classifier_manifest": calderon.evaluation_manifest_path(cfg),
        "classifier_manifest_sha256": calderon.sha256_file(calderon.evaluation_manifest_path(cfg)),
        "classifier_export_sha256": oracle.export_sha256,
        "classifier_role": "all_times_reference_oracle", "start_labels": "Released tissue labels of flow-validation source cells",
        "raw": raw["metrics"], "prob_threshold": 0., "margin_threshold": 0.,
        "checkpoint_times": list(cfg.evaluation.checkpoint_times), "checkpoint_hours": hours.tolist()}
    write_json(evaluation_dir / "tissue_trajectory_metrics.json", tissue_metrics)
    first = raw["first_rejection"]
    pd.DataFrame({"source_cell_id": source_meta.cell_id.to_numpy(), "source_tissue": source_labels,
        "raw_violation": first >= 0, "raw_first_rejection_index": first}).to_csv(
            evaluation_dir / "trajectory_summary.csv.gz", index=False)
    pd.DataFrame({"source_cell_id": np.repeat(source_meta.cell_id.to_numpy(), len(hours)),
        "checkpoint_time": np.tile(cfg.evaluation.checkpoint_times, len(paths)),
        "checkpoint_hours": np.tile(hours, len(paths)),
        "predicted_tissue": np.asarray(oracle.class_names)[raw["predicted_ids"]].reshape(-1)}).to_csv(
            evaluation_dir / "trajectory_predictions.csv.gz", index=False)
    np.savez_compressed(evaluation_dir / "evaluation_inputs.npz", paths=paths, source_x=source_x,
        source_cell_ids=source_meta.cell_id.to_numpy(dtype=str), source_tissues=source_labels,
        class_names=np.asarray(data.class_names), checkpoint_times=np.asarray(cfg.evaluation.checkpoint_times),
        checkpoint_hours=hours, evaluation_protocol=np.asarray("maizels"))
    np.savez_compressed(evaluation_dir / "trajectory_probabilities.npz", probabilities=probabilities,
        class_names=np.asarray(oracle.class_names), source_cell_ids=source_meta.cell_id.to_numpy(dtype=str))
    metrics = {"experiment": "Calderon Maizels OT flow-map protocol", "evaluation_protocol": "maizels",
        "training_windows": list(data.training_times), "intermediate_windows": list(data.heldout_times),
        "source_window": data.source_time, "target_window": data.target_time,
        "training_classifier_manifest": cfg.classifier.manifest_path,
        "evaluation_classifier_manifest": calderon.evaluation_manifest_path(cfg),
        "parameters": "instantaneous", "ema_factor": None, "flowmap_n_steps": cfg.evaluation.flowmap_n_steps,
        "w1_population_cap": cfg.evaluation.marginal_max_points,
        "distribution_eval_samplers": list(cfg.logging.maizels.get("distribution_eval_samplers", ["direct", "flowmap", "euler", "linear"])),
        "distribution_eval_metrics": list(cfg.logging.maizels.get("distribution_eval_metrics", ["emd", "rbf_mmd2"])),
        "trajectory_eval_samplers": list(cfg.logging.maizels.get("trajectory_eval_samplers", ["direct", "flowmap", "euler"])),
        "n_violation_checkpoints": len(cfg.evaluation.checkpoint_times),
        "violation_flowmap_n_steps": trajectory_cfg.flowmap_n_steps,
        "n_heldout_sources_available": len(available), "n_heldout_sources_evaluated": len(source_x),
        "distribution": distribution, "tissue_trajectories": tissue_metrics, "native_maizels_metrics": native_metrics,
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": calderon.sha256_file(checkpoint),
        "training_step": int(jax.device_get(dist_utils.safe_unreplicate(cfg, state).step)),
        "training_config_sha256": calderon.sha256_file(output / "config.json"), "provenance": data.provenance}
    if cfg.problem.maizels_pair_mode == "ot_endpoint_interpolant":
        metrics["training_interpolant_filter"] = {"n_interior_checks": cfg.problem.n_interpolant_check_times,
            "prob_threshold": cfg.problem.classifier_prob_threshold, "margin_threshold": cfg.problem.classifier_margin_threshold,
            "ot_infeasible_fallback": cfg.problem.ot_minibatch_infeasible_fallback}
    if cfg.constraints.enabled:
        metrics["training_lineage_constraint"] = cfg.constraints.to_dict()
    write_json(result_root / "metrics.json", metrics)
    write_json(evaluation_dir / "evaluation_config.json", cfg.to_dict())

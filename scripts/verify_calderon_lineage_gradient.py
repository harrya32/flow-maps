#!/usr/bin/env python3
"""Verify the frozen differentiable decoder and the native lineage gradient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from common.calderon import sha256_file, DEFAULT_SPLIT_MANIFEST
from common.calderon_lineage import setup_lineage_classifier
from common.losses import maizels_lineage_loss_point_constraint
from configs.calderon_constrained_pca100 import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    args = parser.parse_args()
    cfg = get_config()
    classifier = setup_lineage_classifier(cfg)
    model = classifier["params"]
    export = Path(cfg.classifier.jax_export_path)
    verification = export.with_name(export.stem + "_verification.npz")
    with np.load(verification, allow_pickle=False) as stored:
        assert stored["export_sha256"].item() == model.export_sha256
        x, expected = stored["x_pca"], stored["torch_probabilities"]
        endpoint_rows = stored["endpoint_pca_rows"]
        endpoint_windows = stored["endpoint_windows"].astype(str).tolist()
    predict = jax.jit(lambda values: jax.nn.softmax(model.log_probabilities(values), axis=-1))
    actual = np.asarray(predict(jnp.asarray(x)))
    error = float(np.max(np.abs(actual - expected)))
    decisions_match = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
    confidence_match = bool(np.array_equal(actual.max(1) >= .9, expected.max(1) >= .9))
    if error > 3e-5 or not decisions_match or not confidence_match:
        raise ValueError(f"Differentiable decoder differs from the frozen ensemble: error={error}, decisions={decisions_match}, confidence={confidence_match}")
    table = pd.read_csv(args.splits)
    endpoint_ids = []
    lookup = {name: i for i, name in enumerate(model.class_names)}
    split_column = "flow_split" if "flow_split" in table else "classifier_split"
    row_column = "pca_row" if "pca_row" in table else "source_pca_row"
    for window, rows in zip(endpoint_windows, endpoint_rows):
        train = table.loc[table.day.eq(window) & table[split_column].eq("train")].set_index(row_column)
        assert np.isin(rows, train.index).all()
        endpoint_ids.append(train.loc[rows, "broad_tissue"].map(lookup).to_numpy(dtype=np.int32))
    starts, ends = jnp.asarray(x[:8]), jnp.asarray(x[256:264])
    labels = np.stack([ids[:8] for ids in endpoint_ids[:2]], axis=1).astype(np.float32)
    if cfg.problem.pair_time_bounds_in_label:
        labels = np.column_stack([labels, np.zeros(8), np.full(8, .5)])
    labels = jnp.asarray(labels)

    def penalty(values):
        terms = jax.vmap(lambda a, b, label: maizels_lineage_loss_point_constraint(
            a, b, label, cfg=cfg, classifier=classifier))(starts, values, labels)
        return jnp.mean(terms)

    penalty = jax.jit(penalty)
    value = float(penalty(ends))
    gradient = np.asarray(jax.grad(penalty)(ends))
    norm = float(np.linalg.norm(gradient))
    assert np.isfinite(gradient).all() and norm > 1e-5
    direction = jnp.asarray(gradient / norm)
    step = .005
    finite_difference = float((penalty(ends + step * direction) - penalty(ends - step * direction)) / (2 * step))
    relative_error = abs(finite_difference - norm) / max(norm, 1e-6)
    assert relative_error < .05, (norm, finite_difference, relative_error)
    assert float(penalty(ends - step * direction)) < value
    a, b = np.asarray(predict(starts)), np.asarray(predict(ends))
    allowed = 1 - np.asarray(classifier["invalid_transition"])
    explicit = float(cfg.constraints.weight * np.mean(-np.log(np.clip(np.einsum("bi,ij,bj->b", a, allowed, b), 1e-6, 1.))))
    assert np.isclose(value, explicit, rtol=1e-5, atol=.005), (value, explicit)
    result = {"status": "passed", "devices": [str(d) for d in jax.devices()],
              "n_classifier_verification_points": len(x), "endpoint_training_cells_only": True,
              "training_windows": endpoint_windows,
              "max_absolute_probability_difference": error, "argmax_decisions_identical": decisions_match,
              "confidence_0p9_masks_identical": confidence_match,
              "native_transition_nll": value, "explicit_transition_nll": explicit,
              "input_gradient_norm": norm, "directional_finite_difference": finite_difference,
              "directional_relative_error": relative_error, "gradient_descent_reduces_penalty": True,
              "export_sha256": model.export_sha256, "manifest_sha256": model.manifest_sha256,
              "verification_inputs_sha256": sha256_file(verification),
              "differentiable_forward_sha256": sha256_file(ROOT / "py/common/calderon_lineage.py"),
              "filter_forward_sha256": sha256_file(ROOT / "py/common/calderon_classifier.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

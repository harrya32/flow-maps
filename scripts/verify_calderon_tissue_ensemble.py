#!/usr/bin/env python3
"""Verify frozen JAX tissue decoding against the saved PyTorch predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "py"))

import jax
import numpy as np

from common.calderon import sha256_file, DEFAULT_TRAINING_MANIFEST, DEFAULT_TRAINING_EXPORT
from common.calderon_classifier import FrozenTissueEnsemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, default=DEFAULT_TRAINING_EXPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check = args.export.with_name(args.export.stem + "_verification.npz")
    with np.load(check, allow_pickle=False) as stored:
        if stored["export_sha256"].item() != sha256_file(args.export):
            raise ValueError("Verification inputs belong to a different frozen export.")
        x, expected = stored["x_pca"], stored["torch_probabilities"]
        training_only = bool(stored["endpoint_training_cells_only"].item()) if "endpoint_training_cells_only" in stored else True
        scope = str(stored["verification_scope"].item()) if "verification_scope" in stored else "selected_training_cells_and_interpolants"
    model = FrozenTissueEnsemble(args.export, args.manifest)
    started = time.monotonic()
    actual = model.predict_proba(x)
    first_seconds = time.monotonic() - started
    started = time.monotonic()
    repeated = model.predict_proba(x)
    result = {"devices": [str(d) for d in jax.devices()], "n_points": len(x),
              "export_sha256": model.export_sha256, "manifest_sha256": model.manifest_sha256,
              "gelu_implementation": model.gelu_implementation,
              "implementation_sha256": sha256_file(ROOT / "py/common/calderon_classifier.py"),
              "verification_inputs_sha256": sha256_file(check), "endpoint_training_cells_only": training_only,
              "verification_scope": scope,
              "max_absolute_probability_difference": float(np.max(np.abs(actual - expected))),
              "argmax_agreement_pct": float(100 * np.mean(actual.argmax(1) == expected.argmax(1))),
              "confidence_0p9_agreement_pct": float(100 * np.mean((actual.max(1) >= .9) == (expected.max(1) >= .9))),
              "repeated_predictions_identical": bool(np.array_equal(actual, repeated)),
              "first_seconds": first_seconds, "steady_seconds": time.monotonic() - started}
    bucket_errors = []
    for count in (1, 37, 128, 511, 1500, 4096):
        value = model.predict_proba(x[:count])
        bucket_errors.append(float(np.max(np.abs(value - expected[:count]))))
        if not np.array_equal(value.argmax(1), expected[:count].argmax(1)) or not np.array_equal(value.max(1) >= .9, expected[:count].max(1) >= .9):
            raise ValueError("Classifier decisions changed in an adaptive inference batch.")
    result["adaptive_batch_max_probability_difference"] = max(bucket_errors)
    result["adaptive_batch_decisions_identical"] = True
    print(json.dumps(result, indent=2), flush=True)
    if max(result["max_absolute_probability_difference"], max(bucket_errors)) > 3e-5 or result["argmax_agreement_pct"] != 100 or result["confidence_0p9_agreement_pct"] != 100:
        raise ValueError("Frozen JAX decoder does not sufficiently reproduce the selected tissue ensemble.")
    result["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

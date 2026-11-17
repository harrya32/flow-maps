"""Differentiable Calderon adapter for the native Maizels lineage loss.

The filter and final evaluator retain their existing frozen inference paths.
This adapter uses the same frozen weights with float32 JAX preprocessing and
calibration so gradients can pass from the lineage loss to the flow map.
"""

from __future__ import annotations

import numpy as np

from .calderon import load_allowed_pairs
from .calderon_classifier import FrozenTissueEnsemble


class DifferentiableTissueEnsemble(FrozenTissueEnsemble):
    def __init__(self, export_path, manifest_path, batch_size=8192):
        super().__init__(export_path, manifest_path, batch_size)
        import jax.numpy as jnp
        self.jax_mean = jnp.asarray(self.mean, dtype=jnp.float32)
        self.jax_scale = jnp.asarray(self.scale, dtype=jnp.float32)
        self.member_temperatures = jnp.asarray([m["temperature"] for m in self.spec["members"]], dtype=jnp.float32)
        self.ensemble_temperature = jnp.asarray(self.spec["temperature"], dtype=jnp.float32)

    def log_probabilities(self, x_pca):
        import jax
        import jax.numpy as jnp
        x = (x_pca - self.jax_mean) / self.jax_scale
        logits = self._logits(x)
        member_probs = jax.nn.softmax(logits / self.member_temperatures[:, None, None], axis=-1)
        average = jnp.mean(member_probs, axis=0)
        return jax.nn.log_softmax(jnp.log(jnp.maximum(average, 1e-30)) / self.ensemble_temperature, axis=-1)


def setup_lineage_classifier(cfg):
    import jax.numpy as jnp
    model = DifferentiableTissueEnsemble(cfg.classifier.jax_export_path, cfg.classifier.manifest_path,
                                       cfg.problem.classifier_batch_size)
    allowed = load_allowed_pairs(cfg.classifier.allowed_pairs_path, model.class_names, allow_extra_classes=True)
    return {"params": model, "class_names": model.class_names, "scaler_mean": None, "scaler_scale": None,
            "invalid_transition": jnp.asarray(~allowed, dtype=jnp.float32),
            "canonical_to_classifier": jnp.arange(len(model.class_names), dtype=jnp.int32),
            "lineage_transition_mode": "projected_relation_without_additional_closure"}


def jax_classifier_logits(params, scaler_mean, scaler_scale, x_pca):
    # Native Maizels code softmaxes these at temperature 1.0. Calibrated
    # log-probabilities recover the selected ensemble's probabilities.
    return params.log_probabilities(x_pca)


def lineage_soft_terms_from_probs(*args, **kwargs):
    from .maizels import lineage_soft_terms_from_probs as native_terms
    return native_terms(*args, **kwargs)

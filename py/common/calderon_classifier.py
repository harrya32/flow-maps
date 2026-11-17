"""Frozen JAX inference for the released Calderon tissue ensemble export.

Weights are exported from the existing PyTorch checkpoints without fitting.
NumPy preprocessing and calibration match predict_selected; JAX evaluates the
frozen wide/residual trunks so training-time interpolant checks can use a GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calderon import sha256_file


class FrozenTissueEnsemble:
    def __init__(self, export_path, manifest_path, batch_size=8192):
        import jax
        import jax.numpy as jnp
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("Classifier batch size must be positive.")
        with np.load(export_path, allow_pickle=False) as stored:
            metadata = json.loads(stored["metadata"].item())
            self.spec = metadata["manifest"]
            if metadata["manifest_sha256"] != sha256_file(manifest_path):
                raise ValueError("Frozen classifier export no longer matches its selected manifest.")
            self.class_names = tuple(self.spec["class_names"])
            self.families = tuple(member["family"] for member in metadata["members"])
            params = []
            for i, member in enumerate(metadata["members"]):
                path = (Path(manifest_path).parent / self.spec["members"][i]["checkpoint"]).resolve()
                if sha256_file(path) != member["checkpoint_sha256"]:
                    raise ValueError("A frozen ensemble checkpoint changed after export.")
                prefix = f"member{i}__"
                params.append({name[len(prefix):]: jnp.asarray(stored[name]) for name in stored.files if name.startswith(prefix)})
        self.export_sha256 = sha256_file(export_path)
        self.manifest_sha256 = metadata["manifest_sha256"]
        self.member_hashes = [m["checkpoint_sha256"] for m in metadata["members"]]
        self.mean = np.asarray(self.spec["scaler_mean"], dtype=np.float64)
        self.scale = np.asarray(self.spec["scaler_scale"], dtype=np.float64)

        def linear(x, weights, prefix):
            return x @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

        def norm(x, weights, prefix):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            return (x - mean) * jax.lax.rsqrt(variance + 1e-5) * weights[prefix + ".weight"] + weights[prefix + ".bias"]

        metal = any(str(device.platform).lower() == "metal" for device in jax.devices())
        self.gelu_implementation = "verified float32 erf polynomial" if metal else "exact erf"

        def gelu(x):
            if not metal:
                return jax.nn.gelu(x, approximate=False)
            # A&S 7.1.26: mathematical erf approximation error <=1.5e-7.
            # Metal cannot lower mhlo.erf; this remains much closer to the
            # trained exact-GELU network than the usual tanh GELU approximation.
            z = x * (1 / np.sqrt(2.))
            t = 1 / (1 + .3275911 * jnp.abs(z))
            p = (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                    - .284496736) * t + .254829592) * t)
            erf = jnp.sign(z) * (1 - p * jnp.exp(-jnp.square(z)))
            return .5 * x * (1 + erf)

        def logits(x):
            outputs = []
            for family, weights in zip(self.families, params):
                if family == "residual":
                    h = gelu(linear(x, weights, "trunk.0"))
                    for index in (2, 3, 4):
                        prefix = f"trunk.{index}"
                        z = norm(h, weights, prefix + ".norm")
                        z = gelu(linear(z, weights, prefix + ".fc1"))
                        h = h + .5 * linear(z, weights, prefix + ".fc2")
                    h = norm(h, weights, "trunk.5")
                elif family == "wide":
                    h = x
                    for index in (0, 4, 8):
                        h = linear(h, weights, f"trunk.{index}")
                        prefix = f"trunk.{index + 1}"
                        h = ((h - weights[prefix + ".running_mean"])
                             * jax.lax.rsqrt(weights[prefix + ".running_var"] + 1e-5)
                             * weights[prefix + ".weight"] + weights[prefix + ".bias"])
                        h = gelu(h)
                else:
                    raise ValueError(f"Unsupported exported classifier family: {family}")
                outputs.append(linear(h, weights, "heads.broad_tissue"))
            return jnp.stack(outputs)
        self._logits = jax.jit(logits)

    def predict_proba(self, x_pca):
        import jax
        import jax.numpy as jnp
        from scipy.special import softmax
        x = np.asarray(x_pca, dtype=np.float32).copy()
        if x.ndim != 2 or x.shape[1] != 100 or not np.isfinite(x).all():
            raise ValueError("Frozen tissue decoder requires finite original PCA100 inputs.")
        # Match the float64-scaler, float32 in-place steps of predict_selected.
        x -= self.mean
        x /= self.scale
        result = np.empty((len(x), len(self.class_names)), dtype=np.float32)
        for start in range(0, len(x), self.batch_size):
            batch = x[start:start + self.batch_size]
            n = len(batch)
            padded_size = min(self.batch_size, max(32, 1 << (n - 1).bit_length()))
            if n < padded_size:
                batch = np.pad(batch, ((0, padded_size - n), (0, 0)), mode="edge")
            logits = np.asarray(jax.device_get(self._logits(jnp.asarray(batch))))[:, :n]
            probs = [softmax(value.astype(np.float64) / member["temperature"], axis=1).astype(np.float32)
                     for value, member in zip(logits, self.spec["members"])]
            average = np.mean(probs, axis=0)
            result[start:start + n] = softmax(np.log(np.maximum(average, 1e-30)).astype(np.float64)
                                             / self.spec["temperature"], axis=1).astype(np.float32)
        return result

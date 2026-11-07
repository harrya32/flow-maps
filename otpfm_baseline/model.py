"""Sample the upstream mean-velocity model on the benchmark's explicit grid."""

from __future__ import annotations

import numpy as np
import torch


class OTPFMSampler:
    """Return direct, composed-map, or instantaneous-velocity Euler predictions."""

    def __init__(self, model, scaler, *, batch_size=1024, ema=True):
        self.model = model
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("Sampling batch size must be positive.")
        self.ema = bool(ema)
        self.device = next(model.parameters()).device
        self.center = torch.as_tensor(
            (
                np.zeros((1, model.d), dtype=np.float32)
                if scaler is None
                else scaler.mean_
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.scale = 1.0 if scaler is None else float(scaler.scale_)

    @torch.no_grad()
    def sample(
        self,
        source,
        *,
        start_time=0.0,
        end_time=1.0,
        n_steps=50,
        sampler="flowmap",
        paths=False,
    ):
        """Use n_steps over the requested horizon, independent of marginal count."""
        if sampler not in ("direct", "flowmap", "euler"):
            raise ValueError(f"Unknown OTP-FM sampler: {sampler}.")
        if int(n_steps) <= 0 or not 0.0 <= start_time <= end_time <= 1.0:
            raise ValueError(
                "Sampling needs positive steps and 0 <= start <= end <= 1."
            )
        source = np.asarray(source, dtype=np.float32)
        if source.ndim != 2 or len(source) == 0 or source.shape[1] != self.model.d:
            raise ValueError(
                "Sampling needs a non-empty population in the model's PCA space."
            )
        steps = 1 if sampler == "direct" and not paths else int(n_steps)
        times = np.linspace(start_time, end_time, steps + 1, dtype=np.float32)
        velocity = self.model.v_func_ema if self.ema else self.model.v_func
        was_training = self.model.training
        self.model.eval()
        chunks = []
        try:
            for start in range(0, len(source), self.batch_size):
                x0 = (
                    torch.as_tensor(
                        source[start : start + self.batch_size], device=self.device
                    )
                    - self.center
                ) / self.scale
                x = x0
                positions = [source[start : start + self.batch_size]] if paths else None
                for left, right in zip(times[:-1], times[1:]):
                    if sampler == "direct":
                        left = times[0]
                    t1 = torch.full((len(x),), float(left), device=self.device)
                    t2 = torch.full((len(x),), float(right), device=self.device)
                    base = x0 if sampler == "direct" else x
                    v = velocity(base, t1, t1 if sampler == "euler" else t2)
                    x = base + float(right - left) * v
                    if paths:
                        positions.append((x * self.scale + self.center).cpu().numpy())
                prediction = (
                    np.stack(positions, axis=1)
                    if paths
                    else (x * self.scale + self.center).cpu().numpy()
                )
                if not np.isfinite(prediction).all():
                    raise FloatingPointError(
                        f"Non-finite {sampler} OTP-FM predictions."
                    )
                chunks.append(prediction)
        finally:
            self.model.train(was_training)
        return np.concatenate(chunks).astype(np.float32)

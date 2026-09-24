"""Interval-aware TorchCFM adapter for simulation-free SF2M training."""

from __future__ import annotations

import math

import torch
from torchcfm.conditional_flow_matching import (
    SchrodingerBridgeConditionalFlowMatcher,
)


class IntervalSchrodingerBridgeFlowMatcher:
    """Apply TorchCFM's SF2M bridge on intervals of a global time axis.

    TorchCFM parameterizes every bridge on ``[0, 1]``. Biological datasets in
    this repository use a single global clock with possibly unequal retained
    intervals. For a constant global reference diffusion ``sigma``, an
    interval of duration ``delta`` therefore uses TorchCFM sigma
    ``sigma * sqrt(delta)`` and its local velocity target is divided by
    ``delta``.
    """

    alpha = 0

    def __init__(
        self,
        sigma: float,
        *,
        ot_method: str = "exact",
        time_eps: float = 1e-3,
    ):
        sigma = float(sigma)
        time_eps = float(time_eps)
        if sigma <= 0.0:
            raise ValueError("sf2m_sigma must be strictly positive.")
        if ot_method not in {"exact", "sinkhorn"}:
            raise ValueError("sf2m_ot_method must be 'exact' or 'sinkhorn'.")
        if not 0.0 <= time_eps < 0.5:
            raise ValueError("sf2m_time_eps must lie in [0, 0.5).")
        self.sigma = sigma
        self.ot_method = str(ot_method)
        self.time_eps = time_eps
        self._native_matchers = {}

    def _native_matcher(self, duration: float):
        duration = float(duration)
        if duration <= 0.0:
            raise ValueError("SF2M interval duration must be positive.")
        key = round(duration, 12)
        if key not in self._native_matchers:
            self._native_matchers[key] = SchrodingerBridgeConditionalFlowMatcher(
                sigma=self.sigma * math.sqrt(duration),
                ot_method=self.ot_method,
            )
        return self._native_matchers[key]

    def sample_location_flow_and_score(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
    ):
        """Sample an SF2M bridge and return flow and stable score targets."""
        start = float(t_min)
        end = float(t_max)
        duration = end - start
        matcher = self._native_matcher(duration)

        local_t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        if self.time_eps:
            local_t = self.time_eps + (1.0 - 2.0 * self.time_eps) * local_t
        local_t, xt, local_ut, noise = matcher.sample_location_and_conditional_flow(
            x0,
            x1,
            t=local_t,
            return_noise=True,
        )
        global_t = start + duration * local_t
        global_ut = local_ut / duration
        score_scale = matcher.compute_sigma_t(local_t).reshape(
            -1, *([1] * (x0.dim() - 1))
        )
        return global_t, xt, global_ut, score_scale, noise

    def sample_location_and_conditional_flow(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_min: float,
        t_max: float,
    ):
        """Compatibility interface returning only the flow-matching targets."""
        t, xt, ut, _, _ = self.sample_location_flow_and_score(x0, x1, t_min, t_max)
        return t, xt, ut

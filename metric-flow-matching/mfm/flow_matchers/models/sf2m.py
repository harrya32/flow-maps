"""Interval-aware TorchCFM adapter for simulation-free SF2M training."""

from __future__ import annotations

import math

import torch
from torchcfm.conditional_flow_matching import (
    SchrodingerBridgeConditionalFlowMatcher,
)

from mfm.flow_matchers.models.geodesic_ot import (
    GeodesicSinkhornPlanSampler,
    HeatKernelGeodesicCost,
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
        ot_cost: str = "euclidean",
        time_eps: float = 1e-3,
        reference_populations=None,
        geodesic_knn: int = 5,
        geodesic_heat_time: float = 1.0,
        geodesic_eigenvectors: int = 256,
        geodesic_graph_max_points: int = 0,
        geodesic_cache_dir: str = "",
        geodesic_heat_epsilon: float = 1e-12,
        geodesic_sinkhorn_max_iter: int = 5000,
        geodesic_sinkhorn_tol: float = 1e-7,
        seed: int = 0,
    ):
        sigma = float(sigma)
        time_eps = float(time_eps)
        if sigma <= 0.0:
            raise ValueError("sf2m_sigma must be strictly positive.")
        if ot_method not in {"exact", "sinkhorn"}:
            raise ValueError("sf2m_ot_method must be 'exact' or 'sinkhorn'.")
        if ot_cost not in {"euclidean", "geodesic"}:
            raise ValueError("sf2m_ot_cost must be 'euclidean' or 'geodesic'.")
        if ot_cost == "geodesic" and ot_method != "sinkhorn":
            raise ValueError(
                "The paper's geodesic cost is a Geodesic Sinkhorn coupling; "
                "set sf2m_ot_method: sinkhorn."
            )
        if not 0.0 <= time_eps < 0.5:
            raise ValueError("sf2m_time_eps must lie in [0, 0.5).")
        self.sigma = sigma
        self.ot_method = str(ot_method)
        self.ot_cost = str(ot_cost)
        self.time_eps = time_eps
        self._native_matchers = {}
        self.geodesic_sinkhorn_max_iter = int(geodesic_sinkhorn_max_iter)
        self.geodesic_sinkhorn_tol = float(geodesic_sinkhorn_tol)
        self.geodesic_geometry = None
        if self.ot_cost == "geodesic":
            if reference_populations is None:
                raise ValueError(
                    "Geodesic SF2M requires retained training populations."
                )
            self.geodesic_geometry = HeatKernelGeodesicCost(
                reference_populations,
                n_neighbors=geodesic_knn,
                heat_time=geodesic_heat_time,
                n_eigenvectors=geodesic_eigenvectors,
                max_points=geodesic_graph_max_points,
                seed=seed,
                cache_dir=geodesic_cache_dir,
                heat_epsilon=geodesic_heat_epsilon,
            )

    @property
    def geometry_summary(self):
        if self.geodesic_geometry is None:
            return None
        return dict(self.geodesic_geometry.summary)

    def _native_matcher(self, duration: float):
        duration = float(duration)
        if duration <= 0.0:
            raise ValueError("SF2M interval duration must be positive.")
        key = round(duration, 12)
        if key not in self._native_matchers:
            matcher = SchrodingerBridgeConditionalFlowMatcher(
                sigma=self.sigma * math.sqrt(duration),
                ot_method=self.ot_method,
            )
            if self.geodesic_geometry is not None:
                # Proposition 2.1 and Algorithm 5 use epsilon=2*sigma^2.
                # The local bridge sigma already includes sqrt(duration).
                matcher.ot_sampler = GeodesicSinkhornPlanSampler(
                    self.geodesic_geometry,
                    reg=2.0 * matcher.sigma**2,
                    max_iterations=self.geodesic_sinkhorn_max_iter,
                    tolerance=self.geodesic_sinkhorn_tol,
                )
            self._native_matchers[key] = matcher
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

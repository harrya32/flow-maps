"""Diagonal additive-diffusion utilities used by every benchmark method."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def resolve_prior(prior: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Resolve a named learning prior without exposing any other oracle fields."""
    if prior["kind"] == "constant":
        return {"kind": "constant", "sigma": float(prior["sigma"])}
    if prior["kind"] == "ground_truth":
        return dict(dataset["true_diffusion"])
    raise ValueError(f"Unsupported diffusion prior: {prior!r}")


def _gaussian_integral_numpy(start, end, center: float, width: float):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    coefficient = width * math.sqrt(math.pi / 2.0)
    denominator = math.sqrt(2.0) * width
    erf = np.vectorize(math.erf)
    return coefficient * (
        erf((end - center) / denominator) - erf((start - center) / denominator)
    )


def integrated_variance_numpy(start, end, prior: dict[str, Any]) -> np.ndarray:
    """Return coordinate-wise ``integral_start^end G(t)^2 dt``."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    duration = end - start
    if np.any(duration < -1e-12):
        raise ValueError("Diffusion interval has negative duration.")
    if prior["kind"] == "constant":
        return np.repeat((float(prior["sigma"]) ** 2 * duration)[..., None], 2, axis=-1)
    progress = float(prior["progress_diffusion"]) ** 2 * duration
    phenotype = float(prior["diffusion_floor"]) ** 2 * duration
    for prefix in ("first", "second"):
        phenotype = phenotype + float(prior[f"{prefix}_diffusion_peak"]) ** 2 * _gaussian_integral_numpy(
            start,
            end,
            float(prior[f"{prefix}_branch_time"]),
            float(prior[f"{prefix}_diffusion_width"]),
        )
    return np.stack([progress, phenotype], axis=-1)


def instantaneous_variance_numpy(time, prior: dict[str, Any]) -> np.ndarray:
    time = np.asarray(time, dtype=np.float64)
    if prior["kind"] == "constant":
        return np.full((*time.shape, 2), float(prior["sigma"]) ** 2, dtype=np.float64)
    progress = np.full_like(time, float(prior["progress_diffusion"]) ** 2)
    phenotype = np.full_like(time, float(prior["diffusion_floor"]) ** 2)
    for prefix in ("first", "second"):
        peak = float(prior[f"{prefix}_diffusion_peak"])
        center = float(prior[f"{prefix}_branch_time"])
        width = float(prior[f"{prefix}_diffusion_width"])
        phenotype = phenotype + peak**2 * np.exp(-0.5 * ((time - center) / width) ** 2)
    return np.stack([progress, phenotype], axis=-1)


def _gaussian_integral_torch(start, end, center: float, width: float):
    coefficient = width * math.sqrt(math.pi / 2.0)
    denominator = math.sqrt(2.0) * width
    return coefficient * (
        torch.erf((end - center) / denominator)
        - torch.erf((start - center) / denominator)
    )


def integrated_variance_torch(start, end, prior: dict[str, Any]) -> torch.Tensor:
    duration = end - start
    if prior["kind"] == "constant":
        return (float(prior["sigma"]) ** 2 * duration).expand(*duration.shape[:-1], 2)
    progress = float(prior["progress_diffusion"]) ** 2 * duration
    phenotype = float(prior["diffusion_floor"]) ** 2 * duration
    for prefix in ("first", "second"):
        phenotype = phenotype + float(prior[f"{prefix}_diffusion_peak"]) ** 2 * _gaussian_integral_torch(
            start,
            end,
            float(prior[f"{prefix}_branch_time"]),
            float(prior[f"{prefix}_diffusion_width"]),
        )
    return torch.cat([progress, phenotype], dim=-1)


def instantaneous_variance_torch(time, prior: dict[str, Any]) -> torch.Tensor:
    if prior["kind"] == "constant":
        return torch.full(
            (*time.shape[:-1], 2),
            float(prior["sigma"]) ** 2,
            dtype=time.dtype,
            device=time.device,
        )
    progress = torch.full_like(time, float(prior["progress_diffusion"]) ** 2)
    phenotype = torch.full_like(time, float(prior["diffusion_floor"]) ** 2)
    for prefix in ("first", "second"):
        peak = float(prior[f"{prefix}_diffusion_peak"])
        center = float(prior[f"{prefix}_branch_time"])
        width = float(prior[f"{prefix}_diffusion_width"])
        phenotype = phenotype + peak**2 * torch.exp(-0.5 * ((time - center) / width) ** 2)
    return torch.cat([progress, phenotype], dim=-1)

"""Thin adapters around the authors' original vector DSBM implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


def import_original_dsbm(dsbm_dir: str):
    """Load ``DSBM-Gaussian.py`` without copying or modifying upstream code."""
    root = Path(dsbm_dir).expanduser().resolve()
    source = root / "DSBM-Gaussian.py"
    if not source.is_file():
        raise FileNotFoundError(
            "Could not find the original DSBM vector implementation at "
            f"{source}. Set --dsbm_dir or DSBM_DIR to the dsbm-pytorch checkout."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:12]
    module_name = f"_flowmaps_original_dsbm_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not construct an import spec for {source}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        sys.modules.pop(module_name, None)
        raise ModuleNotFoundError(
            "Loading the original DSBM implementation requires both the "
            "flow-maps dependencies and the dependencies listed in the "
            f"upstream bridge.def. Missing module: {exc.name!r}."
        ) from exc
    for required in ("ScoreNetwork", "DSBM", "train_dsbm"):
        if not hasattr(module, required):
            raise AttributeError(
                f"The DSBM checkout at {source} does not define {required}."
            )
    return module


def activation_factory(name: str):
    activations = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    key = str(name).lower()
    if key not in activations:
        raise ValueError(
            f"Unknown DSBM activation {name!r}; choose from {sorted(activations)}."
        )
    return activations[key]


def make_original_dsbm(
    upstream,
    *,
    dim: int,
    hidden_dims: Sequence[int],
    activation: str,
    num_steps: int,
    sigma: float,
    time_eps: float,
    first_coupling: str,
    device: torch.device,
    learning_rate: float,
):
    """Instantiate the upstream MLP and DSBM classes verbatim."""
    activation_type = activation_factory(activation)
    widths = [int(value) for value in hidden_dims] + [int(dim)]

    def make_network():
        return upstream.ScoreNetwork(
            input_dim=int(dim) + 1,
            layer_widths=widths,
            activation_fn=activation_type(),
        ).to(device)

    # The original standalone implementation reads these two module globals.
    upstream.device = device
    upstream.lr = float(learning_rate)
    return upstream.DSBM(
        net_fwd=make_network(),
        net_bwd=make_network(),
        num_steps=int(num_steps),
        sig=float(sigma),
        eps=float(time_eps),
        first_coupling=str(first_coupling),
    ).to(device)


def make_generator(device: torch.device, seed: int) -> tuple[torch.Generator, bool]:
    """Create a deterministic generator, including a safe Apple-MPS fallback."""
    if device.type == "mps":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator, True
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator, False


def randn_like(
    value: torch.Tensor,
    *,
    generator: torch.Generator,
    generate_on_cpu: bool,
) -> torch.Tensor:
    if generate_on_cpu:
        return torch.randn(
            value.shape,
            dtype=value.dtype,
            generator=generator,
            device="cpu",
        ).to(value.device)
    return torch.randn(
        value.shape,
        dtype=value.dtype,
        generator=generator,
        device=value.device,
    )


@torch.no_grad()
def sample_local_forward(
    bridge: nn.Module,
    source: torch.Tensor,
    *,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    """Euler-Maruyama sampling equivalent to upstream ``sample_sde(..., 'f')``."""
    n_steps = max(1, int(n_steps))
    dt = 1.0 / float(n_steps)
    generator, generate_on_cpu = make_generator(source.device, seed)
    x = source
    for step in range(n_steps):
        t = torch.full(
            (x.shape[0], 1),
            float(step) / float(n_steps),
            dtype=x.dtype,
            device=x.device,
        )
        x = x + bridge.net_fwd(x, t) * dt
        if float(bridge.sig) > 0.0:
            x = x + float(bridge.sig) * (dt**0.5) * randn_like(
                x,
                generator=generator,
                generate_on_cpu=generate_on_cpu,
            )
    return x


class PiecewiseDSBM(nn.Module):
    """Compose independently fitted adjacent-marginal DSBM bridges."""

    evaluation_sampler = "dsbm_euler_maruyama"

    def __init__(
        self,
        bridges: Sequence[nn.Module],
        *,
        timepoints: Sequence[str],
        times: Sequence[float],
        center: torch.Tensor,
        scale: float,
    ):
        super().__init__()
        if len(bridges) + 1 != len(times):
            raise ValueError("Piecewise DSBM needs one bridge per adjacent time pair.")
        if len(timepoints) != len(times):
            raise ValueError("DSBM timepoint names and numeric times must align.")
        if any(float(right) <= float(left) for left, right in zip(times[:-1], times[1:])):
            raise ValueError("DSBM segment times must be strictly increasing.")
        self.bridges = nn.ModuleList(bridges)
        self.timepoints = tuple(str(value) for value in timepoints)
        self.segment_times = tuple(float(value) for value in times)
        self.register_buffer("normalization_center", center.detach().float().clone())
        self.register_buffer(
            "normalization_scale",
            torch.tensor(float(scale), dtype=torch.float32),
        )

    @property
    def flow_net(self):
        # Shared benchmark callbacks use this only to toggle train/eval mode.
        return self

    @property
    def device(self) -> torch.device:
        return self.normalization_center.device

    def _segment_index(self, time_value: float) -> int:
        tolerance = 1e-7
        if time_value < self.segment_times[0] - tolerance:
            raise ValueError(
                f"Evaluation starts before the first DSBM time {self.segment_times[0]}."
            )
        if time_value > self.segment_times[-1] + tolerance:
            raise ValueError(
                f"Evaluation ends after the last DSBM time {self.segment_times[-1]}."
            )
        for index, end in enumerate(self.segment_times[1:]):
            if time_value < end - tolerance:
                return index
        return len(self.bridges) - 1

    def _snap_to_boundary(self, time_value: float) -> float:
        for boundary in self.segment_times:
            if abs(float(time_value) - boundary) <= 1e-7:
                return boundary
        return float(time_value)

    @torch.no_grad()
    def evaluation_trajectory(
        self,
        x0: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
        n_steps: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Sample on a uniform global clock, splitting steps at bridge boundaries."""
        start_time = self._snap_to_boundary(float(start_time))
        end_time = self._snap_to_boundary(float(end_time))
        if end_time < start_time:
            raise ValueError("DSBM evaluation end_time precedes start_time.")
        n_steps = max(1, int(n_steps))
        generator, generate_on_cpu = make_generator(
            x0.device,
            0 if seed is None else int(seed),
        )
        scale = self.normalization_scale.to(dtype=x0.dtype)
        center = self.normalization_center.to(dtype=x0.dtype)
        x = (x0 - center) / scale
        path = [x0.clone()]
        global_grid = torch.linspace(
            start_time,
            end_time,
            n_steps + 1,
            dtype=torch.float64,
        ).tolist()

        for left, right in zip(global_grid[:-1], global_grid[1:]):
            current = self._snap_to_boundary(float(left))
            right = self._snap_to_boundary(float(right))
            while current < right - 1e-12:
                segment_index = self._segment_index(current)
                segment_start = self.segment_times[segment_index]
                segment_end = self.segment_times[segment_index + 1]
                substep_end = min(right, segment_end)
                duration = segment_end - segment_start
                local_time = min(max((current - segment_start) / duration, 0.0), 1.0)
                local_dt = (substep_end - current) / duration
                bridge = self.bridges[segment_index]
                t = torch.full(
                    (x.shape[0], 1),
                    local_time,
                    dtype=x.dtype,
                    device=x.device,
                )
                x = x + bridge.net_fwd(x, t) * local_dt
                if float(bridge.sig) > 0.0 and local_dt > 0.0:
                    x = x + float(bridge.sig) * (local_dt**0.5) * randn_like(
                        x,
                        generator=generator,
                        generate_on_cpu=generate_on_cpu,
                    )
                current = substep_end
            path.append(x * scale + center)
        return torch.stack(path, dim=0)

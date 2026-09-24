"""Shared OT bridges, neural fields, SF2M sampling, and direct strong maps."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
from torch import nn

from .dataset import CLASS_NAMES
from .diffusion import (
    instantaneous_variance_torch,
    integrated_variance_numpy,
    integrated_variance_torch,
)


def valid_transition_matrix(device=None, dtype=torch.float32):
    names = list(CLASS_NAMES)
    allowed = {
        "root": set(names),
        "lower": {"lower"},
        "upper": {"upper", "upper_down", "upper_up"},
        "upper_down": {"upper_down"},
        "upper_up": {"upper_up"},
    }
    matrix = torch.zeros((len(names), len(names)), device=device, dtype=dtype)
    for source, source_name in enumerate(names):
        for target_name in allowed[source_name]:
            matrix[source, names.index(target_name)] = 1.0
    return matrix


class FourierTimeEmbedding(nn.Module):
    def __init__(self, frequencies: int, final_time: float):
        super().__init__()
        values = (2.0 ** torch.arange(frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", values, persistent=False)
        self.final_time = float(final_time)

    @property
    def output_dim(self):
        return 1 + 2 * len(self.frequencies)

    def forward(self, time):
        normalized = time / self.final_time
        angles = normalized * self.frequencies[None, :]
        return torch.cat([normalized, torch.sin(angles), torch.cos(angles)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.first = nn.Linear(width, width)
        self.second = nn.Linear(width, width)

    def forward(self, inputs):
        return inputs + self.second(torch.nn.functional.silu(self.first(inputs))) / math.sqrt(2.0)


class VelocityField(nn.Module):
    def __init__(self, width=96, depth=3, time_frequencies=4, final_time=2.0):
        super().__init__()
        self.embedding = FourierTimeEmbedding(int(time_frequencies), float(final_time))
        self.input = nn.Linear(2 + self.embedding.output_dim, int(width))
        self.blocks = nn.ModuleList([ResidualBlock(int(width)) for _ in range(int(depth))])
        self.output = nn.Linear(int(width), 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, time, position):
        hidden = torch.nn.functional.silu(self.input(torch.cat([position, self.embedding(time)], dim=1)))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(torch.nn.functional.silu(hidden))


class Fields(nn.Module):
    """Velocity and score networks for the selected additive diffusion."""

    def __init__(self, mean, scale, prior, architecture, final_time=2.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.prior = dict(prior)
        self.architecture = dict(architecture)
        self.final_time = float(final_time)
        arguments = {**architecture, "final_time": final_time}
        self.velocity = VelocityField(**arguments)
        self.score = VelocityField(**arguments)

    def normalize(self, points):
        return (points - self.mean) / self.scale

    def physical(self, points):
        return points * self.scale + self.mean

    def drift(self, time, position):
        variance = instantaneous_variance_torch(time, self.prior) / self.scale**2
        return self.velocity(time, position) + 0.5 * variance * self.score(time, position)


@dataclass(frozen=True)
class BridgeStatistics:
    mean: torch.Tensor
    variance: torch.Tensor
    mean_derivative: torch.Tensor
    variance_derivative: torch.Tensor


def diagonal_bridge_statistics(source, target, time, left, right, prior, scale):
    scale = torch.as_tensor(scale, dtype=source.dtype, device=source.device).reshape(1, 2)
    interval_start = torch.full_like(time, float(left))
    interval_end = torch.full_like(time, float(right))
    cumulative = integrated_variance_torch(interval_start, time, prior) / scale**2
    total = integrated_variance_torch(interval_start, interval_end, prior) / scale**2
    instantaneous = instantaneous_variance_torch(time, prior) / scale**2
    fraction = cumulative / total.clamp_min(1e-12)
    mean = source + fraction * (target - source)
    variance = torch.clamp(cumulative - cumulative**2 / total.clamp_min(1e-12), min=0.0)
    derivative = instantaneous / total.clamp_min(1e-12) * (target - source)
    variance_derivative = instantaneous * (1.0 - 2.0 * fraction)
    return BridgeStatistics(mean, variance, derivative, variance_derivative)


def make_ot_plan(source, target, prior, left, right):
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("Exact OT requires equal source and target arrays of shape (cells, dimensions).")
    variance = integrated_variance_numpy(left, right, prior)
    normalized_source = (source - source.mean(0)) / np.sqrt(variance)
    normalized_target = (target - target.mean(0)) / np.sqrt(variance)
    cost = 0.5 * cdist(normalized_source, normalized_target, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(cost)
    if not np.array_equal(rows, np.arange(len(source))):
        raise RuntimeError("Unexpected non-canonical OT assignment.")
    plan = np.zeros((len(source), len(target)), dtype=np.float64)
    plan[rows, columns] = 1.0 / len(source)
    marginal_error = max(
        float(np.max(np.abs(plan.sum(0) - 1.0 / len(target)))),
        float(np.max(np.abs(plan.sum(1) - 1.0 / len(source)))),
    )
    if marginal_error > 1e-12:
        raise RuntimeError(f"OT plan failed its balanced-marginal check: {marginal_error}")
    return plan, {
        "cost": float(np.sum(plan * cost)),
        "entropy": float(np.log(len(source))),
        "max_marginal_error": marginal_error,
        "reference_integrated_variance": variance.tolist(),
    }


class PairBank:
    def __init__(self, times, points, plans, labels=None):
        self.times = np.asarray(times, dtype=np.float64)
        self.points = [np.asarray(values, dtype=np.float32) for values in points]
        self.labels = (
            None
            if labels is None
            else [np.asarray(values, dtype=np.int64) for values in labels]
        )
        self.cdfs = [np.cumsum(plan.reshape(-1), dtype=np.float64) for plan in plans]
        for cdf in self.cdfs:
            cdf /= cdf[-1]

    def pair(self, interval, count, rng, device, *, return_labels=False):
        target_size = len(self.points[interval + 1])
        flat = np.searchsorted(self.cdfs[interval], rng.random(count), side="right")
        flat = np.minimum(flat, len(self.cdfs[interval]) - 1)
        source_indices = flat // target_size
        target_indices = flat % target_size
        source = self.points[interval][source_indices]
        target = self.points[interval + 1][target_indices]
        result = (
            torch.as_tensor(source, dtype=torch.float32, device=device),
            torch.as_tensor(target, dtype=torch.float32, device=device),
        )
        if not return_labels:
            return result
        if self.labels is None:
            raise ValueError("Pair labels were requested from an unlabeled PairBank.")
        return result + (
            torch.as_tensor(
                self.labels[interval][source_indices],
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                self.labels[interval + 1][target_indices],
                dtype=torch.long,
                device=device,
            ),
        )

    def bridge_interval(self, model, interval, count, rng, low=None, high=None):
        device = model.mean.device
        source, target = self.pair(interval, count, rng, device)
        left, right = self.times[interval : interval + 2]
        low = left + 0.01 * (right - left) if low is None else float(low)
        high = right - 0.01 * (right - left) if high is None else float(high)
        time = torch.as_tensor(rng.uniform(low, high, (count, 1)), dtype=torch.float32, device=device)
        noise = torch.as_tensor(rng.normal(size=(count, 2)), dtype=torch.float32, device=device)
        stats = diagonal_bridge_statistics(
            model.normalize(source), model.normalize(target), time, left, right, model.prior, model.scale
        )
        std = stats.variance.sqrt().clamp_min(1e-8)
        state = stats.mean + std * noise
        target_velocity = stats.mean_derivative + 0.5 * stats.variance_derivative / std * noise
        return time, state, target_velocity, std, noise

    def bridge(self, model, count, rng):
        intervals = len(self.times) - 1
        counts = [count // intervals] * intervals
        for index in range(count % intervals):
            counts[index] += 1
        chunks = [self.bridge_interval(model, index, n, rng) for index, n in enumerate(counts) if n]
        return tuple(torch.cat([chunk[field] for chunk in chunks]) for field in range(5))


def bridge_losses(model, batch):
    time, state, target, std, noise = batch
    velocity = (model.velocity(time, state) - target).square().mean()
    score = (std * model.score(time, state) + noise).square().mean()
    return velocity, score


def prepare_learning_inputs(training_file, classifier_dir, output, settings, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = np.load(training_file, allow_pickle=False)
    splits = np.load(Path(classifier_dir) / "splits.npz", allow_pickle=False)
    rng = np.random.default_rng(int(seed) + 801)
    arrays = {"times": data["times"]}
    for split, count_key in (
        ("train", "train_cells_per_time"),
        ("validation", "validation_cells_per_time"),
    ):
        count = int(settings[count_key])
        ids = []
        for time_index in range(len(data["times"])):
            available = splits[f"{split}_{time_index}"]
            if count > len(available):
                raise ValueError(f"Requested {count} {split} cells but only {len(available)} are available.")
            ids.append(np.sort(rng.choice(available, count, replace=False)))
        ids = np.stack(ids)
        arrays[f"{split}_indices"] = ids
        arrays[f"{split}_points"] = np.stack(
            [data["points"][time_index, ids[time_index]] for time_index in range(len(ids))]
        )
        arrays[f"{split}_labels"] = np.stack(
            [data["labels"][time_index, ids[time_index]] for time_index in range(len(ids))]
        )
    np.savez_compressed(output / "learning_data.npz", **arrays)


def build_pair_banks(learning_file, prior, output):
    data = np.load(learning_file, allow_pickle=False)
    times = data["times"]
    banks, plans_to_save, diagnostics = {}, {}, {}
    for split in ("train", "validation"):
        points = data[f"{split}_points"]
        labels = data[f"{split}_labels"]
        plans = []
        for interval in range(len(times) - 1):
            plan, values = make_ot_plan(points[interval], points[interval + 1], prior, times[interval], times[interval + 1])
            plans.append(plan)
            plans_to_save[f"{split}_{interval}"] = plan
            allowed = valid_transition_matrix().cpu().numpy().astype(bool)
            forbidden = ~allowed[labels[interval, :, None], labels[interval + 1, None, :]]
            values["hard_label_forbidden_mass"] = float(np.sum(plan * forbidden))
            diagnostics[f"{split}_{interval}"] = values
        banks[split] = PairBank(times, points, plans, labels)
    np.savez_compressed(Path(output) / "couplings.npz", **plans_to_save)
    (Path(output) / "coupling_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    return banks, data


def load_pair_banks(learning_file, coupling_file):
    data = np.load(learning_file, allow_pickle=False)
    plans = np.load(coupling_file, allow_pickle=False)
    times = data["times"]
    return {
        split: PairBank(
            times,
            data[f"{split}_points"],
            [plans[f"{split}_{interval}"] for interval in range(len(times) - 1)],
            data[f"{split}_labels"],
        )
        for split in ("train", "validation")
    }


def save_fields(model, path, metadata):
    torch.save(
        {
            "model_type": "sf2m",
            "state": model.state_dict(),
            "mean": model.mean.detach().cpu().numpy(),
            "scale": model.scale.detach().cpu().numpy(),
            "prior": model.prior,
            "architecture": model.architecture,
            "final_time": model.final_time,
            "metadata": metadata,
        },
        path,
    )


def load_fields(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = Fields(saved["mean"], saved["scale"], saved["prior"], saved["architecture"], saved["final_time"])
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


@torch.no_grad()
def sample_sde(model, initial, start, times, steps_per_unit, seed):
    times = np.asarray(times, dtype=np.float64)
    if int(steps_per_unit) <= 0 or np.any(np.diff(times) < 0) or len(times) == 0 or times[0] < start - 1e-10:
        raise ValueError("SDE sampling times/rate are invalid.")
    device = model.mean.device
    state = model.normalize(torch.as_tensor(initial, dtype=torch.float32, device=device))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    current, output = float(start), []
    for target in times:
        count = max(0, int(np.ceil((target - current) * steps_per_unit - 1e-9)))
        if count:
            dt = (target - current) / count
            for index in range(count):
                time = torch.full((len(state), 1), current + index * dt, dtype=state.dtype, device=device)
                variance = instantaneous_variance_torch(time, model.prior) / model.scale**2
                noise = torch.randn(state.shape, generator=generator, dtype=state.dtype).to(device)
                state = state + model.drift(time, state) * dt + (variance * dt).sqrt() * noise
        output.append(model.physical(state).cpu().numpy().copy())
        current = float(target)
    return np.stack(output, axis=1)


class NoiseResponse(nn.Module):
    def __init__(self, bins, width, depth, time_frequencies, final_time):
        super().__init__()
        self.bins = int(bins)
        self.embedding = FourierTimeEmbedding(int(time_frequencies), float(final_time))
        self.input = nn.Linear(2 + 2 * self.embedding.output_dim + 2 + 2 * self.bins, int(width))
        self.blocks = nn.ModuleList([ResidualBlock(int(width)) for _ in range(int(depth))])
        self.output = nn.Linear(int(width), 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, position, start, end, standardized_noise):
        horizon = (end - start).clamp_min(1e-12)
        features = torch.cat(
            [
                position,
                self.embedding(start),
                self.embedding(end),
                horizon / 2.0,
                (horizon / 2.0).log(),
                standardized_noise.flatten(1),
            ],
            dim=1,
        )
        hidden = torch.nn.functional.silu(self.input(features))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(torch.nn.functional.silu(hidden))


class DirectStrongMap(Fields):
    def __init__(self, mean, scale, prior, architecture, final_time, bins=8, anchor_step=0.0):
        super().__init__(mean, scale, prior, architecture, final_time)
        self.bins = int(bins)
        self.anchor_step = float(anchor_step)
        self.finite = NoiseResponse(bins, final_time=final_time, **architecture)

    def response(self, position, start, end, atoms, variances, *, detach_local=False):
        local = self.drift(start, position)
        if detach_local:
            local = local.detach()
        standardized = atoms / variances.clamp_min(1e-20).sqrt()
        horizon = (end - start).clamp_min(0.0)
        active = (horizon > self.anchor_step * (1.0 + 1e-4)).to(horizon.dtype)
        return local + active * horizon.sqrt() * self.finite(position, start, end, standardized)

    def forward(self, position, start, end, atoms, variances):
        return position + atoms.sum(1) + (end - start) * self.response(
            position, start, end, atoms, variances
        )


def atom_variances(model, start, end, bins=None, *, shared_interval=False):
    bins = model.bins if bins is None else int(bins)
    # Match the reference implementation: integrate in float64 on CPU, then
    # cast once.  This avoids small negative/cancellation errors on fine grids.
    left = (start[:1] if shared_interval else start).detach().cpu().double()
    right = (end[:1] if shared_interval else end).detach().cpu().double()
    if torch.any(right < left):
        raise ValueError("A noise interval cannot have negative duration.")
    unit = torch.linspace(0, 1, bins + 1, dtype=torch.float64)[None, :, None]
    knots = left[:, None, :] + (right - left)[:, None, :] * unit
    variance = integrated_variance_torch(knots[:, :-1], knots[:, 1:], model.prior)
    variance = variance / model.scale.detach().cpu().double()[None, None, :] ** 2
    if torch.any(variance < -1e-12):
        raise FloatingPointError("Integrated covariance is negative.")
    variance = variance.clamp_min(0.0).to(device=start.device, dtype=start.dtype)
    if shared_interval:
        variance = variance.expand(len(start), -1, -1)
    return variance


def draw_atoms(model, start, end, generator, bins=None, *, shared_interval=False):
    variance = atom_variances(model, start, end, bins, shared_interval=shared_interval)
    noise = torch.randn(variance.shape, generator=generator, dtype=variance.dtype).to(variance.device)
    return noise * variance.sqrt(), variance


def coarsen(atoms, variances, bins):
    if atoms.shape[1] % bins:
        raise ValueError("Noise atoms do not divide into complete parent bins.")
    shape = (len(atoms), bins, atoms.shape[1] // bins, 2)
    return atoms.reshape(shape).sum(2), variances.reshape(shape).sum(2)


@dataclass
class CompositionBatch:
    start: torch.Tensor
    end: torch.Tensor
    state: torch.Tensor
    atoms: torch.Tensor
    variances: torch.Tensor


def composition_batch(model, bank, count, horizons, rng, generator):
    chunks = []
    intervals = len(bank.times) - 1
    counts = [count // intervals] * intervals
    for index in range(count % intervals):
        counts[index] += 1
    final_time = float(bank.times[-1])
    for interval, n in enumerate(counts):
        if not n:
            continue
        horizon = torch.as_tensor(rng.choice(horizons, size=(n, 1)), dtype=torch.float32, device=model.mean.device)
        left, right = bank.times[interval : interval + 2]
        high = np.minimum(right - 0.01 * (right - left), final_time - horizon.cpu().numpy())
        low = np.minimum(left + 0.01 * (right - left), high)
        start = torch.as_tensor(rng.uniform(low, high), dtype=torch.float32, device=model.mean.device)
        source, target = bank.pair(interval, n, rng, model.mean.device)
        stats = diagonal_bridge_statistics(
            model.normalize(source), model.normalize(target), start, left, right, model.prior, model.scale
        )
        noise = torch.as_tensor(rng.normal(size=(n, 2)), dtype=torch.float32, device=model.mean.device)
        state = stats.mean + stats.variance.clamp_min(0.0).sqrt() * noise
        chunks.append((start, start + horizon, state))
    start, end, state = [torch.cat([chunk[index] for chunk in chunks]) for index in range(3)]
    atoms, variances = draw_atoms(model, start, end, generator, 2 * model.bins)
    return CompositionBatch(start, end, state, atoms, variances)


def consistency_loss(model, ema, batch):
    middle = (batch.start + batch.end) / 2.0
    parent_atoms, parent_variances = coarsen(batch.atoms, batch.variances, model.bins)
    prediction = model.response(
        batch.state,
        batch.start,
        batch.end,
        parent_atoms,
        parent_variances,
        detach_local=True,
    )
    with torch.no_grad():
        left_atoms, right_atoms = batch.atoms[:, : model.bins], batch.atoms[:, model.bins :]
        left_variances = batch.variances[:, : model.bins]
        right_variances = batch.variances[:, model.bins :]
        first = ema.response(batch.state, batch.start, middle, left_atoms, left_variances)
        midpoint = batch.state + left_atoms.sum(1) + (middle - batch.start) * first
        second = ema.response(midpoint, middle, batch.end, right_atoms, right_variances)
        target = 0.5 * (first + second)
    return (prediction - target).square().mean()


def update_ema(ema, model, decay):
    with torch.no_grad():
        targets = dict(ema.named_parameters())
        for name, source in model.named_parameters():
            if name.startswith("finite."):
                targets[name].mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                targets[name].copy_(source)


def map_fate_batch(model, bank, count, steps, rng, generator, settings):
    interval = len(bank.times) - 2
    start_range = settings.get("fate_start_range", [bank.times[interval], bank.times[interval] + 0.45])
    start, state, _, _, _ = bank.bridge_interval(
        model, interval, count, rng, float(start_range[0]), float(start_range[1])
    )
    minimum = float(settings.get("fate_minimum_horizon", 0.5))
    end_max = float(settings.get("fate_end_time", bank.times[-1]))
    available = end_max - start - minimum
    if torch.any(available < 0):
        raise ValueError("Fate start range and minimum horizon exceed the final time.")
    end = start + minimum + torch.as_tensor(
        rng.random((count, 1)), dtype=torch.float32, device=model.mean.device
    ) * available
    pieces = []
    for index in range(int(steps)):
        left = start + (end - start) * index / steps
        right = start + (end - start) * (index + 1) / steps
        pieces.append(draw_atoms(model, left, right, generator))
    return start, end, state, pieces


def map_fate_loss(model, classifier, batch):
    start, end, state, pieces = batch
    initial_state = state
    for index, (atoms, variance) in enumerate(pieces):
        left = start + (end - start) * index / len(pieces)
        right = start + (end - start) * (index + 1) / len(pieces)
        state = model(state, left, right, atoms, variance)
    return endpoint_lineage_nll(model, classifier, initial_state, state)


def endpoint_lineage_nll(
    model,
    classifier,
    initial_state,
    final_state,
):
    """NLL of the allowed classifier transition between two map states."""
    final = classifier(model.physical(final_state))
    allowed = valid_transition_matrix(device=final_state.device).bool()
    initial = classifier(model.physical(initial_state)).detach()
    joint = initial[:, :, None] + final[:, None, :]
    return -torch.logsumexp(joint.masked_fill(~allowed, -torch.inf).flatten(1), dim=1).mean()


def save_map(model, path, metadata):
    torch.save(
        {
            "model_type": "direct_strong_map",
            "state": model.state_dict(),
            "mean": model.mean.detach().cpu().numpy(),
            "scale": model.scale.detach().cpu().numpy(),
            "prior": model.prior,
            "architecture": model.architecture,
            "final_time": model.final_time,
            "bins": model.bins,
            "anchor_step": model.anchor_step,
            "metadata": metadata,
        },
        path,
    )


def load_map(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = DirectStrongMap(
        saved["mean"],
        saved["scale"],
        saved["prior"],
        saved["architecture"],
        saved["final_time"],
        saved["bins"],
        saved.get("anchor_step", 0.0),
    )
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


@torch.no_grad()
def sample_map(model, initial, start, times, steps_per_unit, seed):
    times = np.asarray(times, dtype=np.float64)
    if int(steps_per_unit) <= 0 or np.any(np.diff(times) < 0) or len(times) == 0 or times[0] < start - 1e-10:
        raise ValueError("Strong-map sampling times/rate are invalid.")
    device = model.mean.device
    state = model.normalize(torch.as_tensor(initial, dtype=torch.float32, device=device))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    current, output = float(start), []
    for target in times:
        count = max(0, int(np.ceil((target - current) * steps_per_unit - 1e-9)))
        for index in range(count):
            left = torch.full(
                (len(state), 1), current + (target - current) * index / count, dtype=state.dtype, device=device
            )
            right = torch.full(
                (len(state), 1), current + (target - current) * (index + 1) / count, dtype=state.dtype, device=device
            )
            atoms, variances = draw_atoms(model, left, right, generator, shared_interval=True)
            state = model(state, left, right, atoms, variances)
        output.append(model.physical(state).cpu().numpy().copy())
        current = float(target)
    return np.stack(output, axis=1)

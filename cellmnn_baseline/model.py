"""The upstream encoder/optimizer with an equivalent economical ODE decoder."""

from __future__ import annotations

import numpy as np
import torch

from lib.model import CellMNN


class BenchmarkCellMNN(CellMNN):
    def __init__(self, *, retained_times, kinetic_dt=0.1, **kwargs):
        super().__init__(
            days_w_data=list(retained_times),
            skip_day=float(retained_times[0]),
            prev_day=float(retained_times[0]),
            train_on_skip_day=True,
            **kwargs,
        )
        # Keep the upstream uniform kinetic quadrature. Add exact supervision
        # times separately: thirds on the equal clock need not lie on this grid.
        base = np.arange(
            retained_times[0], retained_times[-1] + kinetic_dt / 2, kinetic_dt
        )
        grid = list(base)
        for value in retained_times:
            nearest = int(np.argmin(np.abs(np.asarray(grid) - value)))
            if abs(grid[nearest] - value) < 1e-7:
                grid[nearest] = value
            else:
                grid.append(value)
        grid = np.sort(grid)
        self.register_buffer("loss_times", torch.as_tensor(grid, dtype=torch.float32))
        self.register_buffer(
            "population_indices",
            torch.tensor(
                [int(np.argmin(np.abs(grid - value))) for value in retained_times]
            ),
        )
        self.register_buffer(
            "kinetic_indices",
            torch.tensor([int(np.argmin(np.abs(grid - value))) for value in base]),
        )

    def forward(self, x_t, t, decode_ts):
        P, eigenvalues = self.encode(x_t, t)
        z_t = torch.bmm(x_t, P[:, 0])
        # Exactly exp(diag(eigenvalues) * dt), without B*T dense D*D matrices.
        z_traj = z_t * torch.exp(eigenvalues * (decode_ts - t))
        z_dot = eigenvalues * z_traj
        # PyTorch 2.3's batched MPS inverse retains native GPU workspace across
        # calls even after empty_cache. CPU inversion preserves the same gradient.
        inverse_device = torch.device("cpu") if P.device.type == "mps" else P.device
        P_inv = torch.linalg.inv(P.to(inverse_device)).to(P.device)
        x_traj = torch.bmm(z_traj, P_inv[:, 0])
        x_dot = torch.bmm(z_dot, P_inv[:, 0])
        return (
            x_traj,
            P_inv.expand(-1, decode_ts.shape[1], -1, -1),
            eigenvalues,
            P,
            x_dot,
        )

    def loss_terms(self, batch):
        """Upstream training_step's discounted MMD + kinetic + inverse loss."""
        x_t, t, population, population_times = batch
        batch_n, n_days, _ = population.shape
        if n_days != len(self.days_w_data):
            raise ValueError("Population loss requires exactly the retained days.")
        decode_ts = self.loss_times[None, :, None].expand(batch_n, -1, -1)
        predictions, _, _, P, velocity = self(x_t, t, decode_ts)
        # Variable-size indexing/cdist graphs exhaust the PyTorch 2.3 MPS
        # allocator across batches. Reuse the upstream loss on CPU for MPS;
        # differentiable copies retain gradients into the MPS encoder/decoder.
        loss_device = torch.device("cpu") if x_t.device.type == "mps" else x_t.device
        predictions = predictions.to(loss_device).index_select(
            1, self.population_indices.to(loss_device)
        )
        population = population.to(loss_device)
        future = population_times.to(loss_device).squeeze(-1) > t.to(
            loss_device
        ).squeeze(-1)
        ranks = torch.cumsum(future.to(torch.int), dim=1)
        mmd = predictions.sum() * 0.0
        for day in range(n_days):
            for rank in range(1, n_days):
                mask = future[:, day] & (ranks[:, day] == rank)
                if mask.any():
                    mmd = mmd + self.gamma ** (rank - 1) * self.loss_fn(
                        predictions[mask, day], population[mask, day]
                    )
        mmd = mmd.to(x_t.device)
        kinetic = velocity.index_select(1, self.kinetic_indices).square().mean()
        # MPS does not implement det. Use an explicit autograd-preserving CPU
        # operation rather than dispatching through its implicit fallback.
        inverse = (1.0 / (torch.abs(torch.det(P.to(loss_device))) + 1e-4)).mean()
        inverse = inverse.to(x_t.device)
        loss = mmd + self.lambda_kinetic * kinetic + inverse
        return {
            "loss": loss,
            "mmd_loss_future": mmd,
            "kinetic_loss": kinetic,
            "invertibility_loss": inverse,
        }


class CellMNNSampler:
    """Raw-PCA analytical, composed analytical, and instantaneous Euler paths."""

    def __init__(self, model, scaler, *, time_origin, time_scale, batch_size=1024):
        self.model = model
        self.device = next(model.parameters()).device
        self.scale = 1.0 if scaler is None else float(scaler.scale_)
        self.time_origin, self.time_scale = float(time_origin), float(time_scale)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0 or self.time_scale <= 0:
            raise ValueError("Sampling batch size and time scale must be positive.")

    @torch.no_grad()
    def sample(
        self,
        source,
        *,
        start_time=0.0,
        end_time=1.0,
        n_steps=50,
        sampler="direct",
        paths=False,
    ):
        if sampler not in ("direct", "flowmap", "euler"):
            raise ValueError(f"Unknown Cell-MNN sampler: {sampler}.")
        if int(n_steps) <= 0 or not 0.0 <= start_time <= end_time <= 1.0:
            raise ValueError(
                "Sampling requires positive steps and 0 <= start <= end <= 1."
            )
        source = np.asarray(source, dtype=np.float32)
        if (
            source.ndim != 2
            or len(source) == 0
            or source.shape[1] != self.model.latent_dim
        ):
            raise ValueError(
                "Sampling needs a non-empty population in the model PCA space."
            )
        steps = 1 if sampler == "direct" and not paths else int(n_steps)
        times = self.time_origin + self.time_scale * np.linspace(
            start_time, end_time, steps + 1
        )
        was_training = self.model.training
        self.model.eval()
        chunks = []
        try:
            for start in range(0, len(source), self.batch_size):
                raw = source[start : start + self.batch_size]
                x0 = torch.as_tensor(raw, device=self.device) / self.scale
                if sampler == "direct":
                    anchor = torch.full(
                        (len(x0), 1, 1), float(times[0]), device=self.device
                    )
                    decode = torch.tensor(
                        times[1:], dtype=torch.float32, device=self.device
                    )
                    decode = decode[None, :, None].expand(len(x0), -1, -1)
                    predicted = self.model(x0[:, None], anchor, decode)[0] * self.scale
                    prediction = predicted.cpu().numpy()
                    result = (
                        np.concatenate([raw[:, None], prediction], axis=1)
                        if paths
                        else prediction[:, -1]
                    )
                else:
                    x, positions = x0, [raw]
                    for left, right in zip(times[:-1], times[1:]):
                        t = torch.full((len(x), 1, 1), float(left), device=self.device)
                        target = (
                            t
                            if sampler == "euler"
                            else torch.full_like(t, float(right))
                        )
                        decoded = self.model(x[:, None], t, target)
                        x = (
                            x + float(right - left) * decoded[4][:, 0]
                            if sampler == "euler"
                            else decoded[0][:, 0]
                        )
                        if paths:
                            positions.append((x * self.scale).cpu().numpy())
                    result = (
                        np.stack(positions, axis=1)
                        if paths
                        else (x * self.scale).cpu().numpy()
                    )
                if not np.isfinite(result).all():
                    raise FloatingPointError(
                        f"Non-finite Cell-MNN {sampler} predictions."
                    )
                chunks.append(result)
        finally:
            self.model.train(was_training)
        return np.concatenate(chunks).astype(np.float32)

"""
Nicholas M. Boffi
10/5/25

Basic class for a stochastic interpolant.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import pair_times


@dataclasses.dataclass
class Interpolant:
    """Basic class for a stochastic interpolant"""

    alpha: Callable[[float], float]
    beta: Callable[[float], float]
    alpha_dot: Callable[[float], float]
    beta_dot: Callable[[float], float]

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        return self.alpha(t) * x0 + self.beta(t) * x1

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1

    def batch_calc_It(
        self,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            return jax.vmap(lambda tt, x0i, x1i: self.calc_It(tt, x0i, x1i))(
                t, x0, x1
            )
        return jax.vmap(
            lambda tt, x0i, x1i, label_i: self.calc_It(tt, x0i, x1i, label_i)
        )(t, x0, x1, label)

    def batch_calc_It_dot(
        self,
        t: jnp.ndarray,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            return jax.vmap(lambda tt, x0i, x1i: self.calc_It_dot(tt, x0i, x1i))(
                t, x0, x1
            )
        return jax.vmap(
            lambda tt, x0i, x1i, label_i: self.calc_It_dot(tt, x0i, x1i, label_i)
        )(t, x0, x1, label)

    def __hash__(self):
        return hash((self.alpha, self.beta))

    def __eq__(self, other):
        return self.alpha == other.alpha and self.beta == other.beta


class TimeRescaledLinearInterpolant(Interpolant):
    """Linear interpolation on the absolute interval stored in each pair label."""

    def __init__(self):
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("TimeRescaledLinearInterpolant requires pair-time labels.")
        tau = pair_times.global_to_local(t, label, dtype=x0.dtype)
        return (1.0 - tau) * x0 + tau * x1

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("TimeRescaledLinearInterpolant requires pair-time labels.")
        t_start, t_end = pair_times.bounds(label, dtype=x0.dtype)
        duration = jnp.maximum(t_end - t_start, jnp.asarray(1e-8, dtype=x0.dtype))
        return (x1 - x0) / duration

    def __hash__(self):
        return hash("time_rescaled_linear")

    def __eq__(self, other):
        return isinstance(other, TimeRescaledLinearInterpolant)


class TriangleGaussianInterpolant(Interpolant):
    """Piecewise-linear triangular interpolant for 2D Gaussian endpoints."""

    def __init__(self, height: float):
        self.height = float(height)
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        base = super().calc_It(t, x0, x1, label)
        vertical = jnp.zeros_like(x0).at[1].set(
            2.0 * self.height * jnp.minimum(t, 1.0 - t)
        )
        return base + vertical

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        base_dot = super().calc_It_dot(t, x0, x1, label)
        vertical_dot = jnp.zeros_like(x0).at[1].set(
            jnp.where(t <= 0.5, 2.0 * self.height, -2.0 * self.height)
        )
        return base_dot + vertical_dot

    def __hash__(self):
        return hash(("triangle_gaussian", self.height))

    def __eq__(self, other):
        return (
            isinstance(other, TriangleGaussianInterpolant)
            and self.height == other.height
        )


class SpiralInterpolant(Interpolant):
    """Clockwise polar spiral around x0, ending at x1 above it."""

    def __init__(
        self,
        radius_a: float = 2.88,
        radius_b: float = -4.59,
        radius_c: float = 2.71,
        turns: float = 1.5,
        eps: float = 1e-6,
    ):
        self.radius_a = float(radius_a)
        self.radius_b = float(radius_b)
        self.radius_c = float(radius_c)
        self.turns = float(turns)
        self.eps = float(eps)
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def _frame(self, x0: jnp.ndarray, x1: jnp.ndarray):
        delta = x1 - x0
        distance = jnp.maximum(jnp.linalg.norm(delta), self.eps)
        e = delta / distance
        n_left = jnp.stack([-e[1], e[0]]).astype(x0.dtype)
        return distance, e, n_left

    def _radius(self, t: float) -> jnp.ndarray:
        return self.radius_a * t**3 + self.radius_b * t**2 + self.radius_c * t

    def _radius_dot(self, t: float) -> jnp.ndarray:
        return 3.0 * self.radius_a * t**2 + 2.0 * self.radius_b * t + self.radius_c

    def _angle(self, t: float) -> jnp.ndarray:
        return 2.0 * jnp.pi * self.turns * t

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        distance, e, n_left = self._frame(x0, x1)
        angle = self._angle(t)
        direction = -jnp.cos(angle) * e + jnp.sin(angle) * n_left
        return x0 + distance * self._radius(t) * direction

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        distance, e, n_left = self._frame(x0, x1)
        angle = self._angle(t)
        angle_dot = 2.0 * jnp.pi * self.turns
        direction = -jnp.cos(angle) * e + jnp.sin(angle) * n_left
        direction_dot = angle_dot * (jnp.sin(angle) * e + jnp.cos(angle) * n_left)
        return distance * (
            self._radius_dot(t) * direction + self._radius(t) * direction_dot
        )

    def __hash__(self):
        return hash(
            (
                "spiral",
                self.radius_a,
                self.radius_b,
                self.radius_c,
                self.turns,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, SpiralInterpolant)
            and self.radius_a == other.radius_a
            and self.radius_b == other.radius_b
            and self.radius_c == other.radius_c
            and self.turns == other.turns
        )


class HairpinInterpolant(Interpolant):
    """Fast-middle hairpin path with a smooth downward turn."""

    def __init__(
        self,
        out_length: float = 4.0,
        drop: float = 1.6,
        turn_start: float = 0.35,
        turn_end: float = 0.65,
        tangent_speed: float = 0.0,
        endpoint_tangent_speed: float = 0.0,
        speed_scale: float = 6.0,
    ):
        self.out_length = float(out_length)
        self.drop = float(drop)
        self.turn_start = float(turn_start)
        self.turn_end = float(turn_end)
        self.tangent_speed = float(tangent_speed)
        self.endpoint_tangent_speed = float(endpoint_tangent_speed)
        self.speed_scale = float(speed_scale)

        if not 0.0 < self.turn_start < self.turn_end < 1.0:
            raise ValueError(
                "Hairpin turn_start and turn_end must satisfy 0 < start < end < 1."
            )
        if self.speed_scale < 1.0:
            raise ValueError("Hairpin speed_scale must be >= 1.0.")

        if self.tangent_speed <= 0.0:
            self.tangent_speed = self.out_length / self.turn_start
        if self.endpoint_tangent_speed <= 0.0:
            self.endpoint_tangent_speed = self.tangent_speed

        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def _time_warp_fn(self, t: float) -> jnp.ndarray:
        if self.speed_scale == 1.0:
            return t

        k = self.speed_scale
        numerator = t**k
        denominator = numerator + (1.0 - t) ** k
        return numerator / denominator

    def _time_warp_dot(self, t: float) -> jnp.ndarray:
        if self.speed_scale == 1.0:
            return jnp.ones_like(t)

        k = self.speed_scale
        numerator = k * t ** (k - 1.0) * (1.0 - t) ** (k - 1.0)
        denominator = (t**k + (1.0 - t) ** k) ** 2
        return numerator / denominator

    def _hermite(
        self,
        r: jnp.ndarray,
        p0: jnp.ndarray,
        p1: jnp.ndarray,
        m0: jnp.ndarray,
        m1: jnp.ndarray,
    ) -> jnp.ndarray:
        r2 = r * r
        r3 = r2 * r
        return (
            (2.0 * r3 - 3.0 * r2 + 1.0) * p0
            + (r3 - 2.0 * r2 + r) * m0
            + (-2.0 * r3 + 3.0 * r2) * p1
            + (r3 - r2) * m1
        )

    def _hermite_dot(
        self,
        r: jnp.ndarray,
        p0: jnp.ndarray,
        p1: jnp.ndarray,
        m0: jnp.ndarray,
        m1: jnp.ndarray,
    ) -> jnp.ndarray:
        r2 = r * r
        return (
            (6.0 * r2 - 6.0 * r) * p0
            + (3.0 * r2 - 4.0 * r + 1.0) * m0
            + (-6.0 * r2 + 6.0 * r) * p1
            + (3.0 * r2 - 2.0 * r) * m1
        )

    def _path_and_derivative(
        self,
        u: jnp.ndarray,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
    ):
        right = jnp.asarray([1.0, 0.0], dtype=x0.dtype)
        down = jnp.asarray([0.0, -1.0], dtype=x0.dtype)

        turn_in = x0 + self.out_length * right
        turn_out = turn_in + self.drop * down
        turn_start = self.turn_start
        turn_end = self.turn_end
        out_duration = turn_start
        turn_duration = turn_end - turn_start
        return_duration = 1.0 - turn_end

        r_out = u / out_duration
        out_path = x0 + r_out * (turn_in - x0)
        out_dot = (turn_in - x0) / out_duration

        turn_r = (u - turn_start) / turn_duration
        turn_m0 = self.tangent_speed * turn_duration * right
        turn_m1 = -self.tangent_speed * turn_duration * right
        turn_path = self._hermite(turn_r, turn_in, turn_out, turn_m0, turn_m1)
        turn_dot = self._hermite_dot(turn_r, turn_in, turn_out, turn_m0, turn_m1)
        turn_dot = turn_dot / turn_duration

        return_r = (u - turn_end) / return_duration
        return_m0 = -self.tangent_speed * return_duration * right
        return_m1 = -self.endpoint_tangent_speed * return_duration * right
        return_path = self._hermite(return_r, turn_out, x1, return_m0, return_m1)
        return_dot = self._hermite_dot(
            return_r, turn_out, x1, return_m0, return_m1
        )
        return_dot = return_dot / return_duration

        path = jnp.where(
            u < turn_start,
            out_path,
            jnp.where(u < turn_end, turn_path, return_path),
        )
        path_dot = jnp.where(
            u < turn_start,
            out_dot,
            jnp.where(u < turn_end, turn_dot, return_dot),
        )
        return path, path_dot

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        u = self._time_warp_fn(t)
        path, _ = self._path_and_derivative(u, x0, x1)
        return path

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        del label
        u = self._time_warp_fn(t)
        _, path_dot = self._path_and_derivative(u, x0, x1)
        return path_dot * self._time_warp_dot(t)

    def __hash__(self):
        return hash(
            (
                "hairpin",
                self.out_length,
                self.drop,
                self.turn_start,
                self.turn_end,
                self.tangent_speed,
                self.endpoint_tangent_speed,
                self.speed_scale,
            )
        )

    def __eq__(self, other):
        return (
            isinstance(other, HairpinInterpolant)
            and self.out_length == other.out_length
            and self.drop == other.drop
            and self.turn_start == other.turn_start
            and self.turn_end == other.turn_end
            and self.tangent_speed == other.tangent_speed
            and self.endpoint_tangent_speed == other.endpoint_tangent_speed
            and self.speed_scale == other.speed_scale
        )


class BezierBoxInterpolant(Interpolant):
    """Quadratic Bezier interpolant with a signed vertical control point."""

    def __init__(self, height: float):
        self.height = float(height)
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def _control(
        self, x0: jnp.ndarray, x1: jnp.ndarray, label: jnp.ndarray
    ) -> jnp.ndarray:
        sign = jnp.where(label >= 0.0, 1.0, -1.0)
        vertical = jnp.zeros_like(x0).at[1].set(self.height * sign)
        return 0.5 * (x0 + x1) + vertical

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("BezierBoxInterpolant requires a branch-sign label.")
        c = self._control(x0, x1, label)
        return ((1.0 - t) ** 2) * x0 + 2.0 * t * (1.0 - t) * c + (t**2) * x1

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("BezierBoxInterpolant requires a branch-sign label.")
        c = self._control(x0, x1, label)
        return 2.0 * (1.0 - t) * (c - x0) + 2.0 * t * (x1 - c)

    def __hash__(self):
        return hash(("bezier_box", self.height))

    def __eq__(self, other):
        return isinstance(other, BezierBoxInterpolant) and self.height == other.height


class MatchedGatesInterpolant(Interpolant):
    """Piecewise-smooth interpolant through branch-matched midpoint gates."""

    def __init__(
        self,
        midpoint_a,
        midpoint_b,
        tau_mid: float = 0.5,
    ):
        self.midpoint_a = tuple(float(v) for v in midpoint_a)
        self.midpoint_b = tuple(float(v) for v in midpoint_b)
        self.tau_mid = float(tau_mid)
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def _midpoint(self, x0: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        branch_b = label >= 0.0
        midpoint_a = jnp.asarray(self.midpoint_a, dtype=x0.dtype)
        midpoint_b = jnp.asarray(self.midpoint_b, dtype=x0.dtype)
        return jnp.where(branch_b, midpoint_b, midpoint_a)

    @staticmethod
    def _smoothstep(u: jnp.ndarray) -> jnp.ndarray:
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def _smoothstep_dot(u: jnp.ndarray) -> jnp.ndarray:
        return 6.0 * u * (1.0 - u)

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("MatchedGatesInterpolant requires a branch label.")

        midpoint = self._midpoint(x0, label)
        tau_mid = jnp.asarray(self.tau_mid, dtype=x0.dtype)

        u_left = jnp.clip(t / tau_mid, 0.0, 1.0)
        h_left = self._smoothstep(u_left)
        left = (1.0 - h_left) * x0 + h_left * midpoint

        u_right = jnp.clip((t - tau_mid) / (1.0 - tau_mid), 0.0, 1.0)
        h_right = self._smoothstep(u_right)
        right = (1.0 - h_right) * midpoint + h_right * x1

        return jnp.where(t <= tau_mid, left, right)

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        if label is None:
            raise ValueError("MatchedGatesInterpolant requires a branch label.")

        midpoint = self._midpoint(x0, label)
        tau_mid = jnp.asarray(self.tau_mid, dtype=x0.dtype)

        u_left = jnp.clip(t / tau_mid, 0.0, 1.0)
        left_dot = (self._smoothstep_dot(u_left) / tau_mid) * (midpoint - x0)

        u_right = jnp.clip((t - tau_mid) / (1.0 - tau_mid), 0.0, 1.0)
        right_dot = (self._smoothstep_dot(u_right) / (1.0 - tau_mid)) * (
            x1 - midpoint
        )

        return jnp.where(t <= tau_mid, left_dot, right_dot)

    def __hash__(self):
        return hash(("matched_gates", self.midpoint_a, self.midpoint_b, self.tau_mid))

    def __eq__(self, other):
        return (
            isinstance(other, MatchedGatesInterpolant)
            and self.midpoint_a == other.midpoint_a
            and self.midpoint_b == other.midpoint_b
            and self.tau_mid == other.tau_mid
        )


class DiveGateInterpolant(Interpolant):
    """Linear endpoint interpolation with a sharp sample-specific gate dive."""

    def __init__(
        self,
        depth: float,
        tau_down: float = 0.45,
        tau_mid: float = 0.5,
        tau_up: float = 0.55,
    ):
        self.depth = float(depth)
        self.tau_down = float(tau_down)
        self.tau_mid = float(tau_mid)
        self.tau_up = float(tau_up)
        super().__init__(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )

    def _gate_offset(self, x0: jnp.ndarray, label: jnp.ndarray) -> jnp.ndarray:
        base_offset = jnp.zeros_like(x0).at[1].set(-self.depth)
        if label is None:
            return base_offset

        label = jnp.asarray(label, dtype=x0.dtype)
        gate_jitter = jnp.zeros_like(x0)
        gate_jitter = gate_jitter.at[0].set(label[0])
        gate_jitter = gate_jitter.at[1].set(label[1])
        return base_offset + gate_jitter

    def _pulse(self, t: float) -> jnp.ndarray:
        t = jnp.asarray(t)
        down = jnp.asarray(self.tau_down, dtype=t.dtype)
        mid = jnp.asarray(self.tau_mid, dtype=t.dtype)
        up = jnp.asarray(self.tau_up, dtype=t.dtype)
        rise = (t - down) / (mid - down)
        fall = (up - t) / (up - mid)
        return jnp.where(
            t <= down,
            0.0,
            jnp.where(t <= mid, rise, jnp.where(t < up, fall, 0.0)),
        )

    def _pulse_dot(self, t: float) -> jnp.ndarray:
        t = jnp.asarray(t)
        down = jnp.asarray(self.tau_down, dtype=t.dtype)
        mid = jnp.asarray(self.tau_mid, dtype=t.dtype)
        up = jnp.asarray(self.tau_up, dtype=t.dtype)
        rise_dot = 1.0 / (mid - down)
        fall_dot = -1.0 / (up - mid)
        return jnp.where(
            t <= down,
            0.0,
            jnp.where(t <= mid, rise_dot, jnp.where(t < up, fall_dot, 0.0)),
        )

    def calc_It(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        base = super().calc_It(t, x0, x1, label)
        return base + self._pulse(t) * self._gate_offset(x0, label)

    def calc_It_dot(
        self,
        t: float,
        x0: jnp.ndarray,
        x1: jnp.ndarray,
        label: jnp.ndarray = None,
    ) -> jnp.ndarray:
        base_dot = super().calc_It_dot(t, x0, x1, label)
        return base_dot + self._pulse_dot(t) * self._gate_offset(x0, label)

    def __hash__(self):
        return hash(("dive_gate", self.depth, self.tau_down, self.tau_mid, self.tau_up))

    def __eq__(self, other):
        return (
            isinstance(other, DiveGateInterpolant)
            and self.depth == other.depth
            and self.tau_down == other.tau_down
            and self.tau_mid == other.tau_mid
            and self.tau_up == other.tau_up
        )


def setup_interpolant(cfg: config_dict.ConfigDict) -> Interpolant:
    if cfg.problem.interp_type == "linear":
        interp = Interpolant(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            alpha_dot=lambda _: -1.0,
            beta_dot=lambda _: 1.0,
        )
    elif cfg.problem.interp_type in ("time_rescaled_linear", "segment_linear"):
        interp = TimeRescaledLinearInterpolant()
    elif cfg.problem.interp_type == "trig":
        interp = Interpolant(
            alpha=lambda t: jnp.cos(jnp.pi * t / 2),
            beta=lambda t: jnp.sin(jnp.pi * t / 2),
            alpha_dot=lambda t: -0.5 * jnp.pi * jnp.sin(jnp.pi * t / 2),
            beta_dot=lambda t: 0.5 * jnp.pi * jnp.cos(jnp.pi * t / 2),
        )
    elif cfg.problem.interp_type == "triangle":
        interp = TriangleGaussianInterpolant(
            height=float(getattr(cfg.problem, "triangle_height", 3.0))
        )
    elif cfg.problem.interp_type == "spiral":
        interp = SpiralInterpolant(
            radius_a=float(getattr(cfg.problem, "spiral_radius_a", 2.88)),
            radius_b=float(getattr(cfg.problem, "spiral_radius_b", -4.59)),
            radius_c=float(getattr(cfg.problem, "spiral_radius_c", 2.71)),
            turns=float(getattr(cfg.problem, "spiral_turns", 1.5)),
        )
    elif cfg.problem.interp_type == "hairpin":
        interp = HairpinInterpolant(
            out_length=float(getattr(cfg.problem, "hairpin_out_length", 4.0)),
            drop=float(getattr(cfg.problem, "hairpin_drop", 1.6)),
            turn_start=float(getattr(cfg.problem, "hairpin_turn_start", 0.35)),
            turn_end=float(getattr(cfg.problem, "hairpin_turn_end", 0.65)),
            tangent_speed=float(getattr(cfg.problem, "hairpin_tangent_speed", 0.0)),
            endpoint_tangent_speed=float(
                getattr(cfg.problem, "hairpin_endpoint_tangent_speed", 0.0)
            ),
            speed_scale=float(getattr(cfg.problem, "hairpin_speed_scale", 6.0)),
        )
    elif cfg.problem.interp_type == "bezier_box":
        interp = BezierBoxInterpolant(
            height=float(getattr(cfg.problem, "bezier_height", 2.5))
        )
    elif cfg.problem.interp_type == "matched_gates":
        interp = MatchedGatesInterpolant(
            midpoint_a=getattr(cfg.problem, "gate_midpoint_a", [-0.35, 0.0]),
            midpoint_b=getattr(cfg.problem, "gate_midpoint_b", [0.35, 0.0]),
            tau_mid=float(getattr(cfg.problem, "gate_tau_mid", 0.5)),
        )
    elif cfg.problem.interp_type == "dive_gate":
        interp = DiveGateInterpolant(
            depth=float(getattr(cfg.problem, "dive_gate_depth", 0.85)),
            tau_down=float(getattr(cfg.problem, "dive_gate_tau_down", 0.45)),
            tau_mid=float(getattr(cfg.problem, "dive_gate_tau_mid", 0.5)),
            tau_up=float(getattr(cfg.problem, "dive_gate_tau_up", 0.55)),
        )
    else:
        raise NotImplementedError("Interpolant type not implemented.")

    return interp

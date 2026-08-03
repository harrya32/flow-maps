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

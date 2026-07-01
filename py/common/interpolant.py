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
    else:
        raise NotImplementedError("Interpolant type not implemented.")

    return interp

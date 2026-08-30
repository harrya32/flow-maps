"""
Nicholas M. Boffi
10/5/25

Basic routines for flow map class.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.flatten_util import ravel_pytree
from ml_collections import config_dict

from . import edm2_net, network_utils

Parameters = Dict[str, Dict]


class FlowMap(nn.Module):
    """Basic class for a flow map."""

    config: config_dict.ConfigDict

    def setup(self):
        """Set up the flow map."""
        self.flow_map = network_utils.setup_network(self.config)

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map(
            s, t, x, label, train, calc_weight, return_X_and_phi, init_weights
        )

    def partial_t(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to time."""
        Xst, dt_Xst = jax.jvp(
            lambda t: self.flow_map(s, t, x, label, train, calc_weight),
            primals=(t,),
            tangents=(jnp.ones_like(t),),
        )

        return Xst, dt_Xst

    def partial_s(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Compute the partial derivative with respect to space."""
        Xst, ds_Xst = jax.jvp(
            lambda s: self.flow_map(s, t, x, label, train, calc_weight),
            primals=(s,),
            tangents=(jnp.ones_like(s),),
        )

        return Xst, ds_Xst

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        """Compute the weights for the flow map."""
        return self.flow_map.calc_weight(s, t)

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Compute the flow map."""
        return self.flow_map.calc_phi(
            s, t, x, label=label, train=train, calc_weight=calc_weight
        )

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        """Apply the flow map."""
        return self.flow_map.calc_b(
            t, x, label=label, train=train, calc_weight=calc_weight
        )


def sample(
    apply_flow_map: Callable, variables: Dict, x0: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Unconditional sampling."""
    ts = jnp.linspace(0.0, 1.0, N + 1)

    def step(x, idx):
        return (
            apply_flow_map(
                variables,
                ts[idx],
                ts[idx + 1],
                x,
                label=label,
                train=False,
                calc_weight=False,
                return_X_and_phi=False,
            ),
            None,
        )

    final_state, _ = jax.lax.scan(step, x0, jnp.arange(N))
    return final_state


def sample_trajectory(
    apply_flow_map: Callable, variables: Dict, x0: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Track every node in an N-step unconditional sample trajectory."""
    ts = jnp.linspace(0.0, 1.0, N + 1)

    def step(x, idx):
        x_next = apply_flow_map(
            variables,
            ts[idx],
            ts[idx + 1],
            x,
            label=label,
            train=False,
            calc_weight=False,
            return_X_and_phi=False,
        )
        return x_next, x_next

    _, states = jax.lax.scan(step, x0, jnp.arange(N))
    return jnp.concatenate([x0[None, ...], states], axis=0)


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
def batch_sample(
    apply_flow_map: Callable, variables: Dict, x0s: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Batch unconditional sampling."""
    return sample(apply_flow_map, variables, x0s, N, label)


@functools.partial(jax.jit, static_argnums=(0, 3))
@functools.partial(jax.vmap, in_axes=(None, None, 0, None, 0))
def batch_sample_trajectory(
    apply_flow_map: Callable, variables: Dict, x0s: jnp.ndarray, N: int, label: int
) -> jnp.ndarray:
    """Batch trajectory tracking for unconditional sampling."""
    return sample_trajectory(apply_flow_map, variables, x0s, N, label)


@functools.partial(
    jax.pmap,
    in_axes=(None, 0, 0, None, 0),
    static_broadcasted_argnums=(0, 3),
    axis_name="data",
)
def pmap_batch_sample(
    apply_flow_map: Callable, variables: Dict, x0s: jnp.ndarray, N: int, labels
) -> jnp.ndarray:
    """Parallel batch sampling across devices."""
    return batch_sample(apply_flow_map, variables, x0s, N, labels)


def initialize_flow_map(
    network_config: config_dict.ConfigDict, ex_input: jnp.ndarray, prng_key: jnp.ndarray
) -> Tuple[nn.Module, Parameters, jnp.ndarray]:
    # define the network
    net = FlowMap(network_config)

    # Apple's experimental Metal backend cannot currently lower the `erf`
    # operation used by Flax's default truncated-normal Dense initializer.
    # Initialize the same parameters on CPU, then transfer them to Metal; all
    # subsequent optimizer and training work remains on the Metal device.
    training_device = None
    initialization_device = None
    if jax.default_backend().lower() == "metal":
        training_device = jax.devices()[0]
        try:
            initialization_device = jax.devices("cpu")[0]
        except RuntimeError as exc:
            raise RuntimeError(
                "jax-metal requires the CPU backend for parameter "
                "initialization. Set JAX_PLATFORMS=METAL,cpu."
            ) from exc

    ex_s = ex_t = 0.0
    ex_label = 0
    with jax.default_device(initialization_device):
        if initialization_device is not None:
            print(
                "Initializing parameters on CPU for jax-metal compatibility; "
                "training remains on Metal."
            )
            ex_input = jax.device_put(ex_input, initialization_device)
            prng_key = jax.device_put(prng_key, initialization_device)

        prng_key, skey = jax.random.split(prng_key)
        params = net.init(
            {"params": prng_key, "constants": skey},
            ex_s,
            ex_t,
            ex_input,
            ex_label,
            train=False,
            init_weights=True,  # This triggers initialization of all weight params
        )
        prng_key = jax.random.split(prng_key)[0]

    if training_device is not None:
        params = jax.device_put(params, training_device)
        prng_key = jax.device_put(prng_key, training_device)

    print(f"Number of parameters: {ravel_pytree(params)[0].size}")

    if network_config.network_type == "edm2":
        params = edm2_net.project_to_sphere(params)

    return net, params, prng_key

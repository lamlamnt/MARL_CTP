import haiku as hk
import jax
import jax.numpy as jnp
import sys
import flax
import flax.linen as nn
from typing import Sequence


@hk.transform
def simplest_model_hk(x: jnp.ndarray) -> jnp.ndarray:
    # Flatten to 1D array
    x = x.reshape(-1)
    mlp = hk.nets.MLP(output_sizes=[128, 64, 32, 16, 5])
    return mlp(x)


# Custom module
class FCNetwork_HK(hk.Module):
    def __init__(self, output_size, name="custom_linear"):
        super().__init__(name=name)
        self.mlp = hk.nets.MLP(output_sizes=output_size, name="mlp")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1)
        return self.mlp(x)


def _forward_fn_FC(x):
    module = FCNetwork_HK([64, 32, 5])
    return module(x)


forward_fn_FC = hk.without_apply_rng(hk.transform(_forward_fn_FC))


class Flax_FCNetwork(nn.Module):
    hidden_dims: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # Flatten the input
        x = x.reshape(-1)
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x

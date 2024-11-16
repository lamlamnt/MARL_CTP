import jax
import jax.numpy as jnp
import sys
import flax
import flax.linen as nn
from typing import Sequence
from flax.linen.initializers import glorot_normal, lecun_normal


class Flax_CNN(nn.Module):
    num_filters: int
    hidden_dims: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # CNN layer followed by FC layers

        x = jnp.transpose(x, (1, 2, 0))
        x = nn.Conv(
            features=self.num_filters,
            kernel_size=(1, 1),
            kernel_init=lecun_normal(dtype=jnp.float16),
            param_dtype=jnp.float16,
            dtype=jnp.float16,
        )(x)
        x = nn.relu(x)
        x = x.reshape(-1)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=lecun_normal(dtype=jnp.float16),
                param_dtype=jnp.float16,
                dtype=jnp.float16,
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.num_actions,
            kernel_init=lecun_normal(dtype=jnp.float16),
            param_dtype=jnp.float16,
            dtype=jnp.float16,
        )(x)
        return x

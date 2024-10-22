import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Sequence

# Fully connected MLP
class QNetwork(nn.Module):
    hidden_dims: Sequence[int]
    num_actions:int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """
        activation = x
        for i, hidden_dim in enumerate(self.hidden_dims):
            activation = nn.Dense(hidden_dim)(activation)
            activation = nn.relu(activation)
        return activation
        """
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        return x
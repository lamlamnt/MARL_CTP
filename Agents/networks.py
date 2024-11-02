import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from typing import Sequence


# Fully connected MLP
class QNetwork(nn.Module):
    hidden_dims: Sequence[int]
    num_actions: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # If only inputting 1 sample at a time, broadcast so leading dimension size is 1
        # Flatten the input to (batch size, n_nodes*n_nodes*3)
        x = x.reshape((x.shape[0], -1))
        for i, hidden_dim in enumerate(self.hidden_dims):
            # nn.Dense is only applied over the last dimension of the input
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        # Output has shape (batch_size, num_actions)
        # Maybe add softmax here
        return x

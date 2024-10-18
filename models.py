import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

# Fully connected MLP
class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.relu(x)
        return x
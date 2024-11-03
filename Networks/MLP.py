import haiku as hk
import jax
import jax.numpy as jnp
import sys


@hk.transform
def simplest_model(x: jnp.ndarray) -> jnp.ndarray:
    # Flatten to 1D array
    x = x.reshape(-1)
    mlp = hk.nets.MLP(output_sizes=[64, 32, 5])
    return mlp(x)


# Custom module
class FCNetwork(hk.Module):
    def __init__(self, output_size, name="custom_linear"):
        super().__init__(name=name)
        self.mlp = hk.nets.MLP(output_sizes=output_size, name="mlp")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.reshape(-1)
        return self.mlp(x)


def _forward_fn_FC(x):
    module = FCNetwork([64, 32, 5])
    return module(x)


forward_fn_FC = hk.without_apply_rng(hk.transform(_forward_fn_FC))

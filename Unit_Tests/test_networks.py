import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Agents.networks import QNetwork

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # Batch size, number of nodes, number of nodes, 3
    x = jax.random.uniform(key, (10, 5, 5, 3))
    model = QNetwork([128, 64, 32, 16], 6)
    params = model.init(key, x)
    print(jax.tree_map(lambda x: x.shape, params))
    y = model.apply(params, x)
    print(jax.tree_map(lambda x: x.shape, y))

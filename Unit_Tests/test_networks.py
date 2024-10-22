import jax
import jax.numpy as jnp
import sys
sys.path.append('..')
from Agents.networks import QNetwork

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (4,4))
    model = QNetwork([16,8],5)
    params = model.init(key, x)
    y = model.apply(params, x)
    print(y)


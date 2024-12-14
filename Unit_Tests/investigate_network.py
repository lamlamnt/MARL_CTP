import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Networks import densenet, resnet   

if __name__ == "__main__":
    model = 
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((4, 11, 10)))
    output = model.apply(params, jnp.ones((4, 11, 10)))

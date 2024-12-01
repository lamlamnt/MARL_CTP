import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment


def pure_function(key):
    key, subkey = jax.random.split(key)
    graph = CTP_generator.CTPGraph_Realisation(key, 5, 10, 0.9)
    blocking_status = graph.sample_blocking_status(subkey)
    return jnp.array([graph.is_solvable(blocking_status)])


@jax.jit
def test_func(key):
    # dummy lines to simulate jax-jittable code
    a = jnp.array([1, 2, 3])
    # Non-jax-jittable code
    x = jnp.array([jnp.bool_(True)])
    result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    is_solvable = jax.pure_callback(pure_function, result_shape, key)
    return is_solvable


if __name__ == "__main__":
    # Create a CTP environment
    key = jax.random.PRNGKey(0)
    is_solvable = test_func(key)
    print(is_solvable.item())

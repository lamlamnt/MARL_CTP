import jax
import jax.numpy as jnp
import sys
import pytest

sys.path.append("..")
from Environment import CTP_generator, CTP_environment


def pure_function(key, prob):
    key, subkey = jax.random.split(key)
    graph = CTP_generator.CTPGraph_Realisation(key, 5, 10, prob)
    blocking_status = graph.sample_blocking_status(subkey)
    return graph.is_solvable(blocking_status)


@jax.jit
def overall_func(key, prob):
    # dummy lines to simulate jax-jittable code
    a = jnp.array([1, 2, 3])
    # Non-jax-jittable code
    is_solvable = jax.pure_callback(pure_function, jnp.bool_(False), key, prob)
    return is_solvable


def test_callback():
    # Create a CTP environment
    key = jax.random.PRNGKey(0)
    is_solvable = overall_func(key, 0.4)
    assert is_solvable == jnp.bool_(True)

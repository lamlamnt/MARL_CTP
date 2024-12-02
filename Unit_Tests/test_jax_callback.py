import jax
import jax.experimental
import jax.numpy as jnp
import sys
import pytest

sys.path.append("..")
from Environment import CTP_generator, CTP_environment
import os


def pure_function(key, prob):
    key, subkey = jax.random.split(key)
    environment = CTP_environment.CTP(
        1, 1, 10, key, prob, expensive_edge=False, patience=30
    )
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_directory, "Logs/Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(
        log_directory, "callback_graph.png"
    )
    patience_counter = 0
    is_solvable = jnp.bool_(False)
    # switch to using jax.lax.while_loop
    while is_solvable == jnp.bool_(False) and patience_counter < 10:
        key, subkey = jax.random.split(subkey)
        new_blocking_status = environment.graph_realisation.sample_blocking_status(
            subkey
        )
        is_solvable = environment.graph_realisation.is_solvable(new_blocking_status)
        patience_counter += 1
        # error if is_solvable is False
    if is_solvable == jnp.bool_(False):
        raise ValueError(
            "Could not find enough solvable blocking status. Please decrease the prop_stoch."
        )
    return new_blocking_status


@jax.jit
def overall_func(key, prob):
    # dummy lines to simulate jax-jittable code
    a = jnp.array([1, 2, 3])
    # Non-jax-jittable code
    result_shape = jax.ShapeDtypeStruct((10, 10), jnp.float16)
    # new_blocking_status = jax.pure_callback(pure_function, result_shape, key, prob)
    new_blocking_status = jax.experimental.io_callback(
        pure_function, result_shape, key, prob
    )
    return new_blocking_status


# if __name__ == "__main__":
def test_callback():
    # Create a CTP environment
    subkey = jax.random.PRNGKey(61)
    for i in range(10):
        new_blocking_status = overall_func(subkey, 0.9)
        key, subkey = jax.random.split(subkey)
    assert new_blocking_status.shape == (10, 10)

import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest
import pytest_print as pp
import os

if __name__ == "__main__":
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    new_env_state, new_belief_state = environment.reset(subkey)
    print(jnp.array_equal(initial_env_state, new_env_state))
    assert not jnp.array_equal(initial_env_state, new_env_state)
    """
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    environment.graph_realisation.plot_realised_graph(log_directory, "test_graph.png")
    env_state_1, belief_state_1, reward_1, terminate, subkey = (
        environment.step(
            subkey, initial_env_state, initial_belief_state, jnp.array([4])
        )
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = (
        environment.step(subkey, env_state_1, belief_state_1, jnp.array([3]))
    )
    env_state_3, belief_state_3, reward_3, terminate, subkey = (
        environment.step(subkey, env_state_2, belief_state_2, jnp.array([2]))
    )
    """

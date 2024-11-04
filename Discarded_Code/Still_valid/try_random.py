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
    # key = jax.random.PRNGKey(30)
    key = jax.random.PRNGKey(99)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :], log_directory, "test_graph.png"
    )
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([4])
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([3])
    )
    env_state_3, belief_state_3, reward_3, terminate, subkey = environment.step(
        subkey, env_state_2, belief_state_2, jnp.array([2])
    )

import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import os

if __name__ == "__main__":
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")

    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    key, subkey = jax.random.split(key)

    initial_belief_state = environment.reset(subkey)
    environment.graph_realisation.plot_realised_graph(log_directory)
    new_belief_state, reward, terminate = environment.step(
        jnp.array([4]), initial_belief_state
    )
    new_belief_state, reward, terminate = environment.step(
        jnp.array([3]), new_belief_state
    )
    new_belief_state, reward, terminate = environment.step(
        jnp.array([2]), new_belief_state
    )

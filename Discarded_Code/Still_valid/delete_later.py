import jax
import jax.numpy as jnp
import sys
import os

sys.path.append("..")
from Environment import CTP_environment

if __name__ == "__main__":
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([0])
    )
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_directory, "Logs", "Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(log_directory, "delete_later")
    print(initial_env_state)
    print(env_state_1)

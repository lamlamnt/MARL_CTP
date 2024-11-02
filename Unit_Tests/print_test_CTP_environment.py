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

    key = jax.random.PRNGKey(57)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    key, subkey = jax.random.split(key)

    initial_belief_state = environment.reset(subkey)
    print(f"true blocking status: {environment.graph_realisation.blocking_status}")
    environment.graph_realisation.plot_realised_graph(log_directory)
    print(f"Initial belief state: {initial_belief_state}")
    new_belief_state, reward, terminate = environment.step(
        jnp.array([3]), initial_belief_state
    )
    print(f"Agents pos: {environment.agents_pos}")
    print(f"New belief state: {new_belief_state}")
    print(f"Reward: {reward}")
    print(f"Terminate: {terminate}")
    new_belief_state, reward, terminate = environment.step(
        jnp.array([1]), new_belief_state
    )
    print(f"New belief state: {new_belief_state}")
    print(f"Reward: {reward}")
    print(f"Terminate: {terminate}")

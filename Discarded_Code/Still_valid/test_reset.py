import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment, CTP_generator
import os

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_directory, "Logs/Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(log_directory, "test_graph")
    env_key, init_key = jax.random.split(jax.random.PRNGKey(0))
    new_env_state, new_belief_state = environment.reset(init_key)
    timestep_in_episode = 0
    for i in range(10):
        action = jnp.array([4])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, new_env_state, new_belief_state, action
        )
        new_env_state, new_belief_state, reward, timestep_in_episode = jax.lax.cond(
            timestep_in_episode >= 5,
            lambda _: (
                *environment.reset(key),
                jnp.float16(-1.1),
                0,
            ),
            lambda _: (
                new_env_state,
                new_belief_state,
                reward,
                timestep_in_episode + 1,
            ),
            operand=None,
        )
        print(timestep_in_episode)

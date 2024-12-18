import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import os
import sys

sys.path.append("..")
from Baselines.ao_star import AO_Star
from Baselines.optimistic_agent import Optimistic_Agent
from Environment import CTP_generator, CTP_environment


# def test_AO_Star(printer):
if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 5, key, 0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_dir = os.path.join(parent_directory, "Logs", "Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(log_dir, "test_ao_star_graph.png")
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :], log_dir, "test_ao_star_realisation.png"
    )
    optimal_path, expected_cost = AO_Star(
        initial_belief_state,
        environment.graph_realisation.graph.origin.item(),
        environment.graph_realisation.graph.goal.item(),
    )
    print("Results")
    print(optimal_path)
    print(expected_cost)
    # assert not inf


# Try do one episode of optimistic baseline

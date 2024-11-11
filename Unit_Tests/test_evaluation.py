import flax.serialization
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Networks import MLP
from Environment import CTP_environment
from Evaluation.optimal_path_length import dijkstra_shortest_path
import os
import pytest


def test_load_model():
    key = jax.random.PRNGKey(0)
    # Load the parameters from the file
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    file_name = os.path.join(log_directory, "weights.flax")
    with open(file_name, "rb") as f:
        serialized_params = f.read()

    # Initialize your model (assuming you know the input shape or have example inputs)
    flax_model = MLP.Flax_FCNetwork([150, 75, 37, 18], 5)
    example_input = jnp.zeros((3, 6, 5))
    initial_params = flax_model.init(key, example_input)
    random_output = flax_model.apply(initial_params, example_input)

    # Restore the parameters
    restored_params = flax.serialization.from_bytes(initial_params, serialized_params)
    model_output = flax_model.apply(restored_params, example_input)
    assert not jnp.array_equal(random_output, model_output)


def test_optimal_path_length():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    env_state, _ = environment.reset(key)
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    environment.graph_realisation.plot_realised_graph(
        env_state[0, 1:, :], log_directory, "check_dijsktra.png"
    )
    goal = environment.graph_realisation.graph.goal
    origin = environment.graph_realisation.graph.origin
    shortest_path = dijkstra_shortest_path(env_state, origin, goal)
    assert jnp.isclose(shortest_path, 6.36, atol=1e-2)

import flax.serialization
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Networks import MLP
from Environment import CTP_environment, CTP_generator
from Evaluation.optimal_path_length import dijkstra_shortest_path
import os
import pytest
import pytest_print as pp
from Networks.actor_critic_network import ActorCritic_CNN_10


def test_load_model():
    key = jax.random.PRNGKey(0)
    # Load the parameters from the file
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs", "Unit_Tests")
    file_name = os.path.join(log_directory, "weights.flax")
    with open(file_name, "rb") as f:
        serialized_params = f.read()

    # Initialize your model (assuming you know the input shape or have example inputs)
    flax_model = ActorCritic_CNN_10(5)
    example_input = jnp.zeros((4, 6, 5))
    initial_params = flax_model.init(key, example_input)
    random_output = flax_model.apply(initial_params, example_input)

    # Restore the parameters
    restored_params = flax.serialization.from_bytes(initial_params, serialized_params)
    model_output = flax_model.apply(restored_params, example_input)
    assert not jnp.array_equal(random_output, model_output)


def test_optimal_path_length(printer):
    key = jax.random.PRNGKey(30)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys
    environment = CTP_environment.CTP(1, 1, 5, environment_key, prop_stoch=0.4)
    env_state, _ = environment.reset(key)

    env_state = env_state.at[0, 2, 3].set(CTP_generator.UNBLOCKED)
    env_state = env_state.at[0, 4, 1].set(CTP_generator.UNBLOCKED)

    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    environment.graph_realisation.plot_realised_graph(
        env_state[0, 1:, :], log_directory, "check_dijsktra.png"
    )
    goal = environment.graph_realisation.graph.goal
    origin = environment.graph_realisation.graph.origin
    shortest_path = dijkstra_shortest_path(env_state, origin, goal)
    assert jnp.isclose(shortest_path, 1.224, atol=1e-3)


def test_grid_size_dijkstra(printer):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    environment = CTP_environment.CTP(1, 1, 10, key, prop_stoch=0.4, grid_size=10)
    env_state, _ = environment.reset(key)
    goal = environment.graph_realisation.graph.goal
    origin = environment.graph_realisation.graph.origin
    shortest_path_10 = dijkstra_shortest_path(env_state, origin, goal)
    environment.graph_realisation.plot_realised_graph(
        env_state[0, 1:, :], log_directory, "grid_size_10.png"
    )

    different_environment = CTP_environment.CTP(
        1, 1, 10, subkey, prop_stoch=0.4, grid_size=20
    )
    different_env_state, _ = different_environment.reset(subkey)
    different_goal = different_environment.graph_realisation.graph.goal
    different_origin = different_environment.graph_realisation.graph.origin
    shortest_path_20 = dijkstra_shortest_path(
        different_env_state, different_origin, different_goal
    )
    different_environment.graph_realisation.plot_realised_graph(
        different_env_state[0, 1:, :], log_directory, "grid_size_20.png"
    )

    printer(shortest_path_10)
    printer(shortest_path_20)

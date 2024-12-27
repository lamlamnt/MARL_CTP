import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import os
import sys

sys.path.append("..")
from Baselines.ao_star import AO_Star_Planning, AO_Star_Execute, AO_Star_Full
from Baselines.optimistic_agent import Optimistic_Agent
from Environment import CTP_generator, CTP_environment


def test_AO_Star_4_node(printer):
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 4, key, prop_stoch=0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_dir = os.path.join(parent_directory, "Logs", "Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(log_dir, "test_ao_star_graph.png")
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :], log_dir, "test_ao_star_realisation.png"
    )
    root_node = AO_Star_Planning(
        initial_belief_state,
        environment.graph_realisation.graph.origin.item(),
        environment.graph_realisation.graph.goal.item(),
    )
    assert jnp.isclose(root_node.value, 0.464, atol=0.001)
    goal = goal = jnp.unravel_index(
        jnp.argmax(initial_env_state[3, 1:, :]),
        (environment.num_nodes, environment.num_nodes),
    )[0]
    actual_path_length = AO_Star_Execute(initial_env_state, root_node, goal)
    assert jnp.isclose(actual_path_length, 0.447, atol=0.001)

    # Test full AO* algorithm
    origin = jnp.argmax(initial_belief_state[0, :1, :])
    actual_path_length = AO_Star_Full(
        initial_env_state,
        initial_belief_state,
        origin,
        goal,
    )
    assert jnp.isclose(actual_path_length, 0.447, atol=0.001)
    printer(actual_path_length)


def test_AO_Star_custom_small_1(printer):
    initial_belief_state = jnp.array(
        [
            [[1, 0, 0, 0], [1, 0, 1, 1], [0, 1, 0, -1], [1, 0, 1, 0], [1, -1, 0, 1]],
            [
                [0, 0, 0, 0],
                [-1, 1, -1, -1],
                [1, -1, 2, 5],
                [-1, 2, -1, 2],
                [-1, 5, 2, -1],
            ],
            [
                [0, 0, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 0.3],
                [1, 0, 1, 0],
                [1, 0.3, 0, 1],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ],
        ],
        dtype=jnp.float16,
    )
    root_node = AO_Star_Planning(
        initial_belief_state,
        0,
        3,
    )
    assert jnp.isclose(root_node.value, 5.0, atol=0.001)


def test_AO_Star_custom_small_2(printer):
    initial_belief_state = jnp.array(
        [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1],
                [0, 1, 1, 0, 1, -1],
                [0, 1, 1, 1, 0, -1],
                [1, 0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0],
                [1, -1, -1, 0, 0, 1],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [-1, 5, 5, -1, -1, -1],
                [5, -1, -1, 3, -1, 1],
                [5, -1, -1, -1, 1, 1],
                [-1, 3, -1, -1, -1, 6],
                [-1, -1, 1, -1, -1, 1],
                [-1, 1, 1, 6, 1, -1],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1],
                [0, 1, 1, 0, 1, 0.1],
                [0, 1, 1, 1, 0, 0.9],
                [1, 0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0],
                [1, 0.1, 0.9, 0, 0, 1],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ],
        dtype=jnp.float16,
    )
    root_node = AO_Star_Planning(
        initial_belief_state,
        0,
        5,
    )
    assert jnp.isclose(root_node.value, 6.8, atol=0.01)
    env_state_belief_best = jnp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, -1],
            [1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 0, -1, 0, 0, 1],
        ],
        dtype=jnp.float16,
    )
    env_state_best = initial_belief_state.at[0, :, :].set(env_state_belief_best)
    actual_path_length_best = AO_Star_Full(env_state_best, initial_belief_state, 0, 5)
    assert jnp.isclose(actual_path_length_best, 6.0, atol=0.01)

    env_state_belief_worst = jnp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, -1],
            [1, 0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1, 0],
            [1, 1, -1, 0, 0, 1],
        ],
        dtype=jnp.float16,
    )
    env_state_worst = initial_belief_state.at[0, :, :].set(env_state_belief_worst)
    actual_path_length_worst = AO_Star_Full(env_state_worst, initial_belief_state, 0, 5)
    assert jnp.isclose(actual_path_length_worst, 14.0, atol=0.01)


# Check that doesn't time out or memory warning for 30 nodes
# def test_AO_Star_30_node():
if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 10, key, prop_stoch=0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    origin = jnp.argmax(initial_belief_state[0, :1, :])
    goal = jnp.unravel_index(jnp.argmax(initial_env_state[3, 1:, :]), (10, 10))[0]
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_dir = os.path.join(parent_directory, "Logs", "Unit_Tests")
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :], log_dir, "test_ao_star_10_nodes.png"
    )
    actual_path_length = AO_Star_Full(
        initial_env_state,
        initial_belief_state,
        origin.item(),
        goal,
    )
    print("Actual path length: " + str(actual_path_length))
    assert actual_path_length < jnp.inf


def test_one_episode_optimistic_baseline():
    pass

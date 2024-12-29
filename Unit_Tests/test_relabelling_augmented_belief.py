import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import sys
import os

sys.path.append("..")
from Environment import CTP_generator, CTP_environment
from Utils.augmented_belief_state import (
    get_augmented_optimistic_belief,
    get_augmented_optimistic_pessimistic_belief,
)


# Test that the origin is 0 and goal is num_nodes -1
def test_relabelling():
    key = jax.random.PRNGKey(1)
    num_nodes = 5
    graph_realisation = CTP_generator.CTPGraph_Realisation(key, num_nodes, 10, 0.4)
    assert graph_realisation.graph.origin == 0
    assert graph_realisation.graph.goal == num_nodes - 1


# Check that the first part matches the original belief state
# Check diagonals are zero. Matrix is symmetric
def test_augmented_optimistic_belief_state():
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 5, key, 0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_dir = os.path.join(parent_directory, "Logs", "Unit_Tests")
    environment.graph_realisation.graph.plot_nx_graph(
        log_dir, "augmented_optimistic.png"
    )
    augmented_belief_state = get_augmented_optimistic_belief(initial_belief_state)
    assert jnp.array_equal(initial_belief_state, augmented_belief_state[:4, :, :])
    assert jnp.array_equal(
        augmented_belief_state[4, 1:, :], augmented_belief_state[4, 1:, :].T
    )
    assert jnp.all(jnp.diag(augmented_belief_state[4, 1:, :]) == 0)


def test_augmented_optimistic_pessimistic_belief_state(printer):
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 5, key, 0.8)
    initial_env_state, initial_belief_state = environment.reset(key)
    augmented_belief_state = get_augmented_optimistic_pessimistic_belief(
        initial_belief_state
    )
    assert jnp.array_equal(initial_belief_state, augmented_belief_state[:4, :, :])
    assert jnp.array_equal(
        augmented_belief_state[4, 1:, :], augmented_belief_state[4, 1:, :].T
    )
    assert jnp.array_equal(
        augmented_belief_state[5, 1:, :], augmented_belief_state[5, 1:, :].T
    )
    assert jnp.all(jnp.diag(augmented_belief_state[4, 1:, :]) == 0)
    assert jnp.all(jnp.diag(augmented_belief_state[5, 1:, :]) == 0)

    # Asser that all elements in the optimistic belief are equal to or smaller than the pessimistic belief
    assert jnp.all(augmented_belief_state[4, :, :] <= augmented_belief_state[5, :, :])
    printer(augmented_belief_state)

    # for prop_stoch = 0, pessimistic and optimistic the same
    key = jax.random.PRNGKey(1)
    environment = CTP_environment.CTP(1, 1, 5, key, 0)
    initial_env_state, initial_belief_state = environment.reset(key)
    augmented_belief_state = get_augmented_optimistic_pessimistic_belief(
        initial_belief_state
    )
    assert jnp.array_equal(
        augmented_belief_state[4, :, :], augmented_belief_state[5, :, :]
    )

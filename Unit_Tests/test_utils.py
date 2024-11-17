import jax
import jax.numpy as jnp
import pytest
import pytest_print
import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment
from Utils import invalid_action_masking
import os


def test_invalid_action_masking(printer):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    logs_directory = os.path.join(parent_directory, "logs")

    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    initial_env_state, initial_belief_state = environment.reset(key)
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :],
        os.path.join(logs_directory, "Unit_Tests"),
        "test_invalid_action_graph.png",
    )
    valid = invalid_action_masking.decide_validity_of_action_space(initial_belief_state)
    true_valid = jnp.array([-jnp.inf, 1.0, 1.0, 1.0, -jnp.inf])
    assert jnp.array_equal(valid, true_valid)


def test_random_choice_valid_actions(printer):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    invalid_mask = jnp.array([-jnp.inf, 1.0, 1.0, 1.0, -jnp.inf])
    action = invalid_action_masking.random_valid_action(subkey, invalid_mask)
    assert action == 1 or action == 2 or action == 3
    printer(action)

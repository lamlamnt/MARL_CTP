import jax
import jax.numpy as jnp
import pytest
import sys

sys.path.append("..")
from Environment import CTP_environment_generalize
import pytest_print as pp
import os
from Utils.util_generalize import get_origin_expensive_edge


@pytest.fixture
def expensive_edge_environment():
    key = jax.random.PRNGKey(0)
    environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, key, prop_stoch=0.4, expensive_edge=True
    )
    return environment


@pytest.fixture
def non_expensive_edge_environment():
    key = jax.random.PRNGKey(0)
    environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, key, prop_stoch=0.4, expensive_edge=False
    )
    return environment


# Test that each time we reset, we get a different graph
# Test belief state
# Test reach goal
def test_different_graph(
    expensive_edge_environment: CTP_environment_generalize.CTP_General,
):
    key = jax.random.PRNGKey(1)
    subkey1, subkey2 = jax.random.split(key)
    initial_env_state1, initial_belief_state1 = expensive_edge_environment.reset(key)
    initial_env_state2, initial_belief_state2 = expensive_edge_environment.reset(
        subkey1
    )
    assert not jnp.array_equal(initial_env_state1, initial_env_state2)
    assert jnp.array_equal(
        initial_env_state1[1:, :, :], initial_belief_state1[1:, :, :]
    )
    # Not the only way to get the goal node
    goal = jnp.unravel_index(jnp.argmax(initial_env_state2[3, 1:, :]), (5, 5))[0]
    env_state_next, belief_state_next, reward_next, terminate, subkey = (
        expensive_edge_environment.step(
            subkey2, initial_env_state2, initial_belief_state2, jnp.array([goal])
        )
    )
    assert terminate == True
    assert reward_next == -1.0


def test_get_origin(expensive_edge_environment: CTP_environment_generalize.CTP_General):
    key = jax.random.PRNGKey(1)
    subkey1, subkey2 = jax.random.split(key)
    initial_env_state, initial_belief_state = expensive_edge_environment.reset(key)
    actual_origin = jnp.argmax(initial_belief_state[0, 0, :])
    guess_origin = get_origin_expensive_edge(initial_belief_state)
    assert actual_origin == guess_origin

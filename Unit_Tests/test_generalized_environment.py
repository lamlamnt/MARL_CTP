import jax
import jax.numpy as jnp
import pytest
import pytest_print as pp
import sys

sys.path.append("..")
from Environment import CTP_environment_generalize, CTP_environment
import pytest_print as pp
import os
from Utils.util_generalize import get_origin_expensive_edge


@pytest.fixture
def expensive_edge_environment():
    key = jax.random.PRNGKey(0)
    environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, key, prop_stoch=0.4, deal_with_unsolvability="always_expensive_edge"
    )
    return environment


@pytest.fixture
def resample_environment():
    key = jax.random.PRNGKey(0)
    environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, key, prop_stoch=0.4, deal_with_unsolvability="resample"
    )
    return environment


@pytest.fixture
def expensive_if_unsolvable_environment():
    key = jax.random.PRNGKey(0)
    environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, key, prop_stoch=0.9, deal_with_unsolvability="expensive_if_unsolvable"
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


def test_deal_with_unsolvability(
    expensive_edge_environment: CTP_environment_generalize.CTP_General,
    resample_environment: CTP_environment_generalize.CTP_General,
    expensive_if_unsolvable_environment: CTP_environment_generalize.CTP_General,
):
    # Test working
    env_state, belief_state = expensive_edge_environment.reset(jax.random.PRNGKey(0))
    env_state, belief_state = resample_environment.reset(jax.random.PRNGKey(0))

    # Test that all edge weights are less than 1 except for 1 edge with 1.
    env_state, belief_state = expensive_if_unsolvable_environment.reset(
        jax.random.PRNGKey(0)
    )
    weight_matrix = env_state[1, 1:, :]
    all_less_equal_one = jnp.all(weight_matrix <= 1)
    assert all_less_equal_one.item() is True
    num_ones = jnp.sum(weight_matrix == 1)
    exactly_one_equal_one = num_ones == 2
    assert exactly_one_equal_one.item() is True

import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest
import pytest_print as pp
import os

# For single agent


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    return environment


# Check symmetric adjacency matrices
def test_symmetric(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    initial_belief_state = environment.reset(key)
    assert jnp.all(
        initial_belief_state[:, 1:, :]
        == jnp.transpose(initial_belief_state[:, 1:, :], axes=(0, 2, 1))
    )


# Two consecutive resamples are different
def resample(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_belief_state = environment.reset(key)
    next_initial_belief_state = environment.reset(subkey)
    assert not jnp.array_equal(initial_belief_state, next_initial_belief_state)


# Check reward is always negative
def test_reward(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    initial_belief_state = environment.reset(key)
    belief_state_1, reward_1, terminate = environment.step(
        initial_belief_state, jnp.array([3])
    )
    belief_state_2, reward_2, terminate = environment.step(
        belief_state_1, jnp.array([1])
    )
    assert reward_1 < 0
    assert reward_2 < 0


# This test is specific to a certian graph
# Check that belief state's current knowledge same as blocking status after visiting enough nodes
# Check that agent's position is updated
# Check that reward and terminate are correct
def test_belief_state(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    initial_belief_state = environment.reset(key)

    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    environment.graph_realisation.plot_realised_graph(log_directory)
    belief_state_1, reward_1, terminate = environment.step(
        initial_belief_state, jnp.array([4])
    )
    belief_state_2, reward_2, terminate = environment.step(
        belief_state_1, jnp.array([3])
    )
    belief_state_3, reward_3, terminate = environment.step(
        belief_state_2, jnp.array([2])
    )
    # Check that agents position is updated
    assert not jnp.array_equal(belief_state_1[0, :1, :], belief_state_2[0, :1, :])
    assert not jnp.array_equal(belief_state_2[0, :1, :], belief_state_3[0, :1, :])
    # Empty/null for agent_pos part of edge_probs and weights
    assert jnp.sum(belief_state_1[1:, :1, :]) == 0
    assert terminate is True
    assert reward_3 == 0
    assert jnp.all(
        belief_state_3[0, 1:, :] == environment.graph_realisation.blocking_status
    )


# Check invalid action keeps the belief state and agents_pos the same but reward decreases
# Go to the same node twice
def test_invalid_action(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    initial_belief_state = environment.reset(key)
    belief_state_1, reward_1, terminate = environment.step(
        initial_belief_state, jnp.array([4])
    )
    belief_state_2, reward_2, terminate = environment.step(
        belief_state_1, jnp.array([4])
    )
    assert reward_2 < reward_1
    assert terminate is False
    assert jnp.all(belief_state_1 == belief_state_2)

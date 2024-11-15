import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
from Utils import hand_crafted_graphs
import pytest
import pytest_print as pp
import os

# These tests are for single agent only


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    return environment


# Check symmetric adjacency matrices
def test_symmetric(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    initial_env_state, initial_belief_state = environment.reset(key)
    assert jnp.all(
        initial_belief_state[:, 1:, :]
        == jnp.transpose(initial_belief_state[:, 1:, :], axes=(0, 2, 1))
    )
    assert jnp.array_equal(
        environment.graph_realisation.graph.weights,
        jnp.transpose(environment.graph_realisation.graph.weights),
    )
    assert jnp.array_equal(
        environment.graph_realisation.graph.blocking_prob,
        jnp.transpose(environment.graph_realisation.graph.blocking_prob),
    )


# Two consecutive resamples are different
def test_resample(printer, environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(99)
    highly_stochastic_environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.9)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = highly_stochastic_environment.reset(key)
    next_env_state, next_initial_belief_state = highly_stochastic_environment.reset(
        subkey
    )
    assert jnp.array_equal(
        highly_stochastic_environment.graph_realisation.graph.weights,
        highly_stochastic_environment.graph_realisation.graph.weights,
    )
    assert jnp.array_equal(
        highly_stochastic_environment.graph_realisation.graph.blocking_prob,
        highly_stochastic_environment.graph_realisation.graph.blocking_prob,
    )
    # start at the same origin but not necessarily same blocking status
    assert jnp.array_equal(initial_env_state[0, :1, :], next_env_state[0, :1, :])
    assert jnp.array_equal(initial_env_state[1, :, :], next_env_state[1, :, :])
    assert jnp.array_equal(initial_env_state[2, :, :], next_env_state[2, :, :])
    assert not jnp.array_equal(initial_belief_state, next_initial_belief_state)
    assert not jnp.array_equal(initial_env_state, next_env_state)


# Check reward is always negative
def test_reward(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([3])
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([1])
    )
    assert reward_1 < 0
    assert reward_2 < 0


# This test is specific to a certain graph
# Check that belief state's current knowledge same as blocking status after visiting enough nodes
# Check that agent's position is updated
# Check that reward and terminate are correct
def test_belief_state(printer, environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)

    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    environment.graph_realisation.plot_realised_graph(
        initial_env_state[0, 1:, :], log_directory, "test_graph.png"
    )
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([4])
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([3])
    )
    env_state_3, belief_state_3, reward_3, terminate, subkey = environment.step(
        subkey, env_state_2, belief_state_2, jnp.array([2])
    )
    # Check that agents position is updated
    assert not jnp.array_equal(belief_state_1[0, :1, :], belief_state_2[0, :1, :])
    assert not jnp.array_equal(belief_state_2[0, :1, :], belief_state_3[0, :1, :])
    # Empty/null for agent_pos part of edge_probs and weights
    assert jnp.sum(belief_state_1[1:, :1, :]) == 0
    assert terminate == jnp.bool_(True)
    assert reward_3 > 0
    # Test environment automatically reset when episode is done and not reset when episode is not done
    assert (
        jnp.argmax(env_state_3[0, :1, :])
        == environment.graph_realisation.graph.origin[0]
    )
    assert (
        jnp.argmax(env_state_2[0, :1, :])
        != environment.graph_realisation.graph.origin[0]
    )
    assert jnp.array_equal(belief_state_3, initial_belief_state)


# Check invalid action keeps the belief state and agents_pos the same but reward decreases
# Go to the same node twice
# Test environment does not automatically reset when episode is not done
def test_invalid_action(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([4])
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([4])
    )
    assert (reward_2 + reward_1) < reward_1
    assert terminate == jnp.bool_(False)
    assert jnp.all(belief_state_1 == belief_state_2)
    assert jnp.array_equal(env_state_1[0, :1, :], env_state_2[0, :1, :])


# check that weights and blocking probs are floats
def test_float(environment: CTP_environment.CTP):
    assert environment.graph_realisation.graph.weights.dtype == jnp.float32
    assert environment.graph_realisation.graph.blocking_prob.dtype == jnp.float32


# Test reproducibility (same key - call function twice)
def test_reproducibility(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(40)
    # test reproducibility of resampling
    initial_env_state, initial_belief_state = environment.reset(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    assert jnp.array_equal(initial_env_state, initial_env_state)

    # test reproducibility of creation
    key, subkey = jax.random.split(key)
    environment_old = CTP_environment.CTP(1, 1, 5, subkey, prop_stoch=0.4)
    environment_new = CTP_environment.CTP(1, 1, 5, subkey, prop_stoch=0.4)
    assert jnp.array_equal(
        environment_old.graph_realisation.graph.weights,
        environment_new.graph_realisation.graph.weights,
    )
    assert jnp.array_equal(
        environment_old.graph_realisation.graph.blocking_prob,
        environment_new.graph_realisation.graph.blocking_prob,
    )


def test_env_state(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([4])
    )
    assert jnp.array_equal(initial_env_state[0, :1, :], initial_belief_state[0, :1, :])
    assert jnp.array_equal(env_state_1[0, :1, :], belief_state_1[0, :1, :])
    assert jnp.array_equal(
        env_state_1[1, 1:, :], environment.graph_realisation.graph.weights
    )
    assert jnp.array_equal(
        env_state_1[2, 1:, :], environment.graph_realisation.graph.blocking_prob
    )


def test_non_zero(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    new_env_state, new_belief_state = environment.reset(subkey)
    assert jnp.sum(initial_env_state[0, 1:, :]) > 0
    assert jnp.sum(new_env_state[0, 1:, :]) > 0


# Test that if try to go down blocked edges, will get very negative reward
def test_invalid_action(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    initial_env_state, initial_belief_state = environment.reset(key)
    env_state_1, belief_state_1, reward_1, terminate, subkey = environment.step(
        subkey, initial_env_state, initial_belief_state, jnp.array([4])
    )
    env_state_2, belief_state_2, reward_2, terminate, subkey = environment.step(
        subkey, env_state_1, belief_state_1, jnp.array([1])
    )
    assert reward_2 < -100


def test_hand_crafted_graphs():
    key = jax.random.PRNGKey(50)
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs/Unit_Tests")
    n_node_diamond, handcrafted_dict_diamond = (
        hand_crafted_graphs.get_diamond_shaped_graph()
    )
    diamond_environment = CTP_environment.CTP(
        1, 1, n_node_diamond, key, handcrafted_graph=handcrafted_dict_diamond
    )
    diamond_environment.reset(key)
    diamond_environment.graph_realisation.plot_realised_graph(
        diamond_environment.graph_realisation.sample_blocking_status(key),
        log_directory,
        "diamond_graph.png",
    )

    n_node_stochastic_edge, handcrafted_dict_stochastic = (
        hand_crafted_graphs.get_stochastic_edge_graph()
    )
    stochastic_environment = CTP_environment.CTP(
        1, 1, n_node_stochastic_edge, key, handcrafted_graph=handcrafted_dict_stochastic
    )
    stochastic_environment.reset(key)
    stochastic_environment.graph_realisation.plot_realised_graph(
        stochastic_environment.graph_realisation.sample_blocking_status(key),
        log_directory,
        "stochastic_graph.png",
    )

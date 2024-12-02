import pytest
import pytest_print
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment, CTP_generator
from Agents.dqn_masking import DQN_Masking
import os
from Utils import invalid_action_masking
from Networks import CNN


def test_dqn_masking(printer):
    key = jax.random.PRNGKey(0)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    logs_directory = os.path.join(parent_directory, "logs")

    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    env_key, action_key, buffer_key = jax.random.split(subkey, 3)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)

    model = CNN.Flax_CNN(32, [600, 300, 100], 5)
    model_params = model.init(subkey, jnp.ones((4, 6, 5)))

    agent = DQN_Masking(model, 1, 5)
    new_env_state, new_belief_state = environment.reset(env_key)
    environment.graph_realisation.plot_realised_graph(
        new_env_state[0, 1:, :],
        os.path.join(logs_directory, "Unit_Tests"),
        "test_invalid_action_graph.png",
    )
    for i in range(10):
        current_belief_state = new_belief_state
        invalid_action_mask = invalid_action_masking.decide_validity_of_action_space(
            new_env_state
        )
        action, action_key = agent.act(
            action_key, model_params, current_belief_state, 0
        )
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, new_env_state, current_belief_state, action
        )
        assert reward > -190

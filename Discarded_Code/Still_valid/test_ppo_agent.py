import jax
import jax.numpy as jnp
import pytest
import pytest_print as pp
import sys
import os

sys.path.append("..")
from Environment import CTP_generator, CTP_environment
from Agents.ppo import PPO
from Networks.densenet import DenseNet_ActorCritic
from Networks.actor_critic_network import ActorCritic_CNN_10
import flax
from Utils.augmented_belief_state import get_augmented_optimistic_belief


def test_action_masking_augmented(printer):
    key = jax.random.PRNGKey(30)
    key, subkey = jax.random.split(key)
    env_key, action_key, buffer_key = jax.random.split(subkey, 3)
    environment = CTP_environment.CTP(1, 1, 10, key, prop_stoch=0.4)
    model = ActorCritic_CNN_10(10)
    state_shape = (5, 11, 10)
    """
    init_params = model.init(
        jax.random.PRNGKey(0), jax.random.normal(action_key, state_shape)
    )
    """
    # Load params
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_directory, "Logs", "Unit_Tests")

    model_path = os.path.join(
        parent_directory, "Logs", "test_augmented", "weights.flax"
    )
    with open(model_path, "rb") as f:
        init_params = flax.serialization.from_bytes(None, f.read())
    agent = PPO(
        model,
        environment,
        discount_factor=1.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coeff=0.1,
        ent_coeff=0.05,
        batch_size=1,
        num_minibatches=1,
        horizon_length=3,
        reward_exceed_horizon=-200,
        num_loops=2,
        anneal_ent_coeff=True,
        deterministic_inference_policy=False,
        ent_coeff_schedule="linear",
        division_plateau=3,
    )
    new_env_state, new_belief_state = environment.reset(env_key)
    blocking_status = new_env_state[0, 1:, :]
    environment.graph_realisation.plot_realised_graph(
        blocking_status, log_directory, "bug.png"
    )
    augmented_belief_state = get_augmented_optimistic_belief(new_belief_state)
    pi, _ = model.apply(init_params, augmented_belief_state)
    printer(pi.probs)

    """
    for i in range(2):
        action, action_key = agent.act(action_key, init_params, new_belief_state, 0)
        printer("Action")
        printer(action)
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, new_env_state, new_belief_state, action
        )
        printer("Reward")
        printer(reward)
        assert reward > -2
    """

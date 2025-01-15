import pytest
import pytest_print
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment_generalize, CTP_generator
from Networks.big_cnn import Big_CNN_30
from Evaluation.inference_during_training import get_average_testing_stats
from Agents.ppo import PPO
import argparse


def test_get_testing_stats(printer):
    inference_key = jax.random.PRNGKey(0)
    online_key = jax.random.PRNGKey(1)
    testing_environment = CTP_environment_generalize.CTP_General(
        1, 1, 5, inference_key, prop_stoch=0.4
    )
    model = Big_CNN_30(5)
    state_shape = (6, 6, 5)
    init_params = model.init(
        jax.random.PRNGKey(0), jax.random.normal(online_key, state_shape)
    )
    agent = PPO(model, testing_environment)

    simulated_args = [
        "--random_seed_for_inference",
        "1",
        "--factor_testing_timesteps",
        "3",
        "--n_node",
        "5",
    ]
    parser = argparse.ArgumentParser(description="Simulate argparse without CLI")
    parser.add_argument(
        "--reward_exceed_horizon",
        type=float,
        help="Should be equal to or more negative than -1",
        required=False,
        default=-1.5,
    )
    parser.add_argument(
        "--horizon_length_factor",
        type=int,
        help="Factor to multiply with number of nodes to get the maximum horizon length",
        required=False,
        default=2,
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=101
    )
    parser.add_argument(
        "--factor_testing_timesteps",
        type=int,
        required=False,
        default=10,
        help="Factor to multiple with number of nodes to get the number of timesteps to perform testing on during training (in order to plot the learning curve)",
    )
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=False,
        default=5,
    )
    args = parser.parse_args(simulated_args)

    average_competitive_ratio = get_average_testing_stats(
        testing_environment, agent, init_params, args
    )
    printer(average_competitive_ratio)

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
from flax.core.frozen_dict import FrozenDict


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
    agent = PPO(
        model,
        testing_environment,
        discount_factor=1.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coeff=0.15,
        ent_coeff=0.15,
        batch_size=100,
        num_minibatches=1,
        horizon_length=10,
        reward_exceed_horizon=-1.5,
        num_loops=3,
        anneal_ent_coeff=True,
        deterministic_inference_policy=True,
        ent_coeff_schedule="sigmoid",
        division_plateau=5,
    )

    arguments = FrozenDict(
        {
            "factor_testing_timesteps": 7,
            "n_node": 5,
            "reward_exceed_horizon": -1.5,
            "horizon_length_factor": 2,
            "random_seed_for_inference": 1,
        }
    )

    average_competitive_ratio = get_average_testing_stats(
        testing_environment, agent, init_params, arguments
    )
    printer(average_competitive_ratio)
    assert jnp.isclose(average_competitive_ratio, 4.31, atol=0.01)

    loop_count = 1
    frequency_testing = 5
    testing_average_competitive_ratio = jax.lax.cond(
        loop_count % frequency_testing == 0,
        lambda _: get_average_testing_stats(
            testing_environment, agent, init_params, arguments
        ),
        lambda _: jnp.float16(0.0),
        None,
    )
    assert testing_average_competitive_ratio == 0

    loop_count = 5
    frequency_testing = 5
    testing_average_competitive_ratio = jax.lax.cond(
        loop_count % frequency_testing == 0,
        lambda _: get_average_testing_stats(
            testing_environment, agent, init_params, arguments
        ),
        lambda _: jnp.float16(0.0),
        None,
    )
    assert testing_average_competitive_ratio != 0

import jax
import jax.numpy as jnp
from Environment import CTP_generator, CTP_environment
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import optax
from edited_jym import (
    DQN,
    DQN_PER,
    UniformReplayBuffer,
    PrioritizedExperienceReplay,
    deep_rl_rollout,
)
from Networks import MLP

NUM_CHANNELS_IN_BELIEF_STATE = 3


def main(args):
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + args.n_node,
        args.n_node,
    )

    # The * unpacks the tuple
    buffer_state = {
        "states": jnp.empty((args.buffer_size, *state_shape), dtype=jnp.float32),
        "actions": jnp.empty((args.buffer_size,), dtype=jnp.int32),
        "rewards": jnp.empty((args.buffer_size,), dtype=jnp.float32),
        "next_states": jnp.empty((args.buffer_size, *state_shape), dtype=jnp.float32),
        "dones": jnp.empty((args.buffer_size,), dtype=jnp.bool_),
    }

    # Initialize the replay buffer
    replay_buffer = UniformReplayBuffer(args.buffer_size, args.batch_size)

    # Initialize network parameters
    key = jax.random.PRNGKey(args.random_seed)
    subkeys = jax.random.split(key, num=3)
    online_key, target_key, environment_key = subkeys
    online_net_params = MLP.simplest_model.init(
        online_key, jax.random.normal(online_key, state_shape)
    )
    target_net_params = MLP.simplest_model.init(
        target_key, jax.random.normal(target_key, state_shape)
    )
    optimizer = optax.adam(learning_rate=args.learning_rate)
    optimizer_state = optimizer.init(online_net_params)

    # Initialize the environment
    environment = CTP_environment.CTP(1, 1, 5, environment_key, prop_stoch=0.4)

    # Initialize the agent
    agent = DQN(
        MLP.simplest_model,
        args.discount_factor,
        environment.action_spaces.num_categories[0],
    )

    # epsilon decay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=False,
        default=5,
    )
    parser.add_argument(
        "--n_agent",
        type=int,
        help="Number of agents in the environment",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--n_episode",
        type=int,
        help="Number of episodes to run",
        required=False,
        default=3,
    )
    parser.add_argument("--learning_rate", type=str, required=False, default=0.001)
    parser.add_argument("--discount_factor", type=float, required=False, default=0.99)
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", required=False, default=5
    )
    parser.add_argument(
        "--buffer_size", type=int, help="Buffer size", required=False, default=20
    )
    parser.add_argument("--random_seed", type=int, required=False, default=30)
    args = parser.parse_args()

    main(args)

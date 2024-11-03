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


# Function for decaying epsilon
def epsilon_linear_schedule(
    start_e: float, end_e: float, t: int, duration: int
) -> float:
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)


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

    # Choose model based on args
    model = MLP.simplest_model_hk

    # Initialize network parameters
    key = jax.random.PRNGKey(args.random_seed)
    subkeys = jax.random.split(key, num=3)
    online_key, target_key, environment_key = subkeys
    online_net_params = model.init(
        online_key, jax.random.normal(online_key, state_shape)
    )
    target_net_params = model.init(
        target_key, jax.random.normal(target_key, state_shape)
    )
    optimizer = optax.adam(learning_rate=args.learning_rate)
    optimizer_state = optimizer.init(online_net_params)

    # Initialize the environment
    environment = CTP_environment.CTP(1, 1, 5, environment_key, prop_stoch=0.4)

    # Initialize the agent
    agent = DQN(
        model,
        args.discount_factor,
        environment.action_spaces.num_categories[0],
    )

    # Initialize the replay buffer with random samples (not necessary/optional)
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed
    )
    new_state = environment.reset(init_key)

    # Initialize the buffer with random samples.
    # Should start at the same states or let it continue?
    for i in range(args.buffer_size):
        state = new_state
        # set epsilon to 1 for exploration. act returns subkey
        action, action_key = agent.act(action_key, online_net_params, state, 1)
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        new_state, reward, done, env_key = environment.step(env_key, state, action)
        action = action[0]
        experience = (state, action, reward, new_state, done)
        buffer_state = replay_buffer.add(buffer_state, experience, i)
        if done:
            init_key, init_subkey = jax.random.split(init_key)
            new_state = environment.reset(init_subkey)

    # The decay rate in rollout_params is not actually the decay rate. It's duration of epsilon decaying.
    # Named it decay_rate just to minimize changes to jym's deep_rl_rollout function
    rollout_params = {
        "timesteps": args.n_episode,
        "random_seed": args.random_seed,
        "target_net_update_freq": args.target_net_update_freq,
        "model": model,
        "optimizer": optimizer,
        "buffer_state": buffer_state,
        "agent": agent,
        "env": environment,
        "replay_buffer": replay_buffer,
        "state_shape": state_shape,
        "buffer_size": args.buffer_size,
        "epsilon_decay_fn": epsilon_linear_schedule,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "decay_rate": args.epsilon_exploration_rate * args.n_episode * args.n_node,
    }
    out = deep_rl_rollout(**rollout_params)


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
    parser.add_argument("--epsilon_start", type=float, required=False, default=0.3)
    parser.add_argument("--epsilon_end", type=float, required=False, default=0.0)
    parser.add_argument(
        "--epsilon_exploration_rate", type=float, required=False, default=1e-3
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", required=False, default=5
    )
    parser.add_argument(
        "--reward_for_invalid_action", type=float, required=False, default=-200.0
    )

    # Hyperparameters specific to DQN
    parser.add_argument(
        "--buffer_size", type=int, help="Buffer size", required=False, default=20
    )
    parser.add_argument(
        "--target_net_update_freq",
        type=int,
        help="Frequency of updating the target network",
        required=False,
        default=10,
    )
    parser.add_argument("--random_seed", type=int, required=False, default=30)
    args = parser.parse_args()

    main(args)

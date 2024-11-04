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
    print("Setting up the environment ...")
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
    environment = CTP_environment.CTP(
        1,
        1,
        5,
        environment_key,
        prop_stoch=args.prop_stoch,
        k_edges=args.k_edges,
        grid_size=args.grid_size,
        reward_for_invalid_action=args.reward_for_invalid_action,
        num_stored_realisations=args.num_stored_realisations,
        patience_factor=args.patience_factor,
        reward_for_goal=args.reward_for_goal,
    )
    environment.graph_realisation.graph.plot_nx_graph(
        directory=log_directory, file_name="training_graph.png"
    )

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
    new_env_state, new_belief_state = environment.reset(init_key)

    # Initialize the buffer with random samples
    for i in range(args.buffer_size):
        current_belief_state = new_belief_state
        # set epsilon to 1 for exploration. act returns subkey
        action, action_key = agent.act(
            action_key, online_net_params, current_belief_state, 1
        )
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, new_env_state, current_belief_state, action
        )
        action = action[0]
        experience = (current_belief_state, action, reward, new_belief_state, done)
        buffer_state = replay_buffer.add(buffer_state, experience, i)

    # The decay rate in rollout_params is not actually the decay rate. It's duration of epsilon decaying.
    # Named it decay_rate just to minimize changes to jym's deep_rl_rollout function
    rollout_params = {
        "timesteps": args.time_steps,
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
        "decay_rate": args.epsilon_exploration_rate * args.time_steps * args.n_node,
    }
    print("Start training ...")
    out = deep_rl_rollout(**rollout_params)

    # Write to file to check correctness
    all_rewards = out["all_rewards"][-100:]
    # put in logs folder
    file_name = os.path.join(log_directory, "check.txt")
    with open(file_name, "w") as f:
        f.write(f"{all_rewards}\n")

    # Calculate and plot episodic returns
    # Plot losses
    # Store weights in a file


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
        "--time_steps",
        type=int,
        help="Probably around num_episodes you want * num_nodes* 1.2",
        required=False,
        default=30,
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

    # Hyperparameters specific to the environment
    parser.add_argument(
        "--reward_for_invalid_action", type=float, required=False, default=-200.0
    )
    parser.add_argument(
        "--reward_for_goal",
        type=int,
        help="Should be equal to or greater than 0",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--num_stored_realisations",
        type=int,
        help="Number of solvable blocking status (can be the same)",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--patience_factor",
        type=int,
        help="Factor of num_stored_realisations before we give up sampling for more solvable blocking status",
        required=False,
        default=4,
    )
    parser.add_argument(
        "--prop_stoch",
        type=float,
        help="Proportion of edges that are stochastic. Only specify either prop_stoch or k_edges.",
        required=False,
        default=0.4,
    )
    parser.add_argument(
        "--k_edges",
        type=int,
        help="Number of stochastic edges. Only specify either prop_stoch or k_edges",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--grid_size", type=int, help="Size of the grid", required=False, default=10
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

    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    main(args)

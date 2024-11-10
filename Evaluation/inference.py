import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
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
from Evaluation import plotting
import json
import flax

NUM_CHANNELS_IN_BELIEF_STATE = 3


def main(args):
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + args.n_node,
        args.n_node,
    )

    # Choose model based on args. Can use FLAX or HAIKU model
    # model = MLP.simplest_model_hk
    model = MLP.Flax_FCNetwork([128, 64, 32, 16], args.n_node)

    # Initialize network parameters and optimizer
    key = jax.random.PRNGKey(args.random_seed)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys
    online_net_params = model.init(
        online_key, jax.random.normal(online_key, state_shape)
    )

    file_name = os.path.join(log_directory, "weights.flax")
    with open(file_name, "rb") as f:
        serialized_params = f.read()
    online_net_params = flax.serialization.from_bytes(
        online_net_params, serialized_params
    )

    # Initialize the environment
    environment = CTP_environment.CTP(
        args.n_agent,
        1,
        args.n_node,
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
        directory=log_directory, file_name="debug_graph.png"
    )

    # Initialize the agent
    agent = DQN(
        model,
        args.discount_factor,
        environment.action_spaces.num_categories[0],
    )

    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed
    )
    new_env_state, new_belief_state = environment.reset(init_key)

    # Initialize the buffer with random samples
    for i in range(10):
        current_belief_state = new_belief_state
        # set epsilon to 1 for exploration. act returns subkey
        action, action_key = agent.act(
            action_key, online_net_params, current_belief_state, 0
        )
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, new_env_state, current_belief_state, action
        )
        print(action[0], reward, done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command-line arguments")
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
        help="Probably around num_episodes you want * num_nodes* 2",
        required=False,
        default=100000,
    )
    parser.add_argument("--learning_rate", type=str, required=False, default=0.001)
    parser.add_argument("--discount_factor", type=float, required=False, default=0.9)
    parser.add_argument("--epsilon_start", type=float, required=False, default=0.3)
    parser.add_argument("--epsilon_end", type=float, required=False, default=0.0)
    parser.add_argument(
        "--epsilon_exploration_rate", type=float, required=False, default=0.5
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", required=False, default=64
    )

    # Hyperparameters specific to the environment
    parser.add_argument(
        "--reward_for_invalid_action", type=float, required=False, default=-200.0
    )
    parser.add_argument(
        "--reward_for_goal",
        type=int,
        help="Should be 0 or positive",
        required=False,
        default=0,
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
    parser.add_argument("--random_seed", type=int, required=False, default=30)

    # Hyperparameters specific to DQN
    parser.add_argument(
        "--buffer_size", type=int, help="Buffer size", required=False, default=128
    )
    parser.add_argument(
        "--target_net_update_freq",
        type=int,
        help="Frequency of updating the target network",
        required=False,
        default=10,
    )

    # Args related to running/managing experiments
    parser.add_argument(
        "--save_model",
        type=bool,
        help="Whether to save the weights or not",
        required=False,
        default=True,
    )

    args = parser.parse_args()
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    main(args)

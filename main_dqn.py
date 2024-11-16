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
    SumTree,
    deep_rl_rollout,
    per_rollout,
)
from Networks import MLP, CNN
from Evaluation import plotting, visualize_policy
from Agents.ddqn_per import DDQN_PER
from Utils import hand_crafted_graphs
import json
import flax
import ast
import time
from Evaluation.optimal_path_length import dijkstra_shortest_path
from datetime import datetime
import warnings

warnings.simplefilter("error")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in cast"
)

NUM_CHANNELS_IN_BELIEF_STATE = 3
FACTOR_TO_MULTIPLY_NETWORK_SIZE = 5
FACTOR_TO_MULTIPLY_INFERENCE_TIMESTEPS = 100


# Excluding the last layer
def determine_network_size(args):
    if args.network_size is not None:
        return ast.literal_eval(args.network_size)
    else:
        first_layer_num_params = (
            FACTOR_TO_MULTIPLY_NETWORK_SIZE * args.n_node * (args.n_node + args.n_agent)
        )
        return [
            first_layer_num_params,
            int(first_layer_num_params / 2),
            int(first_layer_num_params / 4),
            int(first_layer_num_params / 8),
        ]


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

    if args.replay_buffer_type == "per":
        buffer_state = {
            "state": jnp.empty((args.buffer_size, *state_shape), dtype=jnp.float16),
            "action": jnp.empty((args.buffer_size,), dtype=jnp.uint8),
            "reward": jnp.empty((args.buffer_size,), dtype=jnp.float16),
            "next_state": jnp.empty(
                (args.buffer_size, *state_shape), dtype=jnp.float16
            ),
            "done": jnp.empty((args.buffer_size,), dtype=jnp.bool_),
            "priority": jnp.empty((args.buffer_size,), dtype=jnp.float16),
        }

    else:
        # The * unpacks the tuple
        buffer_state = {
            "states": jnp.empty((args.buffer_size, *state_shape), dtype=jnp.float16),
            "actions": jnp.empty((args.buffer_size,), dtype=jnp.uint8),
            "rewards": jnp.empty((args.buffer_size,), dtype=jnp.float16),
            "next_states": jnp.empty(
                (args.buffer_size, *state_shape), dtype=jnp.float16
            ),
            "dones": jnp.empty((args.buffer_size,), dtype=jnp.bool_),
        }

    # Initialize the replay buffer
    print("Replay buffer type: ", args.replay_buffer_type)
    if args.replay_buffer_type == "per":
        replay_buffer = PrioritizedExperienceReplay(
            args.buffer_size, args.batch_size, args.alpha, args.beta
        )
    else:
        replay_buffer = UniformReplayBuffer(args.buffer_size, args.batch_size)

    # Choose model based on args. Can use FLAX or HAIKU model
    # model = MLP.simplest_model_hk
    network_size = determine_network_size(args)
    print("Network size (excluding the last layer): ", network_size)
    if args.network_type == "CNN":
        model = CNN.Flax_CNN(args.num_filters, network_size, args.n_node)
        print("First layer size - convolutional: ", args.num_filters)
    else:
        model = MLP.Flax_FCNetwork(network_size, args.n_node)

    # Initialize network parameters and optimizer
    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys
    online_net_params = model.init(
        online_key, jax.random.normal(online_key, state_shape)
    )
    optimizer = optax.adam(learning_rate=args.learning_rate)

    # Initialize the environment
    if args.hand_crafted_graph == "diamond":
        n_nodes, defined_graph = CTP_generator.get_diamond_shaped_graph()
        environment = CTP_environment.CTP(
            num_agents=1,
            num_goals=1,
            num_nodes=n_nodes,
            key=environment_key,
            defined_graph=defined_graph,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
        )
    elif args.hand_crafted_graph == "n_stochastic":
        n_nodes, defined_graph = hand_crafted_graphs.get_stochastic_edge_graph()
        environment = CTP_environment.CTP(
            num_agents=1,
            num_goals=1,
            num_nodes=n_nodes,
            key=environment_key,
            defined_graph=defined_graph,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
        )
    else:
        environment = CTP_environment.CTP(
            args.n_agent,
            1,
            args.n_node,
            environment_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=args.grid_size,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
        )
    environment.graph_realisation.graph.plot_nx_graph(
        directory=log_directory, file_name="training_graph.png"
    )

    # Initialize the agent
    if args.replay_buffer_type == "per":
        if args.double_dqn:
            print("Using DDQN with PER")
            agent = DDQN_PER(
                model, args.discount_factor, environment.action_spaces.num_categories[0]
            )
        else:
            agent = DQN_PER(
                model,
                args.discount_factor,
                environment.action_spaces.num_categories[0],
            )
    else:
        agent = DQN(
            model,
            args.discount_factor,
            environment.action_spaces.num_categories[0],
        )

    # Initialize the replay buffer with random samples (not necessary/optional)
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed_for_training
    )
    new_env_state, new_belief_state = environment.reset(init_key)

    start_time = time.time()
    if args.replay_buffer_type == "uniform":
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

    print("Start training ...")
    if args.replay_buffer_type == "per":
        rollout_params = {
            "timesteps": args.time_steps,
            "random_seed": args.random_seed_for_training + 1,
            "target_net_update_freq": args.target_net_update_freq,
            "model": model,
            "optimizer": optimizer,
            "buffer_state": buffer_state,
            "tree_state": jnp.zeros(2 * args.buffer_size - 1),
            "agent": agent,
            "env": environment,
            "state_shape": state_shape,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
            "beta": args.beta,
            "discount": args.discount_factor,
            "epsilon_decay_fn": epsilon_linear_schedule,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "duration": args.epsilon_exploration_rate * args.time_steps,
        }
        out = per_rollout(**rollout_params)
        pass
    else:
        rollout_params = {
            "timesteps": args.time_steps,
            "random_seed": args.random_seed_for_training + 1,
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
            "duration": args.epsilon_exploration_rate * args.time_steps,
        }
        out = deep_rl_rollout(**rollout_params)

    print("Start plotting and storing weights ...")
    # Store weights in a file (for loading in the future)
    # Store to .pickle file if using Haiku model
    if args.save_model:
        # File can have any ending
        with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
            f.write(flax.serialization.to_bytes(out["model_params"]))
    # Put here to ensure timing is correct (plotting time is negligible)
    end_time = time.time()
    elapsed_time = end_time - start_time

    last_average_loss = plotting.plot_loss(
        out["all_done"], out["losses"], log_directory
    )

    training_result_dict = plotting.save_data_and_plotting(
        out["all_done"],
        out["all_rewards"],
        out["all_optimal_path_lengths"],
        log_directory,
        training=True,
    )

    # Evaluate the model
    # Test on the same graph
    print("Start evaluating ...")
    num_steps_for_inference = args.n_node * FACTOR_TO_MULTIPLY_INFERENCE_TIMESTEPS
    test_all_rewards = jnp.zeros([num_steps_for_inference], dtype=jnp.float16)
    test_all_actions = jnp.zeros([num_steps_for_inference], dtype=jnp.uint8)
    test_all_positions = jnp.zeros([num_steps_for_inference], dtype=jnp.int8)
    test_all_done = jnp.zeros([num_steps_for_inference], dtype=jnp.bool_)
    test_all_optimal_path_lengths = jnp.zeros(
        [num_steps_for_inference], dtype=jnp.float16
    )
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed_for_inference
    )
    env_state, belief_state = environment.reset(init_key)

    def _fori_inference(i: int, val: tuple):
        (
            action_key,
            env_key,
            env_state,
            belief_state,
            test_all_actions,
            test_all_positions,
            test_all_rewards,
            test_all_done,
            test_all_optimal_path_lengths,
        ) = val
        current_belief_state = belief_state
        current_env_state = env_state
        action, action_key = agent.act(
            action_key, out["model_params"], current_belief_state, 0
        )
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        env_state, belief_state, reward, done, env_key = environment.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]
        shortest_path = jax.lax.cond(
            done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                environment.graph_realisation.graph.origin,
                environment.graph_realisation.graph.goal,
            ),
            lambda _: 0.0,
            operand=None,
        )
        test_all_rewards = test_all_rewards.at[i].set(reward)
        test_all_done = test_all_done.at[i].set(done)
        test_all_optimal_path_lengths = test_all_optimal_path_lengths.at[i].set(
            shortest_path
        )
        test_all_actions = test_all_actions.at[i].set(action)
        test_all_positions = test_all_positions.at[i].set(
            jnp.argmax(current_env_state[0, : args.n_agent]).astype(jnp.int8)
        )
        val = (
            action_key,
            env_key,
            env_state,
            belief_state,
            test_all_actions,
            test_all_positions,
            test_all_rewards,
            test_all_done,
            test_all_optimal_path_lengths,
        )
        return val

    testing_val_init = (
        action_key,
        env_key,
        env_state,
        belief_state,
        test_all_actions,
        test_all_positions,
        test_all_rewards,
        test_all_done,
        test_all_optimal_path_lengths,
    )
    vals = jax.lax.fori_loop(
        0, num_steps_for_inference, _fori_inference, testing_val_init
    )
    (
        action_key,
        env_key,
        env_state,
        belief_state,
        test_all_actions,
        test_all_positions,
        test_all_rewards,
        test_all_done,
        test_all_optimal_path_lengths,
    ) = vals

    testing_result_dict = plotting.save_data_and_plotting(
        test_all_done,
        test_all_rewards,
        test_all_optimal_path_lengths,
        log_directory,
        training=False,
    )

    # Visualize the policy
    policy = visualize_policy.get_policy(
        args.n_node, test_all_actions, test_all_positions
    )

    # Record hyperparameters and results in JSON file
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = {"current_datetime": current_datetime}
    dict_args = vars(args)
    args_path = os.path.join(log_directory, "Hyperparamters_Results" + ".json")
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)
        fh.write("\n")
        json.dump(date_time, fh, indent=4)
        fh.write("\n")
        json.dump({"Total training time in seconds": elapsed_time}, fh)
        fh.write("\n")
        json.dump({"Last average loss:": last_average_loss}, fh)
        fh.write("\n")
        fh.write("Training results: \n")
        json.dump(training_result_dict, fh, indent=4)
        fh.write("\n")
        fh.write("Testing results: \n")
        json.dump(testing_result_dict, fh, indent=4)
        fh.write("\n")
        fh.write("Policy: \n")
        json_str = (
            "[\n" + ",\n".join(json.dumps(row) for row in policy.tolist()) + "\n]"
        )
        fh.write(json_str)
    print("All done!")


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
        help="Probably around num_episodes you want * num_nodes* 2",
        required=False,
        default=1000000,
    )
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument("--discount_factor", type=float, required=False, default=1.0)
    parser.add_argument("--epsilon_start", type=float, required=False, default=0.8)
    parser.add_argument("--epsilon_end", type=float, required=False, default=0.0)
    parser.add_argument(
        "--epsilon_exploration_rate", type=float, required=False, default=0.6
    )
    parser.add_argument(
        "--batch_size", type=int, help="Batch size", required=False, default=128
    )
    parser.add_argument(
        "--network_size",
        type=str,
        help="Size of each layer of the network (excluding the last layer) as a string. Ex. [128,64,32,16]",
        required=False,
        default=None,
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
        "--factor_expensive_edge", type=float, required=False, default=1.0
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
    parser.add_argument(
        "--random_seed_for_training", type=int, required=False, default=30
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=40
    )

    # Hyperparameters specific to DQN
    parser.add_argument(
        "--buffer_size", type=int, help="Buffer size", required=False, default=2000
    )
    parser.add_argument(
        "--target_net_update_freq",
        type=int,
        help="Frequency of updating the target network",
        required=False,
        default=40,
    )

    # Hyerparameterse related to the network
    parser.add_argument(
        "--network_type", type=str, help="FC,CNN", required=False, default="CNN"
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        help="Number of filters in CNN",
        required=False,
        default=32,
    )

    # Args related to running/managing experiments
    parser.add_argument(
        "--save_model",
        type=bool,
        help="Whether to save the weights or not",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--log_directory", type=str, help="Directory to store logs", required=True
    )
    parser.add_argument(
        "--hand_crafted_graph",
        type=str,
        help="Options: None,diamond,n_stochastic. If anything other than None is specified, all other args relating to environment such as num of nodes are ignored.",
        required=False,
        default="None",
    )

    # Args related to Prioritized Experience Replay
    parser.add_argument(
        "--replay_buffer_type",
        type=str,
        help="Type of replay buffer: uniform, per",
        required=False,
        default="uniform",
    )
    parser.add_argument(
        "--alpha", type=float, help="Alpha value for PER", required=False, default=0.6
    )
    parser.add_argument(
        "--beta", type=float, help="Beta value for PER", required=False, default=1.0
    )

    # DDQN
    parser.add_argument(
        "--double_dqn",
        action="store_true",
        help="Whether to use double DQN or not. Must use with Prioritized Experience Replay",
    )

    args = parser.parse_args()
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    main(args)

import jax
import jax.numpy as jnp
import argparse
import sys
import os

sys.path.append("..")
from Baselines import optimistic_agent
from Environment import CTP_generator, CTP_environment_generalize, CTP_environment
from Evaluation import optimal_path_length
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Evaluation import plotting
from distutils.util import strtobool
import time
from datetime import datetime
from Utils import generate_graphs
from jax_tqdm import scan_tqdm
import json
import wandb


def main(args):
    print("Running optimistic baseline")
    start_time = time.time()
    inference_key = jax.random.PRNGKey(args.random_seed_for_inference)
    if args.generalize:
        environment = CTP_environment_generalize.CTP_General(
            args.n_agent,
            1,
            args.n_node,
            inference_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=args.grid_size,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
            deal_with_unsolvability=args.deal_with_unsolvability,
            patience=args.patience,
            num_stored_graphs=args.factor_inference_timesteps // 2,
            loaded_graphs=inference_graphs,
        )
    else:
        environment = CTP_environment.CTP(
            args.n_agent,
            1,
            args.n_node,
            inference_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=args.grid_size,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
            deal_with_unsolvability=args.deal_with_unsolvability,
            patience=args.patience,
        )
    num_steps_for_inference = args.n_node * args.factor_inference_timesteps
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed_for_inference
    )
    new_env_state, new_belief_state = environment.reset(init_key)
    agent = optimistic_agent.Optimistic_Agent()

    # Don't need to worry about episode exceeding horizon length
    @scan_tqdm(num_steps_for_inference)
    def _one_step_inference(runner_state, unused):
        (
            current_env_state,
            current_belief_state,
            key,
            previous_episode_done,
        ) = runner_state
        _, env_key = jax.random.split(key, 2)
        action = agent.act(current_belief_state)
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]

        env_key, _ = jax.random.split(env_key)

        # Calculate shortest path at the beginning of the episode
        goal = jnp.unravel_index(
            jnp.argmax(current_belief_state[3, 1:, :]),
            (environment.num_nodes, environment.num_nodes),
        )[0]
        origin = jnp.argmax(current_belief_state[0, :1, :])
        shortest_path = jax.lax.cond(
            previous_episode_done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                origin,
                goal,
            ),
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=None,
        )
        position = jnp.argmax(current_env_state[0, : args.n_agent]).astype(jnp.int8)

        runner_state = (
            new_env_state,
            new_belief_state,
            env_key,
            done,
        )
        transition = (done, action, reward, shortest_path, position)
        return runner_state, transition

    runner_state = (
        new_env_state,
        new_belief_state,
        env_key,
        jnp.bool_(True),
    )
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(num_steps_for_inference)
    )
    test_all_done = inference_traj_batch[0]
    test_all_rewards = inference_traj_batch[2]
    test_all_optimal_path_lengths = inference_traj_batch[3]
    testing_result_dict = plotting.save_data_and_plotting(
        test_all_done,
        test_all_rewards,
        test_all_optimal_path_lengths,
        log_directory,
        training=False,
        file_name_excel_sheet_episode="optimistic_baseline_episode_output.xlsx",
        file_name_excel_sheet_timestep="optimistic_baseline_timestep_output.xlsx",
        file_name_regret_episode="optimistic_baseline_episode_regret.png",
        file_name_competitive_ratio_episode="optimistic_baseline_episode_competitive_ratio.png",
        file_name_episodic_reward="optimistic_baseline_episodic_reward.png",
    )
    end_time = time.time()
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = {"current_datetime": current_datetime}
    dict_args = vars(args)
    args_path = os.path.join(log_directory, "Optimistic_baseline" + ".json")
    end_time = time.time()
    total_time = end_time - start_time
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)
        fh.write("\n")
        json.dump(date_time, fh, indent=4)
        fh.write("\n")
        json.dump({"Total time in seconds": total_time}, fh)
        fh.write("\n")
        json.dump(testing_result_dict, fh, indent=4)
    print("All done!")


if __name__ == "__main__":
    # Only include relevant arguments, not an exhaustive list compared to the main_ppo.py script
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=True,
    )
    parser.add_argument(
        "--n_agent",
        type=int,
        help="Number of agents in the environment",
        required=False,
        default=1,
    )
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
        "--reward_exceed_horizon",
        type=float,
        help="Should be equal to or more negative than -1",
        required=False,
        default=-1.1,
    )
    parser.add_argument(
        "--horizon_length_factor",
        type=int,
        help="Factor to multiply with number of nodes to get the maximum horizon length",
        required=False,
        default=5,
    )
    parser.add_argument(
        "--factor_expensive_edge", type=float, required=False, default=1.0
    )
    parser.add_argument(
        "--deal_with_unsolvability",
        type=str,
        default="expensive_if_unsolvable",
        required=False,
        help="Options: always_expensive_edge, expensive_if_unsolvable, resample",
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=5,
        help="Number of times we try to resample a solvable realisation before giving up. If any other options besides resample is chosen for deal_with_unsolvable, then this is not applicable",
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
        "--grid_size", type=int, help="Size of the grid", required=False, default=None
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=40
    )
    parser.add_argument(
        "--log_directory", type=str, help="Directory to store logs", required=True
    )
    parser.add_argument(
        "--generalize",
        type=lambda x: bool(strtobool(x)),
        default=True,
        required=False,
        help="Whether to train and perform inference with multiple different graphs",
    )
    parser.add_argument(
        "--factor_inference_timesteps",
        type=int,
        required=False,
        default=500,
        help="Number to multiply with the number of nodes to get the total number of inference timesteps",
    )
    parser.add_argument(
        "--graph_identifier",
        type=str,
        required=False,
        default="2000_prop_stoch_0.4",
    )
    parser.add_argument(
        "--num_stored_graphs",
        type=int,
        required=False,
        help="How many different graphs will be seen by the agent",
        default=2000,
    )
    args = parser.parse_args()

    # Disable wandb logging
    wandb.init(
        project="baseline",
        name=args.log_directory,
        config=vars(args),
        mode="disabled",
    )

    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    if args.generalize:
        _, inference_graphs = generate_graphs.load_graphs(args)
    main(args)

    # How to ensure same args as ppo. Store dict as .pickle and compare?
    # Pass args as input and call the main function in this script from main_ppo?

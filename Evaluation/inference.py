import time
import flax
import os
import sys
import jax
import jax.numpy as jnp

sys.path.append("..")
from Evaluation import plotting
from Environment import CTP_environment, CTP_environment_generalize
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Evaluation import visualize_policy
from Baselines.optimistic_agent import Optimistic_Agent
from Utils.get_params import extract_params
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import wandb
from jax_tqdm import scan_tqdm
from Evaluation.plot_learning_curve import plot_learning_curve


def plotting_inference(
    log_directory,
    time_info,
    model_params,
    out,
    environment: CTP_environment_generalize.CTP_General,
    agent,
    args,
    n_node,
    total_losses,
    value_loss=None,
    loss_actor=None,
    entropy_loss=None,
):
    print("Start plotting and storing weights ...")
    # Store weights in a file (for loading in the future)
    # File can have any ending
    with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
        f.write(flax.serialization.to_bytes(model_params))
    # Put here to ensure timing is correct (plotting time is negligible)
    end_time = time.time()
    elapsed_time = end_time - time_info["start_training_time"]

    training_result_dict = plotting.save_data_and_plotting(
        out["all_done"],
        out["all_rewards"],
        out["all_optimal_path_lengths"],
        log_directory,
        reward_exceed_horizon=args.reward_exceed_horizon,
        training=True,
    )

    # Plot the learning curve
    plot_learning_curve(out["testing_average_competitive_ratio"], log_directory, args)

    # Evaluate the model
    # Test on the same graph
    print("Start evaluating ...")
    num_steps_for_inference = n_node * args.factor_inference_timesteps
    init_key, action_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(3) + args.random_seed_for_inference
    )
    new_env_state, new_belief_state = environment.reset(init_key)
    optimistic_agent = Optimistic_Agent()

    @scan_tqdm(num_steps_for_inference)
    def _one_step_inference(runner_state, unused):
        (
            current_env_state,
            current_belief_state,
            key,
            timestep_in_episode,
            previous_episode_done,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)
        # Agent acts
        action, action_key = agent.act(
            action_key, model_params, current_belief_state, 0
        )
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]

        # Stop the episode and reset if exceed horizon length
        env_key, reset_key = jax.random.split(env_key)
        # Reset timestep if finish episode
        timestep_in_episode = jax.lax.cond(
            done, lambda _: 0, lambda _: timestep_in_episode, operand=None
        )
        # Reset if exceed horizon length. Otherwise, increment
        new_env_state, new_belief_state, reward, timestep_in_episode, done = (
            jax.lax.cond(
                timestep_in_episode >= args.horizon_length_factor * args.n_node,
                lambda _: (
                    *environment.reset(reset_key),
                    jnp.float16(args.reward_exceed_horizon),
                    0,
                    True,
                ),
                lambda _: (
                    new_env_state,
                    new_belief_state,
                    reward,
                    timestep_in_episode + 1,
                    done,
                ),
                operand=None,
            )
        )

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
        optimistic_baseline = jax.lax.cond(
            previous_episode_done,
            lambda _: optimistic_agent.get_path_length(
                environment, current_belief_state, current_env_state, env_key
            ),
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=None,
        )
        position = jnp.argmax(current_env_state[0, : args.n_agent]).astype(jnp.int8)

        runner_state = (
            new_env_state,
            new_belief_state,
            env_key,
            timestep_in_episode,
            done,
        )
        transition = (
            done,
            action,
            reward,
            shortest_path,
            position,
            optimistic_baseline,
        )
        return runner_state, transition

    runner_state = (
        new_env_state,
        new_belief_state,
        env_key,
        jnp.int32(0),
        jnp.bool_(True),
    )
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(num_steps_for_inference)
    )
    test_all_done = inference_traj_batch[0]
    test_all_actions = inference_traj_batch[1]
    test_all_rewards = inference_traj_batch[2]
    test_all_optimal_path_lengths = inference_traj_batch[3]
    test_all_positions = inference_traj_batch[4]
    test_optimistic_baseline = inference_traj_batch[5]

    testing_result_dict = plotting.save_data_and_plotting(
        test_all_done,
        test_all_rewards,
        test_all_optimal_path_lengths,
        log_directory,
        reward_exceed_horizon=args.reward_exceed_horizon,
        all_optimistic_baseline=test_optimistic_baseline,
        training=False,
    )

    # Plot the loss
    if value_loss is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(total_losses, linestyle="-", color="red", label="Total Weighted Loss")
        plt.plot(
            args.vf_coeff * value_loss,
            linestyle="-",
            color="blue",
            label="Weighted Value Loss",
        )
        plt.plot(loss_actor, linestyle="-", color="green", label="Actor Loss")
        if args.anneal_ent_coeff:
            ent_coeff_values = np.array(
                [
                    agent._ent_coeff_schedule(i)
                    for i in range(args.time_steps // args.num_steps_before_update)
                ]
            )
            ent_coeff_values = np.repeat(
                ent_coeff_values, args.num_update_epochs * args.num_minibatches
            )
        else:
            ent_coeff_values = np.full(entropy_loss.shape, args.ent_coeff)
        weighted_entropy_loss = ent_coeff_values * entropy_loss
        plt.plot(
            weighted_entropy_loss,
            linestyle="-",
            color="orange",
            label="Weighted Entropy Loss",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss Plot")
        plt.legend()
        plt.savefig(os.path.join(log_directory, "PPO_Loss.png"))
        loss = {
            "Total Loss": total_losses[-1].astype(float),
            "Weighted Value Loss": args.vf_coeff * value_loss[-1].astype(float),
            "Actor Loss": loss_actor[-1].astype(float),
            "Weighted Entropy Loss": args.ent_coeff * entropy_loss[-1].astype(float),
        }
    else:
        # DQN
        last_average_loss = plotting.plot_loss(
            out["all_done"], total_losses, log_directory
        )
        loss = {"Last average loss": last_average_loss}

    # Visualize the policy
    if args.generalize is False:
        policy = visualize_policy.get_policy(
            n_node, test_all_actions, test_all_positions
        )

    # Record hyperparameters and results in JSON file
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_time = {"current_datetime": current_datetime}
    dict_args = vars(args)
    args_path = os.path.join(log_directory, "Hyperparameters_Results" + ".json")
    end_time = time.time()
    total_time = end_time - time_info["start_time"]
    with open(args_path, "w") as fh:
        json.dump(dict_args, fh)
        fh.write("\n")
        json.dump(date_time, fh, indent=4)
        fh.write("\n")
        json.dump(
            {
                "Total environment creation time in seconds": time_info[
                    "environment_creation_time"
                ]
            },
            fh,
        )
        fh.write("\n")
        json.dump({"Total training time in seconds": elapsed_time}, fh)
        fh.write("\n")
        json.dump({"Total time in seconds": total_time}, fh)
        fh.write("\n")
        json.dump(loss, fh)
        fh.write("\n")
        fh.write("Training results: \n")
        json.dump(training_result_dict, fh, indent=4)
        fh.write("\n")
        fh.write("Testing results: \n")
        json.dump(testing_result_dict, fh, indent=4)
        fh.write("\n")
        if args.generalize is False:
            fh.write("Policy: \n")
            json_str = (
                "[\n" + ",\n".join(json.dumps(row) for row in policy.tolist()) + "\n]"
            )
            fh.write(json_str)
        # Log the network architecture
        fh.write("\nNetwork architecture: \n")
        for layer_name, weights in extract_params(model_params):
            fh.write(f"{layer_name}: {weights.shape}\n")
        total_num_params = sum(p.size for p in jax.tree_util.tree_leaves(model_params))
        fh.write("Total number of parameters in the network: " + str(total_num_params))
        """
        fh.write("\nGraph Weights: \n")
        fh.write(
            "[\n"
            + ",\n".join(
                json.dumps(row)
                for row in environment.graph_realisation.graph.weights.tolist()
            )
            + "\n]"
        )
        fh.write("\nBlocking Probabilities: \n")
        fh.write(
            "[\n"
            + ",\n".join(
                json.dumps(row)
                for row in environment.graph_realisation.graph.blocking_prob.tolist()
            )
            + "\n]"
        )
        """
    print("All done!")

import time
import flax
import os
import sys
import jax
import jax.numpy as jnp

sys.path.append("..")
from Evaluation import plotting
from Environment import CTP_environment
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Evaluation import visualize_policy
from Utils.get_params import extract_params
from datetime import datetime
import json

FACTOR_TO_MULTIPLY_INFERENCE_TIMESTEPS = 100


def plotting_inference(
    log_directory,
    start_time,
    model_params,
    out,
    environment: CTP_environment.CTP,
    agent,
    args,
    n_node,
):
    print("Start plotting and storing weights ...")
    # Store weights in a file (for loading in the future)
    # File can have any ending
    with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
        f.write(flax.serialization.to_bytes(model_params))
    # Put here to ensure timing is correct (plotting time is negligible)
    end_time = time.time()
    elapsed_time = end_time - start_time

    """
    last_average_loss = plotting.plot_loss(
        out["all_done"], out["losses"], log_directory
    )
    """

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
    num_steps_for_inference = n_node * FACTOR_TO_MULTIPLY_INFERENCE_TIMESTEPS
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
            action_key, model_params, current_belief_state, 0
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
    policy = visualize_policy.get_policy(n_node, test_all_actions, test_all_positions)

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
        # json.dump({"Last average loss:": last_average_loss}, fh)
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
        # Log the network architecture
        fh.write("\nNetwork architecture: \n")
        for layer_name, weights in extract_params(model_params):
            fh.write(f"{layer_name}: {weights.shape}\n")
    print("All done!")

import os
import jax
import jax.numpy as jnp
from Networks.actor_critic_network import ActorCritic
from Environment import CTP_environment, CTP_generator
from Agents.ppo import PPO
from Evaluation import plotting
import argparse
import optax
from flax.training.train_state import TrainState
from typing import Sequence, NamedTuple, Any
import flax
import time
from datetime import datetime
import json

NUM_CHANNELS_IN_BELIEF_STATE = 3


def train_agent(args):
    # Initialize and setting things up
    print("Setting up the environment ...")
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + args.n_node,
        args.n_node,
    )

    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys

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

    model = ActorCritic(args.n_node)
    init_params = model.init(
        jax.random.PRNGKey(0), jax.random.normal(online_key, state_shape)
    )

    # Clip by global norm can be an args
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(args.learning_rate, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=init_params,
        tx=optimizer,
    )
    init_key, env_action_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(2) + args.random_seed_for_training
    )
    agent = PPO(
        model,
        environment,
        args.discount_factor,
        args.gae_lambda,
        args.clip_eps,
        args.vf_coeff,
        args.ent_coeff,
        batch_size=args.num_steps_before_update,
        num_minibatches=args.num_minibatches,
    )

    print("Start training ...")

    def _update_step(runner_state, unused):
        # Collect trajectories
        runner_state, traj_batch = jax.lax.scan(
            agent.env_step, runner_state, None, args.num_steps_before_update
        )
        # Calculate advantages
        train_state, new_env_state, current_belief_state, key = runner_state
        _, last_critic_val = model.apply(train_state.params, current_belief_state)
        advantages, targets = agent.calculate_gae(traj_batch, last_critic_val)
        # advantages and targets are of shape (num_steps_before_update,)

        # Update the network
        update_state = (train_state, traj_batch, advantages, targets, key)
        # traj_batch is a Transition tuple object where for ex.  traj_batch["done"] is of shape (num_steps_before_update,)
        update_state, total_loss = jax.lax.scan(
            agent._update_epoch, update_state, None, args.num_update_epochs
        )
        train_state = update_state[0]
        rng = update_state[-1]
        runner_state = (train_state, new_env_state, current_belief_state, rng)

        # Collect metrics
        metrics = {
            "losses": total_loss,
            "all_rewards": traj_batch.reward,
            "all_done": traj_batch.done,
            "all_optimal_path_lengths": traj_batch.shortest_path,
        }

        return runner_state, metrics

    start_time = time.time()
    new_env_state, new_belief_state = environment.reset(init_key)
    runner_state = (train_state, new_env_state, new_belief_state, env_action_key)
    runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_loops)
    train_state = runner_state[0]
    # Metrics will be stacked
    out = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), metrics)

    print("Start plotting and storing weights ...")
    with open(os.path.join(log_directory, "weights.flax"), "wb") as f:
        f.write(flax.serialization.to_bytes(train_state.params))
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

    print("Start evaluation ...")
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
        # Log the network architecture
        fh.write("\nNetwork architecture: \n")
        # for layer_name, layer_weights in train_state.params.items():
        #    fh.write(f"Layer: {layer_name}, Shape: {layer_weights.shape}\n")
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
        default=1000,
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
    parser.add_argument("--discount_factor", type=float, required=False, default=1.0)
    parser.add_argument("--learning_rate", type=float, required=False, default=0.00025)
    parser.add_argument("--num_update_epochs", type=int, required=False, default=4)

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

    # Args specific to PPO:
    parser.add_argument(
        "--num_steps_before_update", type=int, required=False, default=128
    )
    parser.add_argument("--gae_lambda", type=float, required=False, default=0.95)
    parser.add_argument("--clip_eps", type=float, required=False, default=0.2)
    parser.add_argument("--vf_coeff", type=float, required=False, default=0.5)
    parser.add_argument("--ent_coeff", type=float, required=False, default=0.01)
    parser.add_argument("--num_minibatches", type=int, required=False, default=4)

    args = parser.parse_args()
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    num_loops = args.time_steps // args.num_steps_before_update
    train_agent(args)

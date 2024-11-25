import os
import jax
import jax.numpy as jnp
from Networks.actor_critic_network import ActorCritic_CNN_10, ActorCritic_CNN_30
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
from Utils.get_params import extract_params
from Utils import hand_crafted_graphs
from Evaluation.inference import plotting_inference
import numpy as np
import wandb
from distutils.util import strtobool

NUM_CHANNELS_IN_BELIEF_STATE = 3


def linear_schedule(count):
    frac = 1.0 - (count // args.num_minibatches * args.num_update_epochs) / num_loops
    return args.learning_rate * frac


def main(args):
    # Initialize and setting things up
    print("Setting up the environment ...")
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + n_node,
        n_node,
    )

    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys

    if args.hand_crafted_graph == "diamond":
        _, defined_graph = hand_crafted_graphs.get_diamond_shaped_graph()
        environment = CTP_environment.CTP(
            num_agents=1,
            num_goals=1,
            num_nodes=n_node,
            key=environment_key,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
            handcrafted_graph=defined_graph,
        )
    elif args.hand_crafted_graph == "n_stochastic":
        _, defined_graph = hand_crafted_graphs.get_stochastic_edge_graph()
        environment = CTP_environment.CTP(
            num_agents=1,
            num_goals=1,
            num_nodes=n_node,
            key=environment_key,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
            handcrafted_graph=defined_graph,
        )
    else:
        environment = CTP_environment.CTP(
            args.n_agent,
            1,
            n_node,
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

    if n_node <= 10:
        model = ActorCritic_CNN_10(n_node)
    else:
        model = ActorCritic_CNN_30(n_node)

    init_params = model.init(
        jax.random.PRNGKey(0), jax.random.normal(online_key, state_shape)
    )

    # Clip by global norm can be an args
    if args.anneal_lr:
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(learning_rate=args.learning_rate, eps=1e-5),
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
        horizon_length=args.horizon_length_factor * n_node,
        reward_exceed_horizon=args.reward_exceed_horizon,
        num_loops=num_loops,
        anneal_ent_coeff=args.anneal_ent_coeff,
        deterministic_inference_policy=args.deterministic_inference_policy,
    )

    print("Start training ...")

    def _update_step(runner_state, unused):
        # Collect trajectories
        runner_state, traj_batch = jax.lax.scan(
            agent.env_step, runner_state, None, args.num_steps_before_update
        )
        # Calculate advantages
        # timestep_in_episode is unused here
        (
            train_state,
            new_env_state,
            current_belief_state,
            key,
            timestep_in_episode,
            loop_count,
        ) = runner_state
        _, last_critic_val = model.apply(train_state.params, current_belief_state)
        advantages, targets = agent.calculate_gae(traj_batch, last_critic_val)
        # advantages and targets are of shape (num_steps_before_update,)

        # Update the network
        update_state = (train_state, traj_batch, advantages, targets, key, loop_count)
        # traj_batch is a Transition tuple object where for ex.  traj_batch["done"] is of shape (num_steps_before_update,)
        update_state, total_loss = jax.lax.scan(
            agent._update_epoch, update_state, None, args.num_update_epochs
        )
        train_state = update_state[0]
        rng = update_state[-2]

        # Increment loop count
        loop_count += 1

        runner_state = (
            train_state,
            new_env_state,
            current_belief_state,
            rng,
            timestep_in_episode,
            loop_count,
        )

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
    timestep_in_episode = jnp.int32(0)
    loop_count = jnp.int32(0)
    runner_state = (
        train_state,
        new_env_state,
        new_belief_state,
        env_action_key,
        timestep_in_episode,
        loop_count,
    )
    runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_loops)
    train_state = runner_state[0]
    # Metrics will be stacked
    out = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), metrics)

    total_loss = out["losses"][0]
    value_loss = out["losses"][1][0]
    loss_actor = out["losses"][1][1]
    entropy_loss = out["losses"][1][2]

    plotting_inference(
        log_directory,
        start_time,
        train_state.params,
        out,
        environment,
        agent,
        args,
        n_node,
        np.array(total_loss),
        np.array(value_loss),
        np.array(loss_actor),
        np.array(entropy_loss),
    )


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
        "--random_seed_for_training", type=int, required=False, default=30
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=40
    )
    parser.add_argument("--discount_factor", type=float, required=False, default=1.0)
    parser.add_argument(
        "--anneal_lr",
        type=lambda x: bool(strtobool(x)),
        default=False,
        required=False,
        help="Whether to anneal the learning rate",
    )
    parser.add_argument("--learning_rate", type=float, required=False, default=0.001)
    parser.add_argument(
        "--num_update_epochs",
        type=int,
        help="After collecting trajectories, how many times each minibatch is updated.",
        required=False,
        default=4,
    )

    # Args related to running/managing experiments
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
    parser.add_argument(
        "--wandb_mode",
        type=str,
        help="offline/online/disabled",
        required=False,
        default="disabled",
    )
    parser.add_argument(
        "--wandb_project_name", type=str, required=False, default="no_name"
    )

    # Args specific to PPO:
    parser.add_argument(
        "--num_steps_before_update",
        type=int,
        help="How many timesteps to collect before updating the network",
        required=False,
        default=600,
    )
    parser.add_argument(
        "--gae_lambda",
        help="Control the trade-off between bias and variance in advantage estimates. High = Low bias, High variance as it depends on longer trajectories",
        type=float,
        required=False,
        default=0.95,
    )
    parser.add_argument(
        "--clip_eps",
        help="Related to how big of an update can be made",
        type=float,
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--vf_coeff",
        help="Contribution of the value loss to the total loss",
        type=float,
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--ent_coeff",
        help="Exploration coefficient",
        type=float,
        required=False,
        default=0.05,
    )
    parser.add_argument(
        "--anneal_ent_coeff",
        type=lambda x: bool(strtobool(x)),
        required=False,
        default=True,
        help="Whether to anneal the entropy (exploration) coefficient",
    )
    parser.add_argument(
        "--num_minibatches",
        help="Related to how the trajectory batch is split up for performing updating of the network",
        type=int,
        required=False,
        default=4,
    )
    parser.add_argument(
        "--deterministic_inference_policy",
        type=lambda x: bool(strtobool(x)),
        default=False,
        required=False,
        help="Whether to choose the action with the highest probability instead of sampling from the distribution",
    )

    args = parser.parse_args()
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    num_loops = args.time_steps // args.num_steps_before_update

    # Decide on num of nodes
    if args.hand_crafted_graph == "diamond":
        n_node, defined_graph = hand_crafted_graphs.get_diamond_shaped_graph()
    elif args.hand_crafted_graph == "n_stochastic":
        n_node, defined_graph = hand_crafted_graphs.get_stochastic_edge_graph()
    else:
        n_node = args.n_node

    # Initialize wandb project
    wandb.init(
        project=args.wandb_project_name,
        name=args.log_directory,
        config=vars(args),
        mode=args.wandb_mode,
    )
    main(args)
    wandb.finish()

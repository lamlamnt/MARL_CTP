import os
import jax
import jax.numpy as jnp
from Networks.actor_critic_network import ActorCritic_CNN_10, ActorCritic_CNN_30
from Networks.densenet import DenseNet_ActorCritic, DenseNet_ActorCritic_Same
from Networks.resnet import ResNet_ActorCritic
from Networks.big_cnn import Big_CNN_30
from Networks.densenet_float16 import DenseNet_ActorCritic_Float16
from Environment import CTP_environment, CTP_generator, CTP_environment_generalize
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
from Utils import hand_crafted_graphs, generate_graphs
from Evaluation.inference import plotting_inference
import numpy as np
import wandb
from distutils.util import strtobool
from jax_tqdm import scan_tqdm
import warnings
from Utils.augmented_belief_state import (
    get_augmented_optimistic_belief,
    get_augmented_optimistic_pessimistic_belief,
)
from Evaluation.inference_during_training import get_average_testing_stats
import flax.linen as nn
import sys
import yaml
from flax.core.frozen_dict import FrozenDict

"""
warnings.simplefilter("error")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in cast"
)
"""
NUM_CHANNELS_IN_BELIEF_STATE = 6


def decide_hand_crafted_graph(args):
    if args.hand_crafted_graph == "diamond":
        n_node, defined_graph = hand_crafted_graphs.get_diamond_shaped_graph()
    elif args.hand_crafted_graph == "n_stochastic":
        n_node, defined_graph = hand_crafted_graphs.get_stochastic_edge_graph()
    else:
        n_node = args.n_node
    return n_node, defined_graph


def main(args):
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs", args.log_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    num_loops = args.time_steps // args.num_steps_before_update
    if args.ent_coeff_schedule == "sigmoid_checkpoint":
        assert num_loops < args.sigmoid_total_nums_all // args.num_steps_before_update
        assert args.sigmoid_beginning_offset_num < args.sigmoid_total_nums_all

    def linear_schedule(count):
        frac = (
            1.0 - (count // args.num_minibatches * args.num_update_epochs) / num_loops
        )
        return args.learning_rate * frac

    # Decide on num of nodes
    if args.hand_crafted_graph == "diamond":
        n_node, defined_graph = hand_crafted_graphs.get_diamond_shaped_graph()
    elif args.hand_crafted_graph == "n_stochastic":
        n_node, defined_graph = hand_crafted_graphs.get_stochastic_edge_graph()
    else:
        n_node = args.n_node

    # Initialize and setting things up
    print("Setting up the environment ...")
    print("Add expensive edge: ", args.deal_with_unsolvability)
    if args.deal_with_unsolvability == "resample":
        print("Patience: ", args.patience)
    # Determine belief state shape
    state_shape = (
        NUM_CHANNELS_IN_BELIEF_STATE,
        args.n_agent + n_node,
        n_node,
    )

    key = jax.random.PRNGKey(args.random_seed_for_training)
    subkeys = jax.random.split(key, num=2)
    online_key, environment_key = subkeys

    # Create the training environment
    if args.generalize:
        start_time_environment_creation = time.time()
        environment = CTP_environment_generalize.CTP_General(
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
            deal_with_unsolvability=args.deal_with_unsolvability,
            patience=args.patience,
            num_stored_graphs=num_training_graphs,
            loaded_graphs=training_graphs,
            origin_node=args.origin_node,
        )
        end_time_environment_creation = time.time()
        time_environment_creation = (
            end_time_environment_creation - start_time_environment_creation
        )
    else:
        if args.hand_crafted_graph != "None":
            _, defined_graph = decide_hand_crafted_graph(args)
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
                deal_with_unsolvability=args.deal_with_unsolvability,
                patience=args.patience,
            )
        environment.graph_realisation.graph.plot_nx_graph(
            directory=log_directory, file_name="training_graph.png"
        )

    # Create a new environment to get unseen graphs for testing
    if args.generalize:
        inference_key = jax.random.PRNGKey(args.random_seed_for_inference)
        print("Start generating graphs for inference ...")
        testing_environment = CTP_environment_generalize.CTP_General(
            args.n_agent,
            1,
            n_node,
            inference_key,
            prop_stoch=args.prop_stoch,
            k_edges=args.k_edges,
            grid_size=args.grid_size,
            reward_for_invalid_action=args.reward_for_invalid_action,
            reward_for_goal=args.reward_for_goal,
            factor_expensive_edge=args.factor_expensive_edge,
            deal_with_unsolvability=args.deal_with_unsolvability,
            patience=args.patience,
            num_stored_graphs=num_inference_graphs,
            loaded_graphs=inference_graphs,
            origin_node=args.origin_node,
        )

    if args.network_type == "CNN":
        if n_node <= 10:
            model = ActorCritic_CNN_10(n_node)
        else:
            model = ActorCritic_CNN_30(n_node)
    elif args.network_type == "Densenet" or args.network_type == "Densenet_Same":
        densenet_act_fn_dict = {"leaky_relu": nn.leaky_relu, "tanh": nn.tanh}
        densenet_init_dict = {
            "kaiming_normal": nn.initializers.kaiming_normal(),
            "orthogonal": nn.initializers.orthogonal(jnp.sqrt(2)),
        }
        if args.network_type == "Densenet":
            model = DenseNet_ActorCritic(
                n_node,
                act_fn=densenet_act_fn_dict[args.network_activation_fn],
                densenet_kernel_init=densenet_init_dict[args.network_init],
                bn_size=args.densenet_bn_size,
                growth_rate=args.densenet_growth_rate,
                num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
            )
        else:
            model = DenseNet_ActorCritic_Same(
                n_node,
                act_fn=densenet_act_fn_dict[args.network_activation_fn],
                densenet_kernel_init=densenet_init_dict[args.network_init],
                bn_size=args.densenet_bn_size,
                growth_rate=args.densenet_growth_rate,
                num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
            )
    elif args.network_type == "Densenet_Float16":
        densenet_act_fn_dict = {"leaky_relu": nn.leaky_relu, "tanh": nn.tanh}
        densenet_init_dict = {
            "kaiming_normal": nn.initializers.kaiming_normal(dtype=jnp.float16),
            "orthogonal": nn.initializers.orthogonal(jnp.sqrt(2), dtype=jnp.float16),
        }
        model = DenseNet_ActorCritic_Float16(
            n_node,
            act_fn=densenet_act_fn_dict[args.network_activation_fn],
            densenet_kernel_init=densenet_init_dict[args.network_init],
            bn_size=args.densenet_bn_size,
            growth_rate=args.densenet_growth_rate,
            num_layers=tuple(map(int, (args.densenet_num_layers).split(","))),
        )
    elif args.network_type == "Big_CNN":
        model = Big_CNN_30(n_node)
    else:
        model = ResNet_ActorCritic(n_node)

    init_params = model.init(
        jax.random.PRNGKey(0), jax.random.normal(online_key, state_shape)
    )
    # Load in pre-trained network weights
    if args.load_network_directory is not None:
        network_file_path = os.path.join(
            current_directory, "Logs", args.load_network_directory, "weights.flax"
        )
        with open(network_file_path, "rb") as f:
            init_params = flax.serialization.from_bytes(init_params, f.read())

    # Clip by global norm can be an args
    if args.anneal_lr:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.optimizer_norm_clip),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    elif args.network_type == "Resnet":
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.optimizer_norm_clip),
            optax.sgd(learning_rate=args.learning_rate, momentum=0.9),
        )
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.optimizer_norm_clip),
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
        ent_coeff_schedule=args.ent_coeff_schedule,
        division_plateau=args.division_plateau,
        sigmoid_beginning_offset_num=args.sigmoid_beginning_offset_num
        // args.num_steps_before_update,
        sigmoid_total_nums_all=args.sigmoid_total_nums_all
        // args.num_steps_before_update,
    )

    # For the purpose of plotting the learning curve
    arguments = FrozenDict(
        {
            "factor_testing_timesteps": args.factor_testing_timesteps,
            "n_node": args.n_node,
            "reward_exceed_horizon": args.reward_exceed_horizon,
            "horizon_length_factor": args.horizon_length_factor,
            "random_seed_for_inference": args.random_seed_for_inference,
        }
    )

    print("Start training ...")

    @scan_tqdm(num_loops)
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
            previous_episode_done,
        ) = runner_state
        augmented_state = get_augmented_optimistic_pessimistic_belief(
            current_belief_state
        )
        _, last_critic_val = model.apply(train_state.params, augmented_state)
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
            previous_episode_done,
        )

        # Perform inference (using testing environment) (if loop_count divisible by 50 - tunable)
        # Get average and store in metrics, just like loss
        testing_average_competitive_ratio = jax.lax.cond(
            loop_count % args.frequency_testing == 0,
            lambda _: get_average_testing_stats(
                testing_environment, agent, train_state.params, arguments
            ),
            lambda _: jnp.float16(0.0),
            None,
        )

        # Collect metrics
        metrics = {
            "losses": total_loss,
            "all_rewards": traj_batch.reward,
            "all_done": traj_batch.done,
            "all_optimal_path_lengths": traj_batch.shortest_path,
            "testing_average_competitive_ratio": testing_average_competitive_ratio,
        }

        return runner_state, metrics

    start_training_time = time.time()
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
        jnp.bool_(True),
    )
    runner_state, metrics = jax.lax.scan(
        _update_step, runner_state, jnp.arange(num_loops)
    )
    train_state = runner_state[0]
    # Metrics will be stacked
    out = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,)), metrics)

    total_loss = out["losses"][0]
    value_loss = out["losses"][1][0]
    loss_actor = out["losses"][1][1]
    entropy_loss = out["losses"][1][2]

    # Put the different times in a dictionary
    time_info = {
        "start_training_time": start_training_time,
        "environment_creation_time": time_environment_creation,
        "start_time": start_time_environment_creation,
    }

    if args.generalize is False:
        testing_environment = environment

    plotting_inference(
        log_directory,
        time_info,
        train_state.params,
        out,
        testing_environment,
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
        default=-1.5,
    )
    parser.add_argument(
        "--horizon_length_factor",
        type=int,
        help="Factor to multiply with number of nodes to get the maximum horizon length",
        required=False,
        default=2,
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
        "--random_seed_for_training", type=int, required=False, default=100
    )
    parser.add_argument(
        "--random_seed_for_inference", type=int, required=False, default=101
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
    parser.add_argument(
        "--network_type",
        type=str,
        required=False,
        help="Options: CNN,Densenet,Densenet_Same, Resnet, Big_CNN, Densenet_Float16 (do not choose this one)",
        default="Densenet",
    )
    parser.add_argument(
        "--network_activation_fn",
        type=str,
        required=False,
        help="Options: leaky_relu, tanh",
        default="leaky_relu",
    )
    parser.add_argument(
        "--network_init",
        type=str,
        required=False,
        help="Options: kaiming_normal (often goes with relu/leaky_relu), orthogonal (often goes with tanh activation)",
        default="kaiming_normal",
    )
    parser.add_argument("--densenet_bn_size", type=int, required=False, default=4)
    parser.add_argument("--densenet_growth_rate", type=int, required=False, default=32)
    parser.add_argument(
        "--densenet_num_layers",
        type=str,
        required=False,
        help="Num group of layers for each dense block in string format",
        default="4,4,4",
    )
    parser.add_argument(
        "--optimizer_norm_clip",
        type=float,
        required=False,
        help="optimizer.clip_by_global_norm(value)",
        default=0.5,
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
    parser.add_argument(
        "--yaml_file", type=str, required=False, default="sweep_config_node_10.yaml"
    )
    parser.add_argument(
        "--wandb_sweep",
        type=lambda x: bool(strtobool(x)),
        default=False,
        required=False,
        help="Whether to use yaml file to do hyperparameter sweep (Bayesian optimization)",
    )
    parser.add_argument("--sweep_run_count", type=int, required=False, default=3)
    parser.add_argument(
        "--deal_with_unsolvability",
        type=str,
        default="expensive_if_unsolvable",
        required=False,
        help="Options: always_expensive_edge, expensive_if_unsolvable, resample",
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
        default=1000,
        help="Number to multiply with the number of nodes to get the total number of inference timesteps",
    )
    parser.add_argument(
        "--graph_mode",
        type=str,
        default="load",
        required=False,
        help="Options: generate,store,load",
    )
    parser.add_argument(
        "--graph_identifier",
        type=str,
        required=False,
        default="node_10_relabel_0.4",
    )
    parser.add_argument(
        "--origin_node",
        type=int,
        required=False,
        default=-1,
        help="To facilitate curriculum learning. If -1, then origin will be node 0 (or whatever the original origin is). The higher the node number, the easier it should be.",
    )
    parser.add_argument(
        "--load_network_directory",
        type=str,
        default=None,
        help="Directory to load trained network weights from",
    )
    parser.add_argument(
        "--factor_testing_timesteps",
        type=int,
        required=False,
        default=50,
        help="Factor to multiple with number of nodes to get the number of timesteps to perform testing on during training (in order to plot the learning curve)",
    )
    parser.add_argument(
        "--frequency_testing",
        type=int,
        required=False,
        default=20,
        help="How many updates before performing testing during training to plot the learning curve",
    )
    parser.add_argument(
        "--learning_curve_average_window",
        type=int,
        default=5,
        help="Number of points to average over for the smoothened learning curve plot",
    )

    # Args specific to PPO:
    parser.add_argument(
        "--num_steps_before_update",
        type=int,
        help="How many timesteps to collect before updating the network",
        required=False,
        default=3600,
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
        default=0.14,
    )
    parser.add_argument(
        "--vf_coeff",
        help="Contribution of the value loss to the total loss",
        type=float,
        required=False,
        default=0.128,
    )
    parser.add_argument(
        "--ent_coeff",
        help="Exploration coefficient",
        type=float,
        required=False,
        default=0.174,
    )
    parser.add_argument(
        "--anneal_ent_coeff",
        type=lambda x: bool(strtobool(x)),
        required=False,
        default=True,
        help="Whether to anneal the entropy (exploration) coefficient",
    )
    parser.add_argument(
        "--ent_coeff_schedule",
        type=str,
        required=False,
        help="Options: linear, sigmoid, plateau, sigmoid_checkpoint (for checkpoint training)",
        default="sigmoid",
    )
    parser.add_argument(
        "--division_plateau",
        type=int,
        required=False,
        help="What portion of the training timesteps has max ent_coeff for a period of time at the beginning and 0 ent_coeff at the end of training",
        default=5,
    )
    parser.add_argument(
        "--num_minibatches",
        help="Related to how the trajectory batch is split up for performing updating of the network",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--deterministic_inference_policy",
        type=lambda x: bool(strtobool(x)),
        default=True,
        required=False,
        help="Whether to choose the action with the highest probability instead of sampling from the distribution",
    )
    parser.add_argument(
        "--patience",
        type=int,
        required=False,
        default=5,
        help="Number of times we try to resample a solvable realisation before giving up. If any other options besides resample is chosen for deal_with_unsolvable, then this is not applicable",
    )
    parser.add_argument(
        "--num_stored_graphs",
        type=int,
        required=False,
        help="How many different graphs will be seen by the agent",
        default=2000,
    )
    parser.add_argument(
        "--sigmoid_beginning_offset_num",
        type=int,
        required=False,
        default=0,
        help="For sigmoid ent coeff schedule checkpoint training. Unit: in number of timesteps. In the script, it will be divided by num_steps_before_update to convert to num_loops unit",
    )
    parser.add_argument(
        "--sigmoid_total_nums_all",
        type=int,
        required=False,
        default=10,
        help="For sigmoid ent coeff schedule checkpoint training. Unit: in number of timesteps. In the script, it will be divided by num_steps_before_update to convert to num_loops unit",
    )
    args = parser.parse_args()

    if args.graph_mode == "store":
        print("Generating graphs for storage ...")
        generate_graphs.store_graphs(args)
        sys.exit(0)
    elif args.graph_mode == "generate":
        training_graphs = None
        inference_graphs = None
    else:
        # Load
        print("Checking validity and loading graphs ...")
        # Check args match and load graphs
        training_graphs, inference_graphs, num_training_graphs, num_inference_graphs = (
            generate_graphs.load_graphs(args)
        )
    if args.wandb_sweep == False:
        # Initialize wandb project
        wandb.init(
            project=args.wandb_project_name,
            name=args.log_directory,
            config=vars(args),
            mode=args.wandb_mode,
        )
        main(args)
        wandb.finish()
    else:
        # Hyperparameter sweep
        print("Running hyperparameter sweep ...")
        if args.wandb_mode != "online":
            raise ValueError("Wandb mode must be online for hyperparameter sweep")
        with open(args.yaml_file, "r") as file:
            sweep_config = yaml.safe_load(file)
        sweep_id = wandb.sweep(
            sweep_config,
            project=args.wandb_project_name,
            entity="lam-lam-university-of-oxford",
        )

        def wrapper_function():
            with wandb.init() as run:
                config = run.config
                # Don't need to name the run using config values (run.name = ...) because it will be very long
                # Modify args based on config
                for key in config:
                    setattr(args, key, config[key])
                # Instead of using run.id, can concatenate parameters
                log_directory = os.path.join(
                    os.getcwd(), "Logs", args.wandb_project_name, run.name
                )
                args.log_directory = log_directory
                main(args)

        wandb.agent(sweep_id, function=wrapper_function, count=args.sweep_run_count)

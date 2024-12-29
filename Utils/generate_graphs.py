import numpy as np
import pickle
import sys

sys.path.append("..")
from Environment import CTP_environment_generalize
import os
import jax
import jax.numpy as jnp


def store_graphs(args):
    directory = os.path.join(os.getcwd(), "Generated_graphs", args.graph_identifier)
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save important args into a dictionary and store as a pickle file
    graph_info = {
        "n_node": args.n_node,
        "prop_stoch": args.prop_stoch,
        "k_edges": args.k_edges,
        "grid_size": args.grid_size,
        "factor_expensive_edge": args.factor_expensive_edge,
        "deal_with_unsolvability": args.deal_with_unsolvability,
        "num_stored_graphs": args.num_stored_graphs,
        "factor_inference_timesteps": args.factor_inference_timesteps,
    }
    with open(os.path.join(directory, "graph_info.pkl"), "wb") as f:
        pickle.dump(graph_info, f)

    key = jax.random.PRNGKey(args.random_seed_for_training)
    online_key, environment_key = jax.random.split(key)
    training_environment = CTP_environment_generalize.CTP_General(
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
        deal_with_unsolvability=args.deal_with_unsolvability,
        patience=args.patience,
        num_stored_graphs=args.num_stored_graphs,
        loaded_graphs=None,
    )
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(training_environment.stored_graphs))
    inference_key = jax.random.PRNGKey(args.random_seed_for_inference)
    inference_environment = CTP_environment_generalize.CTP_General(
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
        loaded_graphs=None,
    )
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    np.save(inference_graph_npy_file, np.array(inference_environment.stored_graphs))


def load_graphs(args) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
    directory = os.path.join(os.getcwd(), "Generated_graphs", args.graph_identifier)
    graph_info_file = os.path.join(directory, "graph_info.pkl")
    if os.path.exists(graph_info_file):
        # All graphs have the same prop stoch
        with open(graph_info_file, "rb") as f:
            graph_info = pickle.load(f)
        assert graph_info["n_node"] == args.n_node
        assert graph_info["prop_stoch"] == args.prop_stoch
        assert graph_info["k_edges"] == args.k_edges
        assert graph_info["grid_size"] == args.grid_size
        assert graph_info["factor_expensive_edge"] == args.factor_expensive_edge
        assert graph_info["deal_with_unsolvability"] == args.deal_with_unsolvability
        assert graph_info["num_stored_graphs"] == args.num_stored_graphs
        assert (
            graph_info["factor_inference_timesteps"] == args.factor_inference_timesteps
        )

    # Load the graphs
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    training_graphs = jnp.array(np.load(training_graph_npy_file))
    inference_graph_npy_file = os.path.join(directory, "inference_graphs.npy")
    inference_graphs = jnp.array(np.load(inference_graph_npy_file))

    # Get number of training and inference graphs
    num_training_graphs = training_graphs.shape[0]
    num_inference_graphs = inference_graphs.shape[0]
    return training_graphs, inference_graphs, num_training_graphs, num_inference_graphs

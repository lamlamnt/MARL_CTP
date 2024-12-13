import numpy as np
import pickle
import sys

sys.path.append("../")
from Environment import CTP_environment_generalize
import os

# Potential barriers: file size very big

# Save args into a dictionary and store as a pickle file
# Save graphs into a file

# In the main file, check that args match the args in the pickle file
# Load the graphs
# Environment - 2 options: save or load

# Convert to numpy

# To load: loaded_array = jnp.array(np.load("array.npy"))


def store_graphs(args, environment_key):
    directory = os.path.join(os.getcwd(), "Generated_graphs")
    environment = CTP_environment_generalize.CTP_General(
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
    )
    training_graph_npy_file = os.path.join(directory, "training_graphs.npy")
    np.save(training_graph_npy_file, np.array(environment.stored_graphs))

import os
import numpy as np
import jax.numpy as jnp
import pytest
import pytest_print as pp
import sys

sys.path.append("..")
from Utils import graph_functions


# test that the stored graphs actually have origin closer to the goal
def test_origin_closer_to_goal(printer):
    # Load the graphs in
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    graph_file = os.path.join(
        parent_directory,
        "Generated_graphs",
        "node_30_origin_20_prop_0.4",
        "training_graphs.npy",
    )
    training_graphs = np.load(graph_file)
    # Convert to jax array
    training_graphs = jnp.array(training_graphs)
    first_training_graph = training_graphs[0]
    current_graph_weights = first_training_graph[0, :, :]
    current_graph_blocking_prob = first_training_graph[1, :, :]
    current_graph_origin = first_training_graph[2, 0, 0].astype(jnp.int16)
    current_graph_goal = first_training_graph[2, 0, 1].astype(jnp.int16)
    assert current_graph_origin == 20
    assert current_graph_goal == 29

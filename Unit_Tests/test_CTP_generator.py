import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
import timeit
import time
import argparse
import os

# This is currently used to test the functions in the CTP_generator.py file
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
    args = parser.parse_args()

    # Get directory path to Logs folder
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    key = jax.random.PRNGKey(100)
    graph = CTP_generator.generate_graph(args.n_node, key, prop_stoch=0.4)
    origin, goal = CTP_generator.find_single_goal_and_origin(graph)

    key, subkey1 = jax.random.split(key)

    new_blocking_prob = CTP_generator.get_blocking_prob(graph.n_edge, subkey1)
    new_graph = graph._replace(edges={**graph.edges, "blocked_prob": new_blocking_prob})

    # Get maximum number of connections a node has
    max_connections = CTP_generator.get_max_edges(graph)

    # Convert origin and goal to int first in order to plot the graph
    goal_int = goal.item()
    origin_int = origin.item()

    subkey1, subkey2 = jax.random.split(subkey1)
    complete_graph = CTP_generator.sample_blocking_prob(subkey2, graph)
    solvable = CTP_generator.is_solvable(complete_graph, origin_int, goal_int)

    agent_graph = CTP_generator.get_agent_graph(complete_graph)

    weight_matrix, blocking_prob_matrix = CTP_generator.convert_jraph_to_adj_matrix(
        complete_graph
    )

    new_graph = CTP_generator.add_expensive_edge(complete_graph, 50, origin, goal)
    new_graph_NX = CTP_generator.convert_jraph_to_networkx(new_graph)
    CTP_generator.plot_nx_graph(new_graph_NX, goal_int, origin_int, log_directory)

    # Put the timed code (for large graphs), wrapped in block_until_ready into a function and then timeit that function
    # timeit.timeit(jax.block_until_ready(CTP.generate_graph(5,key)),number=100)

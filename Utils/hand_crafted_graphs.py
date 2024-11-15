import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Environment.CTP_generator import NOT_CONNECTED as NC

# Make sure these graphs are always solvable


# For sanity check, to check that the agent is capable of exploring
# Diamond shaped graph
# For this blocking probability, the optimal path is 0 -> 2 -> 3/ 0 -> 2 -> 0 -> 1 -> 3
def get_diamond_shaped_graph() -> CTP_generator.CTPGraph:
    senders = jnp.array([0, 0, 1, 2])
    receivers = jnp.array([1, 2, 3, 3])
    weights = jnp.array(
        [
            [CTP_generator.NOT_CONNECTED, 1, 1, CTP_generator.NOT_CONNECTED],
            [1, CTP_generator.NOT_CONNECTED, CTP_generator.NOT_CONNECTED, 3],
            [1, CTP_generator.NOT_CONNECTED, CTP_generator.NOT_CONNECTED, 1],
            [CTP_generator.NOT_CONNECTED, 3, 1, CTP_generator.NOT_CONNECTED],
        ]
    )
    node_pos = jnp.array([[2.5, 5], [0, 2.5], [5, 2.5], [2.5, 0]])
    blocking_prob = jnp.array(
        [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0.4], [1, 0, 0.4, 1]]
    )
    goal = 3
    origin = 0
    n_nodes = 4
    # Store in a dictionary
    defined_graph = {
        "n_edges": 4,
        "weights": weights,
        "senders": senders,
        "receivers": receivers,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "origin": origin,
        "goal": goal,
    }
    # Use n_nodes and defined_graph to create a CTPGraph object
    return 1


# For environment, just add option to input a CTPGraph_Realisation
# For CTPGraph_Realisation, add option to input a CTPGraph
# For CTPGraph, add option to input weights, blocking prob, senders, receivers, and node_pos


# N stochastic edge graph - to check that the agent is making the best decision
def get_stochastic_edge_graph() -> CTP_generator.CTPGraph:
    senders = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    receivers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9, 9, 9]
    weights = [
        [NC, 8, 7, 6, 5, 4, 3, 2, 1, NC],
        [8, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [7, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [6, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [5, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [4, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [3, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [2, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [1, NC, NC, NC, NC, NC, NC, NC, NC, 1],
        [NC, 1, 1, 1, 1, 1, 1, 1, NC],
    ]
    node_pos = jnp.array(
        [
            [5, 10],
            [1, 5],
            [2, 5],
            [3, 5],
            [4, 5],
            [5, 5],
            [6, 5],
            [7, 5],
            [8, 5],
            [5, 0],
        ]
    )
    blocking_prob = jnp.array(
        [
            [1, 0, 1 / 8, 2 / 8, 3 / 8, 4 / 8, 5 / 8, 6 / 8, 7 / 8, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [2 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [3 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [4 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [5 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [6 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [7 / 8, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    goal = 9
    origin = 0
    n_nodes = 10
    # Store in a dictionary
    defined_graph = {
        "n_edges": 10,
        "weights": weights,
        "senders": senders,
        "receivers": receivers,
        "node_pos": node_pos,
        "blocking_prob": blocking_prob,
        "origin": origin,
        "goal": goal,
    }
    # Use n_nodes and defined_graph to create a CTPGraph object
    return 1

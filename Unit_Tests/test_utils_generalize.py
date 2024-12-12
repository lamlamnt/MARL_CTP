import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Utils import graph_functions, hand_crafted_graphs, util_generalize


@pytest.fixture
def graphRealisation():
    key = jax.random.PRNGKey(0)
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key, 5, prop_stoch=0.4, expensive_edge=False
    )
    return graphRealisation


def test_sample_blocking_status(graphRealisation: CTP_generator.CTPGraph_Realisation):
    key = jax.random.PRNGKey(101)
    blocking_status = graph_functions.sample_blocking_status(
        key, graphRealisation.graph.blocking_prob
    )
    # check symmetric
    assert jnp.all(blocking_status == blocking_status.T)
    # check blocking_status = 1 if blocking_prob = 1
    for i in range(blocking_status.shape[0]):
        for j in range(blocking_status.shape[1]):
            if graphRealisation.graph.blocking_prob[i, j] == 1:
                assert blocking_status[i, j] == 1


def test_is_solvable_generalize(graphRealisation: CTP_generator.CTPGraph_Realisation):
    key = jax.random.PRNGKey(99)
    blocking_status = graph_functions.sample_blocking_status(
        key, graphRealisation.graph.blocking_prob
    )
    solvable_ground_truth = graphRealisation.is_solvable(blocking_status)
    solvable_test = graph_functions.is_solvable(
        graphRealisation.graph.weights,
        blocking_status,
        graphRealisation.graph.origin[0],
        graphRealisation.graph.goal[0],
    )
    assert solvable_ground_truth == solvable_test


def test_get_expected_optimal_path_length(printer):
    key = jax.random.PRNGKey(1)
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key, 10, prop_stoch=0.4, expensive_edge=False
    )
    expected_path_length = util_generalize.get_expected_optimal_path_length(
        graphRealisation, key
    )
    assert expected_path_length > 0 and expected_path_length < 2
    printer(expected_path_length)

    n_node, hand_crafted_diamond_graph = hand_crafted_graphs.get_diamond_shaped_graph()
    graphRealisation = CTP_generator.CTPGraph_Realisation(
        key,
        n_node,
        handcrafted_graph=hand_crafted_diamond_graph,
    )
    expected_path_length = util_generalize.get_expected_optimal_path_length(
        graphRealisation, key
    )
    assert jnp.isclose(expected_path_length, 1.062, atol=0.2)

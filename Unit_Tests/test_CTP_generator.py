import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
import argparse
import os
import pytest
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def graphRealisation():
    # can be used for all functions
    key = jax.random.PRNGKey(100)
    graphRealisation = CTP_generator.CTPGraph_Realisation(key, 5, prop_stoch=0.4)
    return graphRealisation


def test_no_inf(graphRealisation: CTP_generator.CTPGraph_Realisation):
    assert not jnp.any(jnp.isinf(graphRealisation.graph.blocking_prob))
    assert not jnp.any(jnp.isinf(graphRealisation.graph.weights))
    assert not jnp.any(jnp.isinf(graphRealisation.blocking_status))


def test_symmetric_adjacency_matrices(
    graphRealisation: CTP_generator.CTPGraph_Realisation,
):
    assert jnp.all(graphRealisation.graph.weights == graphRealisation.graph.weights.T)
    assert jnp.all(
        graphRealisation.graph.blocking_prob == graphRealisation.graph.blocking_prob.T
    )
    assert jnp.all(
        graphRealisation.graph.blocking_prob == graphRealisation.graph.blocking_prob.T
    )
    assert jnp.all(
        graphRealisation.blocking_status == graphRealisation.blocking_status.T
    )


def test_plotting(graphRealisation: CTP_generator.CTPGraph_Realisation):
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    graphRealisation.graph.plot_nx_graph(log_directory)
    graphRealisation.plot_realised_graph(log_directory)


def test_is_solvable(graphRealisation: CTP_generator.CTPGraph_Realisation):
    temp = graphRealisation.blocking_status
    graphRealisation.blocking_status = jnp.full(
        (graphRealisation.graph.n_nodes, graphRealisation.graph.n_nodes), True
    )
    assert graphRealisation.is_solvable() is False
    graphRealisation.blocking_status = graphRealisation.blocking_status.at[0, 1].set(
        False
    )
    graphRealisation.blocking_status = graphRealisation.blocking_status.at[1, 0].set(
        False
    )
    assert graphRealisation.is_solvable() is False
    graphRealisation.blocking_status = temp

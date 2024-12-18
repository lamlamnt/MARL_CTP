import pytest
import pytest_print as pp
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


# Test that the origin is 0 and goal is num_nodes -1
def test_relabelling():
    key = jax.random.PRNGKey(1)
    num_nodes = 5
    graph_realisation = CTP_generator.CTPGraph_Realisation(key, num_nodes, 10, 0.4)
    assert graph_realisation.graph.origin == 0
    assert graph_realisation.graph.goal == num_nodes - 1

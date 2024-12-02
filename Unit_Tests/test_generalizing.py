import jax
import jax.numpy as jnp
import pytest
import sys

sys.path.append("..")
from Environment import CTP_environment, CTP_generator
import pytest_print as pp
import os


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.9)
    return environment


# Test that that the graph generated is solvable
def test_solvable_no_expensive_edge(environment: CTP_environment.CTP):
    key = jax.random.PRNGKey(40)
    env_state, _ = environment.reset(key)
    assert environment.graph_realisation.is_solvable(env_state[0, 1:, :]) == jnp.bool_(
        True
    )

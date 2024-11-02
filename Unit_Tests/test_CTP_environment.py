import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import pytest


@pytest.fixture
def environment():
    key = jax.random.PRNGKey(30)
    environment = CTP_environment.CTP(1, 1, 5, key, prop_stoch=0.4)
    return environment


# Check symmetric adjacency matrices
def test_symmetric():
    pass


# Two consecutive resamples are different

# Check reward is always negative

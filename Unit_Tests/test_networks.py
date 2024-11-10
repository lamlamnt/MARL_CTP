import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks import MLP


# Test that the forward pass does not error
def test_mlp(printer):
    key = jax.random.PRNGKey(100)
    key, subkey = jax.random.split(key)
    dummy_input = jnp.ones((3, 6, 5))
    params = MLP.forward_fn_FC.init(key, dummy_input)
    output_1 = MLP.forward_fn_FC.apply(params, dummy_input)

    simple_params = MLP.simplest_model_hk.init(key, dummy_input)
    output_2 = MLP.simplest_model_hk.apply(simple_params, dummy_input)

    flax_model = MLP.Flax_FCNetwork([64, 32], 5)
    flax_params = flax_model.init(key, dummy_input)
    output_3 = flax_model.apply(flax_params, dummy_input)

    # Check that the output has the correct shape
    assert output_1.shape == (5,)
    assert output_2.shape == (5,)
    assert output_3.shape == (5,)

    # Test that Flax network works with other shapes too
    big_flax_model = MLP.Flax_FCNetwork([128, 64, 32, 16], 10)
    big_flax_params = big_flax_model.init(subkey, jnp.ones((3, 11, 10)))
    big_output = big_flax_model.apply(big_flax_params, jnp.ones((3, 11, 10)))
    assert big_output.shape == (10,)

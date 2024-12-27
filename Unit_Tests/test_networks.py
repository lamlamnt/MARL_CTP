import jax
import jax.numpy as jnp
import pytest
import pytest_print
import haiku as hk
import sys

sys.path.append("..")
from Networks import MLP, CNN, actor_critic_network, densenet, resnet
from Networks.big_cnn import Big_CNN_30


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


def test_cnn():
    model = CNN.Flax_CNN(32, [64, 32], 5)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((3, 6, 5)))
    output = model.apply(params, jnp.ones((3, 6, 5)))
    assert output.shape == (5,)


# Test that the weights and outputs of the network have the desired dtype
def test_dtype():
    # float32 one
    model = CNN.Flax_CNN(32, [64, 32], 10)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((3, 11, 10), dtype=jnp.float16))
    output = model.apply(params, jnp.ones((3, 11, 10), dtype=jnp.float16))
    assert output.dtype == jnp.float16
    assert params["params"]["Conv_0"]["kernel"].dtype == jnp.float32
    assert params["params"]["Dense_0"]["kernel"].dtype == jnp.float32

    # float16 one
    model = CNN.Flax_CNN(32, [64, 32], 10, dtype_params=jnp.float16)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((3, 11, 10), dtype=jnp.float16))
    output = model.apply(params, jnp.ones((3, 11, 10), dtype=jnp.float16))
    assert output.dtype == jnp.float16
    assert params["params"]["Conv_0"]["kernel"].dtype == jnp.float16
    assert params["params"]["Dense_0"]["kernel"].dtype == jnp.float16


def test_actor_critic_network(printer):
    model = actor_critic_network.ActorCritic_CNN_10(5)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((3, 6, 5)))
    action_values, critic = model.apply(params, jnp.ones((3, 6, 5)))
    # action_values.probs is the probabilities after applying softmax. action_values.logits is the raw output before softmax
    action = action_values.sample(seed=key)
    log_prob = action_values.log_prob(action)


def test_densenet(printer):
    model = densenet.DenseNet_ActorCritic(10)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((4, 11, 10)))
    action_values, critic = model.apply(params, jnp.ones((4, 11, 10)))
    assert action_values.probs.shape == (10,)
    assert critic.shape == ()


def test_resnet():
    model = resnet.ResNet_ActorCritic(10)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((4, 11, 10)))
    action_values, critic = model.apply(params, jnp.ones((4, 11, 10)))
    assert action_values.probs.shape == (10,)
    assert critic.shape == ()


def test_bigger_cnn():
    model = Big_CNN_30(30)
    key = jax.random.PRNGKey(100)
    params = model.init(key, jnp.ones((5, 30, 30)))
    action_values, critic = model.apply(params, jnp.ones((5, 30, 30)))
    assert action_values.probs.shape == (30,)
    assert critic.shape == ()

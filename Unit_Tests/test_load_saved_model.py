import flax.serialization
import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Networks import MLP  # Replace with your model's actual import
import os
import pytest


def test_load_model():
    key = jax.random.PRNGKey(0)
    # Load the parameters from the file
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    file_name = os.path.join(log_directory, "weights.flax")
    with open(file_name, "rb") as f:
        serialized_params = f.read()

    # Initialize your model (assuming you know the input shape or have example inputs)
    flax_model = MLP.Flax_FCNetwork([128, 64, 32, 16], 5)
    example_input = jnp.zeros((3, 6, 5))
    initial_params = flax_model.init(key, example_input)
    random_output = flax_model.apply(initial_params, example_input)

    # Restore the parameters
    restored_params = flax.serialization.from_bytes(initial_params, serialized_params)
    model_output = flax_model.apply(restored_params, example_input)
    assert not jnp.array_equal(random_output, model_output)

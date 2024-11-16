import flax.serialization
import jax
import jax.numpy as jnp
from typing import Dict, Any
import numpy as np
import os
import sys

sys.path.append("..")
from Networks.CNN import Flax_CNN


def load_checkpoint(file_path: str) -> Dict[str, Any]:
    """Loads weights from a Flax checkpoint file."""
    model = Flax_CNN(32, [550, 275, 137, 68], 10)
    example_input = jax.numpy.ones((3, 11, 10))
    variables = model.init(jax.random.PRNGKey(0), example_input)
    with open(file_path, "rb") as f:
        loaded_params = flax.serialization.from_bytes(variables, f.read())
    return loaded_params


def analyze_weights(weights: Dict[str, Any], threshold: float = 1e-2):
    """
    Analyze weights to find layers with mostly small magnitude weights.

    Args:
        state (dict): Loaded model weights.
        threshold (float): Magnitude below which weights are considered small.

    Returns:
        List of layers with weight statistics.
    """
    results = []
    for layer_name, layer_weights in weights.items():
        for weight_name, weights in layer_weights.items():
            # Flatten weights to analyze their magnitudes
            flat_weights = jnp.ravel(jnp.array(weights))
            total_weights = flat_weights.size
            small_weights_count = jnp.sum(jnp.abs(flat_weights) < threshold)

            # Store stats for this weight tensor
            stats = {
                "layer": layer_name,
                "weight_name": weight_name,
                "shape": weights.shape,
                "total_weights": total_weights,
                "small_weights": int(small_weights_count),
                "small_weights_ratio": float(small_weights_count) / total_weights,
            }
            results.append(stats)

            print(
                f"Layer: {layer_name}, Weight: {weight_name}, "
                f"Shape: {weights.shape}, Small Weights Ratio: {stats['small_weights_ratio']:.2%}"
            )
    return results


# Example Usage
if __name__ == "__main__":
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    logs_dir = os.path.join(parent_dir, "logs")

    checkpoint_file = "weights.flax"  # Replace with your checkpoint path
    model_weights = load_checkpoint(
        os.path.join(logs_dir, "DQN_uniform_10", checkpoint_file)
    )

    print("\nAnalyzing weights...")
    results = analyze_weights(model_weights, threshold=1e-2)

    # Identify layers with >90% small weights
    """
    reducible_layers = [res for res in results if res["small_weights_ratio"] > 0.9]

    print("\nLayers with mostly small weights (potential for reduction):")
    for layer in reducible_layers:
        print(layer)
    """

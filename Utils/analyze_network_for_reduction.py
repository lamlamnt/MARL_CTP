import flax.serialization
import jax
import jax.numpy as jnp
from typing import Dict, Any
import numpy as np
import os
import sys
import argparse

sys.path.append("..")
from Networks.CNN import Flax_CNN
from Networks.actor_critic_network import ActorCritic_CNN_10, ActorCritic_CNN_30


def load_checkpoint(args, file_path: str) -> Dict[str, Any]:
    """Loads weights from a Flax checkpoint file."""
    model = ActorCritic_CNN_10(args.n_node)
    example_input = jax.numpy.ones((3, args.n_node + 1, args.n_node))
    variables = model.init(jax.random.PRNGKey(0), example_input)
    with open(file_path, "rb") as f:
        loaded_params = flax.serialization.from_bytes(variables, f.read())
    return loaded_params


def analyze_weights(args, weights: Dict[str, Any], threshold: float = 1e-2):
    results = []
    for layer_name, layer_weights in weights["params"].items():
        kernel_weights = layer_weights["kernel"]
        # Flatten weights to analyze their magnitudes
        flat_weights = jnp.ravel(jnp.array(kernel_weights))
        total_weights = flat_weights.size
        small_weights_count = jnp.sum(jnp.abs(flat_weights) < threshold)

        # Store stats for this weight tensor
        stats = {
            "layer": layer_name,
            "shape": kernel_weights.shape,
            "total_weights": total_weights,
            "small_weights": int(small_weights_count),
            "small_weights_ratio": float(small_weights_count) / total_weights,
        }
        results.append(stats)

        print(
            f"Layer: {layer_name}"
            f"Shape: {kernel_weights.shape}, Small Weights Ratio: {stats['small_weights_ratio']:.2%}"
        )
    return results


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=False,
        default=5,
    )
    parser.add_argument(
        "--folder_name", type=str, help="Name of the folder", required=True
    )
    args = parser.parse_args()

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    logs_dir = os.path.join(parent_dir, "Logs")

    checkpoint_file = "weights.flax"  # Replace with your checkpoint path
    model_weights = load_checkpoint(
        args, os.path.join(logs_dir, args.folder_name, checkpoint_file)
    )

    print("\nAnalyzing weights...")
    results = analyze_weights(args, model_weights, threshold=1e-2)

import jax
import jax.numpy as jnp
import sys


# Input the belief state and output the origin node
def get_origin_expensive_edge(belief_state):
    # Get deterministic edge with largest weight connected to goal
    num_nodes = belief_state.shape[-1]
    goal = jnp.unravel_index(
        jnp.argmax(belief_state[3, 1:, :]), (num_nodes, num_nodes)
    )[0]
    potential_origin = jnp.argmax(belief_state[1, 1 + goal, :])
    return potential_origin
    """
    if (
        belief_state[1, 1 + goal, potential_origin] == 1
        and belief_state[2, 1 + goal, potential_origin] == 0
    )
    """

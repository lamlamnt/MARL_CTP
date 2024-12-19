import jax
import jax.numpy as jnp
import os
import sys

sys.path.append("..")
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Environment import CTP_generator, CTP_environment
from functools import partial


# Input belief state, output augmented belief state
@jax.jit
def get_augmented_optimistic_belief(belief_state: jnp.ndarray):
    # Assume all unknown stochastic edges are not blocked
    optimistic_belief_state = belief_state.at[0, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.UNKNOWN,
            CTP_generator.UNBLOCKED,
            belief_state[0, 1:, :],
        )
    )
    # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
    optimistic_belief_state = optimistic_belief_state.at[1, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.BLOCKED,
            1,
            belief_state[1, 1:, :],
        )
    )
    num_nodes = belief_state.shape[2]
    num_agents = belief_state.shape[1] - belief_state.shape[2]

    def pairwise_dijkstra(optimistic_belief_state, node_pair):
        return dijkstra_shortest_path(
            optimistic_belief_state, node_pair[0], node_pair[1]
        )

    node_indices = jnp.arange(num_nodes)
    node_pairs = jnp.array(
        [(i, j) for i in node_indices for j in node_indices]
    ).reshape(num_nodes, num_nodes, 2)

    # Use vmap to compute shortest paths for all pairs of nodes
    vmap_func = jax.vmap(jax.vmap(partial(pairwise_dijkstra, optimistic_belief_state)))
    shortest_paths = vmap_func(node_pairs)
    empty = jnp.zeros((num_agents, num_nodes), dtype=jnp.float16)
    shortest_paths = jnp.concatenate((empty, shortest_paths), axis=0)
    augmented_belief_state = jnp.vstack(
        (belief_state, jnp.expand_dims(shortest_paths, axis=0)), dtype=jnp.float16
    )
    return augmented_belief_state

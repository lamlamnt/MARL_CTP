import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


@jax.jit
def sample_blocking_status(
    key: jax.random.PRNGKey, blocking_prob: jnp.ndarray
) -> jnp.ndarray:
    n_nodes = blocking_prob.shape[0]
    blocking_status = jnp.full(
        (n_nodes, n_nodes), CTP_generator.BLOCKED, dtype=jnp.float16
    )
    # Indices for the upper triangular part (excluding the diagonal)
    idx_upper = jnp.triu_indices(n_nodes, k=1)
    keys = jax.random.split(key, num=idx_upper[0].shape)
    for i in range(idx_upper[0].shape[0]):
        element_blocking_status = jax.random.bernoulli(
            keys[i], p=blocking_prob[idx_upper[0][i], idx_upper[1][i]]
        )
        blocking_status = blocking_status.at[idx_upper[0][i], idx_upper[1][i]].set(
            element_blocking_status
        )
        blocking_status = blocking_status.at[idx_upper[1][i], idx_upper[0][i]].set(
            element_blocking_status
        )
    return blocking_status


@jax.jit
def is_solvable(
    weights: jnp.ndarray, blocking_status: jnp.ndarray, origin: int, goal: int
) -> bool:
    graph = weights
    n_nodes = weights.shape[0]
    # Change the weights element where blocking_prob is 1 to -1
    graph = jnp.where(
        blocking_status == CTP_generator.BLOCKED,
        CTP_generator.NOT_CONNECTED,
        graph,
    )
    # Change all -1 elements to infinity
    graph = jnp.where(graph == CTP_generator.NOT_CONNECTED, jnp.inf, graph)

    # Initialize distances with "infinity" and visited nodes
    distances = jnp.inf * jnp.ones(n_nodes)
    distances = distances.at[origin].set(0)
    visited = jnp.zeros(n_nodes, dtype=bool)

    def body_fun(i, carry):
        distances, visited = carry

        # Find the node with the minimum distance that hasn't been visited yet
        unvisited_distances = jnp.where(visited, jnp.inf, distances)
        current_node = jnp.argmin(unvisited_distances)
        current_distance = distances[current_node]

        # Mark this node as visited
        visited = visited.at[current_node].set(True)

        # Update distances to neighboring nodes
        neighbors = graph[current_node, :]
        new_distances = jnp.where(
            (neighbors < jnp.inf) & (~visited),
            jnp.minimum(distances, current_distance + neighbors),
            distances,
        )
        return new_distances, visited

    # Run the loop with `jax.lax.fori_loop`
    distances, visited = jax.lax.fori_loop(0, n_nodes, body_fun, (distances, visited))
    solvable = jax.lax.cond(
        distances[goal] == jnp.inf,
        lambda _: jnp.bool_(False),
        lambda _: jnp.bool_(True),
        operand=None,
    )
    return solvable

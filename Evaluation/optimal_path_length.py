import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment
import os


@jax.jit
def dijkstra_shortest_path(
    env_state: CTP_environment.EnvState, origin: int, goal: int
) -> float:
    # Given the current environment state (perfect knowledge of blocking status),
    # return the shortest path length from the source to the origin
    num_nodes = env_state.shape[2]
    num_agents = env_state.shape[1] - num_nodes
    graph = env_state[1, num_agents:, :]
    # Change the weights element where blocking_prob is 1 to -1
    graph = jnp.where(
        env_state[0, num_agents:, :] == CTP_generator.BLOCKED,
        CTP_generator.NOT_CONNECTED,
        graph,
    )
    # Change all -1 elements to infinity
    graph = jnp.where(graph == CTP_generator.NOT_CONNECTED, jnp.inf, graph)

    # Initialize distances with "infinity" and visited nodes
    distances = jnp.inf * jnp.ones(num_nodes, dtype=jnp.float16)
    distances = distances.at[origin].set(0)
    visited = jnp.zeros(num_nodes, dtype=bool)

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
    distances, visited = jax.lax.fori_loop(0, num_nodes, body_fun, (distances, visited))

    return distances[goal][0]


def dijkstra_with_path(env_state: jnp.ndarray) -> tuple[int, jnp.array]:
    pass

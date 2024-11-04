import jax
import jax.numpy as jnp
from jax import lax

MAX_NODES = 5


# @jax.jit
def is_solvable(adj_matrix: jnp.ndarray, origin: int, goal: int) -> bool:
    """
    Determines if there is a path between two nodes in a graph represented by an adjacency matrix.

    Args:
        adj_matrix (jnp.ndarray): Adjacency matrix of the graph.
        origin (int): Starting node.
        goal (int): Target node.

    Returns:
        bool: True if there is a path between origin and goal, False otherwise.
    """
    # Initialize visited array and queue for BFS
    visited = jnp.zeros(MAX_NODES, dtype=jnp.bool_)
    queue = jnp.full(MAX_NODES, -1, dtype=jnp.int32)  # Queue with -1 indicating empty
    queue = queue.at[0].set(origin)  # Start with the origin in the queue
    visited = visited.at[origin].set(True)  # Mark the origin as visited

    def bfs_step(i, carry):
        queue, visited, found_goal = carry
        current = queue[i]

        # If current node is the goal, set found_goal to True
        found_goal = lax.cond(
            current == goal, lambda _: True, lambda _: found_goal, None
        )
        print(found_goal)

        # Skip processing if the current node is -1 or if we've found the goal
        def skip_step(x):
            return x[0], x[1], found_goal

        carry = lax.cond((current == -1) | found_goal, skip_step, lambda x: x, carry)

        # Explore neighbors of the current node
        def add_neighbors(j, inner_carry):
            queue, visited, insert_idx = inner_carry
            is_connected = adj_matrix[current, j] == 1
            is_unvisited = ~visited[j]
            to_add = is_connected & is_unvisited

            # If the neighbor should be added, add to queue and mark as visited
            queue = lax.cond(
                to_add, lambda q: q.at[insert_idx].set(j), lambda q: q, queue
            )
            visited = visited.at[j].set(to_add | visited[j])  # Mark as visited if added
            insert_idx = insert_idx + to_add  # Increment index if added
            return queue, visited, insert_idx

        queue, visited, _ = lax.fori_loop(
            0, MAX_NODES, add_neighbors, (queue, visited, i + 1)
        )

        return queue, visited, found_goal

    # Use fori_loop to iterate through the queue up to MAX_NODES times
    initial_carry = (queue, visited, False)
    final_queue, final_visited, found_goal = lax.fori_loop(
        0, MAX_NODES, bfs_step, initial_carry
    )

    return found_goal


if __name__ == "__main__":
    adjacency_matrix = jnp.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=jnp.int32,
    )
    origin = 0
    goal = 4
    # This returns False but it should be True
    print(is_solvable(adjacency_matrix, origin, goal))

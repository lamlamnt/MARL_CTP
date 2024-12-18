import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


def distance(a, b):
    return jnp.sqrt(jnp.sum((a - b) ** 2))


def convert_to_grid(i, ymax):
    return (i // (ymax + 1), i % (ymax + 1))


def check(node_pos, origin, goal):
    grid_nodes = jax.vmap(convert_to_grid, in_axes=(0, None))(node_pos, 5)
    grid_nodes = jnp.array(grid_nodes, dtype=jnp.float16).T
    print(grid_nodes)
    origin_pos = grid_nodes[origin]
    goal_pos = grid_nodes[goal]
    grid_nodes_not_origin_goal = jnp.delete(
        grid_nodes, np.array([origin, goal]), axis=0
    )
    distances_from_goal = jax.vmap(lambda x: distance(grid_nodes[goal], x))(
        grid_nodes_not_origin_goal
    )
    sorted_indices = jnp.argsort(-distances_from_goal)
    grid_nodes = jnp.vstack(
        [origin_pos, grid_nodes_not_origin_goal[sorted_indices], goal_pos]
    )
    return grid_nodes


if __name__ == "__main__":
    node_pos = jnp.array([0, 10, 4, 15])
    origin = 0
    goal = 3
    grid_nodes = check(node_pos, origin, goal)
    print(grid_nodes)

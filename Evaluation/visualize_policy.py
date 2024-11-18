import sys
import jax
import jax.numpy as jnp

sys.path.append("..")
from Environment import CTP_generator, CTP_environment


# Returns a n_node x n_node matrix where the element at (i,j) is the probability of moving from node i to node j
def get_policy(num_node, all_actions: jnp.array, all_positions: jnp.array):
    policy = jnp.zeros((num_node, num_node))

    def update_policy(policy, start, end):
        return policy.at[start, end].add(1)

    # Vectorize the update_policy function
    policy = jax.lax.fori_loop(
        0,
        len(all_positions),
        lambda i, policy: update_policy(policy, all_positions[i], all_actions[i]),
        policy,
    )
    # Normalize to get probabilities
    row_sums = policy.sum(axis=1, keepdims=True)
    policy = policy / row_sums
    return policy

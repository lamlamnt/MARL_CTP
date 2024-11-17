import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


def decide_validity_of_action_space(current_belief_state: jnp.ndarray) -> jnp.array:
    # Return an array with size equal to num_nodes in the graph where the element is
    # True if the action is valid and False if the action is invalid
    num_agents = current_belief_state.shape[1] - current_belief_state.shape[2]
    num_nodes = current_belief_state.shape[2]
    weights = current_belief_state[1, num_agents:, :]
    blocking_status = current_belief_state[0, num_agents:, :]
    valid = jnp.zeros(num_nodes, dtype=jnp.float16)
    for i in range(num_nodes):
        is_invalid = jnp.logical_or(
            i == jnp.argmax(current_belief_state[0, :1, :]),
            jnp.logical_or(
                weights[jnp.argmax(current_belief_state[0, :1, :]), i]
                == CTP_generator.NOT_CONNECTED,
                blocking_status[jnp.argmax(current_belief_state[0, :1, :]), i]
                == CTP_generator.BLOCKED,
            ),
        )
        valid = valid.at[i].set(jnp.where(is_invalid, -jnp.inf, 1.0))
    return valid


@jax.jit
def random_valid_action(subkey, invalid_mask):
    # Those with -inf will have probs 0.
    probs = jax.nn.softmax(invalid_mask)
    sampled_action = jax.random.choice(subkey, invalid_mask.shape[0], p=probs)
    return sampled_action

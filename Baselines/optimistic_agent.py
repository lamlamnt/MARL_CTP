import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, lax, random, value_and_grad, vmap
import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment_generalize
from Evaluation import optimal_path_length


class Optimistic_Agent:
    @partial(jit, static_argnums=(0))
    def act(self, belief_state: jnp.ndarray) -> int:
        # Return the next action to take
        # Assume all unknown stochastic edges are not blocked
        belief_state = belief_state.at[0, 1:, :].set(
            jnp.where(
                belief_state[0, 1:, :] == CTP_generator.UNKNOWN,
                CTP_generator.UNBLOCKED,
                belief_state[0, 1:, :],
            )
        )
        # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
        belief_state = belief_state.at[1, 1:, :].set(
            jnp.where(
                belief_state[0, 1:, :] == CTP_generator.BLOCKED,
                1,
                belief_state[1, 1:, :],
            )
        )
        path_length, path = optimal_path_length.dijkstra_with_path(belief_state)
        return path[1]

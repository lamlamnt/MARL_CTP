import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator, CTP_environment_generalize
from Evaluation import optimal_path_length


# Return the next action to take
def _optimistic_best_action(belief_state: jnp.ndarray) -> int:
    # Assume all unknown stochastic edges are not blocked
    belief_state[0, 1:, :] = jnp.where(
        belief_state[0, 1:, :] == CTP_generator.UNKNOWN,
        CTP_generator.UNBLOCKED,
        belief_state[0, 1:, :],
    )
    # dijkstra expects env_state. Need to either modify dijkstra_with_path OR change blocking_prob of known blocked edges to 1.
    belief_state[1, 1:, :] = jnp.where(
        belief_state[0, 1:, :] == CTP_generator.BLOCKED, 1, belief_state[1, 1:, :]
    )
    path_length, path = optimal_path_length.dijkstra_with_path(belief_state)
    return path[1]


# input env_state and return path length
# Optimistic Baselines
def optimistic(
    initial_env_state: jnp.ndarray,
    initial_belief_state: jnp.ndarray,
    environment: CTP_environment_generalize,
) -> int:
    # Take optimistic best action. Check if valid. If valid, proceed. If not, replan until reach goal.
    current_env_state = initial_env_state
    current_belief_state = initial_belief_state
    # Loop here until reach goal
    action = _optimistic_best_action(current_belief_state)
    # env_key is for resetting so not important that it matches here
    new_env_state, new_belief_state, reward, done, env_key = environment.step(
        env_key, current_env_state, current_belief_state, action
    )


# AO* search
def ao_star(belief_state: jnp.ndarray) -> int:
    pass


# Thompson sampling

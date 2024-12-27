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
        path_length, next_node = optimal_path_length.dijkstra_with_path(belief_state)
        return next_node

    @partial(jit, static_argnums=(0, 1))
    def get_path_length(
        self,
        environment: CTP_environment_generalize.CTP_General,
        initial_belief_state: jnp.ndarray,
        initial_env_state: jnp.ndarray,
        env_key: jax.random.PRNGKey,
    ) -> float:
        def cond_fn(carry):
            _, _, _, done, _ = carry
            return ~done  # Continue while not done

        def body_fn(carry):
            env_state, belief_state, total_path_length, done, env_key = carry
            action = self.act(belief_state)
            action = jnp.array([action])
            env_state, belief_state, reward, done, env_key = environment.step(
                env_key, env_state, belief_state, action
            )
            action = action[0]
            total_path_length += -reward
            env_key, _ = jax.random.split(env_key)  # technically not needed
            return env_state, belief_state, total_path_length, done, env_key

        total_path_length = 0.0
        carry = (
            initial_env_state,
            initial_belief_state,
            total_path_length,
            False,
            env_key,
        )
        final_carry = jax.lax.while_loop(cond_fn, body_fn, carry)
        total_path_length = final_carry[2]

        return total_path_length

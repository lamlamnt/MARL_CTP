from functools import partial
import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad, vmap
import sys
import jax

sys.path.append("..")
from edited_jym.agents.dqn import DQN
from Utils.invalid_action_masking import decide_validity_of_action_space


class DQN_Masking(DQN):
    def __init__(
        self,
        model: hk.Transformed,
        discount: float,
        n_actions: int,
    ) -> None:
        super(DQN_Masking, self).__init__(model, discount, n_actions)

    @partial(jit, static_argnums=(0))
    def act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        """
        Epsilon-Greedy policy with respect to the estimated Q-values.
        """

        def _random_action(args):
            subkey, invalid_mask = args
            probs = jax.nn.softmax(invalid_mask)
            sampled_action = jax.random.choice(subkey, invalid_mask.shape[0], p=probs)
            return sampled_action.astype(jnp.uint8)

        def _forward_pass(args):
            subkey, invalid_action_mask = args
            q_values = self.model.apply(online_net_params, state)
            # adjusted_q_values = jnp.minimum(
            #    q_values * invalid_action_mask, q_values * jnp.abs(invalid_action_mask)
            # )
            adjusted_q_values = jnp.where(invalid_action_mask == 1, q_values, -jnp.inf)
            return jnp.argmax(adjusted_q_values).astype(jnp.uint8)

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        invalid_action_mask = decide_validity_of_action_space(state)
        action = lax.cond(
            explore,
            _random_action,
            _forward_pass,
            operand=(subkey, invalid_action_mask),
        )
        return action, subkey

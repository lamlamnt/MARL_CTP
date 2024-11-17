from functools import partial

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad, vmap
import sys
import jax

sys.path.append("..")
from edited_jym.agents.base_agents import BaseDeepRLAgent
from Utils.invalid_action_masking import decide_validity_of_action_space


class DQN_Masking(BaseDeepRLAgent):
    def __init__(
        self,
        model: hk.Transformed,
        discount: float,
        n_actions: int,
    ) -> None:
        super(DQN_Masking, self).__init__(
            discount,
        )
        self.model = model
        self.n_actions = n_actions

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

    @partial(jit, static_argnames=("self", "optimizer"))
    def update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            states: jnp.ndarray,
            actions: jnp.ndarray,
            rewards: jnp.ndarray,
            next_states: jnp.ndarray,
            dones: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, rewards, next_states and done flags
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
            def _loss_fn(
                online_net_params,
                target_net_params,
                state,
                action,
                reward,
                next_state,
                done,
            ):
                target = reward + (1 - done) * self.discount * jnp.max(
                    self.model.apply(target_net_params, next_state),
                )
                prediction = self.model.apply(online_net_params, state)[action]
                return jnp.square(target - prediction)

            return jnp.mean(
                _loss_fn(
                    online_net_params,
                    target_net_params,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ),
                axis=0,
            )

        loss, grads = value_and_grad(_batch_loss_fn)(
            online_net_params, target_net_params, **experiences
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)

        return online_net_params, optimizer_state, loss

    @partial(jit, static_argnums=(0))
    def batch_act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        return vmap(
            DQN_Masking.act,
            in_axes=(None, 0, 0, 0, 0),
        )(self, key, online_net_params, state, epsilon)

    @partial(jit, static_argnames=("self", "optimizer"))
    def batch_update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        return vmap(
            DQN_Masking.update,
            in_axes=(0, 0, None, 0, 0),
        )(
            self,
            online_net_params,
            target_net_params,
            optimizer,
            optimizer_state,
            experiences,
        )

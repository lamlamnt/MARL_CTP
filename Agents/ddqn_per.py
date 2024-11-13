import jax
import jax.numpy as jnp
import sys
from jax import jit, lax, random, value_and_grad, vmap

sys.path.append("..")
from edited_jym.agents import DQN_PER
from functools import partial
import optax


class DDQN_PER(DQN_PER):
    def __init__(
        self,
        model,
        discount: float,
        n_actions: int,
    ) -> None:
        # Initialize the parent class (DQN_PER) with the given arguments
        super(DDQN_PER, self).__init__(
            model=model,
            discount=discount,
            n_actions=n_actions,
        )

    # Only modify the loss_function compared to DQN_PER
    @partial(jit, static_argnames=("self", "optimizer"))
    def update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        importance_weights: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: jnp.ndarray,
            next_state: jnp.ndarray,
            done: jnp.ndarray,
            priority: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, rewards, next_states and done flags
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
            def _loss_fn(
                online_net_params,
                target_net_params,
                state,
                action,
                reward,
                next_state,
                done,
                priority,
            ):
                # Get the action using the online network
                a_targ = jnp.argmax(self.model.apply(online_net_params, next_state))

                # Evaluate the chosen action using the target network
                q_targ = self.model.apply(target_net_params, next_state)[a_targ]

                # Calculate the TD target for DDQN
                target = reward + (1 - done) * self.discount * q_targ

                # target = reward + (1 - done) * self.discount * jnp.max(
                #    self.model.apply(target_net_params, next_state),
                # )

                prediction = self.model.apply(online_net_params, state)[action]
                return jnp.square(target - prediction)

            loss = (
                _loss_fn(
                    online_net_params,
                    target_net_params,
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    priority,
                )
                * importance_weights
            )

            return jnp.mean(loss, axis=0)

        loss, grads = value_and_grad(_batch_loss_fn)(
            online_net_params, target_net_params, **experiences
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)

        return online_net_params, optimizer_state, loss

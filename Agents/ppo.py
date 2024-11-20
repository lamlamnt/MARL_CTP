from functools import partial
import jax.numpy as jnp
import jax
import sys
import optax
from flax.training.train_state import TrainState

sys.path.append("..")
from edited_jym.agents.base_agents import BaseDeepRLAgent
from Networks.actor_critic_network import ActorCritic
from Environment import CTP_environment, CTP_generator
from Evaluation.optimal_path_length import dijkstra_shortest_path
import flax.linen as nn
from typing import Sequence, NamedTuple, Any


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    critic_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    belief_state: jnp.ndarray
    shortest_path: jnp.ndarray


class PPO:
    def __init__(
        self,
        model: nn.Module,
        environment: CTP_environment.CTP,
        discount_factor: float,
        gae_lambda: float,
        clip_eps: float,
        vf_coeff: float,
        ent_coeff: float,
        batch_size: int,
        num_minibatches: int,
    ) -> None:
        self.model = model
        self.environment = environment
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.batch_size = batch_size
        self.num_minibatches = num_minibatches

    # For inference only
    @partial(jax.jit, static_argnums=(0,))
    def act(self, key, params, belief_state, unused):
        pi, _ = self.model.apply(params, belief_state)
        action = pi.sample(seed=key)
        # action = pi.mode()
        # OR pi.mode()
        old_key, new_key = jax.random.split(key)
        return action, new_key

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, runner_state, unused):
        # Collect trajectories
        train_state, new_env_state, current_belief_state, key = runner_state
        action_key, env_key = jax.random.split(key, 2)

        current_env_state = new_env_state
        # Agent acts
        pi, critic_value = self.model.apply(train_state.params, current_belief_state)
        action = pi.sample(seed=key)
        log_prob = pi.log_prob(action)
        original_key, action_key = jax.random.split(action_key)

        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = self.environment.step(
            env_key, new_env_state, current_belief_state, action
        )
        action = action[0]
        shortest_path = jax.lax.cond(
            done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                self.environment.graph_realisation.graph.origin,
                self.environment.graph_realisation.graph.goal,
            ),
            lambda _: 0.0,
            operand=None,
        )

        runner_state = (train_state, new_env_state, new_belief_state, env_key)
        transition = Transition(
            done,
            action,
            critic_value,
            reward,
            log_prob,
            current_belief_state,
            shortest_path,
        )
        return runner_state, transition

    @partial(jax.jit, static_argnums=(0,))
    def calculate_gae(self, traj_batch, last_critic_val):
        def _get_advantages(gae_and_next_value, transition: Transition):
            gae, next_value = gae_and_next_value
            done, critic_value, reward = (
                transition.done,
                transition.critic_value,
                transition.reward,
            )
            delta = (
                reward + self.discount_factor * next_value * (1 - done) - critic_value
            )
            gae = delta + self.discount_factor * self.gae_lambda * (1 - done) * gae
            return (gae, critic_value), gae

        # Apply get_advantage to each element in traj_batch
        _, advantages = jax.lax.scan(
            _get_advantages,
            (jnp.zeros_like(last_critic_val), last_critic_val),
            traj_batch,
            reverse=True,
            unroll=16,
        )
        return advantages, advantages + traj_batch.critic_value

    def _loss_fn(self, params, traj_batch: Transition, gae, targets):
        # RERUN NETWORK
        pi, value = jax.vmap(self.model.apply, in_axes=(None, 0))(
            params, traj_batch.belief_state
        )
        # pi, value = self.model.apply(params, traj_batch.belief_state)
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.critic_value + (
            value - traj_batch.critic_value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = loss_actor + self.vf_coeff * value_loss - self.ent_coeff * entropy
        return total_loss, (value_loss, loss_actor, entropy)

    @partial(jax.jit, static_argnums=(0,))
    def _update_epoch(self, update_state, unused):
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info
            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            grad_fn = jax.value_and_grad(self._loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, self.batch_size)
        batch = (traj_batch, advantages, targets)
        batch = jax.tree_util.tree_map(
            lambda x: x.reshape((self.batch_size,) + x.shape[1:]), batch
        )
        shuffled_batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )

        # Mini-batch Updates
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [self.num_minibatches, -1] + list(x.shape[1:])),
            shuffled_batch,
        )
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss

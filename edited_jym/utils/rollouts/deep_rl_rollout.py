from typing import Callable

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ...agents import BaseDeepRLAgent
from ..replay_buffers import BaseReplayBuffer


def deep_rl_rollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    agent: BaseDeepRLAgent,
    env,
    replay_buffer: BaseReplayBuffer,
    state_shape: int,
    buffer_size: int,
    epsilon_decay_fn: Callable,
    epsilon_start: float,
    epsilon_end: float,
    decay_rate: float,
) -> dict:
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            action_key,
            env_key,
            buffer_key,
            env_state,
            belief_state,
            all_actions,
            all_rewards,
            all_done,
            losses,
        ) = val

        current_belief_state = belief_state
        epsilon = epsilon_decay_fn(epsilon_start, epsilon_end, i, decay_rate)
        # set epsilon to 1 for exploration. act returns subkey
        action, action_key = agent.act(
            action_key, model_params, current_belief_state, epsilon
        )
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        env_state, belief_state, reward, done, env_key = env.step(
            env_key, env_state, current_belief_state, action
        )
        action = action[0]
        experience = (current_belief_state, action, reward, belief_state, done)

        buffer_state = replay_buffer.add(buffer_state, experience, i)
        current_buffer_size = jnp.min(jnp.array([i, buffer_size]))

        experiences_batch, buffer_key = replay_buffer.sample(
            buffer_key,
            buffer_state,
            current_buffer_size,
        )

        model_params, optimizer_state, loss = agent.update(
            model_params,
            target_net_params,
            optimizer,
            optimizer_state,
            experiences_batch,
        )

        # update the target parameters every ``target_net_update_freq`` steps
        target_net_params = lax.cond(
            i % target_net_update_freq == 0,
            lambda _: model_params,
            lambda _: target_net_params,
            operand=None,
        )

        all_actions = all_actions.at[i].set(action)
        all_rewards = all_rewards.at[i].set(reward)
        all_done = all_done.at[i].set(done)
        losses = losses.at[i].set(loss)

        val = (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            action_key,
            env_key,
            buffer_key,
            env_state,
            belief_state,
            all_actions,
            all_rewards,
            all_done,
            losses,
        )

        return val

    init_key, action_key, buffer_key, env_key = vmap(random.PRNGKey)(
        jnp.arange(4) + random_seed
    )
    env_state, belief_state = env.reset(init_key)
    all_actions = jnp.zeros([timesteps])
    all_rewards = jnp.zeros([timesteps], dtype=jnp.float32)
    all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
    losses = jnp.zeros([timesteps], dtype=jnp.float32)

    model_params = model.init(init_key, jnp.zeros(state_shape))
    target_net_params = model.init(action_key, jnp.zeros(state_shape))
    optimizer_state = optimizer.init(model_params)

    val_init = (
        model_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        action_key,
        env_key,
        buffer_key,
        env_state,
        belief_state,
        all_actions,
        all_rewards,
        all_done,
        losses,
    )

    vals = lax.fori_loop(0, timesteps, _fori_body, val_init)
    output_dict = {}
    keys = [
        "model_params",
        "target_net_params",
        "optimizer_state",
        "buffer_state",
        "action_key",
        "env_key",
        "buffer_key",
        "env_state",
        "belief_state",
        "all_actions",
        "all_rewards",
        "all_done",
        "losses",
    ]
    for idx, value in enumerate(vals):
        output_dict[keys[idx]] = value

    return output_dict

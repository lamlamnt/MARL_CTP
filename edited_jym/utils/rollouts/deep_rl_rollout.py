from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ...agents import BaseDeepRLAgent
from ..replay_buffers import BaseReplayBuffer
import sys

sys.path.append("../../..")
from Environment import CTP_generator, CTP_environment
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Utils import invalid_action_masking


def deep_rl_rollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    agent: BaseDeepRLAgent,
    env: CTP_environment.CTP,
    replay_buffer: BaseReplayBuffer,
    state_shape: int,
    buffer_size: int,
    epsilon_decay_fn: Callable,
    epsilon_start: float,
    epsilon_end: float,
    duration: float,
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
            all_optimal_path_lengths,
        ) = val

        current_belief_state = belief_state
        current_env_state = env_state
        epsilon = epsilon_decay_fn(epsilon_start, epsilon_end, i, duration)
        action, action_key = agent.act(
            action_key, model_params, current_belief_state, epsilon
        )
        # For multi-agent, we would concatenate all the agents' actions together here
        action = jnp.array([action])
        env_state, belief_state, reward, done, env_key = env.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]
        shortest_path = jax.lax.cond(
            done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                env.graph_realisation.graph.origin.item(),
                env.graph_realisation.graph.goal.item(),
            ),
            lambda _: 0.0,
            operand=None,
        )
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

        # update the target parameters every `target_net_update_freq` steps
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
        all_optimal_path_lengths = all_optimal_path_lengths.at[i].set(shortest_path)

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
            all_optimal_path_lengths,
        )

        return val

    init_key, action_key, buffer_key, env_key = vmap(random.PRNGKey)(
        jnp.arange(4) + random_seed
    )
    env_state, belief_state = env.reset(init_key)
    all_actions = jnp.zeros([timesteps], dtype=jnp.uint8)
    all_rewards = jnp.zeros([timesteps], dtype=jnp.float16)
    all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
    all_optimal_path_lengths = jnp.zeros([timesteps], dtype=jnp.float16)
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
        all_optimal_path_lengths,
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
        "all_optimal_path_lengths",
    ]
    for idx, value in enumerate(vals):
        output_dict[keys[idx]] = value

    return output_dict

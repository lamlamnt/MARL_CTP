from functools import partial
from typing import Callable, List

import haiku as hk
import jax.numpy as jnp
import optax
import jax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ...agents import BaseDeepRLAgent
from ..replay_buffers import Experience, PrioritizedExperienceReplay
import sys

sys.path.append("..")
from Environment import CTP_environment, CTP_generator
from Evaluation.optimal_path_length import dijkstra_shortest_path


@partial(vmap, in_axes=(None, None, None, None))
def compute_td_error(
    model,
    online_net_params: dict,
    target_net_params: dict,
    discount: float,
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_state: jnp.ndarray,
    done: jnp.ndarray,
    priority: jnp.ndarray,  # unused
) -> List[float]:
    """
    Computes the td errors for a batch of experiences.
    Errors are clipped to [-1, 1] for statibility reasons.
    """
    td_target = (
        (1 - done) * discount * jnp.max(model.apply(target_net_params, next_state))
    )
    prediction = model.apply(online_net_params, state)[action]
    return jnp.clip(reward + td_target - prediction, a_min=-1, a_max=1)


def per_rollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    tree_state: jnp.ndarray,
    agent: BaseDeepRLAgent,
    env: CTP_environment.CTP,
    state_shape: int,
    buffer_size: int,
    batch_size: int,
    alpha: float,
    beta: float,
    discount: float,
    epsilon_decay_fn: Callable,
    epsilon_start: float,
    epsilon_end: float,
    duration: int,
) -> dict[jnp.ndarray | dict]:
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            tree_state,
            env_key,
            action_key,
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
        action = jnp.array([action])
        env_state, belief_state, reward, done, env_key = env.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]
        shortest_path = jax.lax.cond(
            done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                env.graph_realisation.graph.origin,
                env.graph_realisation.graph.goal,
            ),
            lambda _: 0.0,
            operand=None,
        )

        experience = Experience(
            state=current_belief_state,
            action=action,
            reward=reward,
            next_state=belief_state,
            done=done,
        )

        buffer_state, tree_state = replay_buffer.add(
            tree_state, buffer_state, i, experience
        )

        (
            experiences_batch,
            sample_indexes,
            importance_weights,
            buffer_key,
        ) = replay_buffer.sample(buffer_key, buffer_state, tree_state)

        # compute individual td errors for the sampled batch and
        # update the tree state using the batched absolute td errors
        td_errors = compute_td_error(
            model, model_params, target_net_params, discount, **experiences_batch
        )

        tree_state = replay_buffer.sum_tree.batch_update(
            tree_state, sample_indexes, jnp.abs(td_errors)
        )
        model_params, optimizer_state, loss = agent.update(
            model_params,
            target_net_params,
            optimizer,
            optimizer_state,
            importance_weights,
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
        all_optimal_path_lengths = all_optimal_path_lengths.at[i].set(shortest_path)
        losses = losses.at[i].set(loss)

        val = (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            tree_state,
            env_key,
            action_key,
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
    losses = jnp.zeros([timesteps], dtype=jnp.bfloat16)

    model_params = model.init(init_key, jnp.zeros(state_shape))
    target_net_params = model.init(action_key, jnp.zeros(state_shape))
    optimizer_state = optimizer.init(model_params)
    replay_buffer = PrioritizedExperienceReplay(buffer_size, batch_size, alpha, beta)

    val_init = (
        model_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        tree_state,
        env_key,
        action_key,
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
        "tree_state",
        "env_key",
        "action_key",
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

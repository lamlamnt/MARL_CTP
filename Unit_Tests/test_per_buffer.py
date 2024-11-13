import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment, CTP_generator
from edited_jym.utils.replay_buffers import Experience, PrioritizedExperienceReplay
import pytest


# test that it doesn't error
def test_per_replay_buffer():
    batch_size = 5
    buffer_size = 10
    replay_buffer = PrioritizedExperienceReplay(buffer_size, batch_size, 0.6, 1)
    tree_state = jnp.zeros(2 * buffer_size - 1)
    buffer_state = buffer_state = {
        "state": jnp.empty((buffer_size, (1)), dtype=jnp.float32),
        "action": jnp.empty((buffer_size,), dtype=jnp.int32),
        "reward": jnp.empty((buffer_size,), dtype=jnp.float32),
        "next_state": jnp.empty((buffer_size, (1)), dtype=jnp.float32),
        "done": jnp.empty((buffer_size,), dtype=jnp.bool_),
        "priority": jnp.empty((buffer_size,), dtype=jnp.float32),
    }
    current_belief_state = jnp.array([1])
    action = 1
    reward = 1
    next_belief_state = jnp.array([2])
    done = False
    experience = Experience(
        state=current_belief_state,
        action=action,
        reward=reward,
        next_state=next_belief_state,
        done=done,
    )

    for field in experience:
        buffer_state[field] = buffer_state[field].at[0].set(experience[field])

    buffer_state, tree_state = replay_buffer.add(
        tree_state, buffer_state, 0, experience
    )

from abc import ABC, abstractmethod
from functools import partial

from jax import jit
from .uniform_replay_buffer import UniformReplayBuffer


class Action_Masking_Buffer(UniformReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        super(Action_Masking_Buffer, self).__init__(buffer_size, batch_size)

    @partial(jit, static_argnums=(0))
    def add(
        self,
        buffer_state: dict,
        experience: tuple,
        idx: int,
    ):
        state, action, reward, next_state, done, invalid_action_mask = experience
        idx = idx % self.buffer_size

        buffer_state["states"] = buffer_state["states"].at[idx].set(state)
        buffer_state["actions"] = buffer_state["actions"].at[idx].set(action)
        buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward)
        buffer_state["next_states"] = (
            buffer_state["next_states"].at[idx].set(next_state)
        )
        buffer_state["dones"] = buffer_state["dones"].at[idx].set(done)
        buffer_state["invalid_action_masks"] = (
            buffer_state["invalid_action_masks"].at[idx].set(invalid_action_mask)
        )

        return buffer_state

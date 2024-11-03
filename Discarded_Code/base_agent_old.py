import jax
import jax.numpy as jnp
from Environment import CTP_environment_old
from functools import partial


class BaseAgent:
    def __init__(self):
        pass

    def reset(self):
        # Reset network and hyperparameters
        pass

    @partial(jax.jit, static_argnums=(0,))
    def act(
        self,
        key: jax.random.PRNGKey,
        state: CTP_environment_old.EnvState,
        observation: CTP_environment_old.Observation,
    ) -> int:
        # Return the action to take
        pass

    def update(
        self,
        state: CTP_environment_old.EnvState,
        observation: CTP_environment_old.Observation,
        action: int,
        reward: float,
        next_state: CTP_environment_old.EnvState,
        next_observation: CTP_environment_old.Observation,
        terminate: bool,
    ):
        # Update the network
        pass

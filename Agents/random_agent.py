from Agents.base_agent import BaseAgent
import jax
import jax.numpy as jnp
import jaxmarl
from functools import partial
from Environment import CTP_environment_old


class RandomAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    @partial(jax.jit, static_argnums=(0,))
    def act(
        self,
        key: jax.random.PRNGKey,
        state: CTP_environment_old.EnvState,
        observation: CTP_environment_old.Observation,
    ) -> int:
        return self.action_space.sample(key)

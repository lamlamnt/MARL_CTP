import agent
import jax 
import jax.numpy as jnp
import jaxmarl
from functools import partial
import CTP_environment

class RandomAgent(agent.Agent):
    def __init__(self, action_space):
        self.action_space = action_space
    
    @partial(jax.jit, static_argnums=(0,))
    def act(self, key: jax.random.PRNGKey,state:CTP_environment.EnvState,observation:CTP_environment.Observation) -> int:
        return self.action_space.sample(key)

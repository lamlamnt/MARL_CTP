import agent
import jax 
import jax.numpy as jnp
import jaxmarl
import CTP_environment

class RandomAgent(agent.Agent):
    def __init__(self, action_space):
        super().__init__(action_space)
    
    def act(self, state:CTP_environment.EnvState,observation:CTP_environment.Observation, key: jax.random.PRNGKey) -> int:
        return self.action_space.sample(key)

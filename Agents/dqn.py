from Agents.base_agent import BaseAgent
import jax 
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments import spaces
from Environment import CTP_environment
from functools import partial
import optax

class DQN_Agent(BaseAgent):
    def __init__(self, n_actions:int,discount_factor=0.99, learning_rate=0.1,epsilon=0.1):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon #epsilon-greedy policy - epsilon decreases over time
        self.n_actions = n_actions

        #data structure for replay buffer (use library)
        #replay buffer contains tuples of (state, action, reward, next_state, terminate)

    def reset(self):
        # Reset network and hyperparameters
        pass

    @partial(jax.jit, static_argnums=(0,))
    def act(self, key: jax.random.PRNGKey,state:CTP_environment.EnvState,observation:CTP_environment.Observation) -> int:
        # Epsilon greedy policy
        explore = jax.random.uniform(key) < self.epsilon
        key, subkey = jax.random.split(key)
        action = jax.lax.cond(explore,lambda subkey:jax.random.randint(subkey, shape=(1,), minval=0, maxval=self.n_actions).astype(jnp.int32),lambda _:self._forward_pass(state,observation),subkey)
        return action
    
    def _forward_pass(self,state:CTP_environment.EnvState,observation:CTP_environment.Observation):
        # Use online network to do action selection (max Q value)
        #q_values = self.model.apply(online_net_params, None, state)
        #return jnp.argmax(q_values)
        return jnp.array([2])
    
    def update(self,state:CTP_environment.EnvState,observation:CTP_environment.Observation,action:int,reward:float,next_state:CTP_environment.EnvState,next_observation:CTP_environment.Observation,terminate:bool):
        # Update the network
        pass


import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
import CTP_generator
import jraph 
from typing import Dict,Tuple
from flax.core import FrozenDict

@chex.dataclass
class EnvState:
    agents_pos: jnp.ndarray

#Each observation is a dictionary of dictionaries, where the first key is the agent index and 
#the second key is the edge (sender,receiver) and the value is the blocking status
# Make JAX compatible
@chex.dataclass
class Observation:
    observation: FrozenDict[int, FrozenDict[Tuple[int, int], bool]]

#An action is just the index of the next node to move to
#Action space is a discrete space of size num_nodes

class CTP(MultiAgentEnv):
    def __init__(self, num_agents:int, num_goals:int, num_nodes:int, key:chex.PRNGKey,prop_stoch=0.4,grid_size=10):
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.grid_size = grid_size
        self.num_nodes = num_nodes
        self.prop_stoch = prop_stoch
        
        # Generate the graph and get origin and goal
        self.graph,self.origin,self.goal = CTP_generator.generate_graph(self.num_nodes, key)
        self.graph = CTP_generator.make_edges_blocked(self.graph,key,self.prop_stoch)

        # Define observation space and action space
        self.action_space = spaces.Discrete(self.num_nodes)
        
    def reset(self,key:chex.PRNGKey) -> tuple[Observation,EnvState]:
        # Resample the blocking status
        self.graph = CTP_generator.sample_blocking_prob(key,self.graph)

        # Return origin as starting environment state
        starting_env_state = EnvState(agents_pos=self.origin)
        return self._state_to_observation(starting_env_state), starting_env_state

    def step(self, state:EnvState,action:int) -> tuple[Observation,EnvState,int,bool]:
        # Return observation, state, reward, and whether the episode is done

        # Check if action is valid
        pass

    def _state_to_observation(self, state:EnvState) -> Observation:
        # Get the blocking status of the edges connected to a node
        pass


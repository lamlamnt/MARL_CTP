from typing import Tuple
import jax.numpy as jnp
from flax.core import FrozenDict

#Each observation is a dictionary of dictionaries, where the first key is the agent index and 
#the second key is the edge (sender,receiver) and the value is the blocking status (0 means not blocked).
Observation = FrozenDict[int, FrozenDict[Tuple[int, int], bool]]

def get_obs(self, state:EnvState) -> Observation:
        # Get the blocking status of the stochastic edges connected to a node
        for agent_num in range(self.num_agents):                
            edge_indices = jnp.where(jnp.logical_and(jnp.logical_or(self.true_graph.senders == state.agents_pos[agent_num],self.true_graph.receivers == state.agents_pos[agent_num]), self.true_graph.edges['blocked_prob'] > 0))
            observation = FrozenDict({
                agent_num: FrozenDict({
                    (int(self.true_graph.senders[edge_indices[0][i]]), int(self.true_graph.receivers[edge_indices[0][i]])): 
                    int(self.true_graph.edges['blocked_status'][edge_indices[0][i]])  # Converting the edge status to int as well
                    for i in range(len(edge_indices[0]))
                })
            })
        return observation
                

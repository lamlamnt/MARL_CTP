import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from Environment import CTP_generator
from typing import TypeAlias

Belief_State: TypeAlias = jnp.ndarray


@chex.dataclass
class EnvState:
    weights_and_agents_pos: jnp.ndarray
    blocking_prob: jnp.ndarray
    goals_progress: jnp.ndarray


class CTP(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int,
        num_goals: int,
        num_nodes: int,
        key: chex.PRNGKey,
        prop_stoch=None,
        k_edges=None,
        grid_size=10,
        reward_for_invalid_action=-200,
    ):
        """
        List of attributes:
        graph_realisation: a CTPGraph_Realisation object
        num_agents: int
        reward_for_invalid_action: int
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = reward_for_invalid_action

        # Instantiate a CTPGraph_Realisation object
        self.graph_realisation = CTP_generator.CTPGraph_Realisation(
            key,
            num_nodes,
            grid_size=grid_size,
            prop_stoch=prop_stoch,
            k_edges=k_edges,
            num_goals=num_goals,
        )

        actions = [num_nodes for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

    def reset(self, key: chex.PRNGKey) -> Belief_State:
        self.current_state = EnvState()
        pass

    def step(self, actions: jnp.ndarray) -> tuple[Belief_State, int, bool]:
        pass

    # Need to input previous belief state and current
    def get_belief_state(self) -> Belief_State:
        pass

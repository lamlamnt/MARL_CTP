import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from Environment import CTP_generator
import jraph
from typing import Tuple
from flax.core import FrozenDict


@chex.dataclass
class EnvState:
    # agents_pos and list_of_goals contain indices of the nodes
    agents_pos: jnp.array
    # list_of_goals: jnp.array
    # graph: jraph.GraphsTuple #globals = 0 (does not contain blocking status)


# Each observation is a dictionary of dictionaries, where the first key is the agent index and
# the second key is the edge (sender,receiver) and the value is the blocking status (0 means not blocked).
# Observation = FrozenDict[int, FrozenDict[Tuple[int, int], bool]]


@chex.dataclass
class Observation:
    location: jnp.ndarray
    blocked_status: jnp.ndarray


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
        add_expensive_edge=True,
        reward_for_invalid_action=-500,
        factor_for_expensive_edge=3,
    ):
        super().__init__(num_agents=num_agents)
        self.num_goals = num_goals
        self.add_expensive_edge = add_expensive_edge
        self.reward_for_invalid_action = reward_for_invalid_action

        # Generate the graph and get origin and goal
        # Currently, choosing origin and goal for one agent only.
        self.agent_graph = CTP_generator.generate_graph(
            num_nodes, key, grid_size=grid_size, prop_stoch=prop_stoch, k_edges=k_edges
        )

        self.origin, self.goal = CTP_generator.find_single_goal_and_origin(
            self.agent_graph
        )

        # For multiple agents, origin will be an array of size n_agents and goal will be an array that can be of any size
        self.origin = jnp.array([self.origin])
        self.goal = jnp.array([self.goal])

        # An action is just the index of the next node to move to.
        # An action changes agent_pos to a different node.
        # Jax.lax.cond doesn't work here because spaces. Discrete is not a valid jax type
        if self.add_expensive_edge:
            actions = [num_nodes for _ in range(self.num_agents)]
            # Add an expensive edge.
            largest_edge_weight = jnp.max(self.agent_graph.edges["weight"])
            self.agent_graph = CTP_generator.add_expensive_edge(
                self.agent_graph,
                largest_edge_weight * factor_for_expensive_edge,
                self.origin,
                self.goal,
            )
        else:
            # If action = self.num_nodes, the agent is saying it's not solvable
            actions = [num_nodes + 1 for _ in range(self.num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)
        self.max_edges = CTP_generator.get_max_edges(self.agent_graph)

    def reset(self, key: chex.PRNGKey) -> tuple[Observation, EnvState]:
        starting_env_state = EnvState(agents_pos=self.origin)
        # Sample the blocking status
        # self.agent_graph is the graph without the blocking status. self.true_graph contains blocking status
        self.true_graph = CTP_generator.sample_blocking_prob(key, self.agent_graph)
        # This if-else statement cannot be converted to jax.lax.cond because requires converting to networkx graph
        if (self.add_expensive_edge) is True:
            self.solvable = True
        else:
            self.solvable = CTP_generator.is_solvable(
                self.true_graph, self.origin.item(), self.goal.item()
            )
        return self.get_obs(starting_env_state), starting_env_state

    # change the if statements in this function to jax.lax.cond/make this important function more jax-compatible
    # Whether returning different types like this affect performance?
    def step(
        self, state: EnvState, action: jnp.array
    ) -> tuple[Observation, EnvState, int, bool]:
        # Return observation, state, reward, and whether the episode is done
        index_of_edge = jax.vmap(self._valid_action, in_axes=(0, 0, None))(
            state.agents_pos, action, self.true_graph
        )

        # Can easily be converted to jax.lax.cond but less readable
        if (
            self.add_expensive_edge is False
            and action == self.true_graph.n_node
            and self.solvable is True
        ):
            terminate = True
            reward = 0
        elif (
            self.add_expensive_edge is False
            and action == self.true_graph.n_node
            and self.solvable is False
        ):
            terminate = False
            reward = self.reward_for_invalid_action
        elif index_of_edge == self.true_graph.n_edge:
            # Attempt to move through an edge that does not exist or is a blocked edge
            # Need to Loop over all agents here! If any of the agent has invalid action, penalty
            # Does not change the state of the agent
            terminate = False
            reward = self.reward_for_invalid_action
        elif action == self.goal[0]:
            # at the goal
            terminate = True
            reward = 0
        else:
            terminate = False
            reward = (-self.true_graph.edges["weight"][index_of_edge]).item()
            # If at goal, episode is done.
            state.agents_pos = action

        observation = self.get_obs(state)

        return observation, state, reward, terminate

    def _valid_action(
        self, node1: int, node2: int, true_graph: jraph.GraphsTuple
    ) -> Tuple[bool, int]:
        index_of_edge = jax.lax.cond(
            node2 > node1,
            lambda node1, node2: jnp.argmax(
                jnp.append(
                    jnp.logical_and(
                        jnp.logical_and(
                            true_graph.senders == node1,
                            true_graph.receivers == node2,
                        ),
                        true_graph.edges["blocked_status"] == 0,
                    ),
                    True,
                )
            ),
            lambda node1, node2: jnp.argmax(
                jnp.append(
                    jnp.logical_and(
                        jnp.logical_and(
                            true_graph.senders == node2,
                            true_graph.receivers == node1,
                        ),
                        true_graph.edges["blocked_status"] == 0,
                    ),
                    True,
                )
            ),
            node1,
            node2,
        )
        return index_of_edge

    """
    # This is currently not jax-compatible because edge_indices size varies
    def get_obs(self, state: EnvState) -> Observation:
        # Get the blocking status of the stochastic edges connected to a node
        for agent_num in range(self.num_agents):
            edge_indices = jnp.where(
                jnp.logical_and(
                    jnp.logical_or(
                        self.true_graph.senders == state.agents_pos[agent_num],
                        self.true_graph.receivers == state.agents_pos[agent_num],
                    ),
                    self.true_graph.edges["blocked_prob"] > 0,
                )
            )
            observation = FrozenDict(
                {
                    agent_num: FrozenDict(
                        {
                            (
                                int(self.true_graph.senders[edge_indices[0][i]]),
                                int(self.true_graph.receivers[edge_indices[0][i]]),
                            ): int(
                                self.true_graph.edges["blocked_status"][
                                    edge_indices[0][i]
                                ]
                            )  # Converting the edge status to int as well
                            for i in range(len(edge_indices[0]))
                        }
                    )
                }
            )
        return observation
        """

    # Loop over all agents, not just agent 0
    def get_obs(self, state: EnvState) -> Observation:
        # Get the blocking status of the stochastic edges connected to a node
        current_location = state.agents_pos[0]
        current_observation = Observation(
            location=current_location, blocked_status=jnp.full((self.max_edges + 1,), 2)
        )
        # Check for node in both sender and receiver (get indices)
        edge_indices = jnp.where(
            jnp.logical_or(
                self.true_graph.senders == current_location,
                self.true_graph.receivers == current_location,
            )
        )
        for i in range(len(edge_indices[0])):
            # Get the blocking status
            current_observation.blocked_status = current_observation.blocked_status.at[
                i
            ].set(self.true_graph.edges["blocked_status"][edge_indices[0][i]])
        """
        current_observation.blocked_status = jax.vmap(
            lambda i: current_observation.blocked_status.at[i].set(
                self.true_graph.edges["blocked_status"][edge_indices][0][i]
            )
        )(jnp.arange(len(edge_indices[0])))
        """
        return current_observation

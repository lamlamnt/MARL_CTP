from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from chex import dataclass
from Environment import CTP_generator
from typing import TypeAlias
import sys
from tqdm import tqdm

sys.path.append("..")
from Utils import graph_functions, util_generalize

# Belief_state contains the agents' positions + current knowledge about blocked status, edge_weights, edge_probs, and goals in this order
# 4D tensor where each channel is size (num_agents+num_nodes, num_agents+num_nodes)
Belief_State: TypeAlias = jnp.ndarray
# current location of agents and knowledge of blocking status of connected edges
Observation: TypeAlias = jnp.ndarray
EnvState: TypeAlias = jnp.ndarray


class CTP_General(MultiAgentEnv):
    def __init__(
        self,
        num_agents: int,
        num_goals: int,
        num_nodes: int,
        key: chex.PRNGKey,
        prop_stoch=None,
        k_edges=None,
        grid_size=None,
        reward_for_invalid_action=-200.0,
        reward_for_goal=0,
        factor_expensive_edge=1.0,
        handcrafted_graph=None,
        deal_with_unsolvability="always_expensive_edge",
        patience=5,
        num_stored_graphs=10,
        loaded_graphs=None,
        origin_node=-1,
    ):
        """
        List of attributes:
        num_agents: int
        num_nodes: int
        reward_for_invalid_action: int
        reward_for_goal:int
        action_spaces
        graph_realisation: CTPGraph_Realisation
        patience: int # How many times we try to find a solvable blocking status before giving up
        num_stored_graphs: int # Number of stored graphs to choose from
        factor_expensive_edge: float
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = jnp.float16(reward_for_invalid_action)
        self.reward_for_goal = jnp.float16(reward_for_goal)
        self.num_nodes = num_nodes
        self.patience = patience
        self.num_stored_graphs = num_stored_graphs
        self.factor_expensive_edge = factor_expensive_edge
        self.origin_node = origin_node

        assert origin_node < num_nodes

        if handcrafted_graph is not None:
            raise ValueError("Generalizing. Cannot use handcrafted graph.")

        actions = [num_nodes for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

        if deal_with_unsolvability == "resample":
            raise ValueError("Not implemented yet.")
        elif deal_with_unsolvability == "always_expensive_edge":
            auto_expensive_edge = True
        else:
            auto_expensive_edge = False

        # Generate graphs
        if loaded_graphs is not None:
            self.stored_graphs = loaded_graphs
        else:
            key, subkey = jax.random.split(key)
            self.stored_graphs = jnp.zeros(
                (num_stored_graphs, 3, num_nodes, num_nodes), dtype=jnp.float16
            )
            print("Generating graphs...")
            for i in tqdm(range(num_stored_graphs)):
                key, subkey = jax.random.split(subkey)
                graph_realisation = CTP_generator.CTPGraph_Realisation(
                    subkey,
                    self.num_nodes,
                    grid_size=grid_size,
                    prop_stoch=prop_stoch,
                    k_edges=k_edges,
                    num_goals=num_goals,
                    factor_expensive_edge=factor_expensive_edge,
                    expensive_edge=auto_expensive_edge,
                )

                # Change origin here so that the expensive edge is added if not solvable in the function below
                graph_realisation.graph.origin = jax.lax.cond(
                    self.origin_node == -1,
                    lambda _: graph_realisation.graph.origin,
                    lambda _: self.origin_node.astype(jnp.int16),
                    operand=None,
                )

                # Normalize the weights using expected optimal path length
                expected_optimal_path_length = (
                    util_generalize.get_expected_optimal_path_length(
                        graph_realisation, key, self.factor_expensive_edge
                    )
                )
                # normalizing_factor = 0.5 * self.num_nodes / 2

                graph_realisation.graph.weights = jnp.where(
                    graph_realisation.graph.weights != CTP_generator.NOT_CONNECTED,
                    graph_realisation.graph.weights / expected_optimal_path_length,
                    CTP_generator.NOT_CONNECTED,
                )

                # Store the matrix of weights, blocking probs, and origin/goal
                self.stored_graphs = self.stored_graphs.at[i, 0, :, :].set(
                    graph_realisation.graph.weights
                )
                self.stored_graphs = self.stored_graphs.at[i, 1, :, :].set(
                    graph_realisation.graph.blocking_prob
                )
                self.stored_graphs = self.stored_graphs.at[i, 2, 0, 0].set(
                    graph_realisation.graph.origin[0]
                )
                self.stored_graphs = self.stored_graphs.at[i, 2, 0, 1].set(
                    graph_realisation.graph.goal[0]
                )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[EnvState, Belief_State]:
        key, subkey = jax.random.split(key)

        # Sample from list of stored graph realisations
        index = jax.random.randint(
            subkey, shape=(), minval=0, maxval=self.num_stored_graphs - 1
        )
        current_graph_weights = self.stored_graphs[index, 0, :, :]
        current_graph_blocking_prob = self.stored_graphs[index, 1, :, :]
        current_graph_origin = self.stored_graphs[index, 2, 0, 0].astype(jnp.int16)
        current_graph_goal = self.stored_graphs[index, 2, 0, 1].astype(jnp.int16)

        # Get solvable realisation - add expensive edge if unsolvable
        new_blocking_status = graph_functions.sample_blocking_status(
            subkey, current_graph_blocking_prob
        )
        is_solvable = graph_functions.is_solvable(
            current_graph_weights,
            new_blocking_status,
            current_graph_origin,
            current_graph_goal,
        )

        new_blocking_status, current_graph_weights, current_graph_blocking_prob = (
            jax.lax.cond(
                is_solvable == jnp.bool_(False),
                lambda _: util_generalize.add_expensive_edge(
                    new_blocking_status,
                    current_graph_weights,
                    current_graph_blocking_prob,
                    current_graph_goal,
                    current_graph_origin,
                    self.factor_expensive_edge,
                ),
                lambda _: (
                    new_blocking_status,
                    current_graph_weights,
                    current_graph_blocking_prob,
                ),
                operand=None,
            )
        )

        env_state = self.__convert_graph_realisation_to_state(
            current_graph_origin,
            current_graph_goal,
            new_blocking_status,
            current_graph_weights,
            current_graph_blocking_prob,
        )

        # return the initial belief state
        original_observation = self.get_obs(env_state)
        # Incorporate info that non-existent edges are blocked and deterministic edges are not blocked
        blocking_status_knowledge = jnp.where(
            jnp.logical_or(
                env_state[2, self.num_agents :, :] == 0,
                env_state[2, self.num_agents :, :] == 1,
            ),
            env_state[0, self.num_agents :, :],
            original_observation[self.num_agents :, :],
        )
        pos_and_blocking_status = original_observation.at[self.num_agents :, :].set(
            blocking_status_knowledge
        )
        initial_belief_state = jnp.stack(
            (
                pos_and_blocking_status,
                env_state[1, :, :],
                env_state[2, :, :],
                env_state[3, :, :],
            ),
            axis=0,
            dtype=jnp.float16,
        )
        return env_state, initial_belief_state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.random.PRNGKey,
        current_env_state: EnvState,
        current_belief_state: Belief_State,
        actions: jnp.ndarray,
    ) -> tuple[EnvState, Belief_State, int, bool]:
        # return the new environment state, next belief state, reward, and whether the episode is done
        weights = current_env_state[1, self.num_agents :, :]
        blocking_status = current_env_state[0, self.num_agents :, :]

        # Use environment state and actions to determine if the action is valid
        def _is_invalid_action(
            actions: jnp.ndarray, current_env_state: jnp.array
        ) -> bool:
            return jnp.logical_or(
                actions[0] == jnp.argmax(current_env_state[0, :1, :]),
                jnp.logical_or(
                    weights[jnp.argmax(current_env_state[0, :1, :]), actions[0]]
                    == CTP_generator.NOT_CONNECTED,
                    blocking_status[jnp.argmax(current_env_state[0, :1, :]), actions[0]]
                    == CTP_generator.BLOCKED,
                ),
            )

        # If invalid action, then return the same state, reward is very negative, and terminate=False
        def _step_invalid_action(args) -> tuple[jnp.array, int, bool]:
            current_env_state, actions = args
            reward = self.reward_for_invalid_action
            terminate = jnp.bool_(False)
            return current_env_state, reward, terminate

        # Function that gets called if at goal
        def _at_goal(args) -> tuple[jnp.array, int, bool]:
            current_env_state, actions = args
            reward = (
                -(weights[jnp.argmax(current_env_state[0, :1, :]), actions[0]])
                + self.reward_for_goal
            )
            agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
            agents_pos = agents_pos.at[0, actions[0]].set(1)
            new_env_state = current_env_state.at[0, : self.num_agents, :].set(
                agents_pos
            )
            new_env_state = new_env_state.at[3, : self.num_agents, actions[0]].add(1)
            terminate = jnp.bool_(True)
            return new_env_state, reward, terminate

        # Function that gets called if valid action and not at goal -> move to new node
        def _move_to_new_node(args) -> tuple[jnp.array, int, bool]:
            current_env_state, actions = args
            reward = -(weights[jnp.argmax(current_env_state[0, :1, :]), actions[0]])
            agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
            agents_pos = agents_pos.at[0, actions[0]].set(1)
            new_env_state = current_env_state.at[0, : self.num_agents, :].set(
                agents_pos
            )
            terminate = jnp.bool_(False)
            return new_env_state, reward, terminate

        new_env_state, reward, terminate = jax.lax.cond(
            _is_invalid_action(actions, current_env_state),
            _step_invalid_action,
            lambda args: jax.lax.cond(
                current_env_state[3, 1:, :][actions[0], actions[0]] > 0,
                _at_goal,
                _move_to_new_node,
                args,
            ),
            (current_env_state, actions),
        )
        new_observation = self.get_obs(new_env_state)
        next_belief_state = self.get_belief_state(current_belief_state, new_observation)
        key, subkey = jax.random.split(key)

        new_env_state, next_belief_state = jax.lax.cond(
            terminate,
            lambda x: self.reset(x),
            lambda x: (new_env_state, next_belief_state),
            key,
        )

        return new_env_state, next_belief_state, reward, terminate, subkey

    # use current state to get observation
    def get_obs(self, env_state: EnvState) -> jnp.ndarray:
        agents_pos = env_state[0, : self.num_agents, :]
        blocking_status = env_state[0, self.num_agents :, :]
        # Get edges connected to agent's current position
        obs_blocking_status = jnp.full(
            (
                self.num_nodes,
                self.num_nodes,
            ),
            CTP_generator.UNKNOWN,
            dtype=jnp.float16,
        )
        # replace 1 row and column corresponding to agent's position
        obs_blocking_status = obs_blocking_status.at[jnp.argmax(agents_pos[0]), :].set(
            blocking_status[jnp.argmax(agents_pos[0]), :]
        )
        obs_blocking_status = obs_blocking_status.at[:, jnp.argmax(agents_pos[0])].set(
            blocking_status[jnp.argmax(agents_pos[0]), :]
        )
        # Concatenate
        new_observation = jnp.concatenate((agents_pos, obs_blocking_status), axis=0)
        return new_observation

    # Need to use previous belief state and current state
    # @partial(jax.jit, static_argnums=(0,))
    def get_belief_state(
        self, old_belief_state: Belief_State, new_observation: Observation
    ) -> Belief_State:
        # Combine current_blocking_status with new_observation
        new_blocking_knowledge = jnp.where(
            old_belief_state[0, self.num_agents :, :] == CTP_generator.UNKNOWN,
            new_observation[self.num_agents :, :],
            old_belief_state[0, self.num_agents :, :],
        )

        # Replace the right channel with new_blocking_knowledge
        new_belief_state = old_belief_state.at[0, self.num_agents :, :].set(
            new_blocking_knowledge
        )

        # Update agent's position from new observation (for single agent!)
        new_belief_state = new_belief_state.at[0, : self.num_agents, :].set(
            new_observation[: self.num_agents, :]
        )

        return new_belief_state

    def __convert_graph_realisation_to_state(
        self,
        origin: int,
        goal: int,
        blocking_status: jnp.ndarray,
        graph_weights: jnp.ndarray,
        blocking_prob: jnp.ndarray,
    ) -> EnvState:
        agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        agents_pos = agents_pos.at[0, origin].set(1)
        empty = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        edge_weights = jnp.concatenate((empty, graph_weights), axis=0)
        edge_probs = jnp.concatenate((empty, blocking_prob), axis=0)
        pos_and_blocking_status = jnp.concatenate((agents_pos, blocking_status), axis=0)

        # Top part is each agent's service history. Bottom part is number of times each goal needs to
        # be serviced
        goal_matrix = jnp.zeros_like(pos_and_blocking_status)
        goal_matrix = goal_matrix.at[
            self.num_agents + goal,
            goal,
        ].set(1)

        return jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs, goal_matrix),
            axis=0,
            dtype=jnp.float16,
        )

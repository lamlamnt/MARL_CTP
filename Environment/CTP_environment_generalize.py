from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from chex import dataclass
from Environment import CTP_generator
from typing import TypeAlias

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

        num_goals: int
        prop_stoch: float
        k_edges: int
        grid_size: int
        deal_with_unsolvability: str (one of 3 options)
        factor_expensive_edge: float
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = jnp.float16(reward_for_invalid_action)
        self.reward_for_goal = jnp.float16(reward_for_goal)
        self.num_nodes = num_nodes
        self.patience = patience

        if handcrafted_graph is not None:
            raise ValueError("Generalizing. Cannot use handcrafted graph.")

        # Arguments for generating graphs. Maybe put into more compact representation, like a dictionary
        self.num_goals = num_goals
        self.prop_stoch = prop_stoch
        self.k_edges = k_edges
        self.grid_size = grid_size
        self.deal_with_unsolvability = deal_with_unsolvability
        self.factor_expensive_edge = factor_expensive_edge

        actions = [num_nodes for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[EnvState, Belief_State]:
        key, subkey = jax.random.split(key)

        # Pure callback function to return the env state
        result_shape = jax.ShapeDtypeStruct(
            (4, (self.num_nodes + self.num_agents), self.num_nodes), jnp.float16
        )
        env_state = jax.pure_callback(self.get_initial_env_state, result_shape, subkey)

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

    def get_initial_env_state(self, key: jax.random.PRNGKey):
        if self.deal_with_unsolvability == "always_expensive_edge":
            auto_expensive_edge = True
        else:
            auto_expensive_edge = False
        graph_realisation = CTP_generator.CTPGraph_Realisation(
            key,
            self.num_nodes,
            grid_size=self.grid_size,
            prop_stoch=self.prop_stoch,
            k_edges=self.k_edges,
            num_goals=self.num_goals,
            factor_expensive_edge=self.factor_expensive_edge,
            expensive_edge=auto_expensive_edge,
        )
        _, subkey = jax.random.split(key)

        # Resample until we get a solvable realisation
        if self.deal_with_unsolvability == "resample":
            patience_counter = 0
            is_solvable = jnp.bool_(False)
            while is_solvable == jnp.bool_(False) and patience_counter < self.patience:
                key, subkey = jax.random.split(subkey)
                new_blocking_status = graph_realisation.sample_blocking_status(subkey)
                is_solvable = graph_realisation.is_solvable(new_blocking_status)
                patience_counter += 1
            # error if is_solvable is False
            if is_solvable == jnp.bool_(False):
                raise ValueError(
                    "Could not find enough solvable blocking status. Please decrease the prop_stoch."
                )
        elif self.deal_with_unsolvability == "always_expensive_edge":
            new_blocking_status = graph_realisation.sample_blocking_status(subkey)
        else:
            new_blocking_status = graph_realisation.sample_blocking_status(subkey)
            is_solvable = graph_realisation.is_solvable(new_blocking_status)
            # Add expensive edge if unsolvable
            if is_solvable == jnp.bool_(False):
                upper_bound = (
                    (self.num_nodes - 1)
                    * jnp.max(graph_realisation.graph.weights)
                    * self.factor_expensive_edge
                )
                graph_realisation.graph.weights = graph_realisation.graph.weights.at[
                    graph_realisation.graph.origin, graph_realisation.graph.goal
                ].set(upper_bound)
                graph_realisation.graph.weights = graph_realisation.graph.weights.at[
                    graph_realisation.graph.goal, graph_realisation.graph.origin
                ].set(upper_bound)
                graph_realisation.graph.blocking_prob = (
                    graph_realisation.graph.blocking_prob.at[
                        graph_realisation.graph.origin, graph_realisation.graph.goal
                    ].set(0)
                )
                graph_realisation.graph.blocking_prob = (
                    graph_realisation.graph.blocking_prob.at[
                        graph_realisation.graph.goal, graph_realisation.graph.origin
                    ].set(0)
                )
                new_blocking_status = new_blocking_status.at[
                    graph_realisation.graph.origin, graph_realisation.graph.goal
                ].set(CTP_generator.UNBLOCKED)
                new_blocking_status = new_blocking_status.at[
                    graph_realisation.graph.goal, graph_realisation.graph.origin
                ].set(CTP_generator.UNBLOCKED)

                # renormalize the edge weights by the expensive edge
                max_weight = jnp.max(graph_realisation.graph.weights)
                graph_realisation.graph.weights = jnp.where(
                    graph_realisation.graph.weights != CTP_generator.NOT_CONNECTED,
                    graph_realisation.graph.weights / max_weight,
                    CTP_generator.NOT_CONNECTED,
                )

        # Put into env state
        empty = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        edge_weights = jnp.concatenate((empty, graph_realisation.graph.weights), axis=0)
        edge_probs = jnp.concatenate(
            (empty, graph_realisation.graph.blocking_prob), axis=0
        )
        agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        agents_pos = agents_pos.at[0, graph_realisation.graph.origin[0]].set(1)
        pos_and_blocking_status = jnp.concatenate(
            (agents_pos, new_blocking_status), axis=0
        )

        # Top part is each agent's service history. Bottom part is number of times each goal needs to
        # be serviced
        goal_matrix = jnp.zeros_like(pos_and_blocking_status)
        goal_matrix = goal_matrix.at[
            self.num_agents + graph_realisation.graph.goal[0],
            graph_realisation.graph.goal[0],
        ].set(1)

        return jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs, goal_matrix),
            axis=0,
            dtype=jnp.float16,
        )

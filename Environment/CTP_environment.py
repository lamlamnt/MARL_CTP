from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from chex import dataclass
from Environment import CTP_generator
from typing import TypeAlias

# Belief_state contains the agents' positions + current knowledge about blocked status, edge_weights, and edge_probs in this order
# 3D tensor where each channel is size (num_agents+num_nodes, num_agents+num_nodes)
Belief_State: TypeAlias = jnp.ndarray
# current location of agents and knowledge of blocking status of connected edges
Observation: TypeAlias = jnp.ndarray
EnvState: TypeAlias = jnp.ndarray


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
        reward_for_invalid_action=-200.0,
        reward_for_goal=10,
        factor_expensive_edge=1.0,
        handcrafted_graph=None,
    ):
        """
        List of attributes:
        num_agents: int
        num_nodes: int
        reward_for_invalid_action: int
        reward_for_goal:int
        action_spaces
        graph_realisation: CTPGraph_Realisation
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = reward_for_invalid_action
        self.reward_for_goal = reward_for_goal
        self.num_nodes = num_nodes
        # Instantiate a CTPGraph_Realisation object
        if handcrafted_graph is not None:
            self.graph_realisation = CTP_generator.CTPGraph_Realisation(
                key, self.num_nodes, handcrafted_graph=handcrafted_graph
            )
        else:
            self.graph_realisation = CTP_generator.CTPGraph_Realisation(
                key,
                self.num_nodes,
                grid_size=grid_size,
                prop_stoch=prop_stoch,
                k_edges=k_edges,
                num_goals=num_goals,
                factor_expensive_edge=factor_expensive_edge,
            )
        actions = [num_nodes for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[EnvState, Belief_State]:
        key, subkey = jax.random.split(key)
        new_blocking_status = self.graph_realisation.sample_blocking_status(subkey)

        # update agents' positions to origin
        agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.int32)
        agents_pos = agents_pos.at[0, self.graph_realisation.graph.origin[0]].set(1)
        env_state = self.__convert_graph_realisation_to_matrix(
            self.graph_realisation, new_blocking_status, agents_pos
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
            (pos_and_blocking_status, env_state[1, :, :], env_state[2, :, :]),
            axis=0,
            dtype=jnp.float32,
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
        blocking_prob = current_env_state[2, self.num_agents :, :]
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

        # Function that gets called if at goal -> reset to origin
        def _at_goal(args) -> tuple[jnp.array, int, bool]:
            current_env_state, actions = args
            reward = (
                -(weights[jnp.argmax(current_env_state[0, :1, :]), actions[0]])
                + self.reward_for_goal
            )
            agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.int32)
            agents_pos = agents_pos.at[0, actions[0]].set(1)
            new_env_state = current_env_state.at[0, : self.num_agents, :].set(
                agents_pos
            )
            terminate = jnp.bool_(True)
            return new_env_state, reward, terminate

        # Function that gets called if valid action and not at goal -> move to new node
        def _move_to_new_node(args) -> tuple[jnp.array, int, bool]:
            current_env_state, actions = args
            reward = -(weights[jnp.argmax(current_env_state[0, :1, :]), actions[0]])
            agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.int32)
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
                actions[0] == self.graph_realisation.graph.goal[0],
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

    def __convert_graph_realisation_to_matrix(
        self,
        graph_realisation: CTP_generator.CTPGraph_Realisation,
        blocking_status: jnp.ndarray,
        agents_pos: jnp.ndarray,
    ) -> EnvState:
        # Convert graph realisation to matrix
        empty = jnp.zeros((self.num_agents, self.num_nodes))
        edge_weights = jnp.concatenate((empty, graph_realisation.graph.weights), axis=0)
        edge_probs = jnp.concatenate(
            (empty, graph_realisation.graph.blocking_prob), axis=0
        )
        pos_and_blocking_status = jnp.concatenate((agents_pos, blocking_status), axis=0)
        return jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs),
            axis=0,
            dtype=jnp.float32,
        )

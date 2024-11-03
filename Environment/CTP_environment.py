from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from chex import dataclass
from Environment import CTP_generator
from typing import TypeAlias

# If exceed this number, give up and error
Max_Times_Resample_For_Solvability = 10

# Belief_state contains the current knowledge about blocked status, edge_weights, and edge_probs in this order
# 3D tensor where each channel is size (num_agents+num_nodes, num_agents+num_nodes)
Belief_State: TypeAlias = jnp.ndarray
# current location of agents and knowledge of blocking status of connected edges
Observation: TypeAlias = jnp.ndarray
EnvState_agents_pos: TypeAlias = jnp.array
# Technically, env_state should contain the graph_realisation and agents_pos, but I so far cannot jax jit
# a function when graph realisation is passed in as part of the argument, so graph_realisation will be a
# class attribute for now


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
    ):
        """
        List of attributes:
        num_agents: int
        num_nodes
        reward_for_invalid_action: int
        action_spaces
        graph_realisation
        """
        super().__init__(num_agents=num_agents)
        self.num_agents = num_agents
        self.reward_for_invalid_action = reward_for_invalid_action
        self.num_nodes = num_nodes
        # Instantiate a CTPGraph_Realisation object
        self.graph_realisation = CTP_generator.CTPGraph_Realisation(
            key,
            self.num_nodes,
            grid_size=grid_size,
            prop_stoch=prop_stoch,
            k_edges=k_edges,
            num_goals=num_goals,
        )
        actions = [num_nodes for _ in range(num_agents)]
        self.action_spaces = spaces.MultiDiscrete(actions)

    # Cannot not jax.jit or use jax.while_loop because is_solvable the way unblocked_senders and unblocked_receivers are computed is not jax compatible (not static shape)
    def reset(self, key: chex.PRNGKey) -> tuple[EnvState_agents_pos, Belief_State]:
        key, subkey = jax.random.split(key)
        # Resample until the graph is solvable
        times_resample = 0
        while (
            self.graph_realisation.is_solvable() == False
            and times_resample <= Max_Times_Resample_For_Solvability
        ):
            self.graph_realisation.resample_blocking_prob(subkey)
            key, subkey = jax.random.split(subkey)
            times_resample += 1
        # update agents' positions (array)
        env_state_agents_pos = self.graph_realisation.graph.origin

        # return the initial belief state
        original_observation = self.get_obs(env_state_agents_pos)
        # Incorporate info that non-existent edges are blocked and deterministic edges are not blocked
        blocking_status_knowledge = jnp.where(
            jnp.logical_or(
                self.graph_realisation.graph.blocking_prob == 0,
                self.graph_realisation.graph.blocking_prob == 1,
            ),
            self.graph_realisation.blocking_status,
            original_observation[self.num_agents :, :],
        )
        pos_and_blocking_status = original_observation.at[self.num_agents :, :].set(
            blocking_status_knowledge
        )

        empty = jnp.zeros((self.num_agents, self.graph_realisation.graph.n_nodes))
        edge_weights = jnp.concatenate(
            (empty, self.graph_realisation.graph.weights), axis=0
        )
        edge_probs = jnp.concatenate(
            (empty, self.graph_realisation.graph.blocking_prob), axis=0
        )
        initial_belief_state = jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs),
            axis=0,
            dtype=jnp.float32,
        )
        return env_state_agents_pos, initial_belief_state

    # If want to speed up, then don't need to recompute belief state for invalid actions
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.random.PRNGKey,
        old_env_state_agents_pos: EnvState_agents_pos,
        current_belief_state,
        actions: jnp.ndarray,
    ) -> tuple[EnvState_agents_pos, Belief_State, int, bool]:
        # return the new environment state, next belief state, reward, and whether the episode is done

        # Use environment state and actions to determine if the action is valid
        def _is_invalid_action(actions: jnp.ndarray, agents_pos: jnp.array) -> bool:
            return jnp.logical_or(
                actions[0] == agents_pos[0],
                jnp.logical_or(
                    self.graph_realisation.graph.weights[agents_pos[0], actions[0]]
                    == CTP_generator.NOT_CONNECTED,
                    self.graph_realisation.blocking_status[agents_pos[0], actions[0]]
                    == CTP_generator.BLOCKED,
                ),
            )

        # If invalid action, then return the same state, reward is very negative, and terminate=False
        def _step_invalid_action(args) -> tuple[jnp.array, int, bool]:
            agents_pos, actions = args
            reward = self.reward_for_invalid_action
            terminate = jnp.bool_(False)
            return agents_pos, reward, terminate

        # Function that gets called if at goal -> reset to origin
        def _at_goal(args) -> tuple[jnp.array, int, bool]:
            agents_pos, actions = args
            agents_pos = agents_pos.at[0].set(actions[0])
            reward = 0.0
            terminate = jnp.bool_(True)
            return agents_pos, reward, terminate

        # Function that gets called if valid action and not at goal -> move to new node
        def _move_to_new_node(args) -> tuple[jnp.array, int, bool]:
            agents_pos, actions = args
            reward = -(self.graph_realisation.graph.weights[agents_pos[0], actions[0]])
            agents_pos = agents_pos.at[0].set(actions[0])
            terminate = jnp.bool_(False)
            return agents_pos, reward, terminate

        new_agents_pos, reward, terminate = jax.lax.cond(
            _is_invalid_action(actions, old_env_state_agents_pos),
            _step_invalid_action,
            lambda args: jax.lax.cond(
                actions[0] == self.graph_realisation.graph.goal[0],
                _at_goal,
                _move_to_new_node,
                args,
            ),
            (old_env_state_agents_pos, actions),
        )
        new_env_state_agent_pos = new_agents_pos
        new_observation = self.get_obs(new_env_state_agent_pos)
        next_belief_state = self.get_belief_state(current_belief_state, new_observation)

        key, subkey = jax.random.split(key)

        return new_env_state_agent_pos, next_belief_state, reward, terminate, subkey

    # use current state to get observation
    def get_obs(self, env_state_agents_pos: EnvState_agents_pos) -> jnp.ndarray:
        # Get edges connected to agent's current position
        obs_blocking_status = jnp.full(
            (
                self.graph_realisation.graph.n_nodes,
                self.graph_realisation.graph.n_nodes,
            ),
            CTP_generator.UNKNOWN,
        )
        # replace 1 row and column corresponding to agent's position
        obs_blocking_status = obs_blocking_status.at[env_state_agents_pos[0], :].set(
            self.graph_realisation.blocking_status[env_state_agents_pos[0], :]
        )
        obs_blocking_status = obs_blocking_status.at[:, env_state_agents_pos[0]].set(
            self.graph_realisation.blocking_status[env_state_agents_pos[0], :]
        )

        # Convert agent's position to one-hot encoding
        obs_agent_pos = jnp.zeros(
            (self.num_agents, self.graph_realisation.graph.n_nodes)
        )
        obs_agent_pos = obs_agent_pos.at[0, env_state_agents_pos[0]].add(1)

        # Concatenate
        new_observation = jnp.concatenate((obs_agent_pos, obs_blocking_status), axis=0)
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

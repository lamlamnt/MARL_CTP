from functools import partial
import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from Environment import CTP_generator
from typing import TypeAlias

# If exceed this number, give up and error
Max_Times_Resample_For_Solvability = 10

# Belief_state contains the current knowledge about blocked status, edge_weights, and edge_probs in this order
# 3D tensor where each channel is size (num_agents+num_nodes, num_agents+num_nodes)
Belief_State: TypeAlias = jnp.ndarray
# current location of agents and knowledge of blocking status of connected edges
Observation: TypeAlias = jnp.ndarray


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
        Part of the environment state:
        num_agents: int
        graph_realisation: a CTPGraph_Realisation object
        agents_pos: 1d array of size num_agents, where each element is the position of the agent

        Hyperparameters:
        reward_for_invalid_action: int

        Others:
        action_spaces
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

    # Cannot not jax.jit or use jax.while_loop because is_solvable the way unblocked_senders and unblocked_receivers are computed is not jax compatible (not static shape)
    def reset(self, key: chex.PRNGKey) -> Belief_State:
        key, subkey = jax.random.split(key)
        times_resample = 0
        while (
            self.graph_realisation.is_solvable() == False
            and times_resample <= Max_Times_Resample_For_Solvability
        ):
            self.graph_realisation.resample_blocking_prob(subkey)
            key, subkey = jax.random.split(subkey)
            times_resample += 1
        # update agents' positions (array)
        self.agents_pos = self.graph_realisation.graph.origin

        # return the initial belief state
        original_observation = self.get_obs()
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
            (pos_and_blocking_status, edge_weights, edge_probs), axis=0
        )
        return initial_belief_state

    # If want to speed up, then don't need to recompute belief state for invalid actions
    # @partial(jax.jit, static_argnums=(0,))
    def step(
        self, actions: jnp.ndarray, current_belief_state
    ) -> tuple[Belief_State, int, bool]:
        # return the next belief state, reward, and whether the episode is done
        # Use environment state and actions to determine if the action is valid
        if jnp.logical_or(
            actions[0] == self.agents_pos[0],
            jnp.logical_or(
                self.graph_realisation.graph.weights[self.agents_pos[0], actions[0]]
                == -1,
                self.graph_realisation.blocking_status[self.agents_pos[0], actions[0]]
                == CTP_generator.blocked,
            ),
        ):
            reward = self.reward_for_invalid_action
            terminate = False
        # if at goal
        elif actions[0] == self.graph_realisation.graph.goal[0]:
            self.agents_pos.at[0].set(actions[0])
            reward = 0
            terminate = True
        else:
            reward = -(
                self.graph_realisation.graph.weights[self.agents_pos[0], actions[0]]
            )
            self.agents_pos = self.agents_pos.at[0].set(actions[0])
            terminate = False
        new_observation = self.get_obs()
        next_belief_state = self.get_belief_state(current_belief_state, new_observation)
        return next_belief_state, reward, terminate

    # use current state to get observation
    def get_obs(self) -> jnp.ndarray:
        # Get edges connected to agent's current position
        obs_blocking_status = jnp.full(
            (
                self.graph_realisation.graph.n_nodes,
                self.graph_realisation.graph.n_nodes,
            ),
            CTP_generator.unknown,
        )
        # replace 1 row and column corresponding to agent's position
        obs_blocking_status = obs_blocking_status.at[self.agents_pos[0], :].set(
            self.graph_realisation.blocking_status[self.agents_pos[0], :]
        )
        obs_blocking_status = obs_blocking_status.at[:, self.agents_pos[0]].set(
            self.graph_realisation.blocking_status[self.agents_pos[0], :]
        )

        # Convert agent's position to one-hot encoding
        obs_agent_pos = jnp.zeros(
            (self.num_agents, self.graph_realisation.graph.n_nodes)
        )
        obs_agent_pos = obs_agent_pos.at[0, self.agents_pos[0]].add(1)

        # Concatenate
        new_observation = jnp.concatenate((obs_agent_pos, obs_blocking_status), axis=0)
        return new_observation

    # Need to use previous belief state and current state
    @partial(jax.jit, static_argnums=(0,))
    def get_belief_state(
        self, old_belief_state: Belief_State, new_observation: Observation
    ) -> Belief_State:
        # Combine current_blocking_status with new_observation
        new_blocking_knowledge = jnp.where(
            old_belief_state[0, self.num_agents :, :] == CTP_generator.unknown,
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

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

    def reset(self, key: chex.PRNGKey) -> Belief_State:
        def resample(loop_vars):
            subkey, times_resample = loop_vars
            # Perform resampling
            self.graph_realisation.resample_blocking_prob(subkey)
            key, subkey = jax.random.split(subkey)
            times_resample += 1
            return subkey, times_resample

        key, subkey = jax.random.split(key)
        times_resample = 0
        """
        while (
            self.graph_realisation.is_solvable() == False
            and times_resample <= Max_Times_Resample_For_Solvability
        ):
            self.graph_realisation.resample_blocking_prob(subkey)
            key, subkey = jax.random.split(subkey)
            times_resample += 1
        """
        subkey, times_resample = jax.lax.while_loop(
            self.graph_realisation.solvable == False
            and times_resample <= Max_Times_Resample_For_Solvability,
            resample,
            (subkey, times_resample),
        )
        # update agents' positions (array)
        self.agents_pos = self.graph_realisation.graph.origin

        # return the initial belief state
        original_observation = self.get_obs()
        empty = jnp.zeros(self.num_agents, self.graph_realisation.graph.n_nodes)
        edge_weights = jnp.concatenate(
            (empty, self.graph_realisation.graph.weights), axis=0
        )
        edge_probs = jnp.concatenate(
            (empty, self.graph_realisation.graph.blocking_prob), axis=0
        )
        initial_belief_state = jnp.stack(
            (original_observation, edge_weights, edge_probs), axis=0
        )
        return initial_belief_state

    # If want to speed up, then don't need to recompute belief state for invalid actions
    def step(
        self, actions: jnp.ndarray, current_belief_state
    ) -> tuple[Belief_State, int, bool]:
        # return the next belief state, reward, and whether the episode is done
        # Use environment state and actions to determine if the action is valid
        if (
            actions[0] == self.agents_pos[0]
            or self.graph_realisation.graph.weights[self.agents_pos[0], actions[0]]
            == -1
            or self.graph_realisation.blocking_status[self.agents_pos[0], actions[0]]
            == CTP_generator.blocked
        ):
            reward = self.reward_for_invalid_action
            terminate = False
        # if at goal
        elif actions[0] == self.graph_realisation.graph.goal[0]:
            self.agents_pos[0] = actions[0]
            reward = 0
            terminate = True
        else:
            self.agents_pos[0] = actions[0]
            reward = -self.graph_realisation.graph.weights[
                self.agents_pos[0], actions[0]
            ]
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
            self.graph_realisation.blocking_status[:, self.agents_pos[0]]
        )

        # Convert agent's position to one-hot encoding
        obs_agent_pos = jnp.zeros(self.graph_realisation.graph.n_nodes)
        obs_agent_pos = obs_agent_pos.at[self.agents_pos[0]].add(1)

        # Concatenate
        new_observation = jnp.concatenate((obs_agent_pos, obs_blocking_status), axis=0)
        return new_observation

    # Need to use previous belief state and current state
    def get_belief_state(
        self, old_belief_state: Belief_State, new_observation: Observation
    ) -> Belief_State:
        # Combine current_blocking_status with new_observation
        new_blocking_knowledge = jnp.where(
            old_belief_state[0, :, :] == CTP_generator.unknown,
            new_observation,
            old_belief_state[0, :, :],
        )

        # Replace the right channel with new_blocking_knowledge
        new_belief_state = old_belief_state.at[0, :, :].set(new_blocking_knowledge)
        return new_belief_state

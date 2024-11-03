# First part of Environment's step function in non-jax-like code
if _is_invalid_action(actions):
    reward = self.reward_for_invalid_action
    terminate = False
# if at goal
elif actions[0] == self.graph_realisation.graph.goal[0]:
    self.agents_pos = self.agents_pos.at[0].set(actions[0])
    reward = 0
    terminate = True
else:
    reward = -(self.graph_realisation.graph.weights[self.agents_pos[0], actions[0]])
    self.agents_pos = self.agents_pos.at[0].set(actions[0])
    terminate = False


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
        agents_pos, actions, current_belief_state, key = args
        reward = self.reward_for_invalid_action
        terminate = jnp.bool_(False)
        return agents_pos, reward, terminate, current_belief_state

    # Function that gets called if at goal -> reset to origin
    def _at_goal(args) -> tuple[jnp.array, int, bool]:
        agents_pos, actions, current_belief_state, key = args
        reward = 0.0
        terminate = jnp.bool_(True)
        start_env_state_agent_pos, start_belief_state = self.reset(key)
        return start_env_state_agent_pos, reward, terminate, start_belief_state

    # Function that gets called if valid action and not at goal -> move to new node
    def _move_to_new_node(args) -> tuple[jnp.array, int, bool]:
        agents_pos, actions, current_belief_state, key = args
        reward = -(self.graph_realisation.graph.weights[agents_pos[0], actions[0]])
        agents_pos = agents_pos.at[0].set(actions[0])
        new_env_state_agent_pos = agents_pos
        terminate = jnp.bool_(False)
        new_observation = self.get_obs(new_env_state_agent_pos)
        next_belief_state = self.get_belief_state(current_belief_state, new_observation)
        return new_env_state_agent_pos, reward, terminate, next_belief_state

    new_env_state_agent_pos, reward, terminate, next_belief_state = jax.lax.cond(
        _is_invalid_action(actions, old_env_state_agents_pos),
        _step_invalid_action,
        lambda args: jax.lax.cond(
            actions[0] == self.graph_realisation.graph.goal[0],
            _at_goal,
            _move_to_new_node,
            args,
        ),
        (old_env_state_agents_pos, actions, current_belief_state, key),
    )
    # new_observation = self.get_obs(new_env_state_agent_pos)
    # next_belief_state = self.get_belief_state(current_belief_state, new_observation)

    key, subkey = jax.random.split(key)

    return new_env_state_agent_pos, next_belief_state, reward, terminate, subkey

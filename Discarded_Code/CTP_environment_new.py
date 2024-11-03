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

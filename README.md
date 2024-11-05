**Things currently in progress:**
<br>-Modify the training loop (taken from jym with minor modifications) to be able to run as many steps as desired without ever running out of RAM (do the rollout, store needed stuff, and do the rollout again and so on). Do this if I run out of RAM (for a reasonably-sized buffer size)
<br>-Used JYM DQN with Uniform Sampling Replay Buffer. -> Will change to JYM DQN with Prioritized Experience Replay Buffer.
<br>-Right now, I haven't included the goal as part of the state. It's treated right now as part of the environment. I will include the goal as part of the state like we discussed (expanded adjacency matrix form) soon.
<br>-I am doing [1,0,-1] corresponding to [Blocked, Not Blocked,Unknown]. When updating the belief, I am checking each element instead of adding them up, which would have been easier with Unknown = 0. -> Will consider changing this in the future for multi-agent. Speed is not an issue rn, so this is not a big issue rn.  
<br>-Inside the step function, when the agent reaches the goal, I calculated the observation, belief state, and env state for the goal, then reset the environment, and then returned the starting env state and belief. If we need to speed things up, remove the calculation of observation and belief state, which are not used. 

**Python/code management in progress:**
<br>-Pycache keeps being uploaded despite .gitignore. 
<br>-requirements.txt is probably not complete. Use setup.py files and make the project into a package and removing sys.path.append(".."). 
<br>-Fix the warnings when running pytest and main_training

**Note:**
<br>-In order to follow JAX's pure functional programming paradigm, I am passing and returning the EnvState. This is not technically necessary in terms of RL and is just an implementation detail. 
<br>-In order to take advantage of JAX speedup, the reset function, which gets called in the step function if terminate is True, needs to JAX jittable. The is_solvable() function is not JAX-compatible right now because of the networkx implementation and the operations used to convert to a networkx representation. To avoid calling is_solvable() in reset, I sampled a certain number of solvable blocking statuses and stored them first before running the training loop. The reset function samples from one of these stored solvable blocking statuses. Currently restrict the problem definition to only solvable graphs. 
<br>-Ignore the "Discarded_Code" folder. 
<br>-Files edited in JYM (minor changes): deep_rl_rollout.py and dqn.py 
<br>-When the agent reaches the goal, the reward received is the negative of the weight between the current node and the goal node + a tunable positive number (default = 10).

**Current progress status: (Nov 4th)**
<br>-I trained a DQN policy on a 5-node graph. The learned policy seems reasonable. I have not done any hyperparameter tuning or trained on other graphs/using different seeds and number of nodes.





Things currently in progress:
<br>-Used JYM DQN with Uniform Sampling Replay Buffer. -> Will change to JYM DQN with Prioritized Experience Replay Buffer.
<br> Right now, I haven't included the goal as part of the state. It's treated right now as part of the environment. I will include the goal as part of the state like we discussed (expanded adjacency matrix form) later (when we need to generalize for single agent)
<br> I am doing [1,0,-1] corresponding to [Blocked, Not Blocked,Unknown]. When updating the belief, I am checking each element instead of adding them up, which would have been easier with Unknown = 0. -> Will consider changing this in the future for multi-agent. Speed is not an issue rn, so this is not a big issue rn.  

Python/code management in progress: 
<br>-Pycache keeps being uploaded despite .gitignore. 
<br>-Check that libraries in requirements.txt are enough and correct. Use setup.py files and make the project into a package and removing sys.path.append("..") 
<br>-Fix the warnings when running pytest and main_training

Note:
<br>-In order to follow JAX's pure functional programming paradigm, I am passing and returning the EnvState. This is not technically necessary in terms of RL and is just an implementation detail. 
<br>-In order to take advantage of JAX speedup, the reset function, which gets called in the step function if terminate is True, needs to JAX jittable. The is_solvable() function is not JAX-compatible right now because of the networkx implementation and the operations used to convert to a networkx representation. To avoid calling is_solvable() in reset, I sampled a certain number of solvable blocking statuses and stored them first before running the training loop. The reset function samples from one of these stored solvable blocking statuses. Currently restrict the problem definition to only solvable graphs. 
<br>-Ignore the "Discarded_Code" folder. 

Current progress status:
<br>-





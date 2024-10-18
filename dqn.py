import agent
import jax 
import jax.numpy as jnp
import jaxmarl
import CTP_environment

class DQN_Agent(agent.Agent):
    def __init__(self, action_space):
        super().__init__(action_space)

    def reset(self):
        # Reset network and hyperparameters
        pass

    def act(self, state:CTP_environment.EnvState,observation:CTP_environment.Observation) -> int:
        # Return the action to take
        pass

    def update(self,state:CTP_environment.EnvState,observation:CTP_environment.Observation,action:int,reward:float,next_state:CTP_environment.EnvState,next_observation:CTP_environment.Observation,terminate:bool):
        # Update the network
        pass


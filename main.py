import jax
import jax.numpy as jnp
import CTP_environment 
import timeit
import time
import argparse

#This is currently used to test the functions in the CTP_generator.py file
if __name__ == "__main__":
    key = jax.random.PRNGKey(40)
    environment = CTP_environment.CTP(1,1,5,key)

    #Reset the environment
    key,subkey1=jax.random.split(key)
    initial_observation, initial_state = environment.reset(subkey1)
    current_observation, current_state, current_reward, terminate = environment.step(initial_state,jnp.array([2]))
    print(current_reward,terminate)
    print(current_state.agents_pos)
    print(current_observation)
    # log the action taken and the accumulated reward

    #While not at the goal
        # Each agent chooses an action based on a policy
        # Combine together to form a joint action
        # Perform the step transition.
        # Assign observation and position to each agent

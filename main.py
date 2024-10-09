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

    #While not at the goal
        #The agent choose an action based on a policy
        # Perform the step transition.

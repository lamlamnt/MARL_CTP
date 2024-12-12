import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


# input env_state and return path length
# Optimistic Baselines
def optimistic(env_state: jnp.ndarray) -> int:
    # Env state is for starting from the origin
    pass


# AO* search
def ao_star(env_state: jnp.ndarray) -> int:
    pass

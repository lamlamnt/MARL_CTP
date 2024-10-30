import jax
import jax.numpy as jnp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
import chex
from Environment import CTP_generator
from typing import Tuple

@chex.dataclass
class EnvState:
    
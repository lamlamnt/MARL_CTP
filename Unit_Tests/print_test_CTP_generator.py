import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator


if __name__ == "__main__":
    key = jax.random.PRNGKey(100)
    graphRealisation = CTP_generator.CTPGraph_Realisation(key, 5, prop_stoch=0.4)
    print(graphRealisation.blocking_status)

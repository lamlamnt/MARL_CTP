import jax
import jax.numpy as jnp
import CTP_generator as CTP
import timeit
import time

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    tree = CTP.generate_graph(5,key)
    jax.block_until_ready(tree)

    #Put the timed code, including block_until_ready into a function and then timeit that function
    #timeit.timeit(jax.block_until_ready(CTP.generate_graph(5,key)),number=100)
    
    #Compare timing with Alex's numpy code for large graphs

    #use timeit with jax.block_until_ready to time the function generate_graph

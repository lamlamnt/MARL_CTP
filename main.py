import jax
import jax.numpy as jnp
import CTP_generator as CTP
import timeit
import time

if __name__ == "__main__":
    key = jax.random.PRNGKey(40)
    #jax.jit this function
    #generate_graph_jit = jax.jit(CTP.generate_graph,static_argnums=(0,))
    #tree = generate_graph_jit(5,key)
    graph,origin,goal = CTP.generate_graph(5,key)
    key,subkey=jax.random.split(key)
    graph_blocked = CTP.make_edges_blocked(graph,subkey)
    jax.block_until_ready(graph_blocked)
    graph_NX = CTP.convert_jraph_to_networkx(graph_blocked)
    
    #Convert origin and goal to int first in order to plot the graph
    goal_int = goal.item()
    origin_int = origin.item()
    #CTP.plot_nx_graph(graph_NX,goal_int,origin_int)

    #Put the timed code, including block_until_ready into a function and then timeit that function
    #timeit.timeit(jax.block_until_ready(CTP.generate_graph(5,key)),number=100)
    
    #Compare timing with Alex's numpy code for large graphs

    #use timeit with jax.block_until_ready to time the function generate_graph

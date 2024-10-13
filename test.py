import jax
import jax.numpy as jnp
import CTP_generator 
import timeit
import time

#This is currently used to test the functions in the CTP_generator.py file
if __name__ == "__main__":
    key = jax.random.PRNGKey(40)
    #jax.jit these functions
    graph,origin,goal = CTP_generator.generate_graph(5,key)
    key,subkey1=jax.random.split(key)
    new_blocking_prob = CTP_generator.get_blocking_prob(graph.n_edge,subkey1)
    new_graph = graph._replace(edges={**graph.edges,'blocked_prob':new_blocking_prob})
    
    #Convert origin and goal to int first in order to plot the graph
    goal_int = goal.item()
    origin_int = origin.item()

    subkey1,subkey2=jax.random.split(subkey1)
    #With blocking status
    complete_graph = CTP_generator.sample_blocking_prob(subkey2,graph)
    solvable = CTP_generator.is_solvable(complete_graph,origin_int,goal_int)

    #Get agent graph
    agent_graph = CTP_generator.get_agent_graph(complete_graph)   

    weight_matrix,blocking_prob_matrix = CTP_generator.convert_jraph_to_adj_matrix(complete_graph)

    new_graph = CTP_generator.add_expensive_edge(complete_graph,50,origin,goal)
    new_graph_NX = CTP_generator.convert_jraph_to_networkx(new_graph)
    CTP_generator.plot_nx_graph(new_graph_NX,goal_int,origin_int)

    #Put the timed code, including block_until_ready into a function and then timeit that function
    #timeit.timeit(jax.block_until_ready(CTP.generate_graph(5,key)),number=100)
    
    #Compare timing with Alex's numpy code for large graphs
    #use timeit with jax.block_until_ready to time the function generate_graph

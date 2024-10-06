import jax 
import jax.numpy as jnp
import jraph 
import numpy as np
from scipy.spatial import Delaunay

#Generate a random graph 
#Maybe should not jit this function because the shape changes with n_nodes
#Consider jitting or partially jitting
def generate_graph(n_nodes: int, key: int, use_edge_weights=False, prop_stoch=0.4, plot=False, grid_size=10,
                   ) -> tuple[jraph.GraphsTuple, int, int, bool]:

    xmax = grid_size
    ymax = grid_size
    #Generate random points in the grid
    node_pos = jax.random.choice(key, xmax * ymax, (n_nodes,), replace=False)
    #Not in any order, and also is it guaranteed to give unique nodes?
    grid_nodes = jax.vmap(convert_to_grid,in_axes=(0,None))(node_pos,ymax) 
    grid_nodes_jax = jnp.array(grid_nodes).T

    #Choose orign and goal nodes
    origin,goal = find_goal_and_origin(grid_nodes_jax)

    # Apply Delauney triangulation to get edges
    #delaunay = Delaunay(np.asarray(grid_nodes_jax))
    delaunay = Delaunay(grid_nodes_jax)

    #The simplices 
    simplices = delaunay.simplices
    print(simplices)
    print(type(simplices))
    simplices_jax = jnp.array(simplices)
    
    #Use jraph instead of networkx

    return [1,2,3]

def convert_to_grid(i,ymax):
    return (i // (ymax + 1), i % (ymax + 1))

def distance(a, b):
    return jnp.sqrt(jnp.sum((a - b) ** 2))

#Does not guarantee that the goal and origin are the furthest apart nodes?
def find_goal_and_origin(grid_nodes):
    #Vectorize distance calculation from the origin node to all others
    distances_from_origin = jax.vmap(lambda x: distance(grid_nodes[0], x))(grid_nodes)

    # Get the goal node using argmax on the distances
    goal = int(jnp.argmax(distances_from_origin))

    # Vectorize distance calculation from the goal node to all others
    distances_from_goal = jax.vmap(lambda x: distance(grid_nodes[goal], x))(grid_nodes)
    print(distances_from_goal)

    # Get the origin node using argmax on the distances
    origin = int(jnp.argmax(distances_from_goal))
    return goal,origin
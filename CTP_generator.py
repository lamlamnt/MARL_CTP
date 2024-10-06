import jax 
import jax.numpy as jnp
import jraph 
import numpy as np
from scipy.spatial import Delaunay

#Generate a random graph 
#Maybe should not jit this function because the shape changes with n_nodes
#Consider jitting or partially jitting
def generate_graph(n_nodes: int, key: int, use_edge_weights=True, plot=False, grid_size=10,
                   ) -> tuple[jraph.GraphsTuple, int, int]:

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
    delaunay = Delaunay(grid_nodes_jax)

    #The simplices 
    simplices = delaunay.simplices
    simplices_jax = jnp.array(simplices)
    
    # Use jraph instead of networkx
    # Extract edges from the simplices, treat them as undirected edges
    senders = []
    receivers = []
    for simplex in simplices:
        # Add edges for each triangle (3 edges per triangle)
        senders.extend([simplex[0], simplex[1], simplex[2]])
        receivers.extend([simplex[1], simplex[2], simplex[0]])

    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    #The edges are the actual distances between the nodes
    #Make the edge feature include both the weight and blocking probability
    if(use_edge_weights):
        edges = jnp.linalg.norm(grid_nodes_jax[senders] - grid_nodes_jax[receivers], axis=1)
    else:
        edges = jnp.ones_like(senders)

    #Global context = 0 means all edges are not blocked
    global_context = jnp.array([[0]])
    graph = jraph.GraphsTuple(nodes=grid_nodes_jax, senders=senders, receivers=receivers,edges=edges, n_node=n_nodes, n_edge=len(senders), globals=global_context)
    print(graph.nodes)
    print(graph.edges)

    #All the edges in the graph are currently not blocked
    return [1,2,3]

def make_edges_blocked():
    pass

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

    # Get the origin node using argmax on the distances
    origin = int(jnp.argmax(distances_from_goal))
    return goal,origin
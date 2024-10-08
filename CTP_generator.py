import jax 
import jax.numpy as jnp
import jraph 
import numpy as np
from scipy.spatial import Delaunay
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt

#Generate a random graph 
#With the same number of nodes, the total number of edges can be different.
#Cannot jit this function because of the conversion to numpy array for Delauney
def generate_graph(n_nodes: int, key: int, use_edge_weights=True, grid_size=10,
                   ) -> tuple[jraph.GraphsTuple, int, int]:

    xmax = grid_size
    ymax = grid_size
    #Generate random points in the grid
    node_pos = jax.random.choice(key, xmax * ymax, (n_nodes,), replace=False)
    #Not in any order, and also is it guaranteed to give unique nodes?
    grid_nodes = jax.vmap(_convert_to_grid,in_axes=(0,None))(node_pos,ymax) 
    grid_nodes_jax = jnp.array(grid_nodes).T

    #Choose orign and goal nodes
    origin,goal = find_goal_and_origin(grid_nodes_jax)

    # Apply Delauney triangulation to get edges
    delaunay = Delaunay(grid_nodes_jax)
    simplices = delaunay.simplices
    simplices_jax = jnp.array(simplices)
    
    # Use jraph instead of networkx
    # Extract edges from the simplices, treat them as undirected edges
    # Remove duplicate edges
    all_edges = jnp.concatenate(jax.vmap(_extract_edges)(simplices_jax))
    unique_edges = jnp.unique(all_edges, axis=0)
    senders = unique_edges[:, 0]
    receivers = unique_edges[:, 1]

    #Convert to jax.lax.cond
    #Make the edge feature include both the weight, which is the distance between the nodes and blocking probability
    if(use_edge_weights):
        edges = jnp.linalg.norm(grid_nodes_jax[senders] - grid_nodes_jax[receivers], axis=1)
    else:
        edges = jnp.ones_like(senders)
    
    blocking_probability = jnp.zeros_like(edges)
    #The first n_nodes elements of edge features are the edge weights and the last n_nodes elements are the blocking probabilities
    edge_features = jnp.concatenate([edges, blocking_probability], axis=0)

    #Global context = 0 means all edges are not blocked. 1 for all edges
    global_context = jnp.array([[0]])
    graph = jraph.GraphsTuple(nodes=grid_nodes_jax, senders=senders, receivers=receivers,edges=edge_features, n_node=n_nodes, n_edge=len(senders), globals=global_context)

    #All the edges in the graph are currently not blocked
    return graph, origin, goal

def _extract_edges(simplex):
    # Create edges as pairs of nodes, ensuring smaller node first
        edges = jnp.array([[jnp.minimum(simplex[0], simplex[1]), jnp.maximum(simplex[0], simplex[1])],
                            [jnp.minimum(simplex[1], simplex[2]), jnp.maximum(simplex[1], simplex[2])],
                            [jnp.minimum(simplex[0], simplex[2]), jnp.maximum(simplex[0], simplex[2])]])
        return edges

def convert_jraph_to_networkx(graph:jraph.GraphsTuple) -> nx.Graph:
    graph_NX = nx.Graph()
    node_index = jnp.arange(graph.n_node)
    jax.vmap(graph_NX.add_node,in_axes=(0))(node_index,pos=graph.nodes)
    jax.vmap(graph_NX.add_edge,in_axes=(0,0))(graph.senders,graph.receivers)
    nx.set_edge_attributes(graph_NX,values=graph.edges[:graph.n_edge], name='weight')
    nx.set_edge_attributes(graph_NX,values=graph.edges[graph.n_edge:], name='blocked_prob')
    return graph_NX

#This is a separate function from generate_graph because at some stage, we want to give the agent the same graph but
#with different edge blocking probabilities
def make_edges_blocked(graph:jraph.GraphsTuple,key,prop_stoch=0.4) -> jraph.GraphsTuple:
    #Global context = 1 means the second edge feature stored in the graph is the blocking probability
    global_context = jnp.array([[1]])

    #Assign blocking probability to each edge
    num_stoch_edges = jnp.round(prop_stoch * graph.n_edge).astype(int)
    stoch_edge_idx = jax.random.choice(key, graph.n_edge, shape=(num_stoch_edges,), replace=False)
    edge_indices = jnp.arange(graph.n_edge)
    keys = jax.random.split(key, num=graph.n_edge)
    is_stochastic_edges = jnp.isin(edge_indices, stoch_edge_idx)
    edge_probs = jax.vmap(_assign_prob_edge, in_axes=(0,0))(keys,is_stochastic_edges)

    #Update the edge features in graph
    graph = graph._replace(edges=jnp.concatenate([graph.edges[:graph.n_edge],edge_probs],axis=0), globals=global_context)
    return graph

#Add an edge feature to the graph that stores whether an edge is blocked or not
def sample_blocking_prob(key:jax.jax.random.PRNGKey,graph:jraph.GraphsTuple) -> jraph.GraphsTuple:
    #Global context = 2 means that the second edge feature stored in the graph is whether an edge is blocked or not
    global_context = jnp.array([[2]])
    return graph

def _assign_prob_edge(subkey, is_stochastic_edge):
    prob = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
    prob = jnp.round(prob, 2) #Round to 2 decimal places
    return jax.lax.cond(is_stochastic_edge, lambda _:prob, lambda _: 1.0,prob)

#After the blocking status is assigned, check whether it's possible to reach the goal from the origin
def solvability_check(graph:jraph.GraphsTuple,origin:int,goal:int) -> bool:
    pass

#Copy the same from Alex's code
#Currently does not work because of tracing issues
def plot_nx_graph(G: nx.Graph, origin, goal):
    # Plot graph
    node_colour = []
    for node in G.nodes:
        c = "white"
        """
        if node == goal:
            c = "#2ca02c"
        elif node == origin:
            c = "#ff7f0e"
        """
        node_colour.append(c)
    edge_labels = []
    probs = nx.get_edge_attributes(G, "blocked_prob")
    weights = nx.get_edge_attributes(G, "weight")
    edge_labels = {
        e: (f"{w}\np: {probs[e]}" if e in probs else f"{w}") for e, w in weights.items()
    }
    edge_style = ["dashed" if edge in probs.keys() else "solid" for edge in G.edges]
    pos = nx.get_node_attributes(G, "pos")
    nx.draw(
        G,
        with_labels=True,
        node_size=500,
        node_color=node_colour,
        edgecolors="black",
        pos=pos,
        style=edge_style,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos={p: (v[0], v[1]) for p, v in pos.items()},
        edge_labels=edge_labels,
        bbox={"boxstyle": "square", "pad": 0, "color": "white"},
        rotate=False,
        font_size=8,
        verticalalignment="baseline",
        clip_on=False,
    )
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.show()

def _convert_to_grid(i,ymax):
    return (i // (ymax + 1), i % (ymax + 1))

def _distance(a, b):
    return jnp.sqrt(jnp.sum((a - b) ** 2))

#Does not guarantee that the goal and origin are the furthest apart nodes?
def find_goal_and_origin(grid_nodes):
    #Vectorize distance calculation from the origin node to all others
    distances_from_origin = jax.vmap(lambda x: _distance(grid_nodes[0], x))(grid_nodes)

    # Get the goal node using argmax on the distances
    goal = jnp.argmax(distances_from_origin)

    # Vectorize distance calculation from the goal node to all others
    distances_from_goal = jax.vmap(lambda x: _distance(grid_nodes[goal], x))(grid_nodes)

    # Get the origin node using argmax on the distances
    origin = jnp.argmax(distances_from_goal)
    return goal,origin
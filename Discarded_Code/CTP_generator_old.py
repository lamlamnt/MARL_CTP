import jax
import jax.numpy as jnp
import jraph
import numpy as np
from scipy.spatial import Delaunay
from functools import partial
import networkx as nx
import matplotlib.pyplot as plt
import os


# Generate a random graph
# With the same number of nodes, the total number of edges can be different.
# Cannot currently jit this function because of the conversion to numpy array for Delauney
def generate_graph(
    n_nodes: int,
    key: int,
    use_edge_weights=True,
    grid_size=0,
    prop_stoch=None,
    k_edges=None,
) -> tuple[jraph.GraphsTuple, int, int]:
    # Check arguments to function
    # grid_size=0 means no grid_size was specified, in which case the grid size will be equal to n_node
    if grid_size == 0:
        grid_size = n_nodes
    if prop_stoch is None and k_edges is None:
        raise ValueError("Either prop_stoch or k_edges must be specified")

    xmax = grid_size
    ymax = grid_size
    # Generate random points in the grid
    node_pos = jax.random.choice(key, xmax * ymax, (n_nodes,), replace=False)
    grid_nodes = jax.vmap(_convert_to_grid, in_axes=(0, None))(node_pos, ymax)
    grid_nodes_jax = jnp.array(grid_nodes).T

    # Apply Delauney triangulation to get edges
    delaunay = Delaunay(grid_nodes_jax)
    simplices = delaunay.simplices
    simplices_jax = jnp.array(simplices)

    # Extract edges from the simplices, treat them as undirected edges
    # Remove duplicate edges. Sender node number always > receiver node number now.
    all_edges = jnp.concatenate(jax.vmap(_extract_edges)(simplices_jax))
    unique_edges = jnp.unique(all_edges, axis=0)
    senders = unique_edges[:, 0]
    receivers = unique_edges[:, 1]

    # The weight is the Euclidean distance between the nodes, unless use_edge_weights is False
    edge_weights = jax.lax.cond(
        use_edge_weights,
        lambda _: jnp.linalg.norm(
            grid_nodes_jax[senders] - grid_nodes_jax[receivers], axis=1
        ),
        lambda _: jnp.ones_like(senders, dtype=jnp.float32),
        operand=None,
    )

    blocking_probability = get_blocking_prob(len(senders), key, prop_stoch=prop_stoch)

    # Global context = 0 means no blocking status
    global_context = jnp.array([[0]])
    graph = jraph.GraphsTuple(
        nodes=grid_nodes_jax,
        senders=senders,
        receivers=receivers,
        edges={"weight": edge_weights, "blocked_prob": blocking_probability},
        n_node=n_nodes,
        n_edge=len(senders),
        globals=global_context,
    )

    return graph


@jax.jit
def _extract_edges(simplex):
    # Create edges from simplices, ensuring smaller node first
    edges = jnp.array(
        [
            [jnp.minimum(simplex[0], simplex[1]), jnp.maximum(simplex[0], simplex[1])],
            [jnp.minimum(simplex[1], simplex[2]), jnp.maximum(simplex[1], simplex[2])],
            [jnp.minimum(simplex[0], simplex[2]), jnp.maximum(simplex[0], simplex[2])],
        ]
    )
    return edges


# Don't use JAX (vmap or otherwise) here because it will trigger tracing
def convert_jraph_to_networkx(graph: jraph.GraphsTuple) -> nx.Graph:
    graph_NX = nx.Graph()
    for i in range(graph.n_node):
        graph_NX.add_node(i, pos=tuple(graph.nodes[i].tolist()))
    for i in range(graph.n_edge):
        graph_NX.add_edge(graph.senders[i].item(), graph.receivers[i].item())

    # Should be of the format {(sender,receiver):weight}
    weight_edge_dict = {
        (s, r): w
        for s, r, w in zip(
            graph.senders.tolist(),
            graph.receivers.tolist(),
            graph.edges["weight"].tolist(),
        )
    }
    # Only add to the blocking_prob attribute of the networkx graph if the blocking probability is greater than 0 (stochastic edge)
    blocking_prob_dict = {
        (s, r): w
        for s, r, w in zip(
            graph.senders.tolist(),
            graph.receivers.tolist(),
            graph.edges["blocked_prob"].tolist(),
        )
        if w > 0
    }
    nx.set_edge_attributes(graph_NX, values=weight_edge_dict, name="weight")
    nx.set_edge_attributes(graph_NX, values=blocking_prob_dict, name="blocked_prob")
    return graph_NX


def get_blocking_prob(n_edge: int, key, prop_stoch=0.4) -> jnp.array:
    # Assign blocking probability to each edge
    num_stoch_edges = jnp.round(prop_stoch * n_edge).astype(int)
    stoch_edge_idx = jax.random.choice(
        key, n_edge, shape=(num_stoch_edges,), replace=False
    )
    edge_indices = jnp.arange(n_edge)
    keys = jax.random.split(key, num=n_edge)
    is_stochastic_edges = jnp.isin(edge_indices, stoch_edge_idx)
    edge_probs = jax.vmap(_assign_prob_edge, in_axes=(0, 0))(keys, is_stochastic_edges)
    return edge_probs


# Add an edge feature to the graph that stores whether an edge is blocked or not
def sample_blocking_prob(
    key: jax.random.PRNGKey, graph: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    # Global context = 1 means the Graph contains the blocking status of all edges
    global_context = jnp.array([[1]])
    # This is bad but leave it for now
    if isinstance(graph.n_edge, jnp.ndarray):
        num_edges = graph.n_edge.astype(int).item()
    else:
        num_edges = graph.n_edge
    keys = jax.random.split(key, num=num_edges)
    # 0 means not blocked, 1 means blocked
    blocking_status = jax.vmap(jax.random.bernoulli, in_axes=(0,))(
        keys, p=graph.edges["blocked_prob"]
    )
    graph = graph._replace(
        globals=global_context, edges={**graph.edges, "blocked_status": blocking_status}
    )
    return graph


def _assign_prob_edge(subkey, is_stochastic_edge):
    prob = jax.random.uniform(subkey, minval=0.0, maxval=1.0)
    prob = jnp.round(prob, 2)  # Round to 2 decimal places
    return jax.lax.cond(is_stochastic_edge, lambda _: prob, lambda _: 0.0, prob)


# Copy from Alex's code
def plot_nx_graph(G: nx.Graph, origin, goal, directory, file_name="graph.png"):
    node_colour = []
    for node in G.nodes:
        c = "white"
        if node == goal:
            c = "#2ca02c"  # orange
        elif node == origin:
            c = "#ff7f0e"  # green
        node_colour.append(c)
    edge_labels = []
    probs = nx.get_edge_attributes(G, "blocked_prob")
    weights = nx.get_edge_attributes(G, "weight")
    edge_labels = {
        e: (
            f"{np.round(w,2)}\np: {np.round(probs[e],2)}"
            if e in probs
            else f"{np.round(w,2)}"
        )
        for e, w in weights.items()
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
    plt.savefig(os.path.join(directory, file_name))
    plt.close()


def _convert_to_grid(i, ymax):
    return (i // (ymax + 1), i % (ymax + 1))


def _distance(a, b):
    return jnp.sqrt(jnp.sum((a - b) ** 2))


# One way to find 1 origin and 1 goal node
@jax.jit
def find_single_goal_and_origin(graph: jraph.GraphsTuple) -> tuple[int, int]:
    # Extract grid_nodes from the jraph
    grid_nodes = graph.nodes
    distances_from_origin = jax.vmap(lambda x: _distance(grid_nodes[0], x))(grid_nodes)
    goal = jnp.argmax(distances_from_origin)
    distances_from_goal = jax.vmap(lambda x: _distance(grid_nodes[goal], x))(grid_nodes)
    origin = jnp.argmax(distances_from_goal)
    return goal, origin


def convert_jraph_to_adj_matrix(
    graph: jraph.GraphsTuple,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    weight_matrix = jnp.full((graph.n_node, graph.n_node), jnp.inf)
    # 0 means definitely not blocked, 1 means definitely blocked, inf means no edge
    blocking_prob_matrix = jnp.full((graph.n_node, graph.n_node), jnp.inf)
    # Set the diagonals to zero (a node to itself)
    weight_matrix = weight_matrix.at[jnp.diag_indices(graph.n_node)].set(0)
    blocking_prob_matrix = blocking_prob_matrix.at[jnp.diag_indices(graph.n_node)].set(
        0
    )
    for i in range(graph.n_edge):
        sender = graph.senders[i]
        receiver = graph.receivers[i]
        weight_matrix = weight_matrix.at[sender, receiver].set(graph.edges["weight"][i])
        weight_matrix = weight_matrix.at[receiver, sender].set(graph.edges["weight"][i])
        blocking_prob_matrix = blocking_prob_matrix.at[sender, receiver].set(
            graph.edges["blocked_prob"][i]
        )
        blocking_prob_matrix = blocking_prob_matrix.at[receiver, sender].set(
            graph.edges["blocked_prob"][i]
        )
    return weight_matrix, blocking_prob_matrix


# Get starting agent graph given the true graph with true blocking status
@jax.jit
def get_agent_graph(true_graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # 0 means not blocked. 1 means blocked. 2 means unknown.
    stochastic_edges = jnp.where(true_graph.edges["blocked_prob"] == 0, 0, 2)
    # Global context = 3 means the blocking status of some edges are unknown
    global_context = jnp.array([[3]])
    agent_graph = true_graph._replace(
        edges={**true_graph.edges, "blocked_status": stochastic_edges},
        globals=global_context,
    )
    return agent_graph


def is_solvable(graph: jraph.GraphsTuple, origin: int, goal: int) -> bool:
    # Return whether an unblocked path exists from origin to goal
    networkx_graph = convert_jraph_to_networkx(graph)
    solvable = nx.has_path(networkx_graph, origin, goal)
    return solvable


# Add expensive edge between origin and goal
# Need to add so that it's sorted
# @jax.jit
def add_expensive_edge(
    graph: jraph.GraphsTuple, weight: float, origin: int, goal: int
) -> jraph.GraphsTuple:
    (origin, goal) = jax.lax.cond(
        (origin < goal).item(), lambda _: (origin, goal), lambda _: (goal, origin), None
    )

    senders = jnp.append(graph.senders, origin)
    receivers = jnp.append(graph.receivers, goal)
    weights = jnp.append(graph.edges["weight"], weight)
    blocking_prob = jnp.append(graph.edges["blocked_prob"], 0)

    # Make sure that sender, receiver still sorted after expensive edge is added
    sort_indices = jnp.lexsort([receivers, senders])
    senders = senders[sort_indices]
    receivers = receivers[sort_indices]
    weights = weights[sort_indices]
    blocking_prob = blocking_prob[sort_indices]

    new_graph = graph._replace(
        n_edge=graph.n_edge + 1,
        senders=senders,
        receivers=receivers,
        edges={"weight": weights, "blocked_prob": blocking_prob},
    )
    return new_graph


# Get the maximum number of edges in a graph
def get_max_edges(graph: jraph.GraphsTuple) -> int:
    count = jnp.zeros(graph.n_node, dtype=jnp.int32)
    count = count.at[graph.senders].add(1)
    count = count.at[graph.receivers].add(1)
    max_count = jnp.max(count)
    return max_count

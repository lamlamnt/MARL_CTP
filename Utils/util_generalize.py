import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Evaluation import optimal_path_length

NUM_SAMPLES_FACTOR = 10


# Sample blocking status for several times, perform dijkstra and get the average path length
# Need to be solvable realisation
def get_expected_optimal_path_length(
    graphRealisation: CTP_generator.CTPGraph_Realisation,
    key: jax.random.PRNGKey,
    factor_expensive_edge=1,
) -> int:
    num_agents = 1
    num_samples = NUM_SAMPLES_FACTOR * graphRealisation.graph.n_nodes
    key = jax.random.split(key, num=num_samples)
    empty = jnp.zeros((num_agents, graphRealisation.graph.n_nodes), dtype=jnp.float16)
    path_lengths = jnp.zeros(num_samples, dtype=jnp.float16)

    # Do not need to add expensive edge
    edge_weights_not_expensive = jnp.concatenate(
        (empty, graphRealisation.graph.weights), axis=0
    )
    edge_probs_not_expensive = jnp.concatenate(
        (empty, graphRealisation.graph.blocking_prob), axis=0
    )

    # Add expensive edge
    upper_bound = jnp.max(graphRealisation.graph.weights) * factor_expensive_edge
    graph_weights = graphRealisation.graph.weights.copy()
    blocking_prob = graphRealisation.graph.blocking_prob.copy()
    graph_weights = graph_weights.at[
        graphRealisation.graph.origin,
        graphRealisation.graph.goal,
    ].set(upper_bound)
    graph_weights = graph_weights.at[
        graphRealisation.graph.goal,
        graphRealisation.graph.origin,
    ].set(upper_bound)
    blocking_prob = blocking_prob.at[
        graphRealisation.graph.origin,
        graphRealisation.graph.goal,
    ].set(0)
    blocking_prob = blocking_prob.at[
        graphRealisation.graph.goal,
        graphRealisation.graph.origin,
    ].set(0)
    edge_weights_expensive = jnp.concatenate((empty, graph_weights), axis=0)
    edge_probs_expensive = jnp.concatenate((empty, blocking_prob), axis=0)

    for i in range(num_samples):
        blocking_status = graphRealisation.sample_blocking_status(key[i])
        is_solvable = graphRealisation.is_solvable(blocking_status)

        if is_solvable == jnp.bool_(False):
            blocking_status = blocking_status.at[
                graphRealisation.graph.origin,
                graphRealisation.graph.goal,
            ].set(CTP_generator.UNBLOCKED)
            blocking_status = blocking_status.at[
                graphRealisation.graph.goal,
                graphRealisation.graph.origin,
            ].set(CTP_generator.UNBLOCKED)
            edge_weights = edge_weights_expensive
            edge_probs = edge_probs_expensive
        else:
            edge_weights = edge_weights_not_expensive
            edge_probs = edge_probs_not_expensive

        # Convert graphRealisation to env_state (not exactly the right format for env_state, just include the info that dijkstra needs)
        blocking_status = jnp.concatenate((empty, blocking_status), axis=0)
        env_state_simplified = jnp.stack(
            (blocking_status, edge_weights, edge_probs),
            axis=0,
            dtype=jnp.float16,
        )
        path_length = optimal_path_length.dijkstra_shortest_path(
            env_state_simplified,
            graphRealisation.graph.goal,
            graphRealisation.graph.origin,
        )
        path_lengths = path_lengths.at[i].set(path_length)
    return jnp.mean(path_lengths)


def add_expensive_edge(
    blocking_status, graph_weights, blocking_prob, goal, origin, factor_expensive_edge
):
    upper_bound = jnp.max(graph_weights) * factor_expensive_edge
    graph_weights = graph_weights.at[
        origin,
        goal,
    ].set(upper_bound)
    graph_weights = graph_weights.at[
        goal,
        origin,
    ].set(upper_bound)
    blocking_prob = blocking_prob.at[
        origin,
        goal,
    ].set(0)
    blocking_prob = blocking_prob.at[
        goal,
        origin,
    ].set(0)
    blocking_status = blocking_status.at[
        origin,
        goal,
    ].set(CTP_generator.UNBLOCKED)
    blocking_status = blocking_status.at[
        goal,
        origin,
    ].set(CTP_generator.UNBLOCKED)
    return blocking_status, graph_weights, blocking_prob

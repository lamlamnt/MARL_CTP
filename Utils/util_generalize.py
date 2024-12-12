import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_generator
from Evaluation import optimal_path_length

NUM_SAMPLES_FACTOR = 20


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

    # def scan_body(carry, i):
    for i in range(num_samples):
        blocking_status = graphRealisation.sample_blocking_status(key[i])
        is_solvable = graphRealisation.is_solvable(blocking_status)

        if is_solvable == jnp.bool_(False):
            blocking_status, graph_weights, graph_blocking_prob = add_expensive_edge(
                blocking_status,
                graphRealisation.graph.weights,
                graphRealisation.graph.blocking_prob,
                graphRealisation.graph.goal[0],
                graphRealisation.graph.origin[0],
                factor_expensive_edge=factor_expensive_edge,
            )
        else:
            graph_weights = graphRealisation.graph.weights
            graph_blocking_prob = graphRealisation.graph.blocking_prob
        """
        # If not solvable, add expensive edge
        (
            blocking_status,
            graph_weights,
            graph_blocking_prob,
        ) = jax.lax.cond(
            is_solvable == jnp.bool_(False),
            lambda _: add_expensive_edge(
                blocking_status,
                graphRealisation.graph.weights,
                graphRealisation.graph.blocking_prob,
                graphRealisation.graph.goal[0],
                graphRealisation.graph.origin[0],
                factor_expensive_edge=factor_expensive_edge,
            ),
            lambda _: (
                blocking_status,
                graphRealisation.graph.weights,
                graphRealisation.graph.blocking_prob,
            ),
            operand=None,
        )
        """

        # Convert graphRealisation to env_state (not exactly the right format for env_state, just include the info that dijkstra needs)
        edge_weights = jnp.concatenate((empty, graph_weights), axis=0)
        edge_probs = jnp.concatenate((empty, graph_blocking_prob), axis=0)
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
        # return None, path_length

    # _, path_lengths = jax.lax.scan(scan_body, None, jnp.arange(num_samples))

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

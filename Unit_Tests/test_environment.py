import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Environment import CTP_generator
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--n_agent",
        type=int,
        help="Number of agents in the environment",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=False,
        default=5,
    )
    args = parser.parse_args()

    # Get directory path to Logs folder
    current_directory = os.getcwd()
    parent_dir = os.path.dirname(current_directory)
    log_directory = os.path.join(parent_dir, "Logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    key = jax.random.PRNGKey(30)
    # Each episode uses the same graph (same connectivity and blocking probabilities)
    environment = CTP_environment.CTP(
        args.n_agent, 1, args.n_node, key, prop_stoch=0.8, add_expensive_edge=True
    )
    key, subkey = jax.random.split(key)

    nx_graph = CTP_generator.convert_jraph_to_networkx(environment.agent_graph)
    CTP_generator.plot_nx_graph(
        nx_graph, environment.goal.item(), environment.origin.item(), log_directory
    )
    observation, state = environment.reset(subkey)
    print(environment.true_graph.senders)
    print(environment.true_graph.receivers)
    print(environment.true_graph.edges["blocked_status"])
    observation, state, current_reward, terminate = environment.step(
        state, jnp.array([0])
    )
    print(state.agents_pos)
    print(observation)

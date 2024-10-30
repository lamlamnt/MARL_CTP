import jax
import jax.numpy as jnp
from Environment import CTP_environment_old, CTP_generator
import timeit
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Agents.random_agent import RandomAgent
from Agents.dqn import DQN_Agent
from Agents.networks import QNetwork
import os


# jit this function (many issues - including passing the environment object as an argument to the function)
# instead of passing environment in, treat it as global variable
def run_episode(
    episode_num: int,
    environment: CTP_environment_old.CTP,
    subkey: jax.random.PRNGKey,
    model_params,
) -> float:
    observation, state = environment.reset(subkey)
    terminate = False
    cumulative_reward = 0
    action_sequence = jnp.array([])

    while terminate is False:
        key, subkey = jax.random.split(subkey)
        # Loop over agents to get agents' actions and combine together into a joint action
        # Use observation to get the action
        if args.agent_algorithm == "Random":
            action = agent.act(subkey, state, observation)
        else:
            action = agent.act(subkey, state, observation, model_params)
        observation, state, current_reward, terminate = environment.step(
            state, jnp.array(action)
        )
        cumulative_reward += current_reward
        # Assign observation and position to each agent
        action_sequence = jnp.concatenate([action_sequence, jnp.array(action)])

        # Update the agent

    # To replace this while loop with jax.lax.cond, need to fix the step function in CTP_environment.py first
    """
    initial_loop_vars = (observation,state,terminate,cumulative_reward, action_sequence,subkey)
    def run_step(loop_vars):
        observation, state, terminate, cumulative_reward, action_sequence, subkey = loop_vars
        #...the rest of the code in the while loop
        return observation, state, terminate, cumulative_reward, action_sequence, subkey

    def condition(loop_vars):
        observation, state, cumulative_reward, action_sequence, subkey, terminate = loop_vars
        return jnp.logical_not(terminate)
    jax.lax.while_loop(condition, run_step, initial_loop_vars)
    """
    return cumulative_reward, action_sequence


def write_to_file(
    episode_num: int,
    cumulative_reward: float,
    action_sequence: jnp.array,
    file_name="output.txt",
) -> None:
    # Write the action_sequence and cumulative_reward to file
    file = os.path.join(log_directory, file_name)
    with open(file, "a") as file:
        file.write("Episode " + str(episode_num) + ":\n")
        file.write("Total reward:" + str(cumulative_reward) + "\n")
        file.write(str(action_sequence) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for this unit test"
    )
    parser.add_argument(
        "--n_node",
        type=int,
        help="Number of nodes in the graph",
        required=False,
        default=5,
    )
    parser.add_argument(
        "--n_agent",
        type=int,
        help="Number of agents in the environment",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--n_episode",
        type=int,
        help="Number of episodes to run",
        required=False,
        default=3,
    )
    parser.add_argument(
        "--agent_algorithm", type=str, help="Random, DQN", required=False, default="DQN"
    )
    parser.add_argument(
        "--reward_for_invalid_action",
        type=int,
        help="Reward if the agent attempts to move to a blocked edge, non-existing edge, or says is not solvable when solvable",
        required=False,
        default=-500,
    )

    # Hyperparameters for DQN
    parser.add_argument(
        "--batch_size", type=int, help="Batch size for DQN", required=False, default=1
    )
    args = parser.parse_args()

    # Get directory path to Logs folder
    current_directory = os.getcwd()
    log_directory = os.path.join(current_directory, "Logs")
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    key = jax.random.PRNGKey(30)
    # Each episode uses the same graph (same connectivity and blocking probabilities)
    environment = CTP_environment_old.CTP(
        args.n_agent,
        1,
        args.n_node,
        key,
        prop_stoch=0.4,
        reward_for_invalid_action=args.reward_for_invalid_action,
    )
    subkeys = jax.random.split(key, args.n_episode)

    nx_graph = CTP_generator.convert_jraph_to_networkx(environment.agent_graph)
    CTP_generator.plot_nx_graph(
        nx_graph, environment.goal.item(), environment.origin.item(), log_directory
    )

    # Intialize the agent
    if args.agent_algorithm == "Random":
        agent = RandomAgent(environment.action_spaces)
        model_params = 0
    elif args.agent_algorithm == "DQN":
        num_actions = environment.action_spaces.num_categories[0]
        model = QNetwork([128, 64, 32, 16], num_actions)
        agent = DQN_Agent(num_actions, model)

        # Split key, don't hardcode x (placeholder for now)
        x = jax.random.uniform(key, (args.batch_size, args.n_node, args.n_node, 3))
        model_params = model.init(key, x)
    else:
        raise ValueError("Invalid agent algorithm")

    with open("Logs/output.txt", "w") as file:
        file.write(args.agent_algorithm + " agent \n")

    # Involves converting to networkx to determine solvability -> issues with tracing -> can't make jax.lax.fori_loop work yet
    reward_sequence = []
    for i in range(1, args.n_episode + 1):
        episode_reward, episode_action_sequence = run_episode(
            i, environment, subkeys[i], model_params
        )
        write_to_file(i, episode_reward, episode_action_sequence)
        reward_sequence.append(episode_reward)

    # Plot total reward over episodes
    plt.plot(np.arange(1, args.n_episode + 1), reward_sequence)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total reward over episodes")
    plt.savefig("Logs/total_reward.png")
    plt.close()

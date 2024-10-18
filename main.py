import jax
import jax.numpy as jnp
import CTP_environment 
import CTP_generator
import timeit
import time
import argparse
import numpy as np    
import matplotlib.pyplot as plt
import random_agent
import dqn

def run_episode(episode_num:int,environment:CTP_environment.CTP,subkey:jax.random.PRNGKey) -> float:
    observation, state = environment.reset(subkey)
    terminate = False
    cumulative_reward = 0
    action_sequence = jnp.array([])

    while terminate is False:
        key,subkey=jax.random.split(subkey)
        #Loop over agents to get agents' actions and combine together into a joint action
        # Use observation to get the action
        action = agent.act(subkey,state,observation)
        observation, state, current_reward, terminate = environment.step(state,jnp.array(action))
        cumulative_reward += current_reward
        # Assign observation and position to each agent
        action_sequence = jnp.concatenate([action_sequence, jnp.array(action)])
    
    #To replace this while loop with jax.lax.cond, need to fix the step function in CTP_environment.py first 
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

    # Write the action_sequence and cumulative_reward to file
    with open("Logs/output.txt", 'a') as file:
        file.write("Episode " + str(episode_num) + ":\n")
        file.write("Total reward:" + str(cumulative_reward) + "\n")
        file.write(str(action_sequence) + "\n")
    return cumulative_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse command-line arguments for this unit test")
    parser.add_argument('--n_node', type=int, help='Number of nodes in the graph', required=False, default=5)
    parser.add_argument('--n_agent', type=int, help='Number of agents in the environment', required=False, default=1)
    parser.add_argument('--n_episode', type=int, help='Number of episodes to run', required=False, default=10)
    parser.add_argument('--agent_algorithm',type=str,help='Random, DQN',required=False,default='DQN')
    args = parser.parse_args()

    key = jax.random.PRNGKey(40)
    #Each episode uses the same graph (same connectivity and blocking probabilities)
    environment = CTP_environment.CTP(args.n_agent,1,args.n_node,key)
    subkeys = jax.random.split(key, args.n_episode)
    
    nx_graph = CTP_generator.convert_jraph_to_networkx(environment.agent_graph)
    CTP_generator.plot_nx_graph(nx_graph,environment.goal.item(),environment.origin.item(),file_name="Logs/graph.png")
    
    #Intialize the agent
    if args.agent_algorithm == "Random":
        agent = random_agent.RandomAgent(environment.action_spaces)
    elif args.agent_algorithm == "DQN":
        agent = dqn.DQN_Agent(environment.action_spaces.num_categories[0])
    else:
        raise ValueError("Invalid agent algorithm")

    with open("Logs/output.txt", 'w') as file:
        file.write(args.agent_algorithm + " agent \n")
    
    # Involves converting to networkx to determine solvability -> issues with tracing -> can't make jax.lax.fori_loop work yet
    reward_sequence = []
    for i in range(1,args.n_episode+1):
        episode_reward = run_episode(i,environment,subkeys[i])
        reward_sequence.append(episode_reward)

    #Plot total reward over episodes
    plt.plot(np.arange(1,args.n_episode+1),reward_sequence)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total reward over episodes")
    plt.savefig("Logs/total_reward.png")
    plt.close()

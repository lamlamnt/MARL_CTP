import jax
import jax.numpy as jnp
import sys

sys.path.append("..")
from Environment import CTP_environment
from Evaluation.optimal_path_length import dijkstra_shortest_path
from Baselines.optimistic_agent import Optimistic_Agent
from jax_tqdm import scan_tqdm


# For the purpose of plotting the learning curve
# Deterministic inference
def get_average_testing_stats(
    environment: CTP_environment, agent, model_params, args
) -> float:
    init_key, env_key = jax.vmap(jax.random.PRNGKey)(
        jnp.arange(2) + args.random_seed_for_inference
    )
    new_env_state, new_belief_state = environment.reset(init_key)
    num_testing_timesteps = args.factor_testing_timesteps * args.n_node

    @scan_tqdm(num_testing_timesteps)
    def _one_step_inference(runner_state, unused):
        (
            current_env_state,
            current_belief_state,
            key,
            timestep_in_episode,
            previous_episode_done,
        ) = runner_state
        action_key, env_key = jax.random.split(key, 2)
        # Agent acts
        action, action_key = agent.act(
            action_key, model_params, current_belief_state, 0
        )
        action = jnp.array([action])
        new_env_state, new_belief_state, reward, done, env_key = environment.step(
            env_key, current_env_state, current_belief_state, action
        )
        action = action[0]

        # Stop the episode and reset if exceed horizon length
        env_key, reset_key = jax.random.split(env_key)
        # Reset timestep if finish episode
        timestep_in_episode = jax.lax.cond(
            done, lambda _: 0, lambda _: timestep_in_episode, operand=None
        )
        # Reset if exceed horizon length. Otherwise, increment
        new_env_state, new_belief_state, reward, timestep_in_episode, done = (
            jax.lax.cond(
                timestep_in_episode >= args.horizon_length_factor * args.n_node,
                lambda _: (
                    *environment.reset(reset_key),
                    jnp.float16(args.reward_exceed_horizon),
                    0,
                    True,
                ),
                lambda _: (
                    new_env_state,
                    new_belief_state,
                    reward,
                    timestep_in_episode + 1,
                    done,
                ),
                operand=None,
            )
        )

        # Calculate shortest path at the beginning of the episode
        goal = jnp.unravel_index(
            jnp.argmax(current_belief_state[3, 1:, :]),
            (environment.num_nodes, environment.num_nodes),
        )[0]
        origin = jnp.argmax(current_belief_state[0, :1, :])
        shortest_path = jax.lax.cond(
            previous_episode_done,
            lambda _: dijkstra_shortest_path(
                current_env_state,
                origin,
                goal,
            ),
            lambda _: jnp.array(0.0, dtype=jnp.float16),
            operand=None,
        )
        runner_state = (
            new_env_state,
            new_belief_state,
            env_key,
            timestep_in_episode,
            done,
        )
        transition = (
            done,
            reward,
            shortest_path,
        )
        return runner_state, transition

    runner_state = (
        new_env_state,
        new_belief_state,
        env_key,
        jnp.int32(0),
        jnp.bool_(True),
    )
    runner_state, inference_traj_batch = jax.lax.scan(
        _one_step_inference, runner_state, jnp.arange(num_testing_timesteps)
    )

    test_all_done = inference_traj_batch[0]
    test_all_rewards = inference_traj_batch[1]
    test_all_optimal_path_lengths = inference_traj_batch[2]

    # Eliminate the last few timesteps that did not finish the episode
    # Calculate competitive ratio without using pandas
    episode_numbers = jnp.cumsum(test_all_done)
    shifted_episode_numbers = jnp.concatenate([jnp.array([0]), episode_numbers[:-1]])

    def aggregate_by_episode(episode_numbers, values):
        num_episodes = jnp.max(episode_numbers) + 1
        aggregated = jax.vmap(
            lambda ep: jnp.sum(jnp.where(episode_numbers == ep, values, 0))
        )(jnp.arange(num_episodes))
        return aggregated

    aggregated_rewards = aggregate_by_episode(shifted_episode_numbers, test_all_rewards)
    aggregated_optimal_path_lengths = aggregate_by_episode(
        shifted_episode_numbers, test_all_optimal_path_lengths
    )
    episodes_to_keep = aggregated_rewards.shape[0] - 1
    aggregated_rewards = aggregated_rewards[:episodes_to_keep]
    aggregated_optimal_path_lengths = aggregated_optimal_path_lengths[:episodes_to_keep]
    competitive_ratio = jnp.abs(aggregated_rewards) / aggregated_optimal_path_lengths

    # Get average competitive ratio
    average_competitive_ratio = jnp.mean(competitive_ratio)
    return average_competitive_ratio

for i in range(num_steps_for_inference):
    current_belief_state = belief_state
    current_env_state = env_state
    action, action_key = agent.act(
        action_key, out["model_params"], current_belief_state, 0
    )
    # For multi-agent, we would concatenate all the agents' actions together here
    action = jnp.array([action])
    env_state, belief_state, reward, done, env_key = environment.step(
        env_key, current_env_state, current_belief_state, action
    )
    action = action[0]
    shortest_path = jax.lax.cond(
        done,
        lambda _: dijkstra_shortest_path(
            current_env_state,
            environment.graph_realisation.graph.origin,
            environment.graph_realisation.graph.goal,
        ),
        lambda _: 0.0,
        operand=None,
    )
    test_all_rewards = test_all_rewards.at[i].set(reward)
    test_all_done = test_all_done.at[i].set(done)
    test_all_optimal_path_lengths = test_all_optimal_path_lengths.at[i].set(
        shortest_path
    )
    test_all_actions = test_all_actions.at[i].set(action)
    test_all_positions = test_all_positions.at[i].set(
        jnp.argmax(current_env_state[0, : args.n_agent])
    )

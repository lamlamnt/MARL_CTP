# For generating a new graph on the fly for every episode
    def get_initial_env_state(self, key: jax.random.PRNGKey):
        if self.deal_with_unsolvability == "always_expensive_edge":
            auto_expensive_edge = True
        else:
            auto_expensive_edge = False
        graph_realisation = CTP_generator.CTPGraph_Realisation(
            key,
            self.num_nodes,
            grid_size=self.grid_size,
            prop_stoch=self.prop_stoch,
            k_edges=self.k_edges,
            num_goals=self.num_goals,
            factor_expensive_edge=self.factor_expensive_edge,
            expensive_edge=auto_expensive_edge,
        )
        _, subkey = jax.random.split(key)

        # Resample until we get a solvable realisation
        if self.deal_with_unsolvability == "resample":
            patience_counter = 0
            is_solvable = jnp.bool_(False)
            while is_solvable == jnp.bool_(False) and patience_counter < self.patience:
                key, subkey = jax.random.split(subkey)
                new_blocking_status = graph_realisation.sample_blocking_status(subkey)
                is_solvable = graph_realisation.is_solvable(new_blocking_status)
                patience_counter += 1
            # error if is_solvable is False
            if is_solvable == jnp.bool_(False):
                raise ValueError(
                    "Could not find enough solvable blocking status. Please decrease the prop_stoch."
                )
        elif self.deal_with_unsolvability == "always_expensive_edge":
            new_blocking_status = graph_realisation.sample_blocking_status(subkey)
        else:
            new_blocking_status = graph_realisation.sample_blocking_status(subkey)
            is_solvable = graph_realisation.is_solvable(new_blocking_status)
            # Add expensive edge if unsolvable
            if is_solvable == jnp.bool_(False):
                upper_bound = (
                    (self.num_nodes - 1)
                    * jnp.max(graph_realisation.graph.weights)
                    * self.factor_expensive_edge
                )
                graph_realisation.graph.weights = graph_realisation.graph.weights.at[
                    graph_realisation.graph.origin, graph_realisation.graph.goal
                ].set(upper_bound)
                graph_realisation.graph.weights = graph_realisation.graph.weights.at[
                    graph_realisation.graph.goal, graph_realisation.graph.origin
                ].set(upper_bound)
                graph_realisation.graph.blocking_prob = (
                    graph_realisation.graph.blocking_prob.at[
                        graph_realisation.graph.origin, graph_realisation.graph.goal
                    ].set(0)
                )
                graph_realisation.graph.blocking_prob = (
                    graph_realisation.graph.blocking_prob.at[
                        graph_realisation.graph.goal, graph_realisation.graph.origin
                    ].set(0)
                )
                new_blocking_status = new_blocking_status.at[
                    graph_realisation.graph.origin, graph_realisation.graph.goal
                ].set(CTP_generator.UNBLOCKED)
                new_blocking_status = new_blocking_status.at[
                    graph_realisation.graph.goal, graph_realisation.graph.origin
                ].set(CTP_generator.UNBLOCKED)

                # renormalize the edge weights by the expensive edge
                max_weight = jnp.max(graph_realisation.graph.weights)
                graph_realisation.graph.weights = jnp.where(
                    graph_realisation.graph.weights != CTP_generator.NOT_CONNECTED,
                    graph_realisation.graph.weights / max_weight,
                    CTP_generator.NOT_CONNECTED,
                )

        # Put into env state
        empty = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        edge_weights = jnp.concatenate((empty, graph_realisation.graph.weights), axis=0)
        edge_probs = jnp.concatenate(
            (empty, graph_realisation.graph.blocking_prob), axis=0
        )
        agents_pos = jnp.zeros((self.num_agents, self.num_nodes), dtype=jnp.float16)
        agents_pos = agents_pos.at[0, graph_realisation.graph.origin[0]].set(1)
        pos_and_blocking_status = jnp.concatenate(
            (agents_pos, new_blocking_status), axis=0
        )

        # Top part is each agent's service history. Bottom part is number of times each goal needs to
        # be serviced
        goal_matrix = jnp.zeros_like(pos_and_blocking_status)
        goal_matrix = goal_matrix.at[
            self.num_agents + graph_realisation.graph.goal[0],
            graph_realisation.graph.goal[0],
        ].set(1)

        return jnp.stack(
            (pos_and_blocking_status, edge_weights, edge_probs, goal_matrix),
            axis=0,
            dtype=jnp.float16,
        )

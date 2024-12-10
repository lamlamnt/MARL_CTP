# Return whether an unblocked path exists from origin to goal using networkx library
    def is_solvable_old(self, blocking_status) -> bool:
        # Remove the blocked edges from the graph before converting to networkx
        # New senders and receivers that are unblocked edges
        def __filter_unblocked_senders(sender, receiver, status):
            return jax.lax.cond(
                status[sender, receiver] == False,  # Check if the edge is not blocked
                lambda _: sender,  # Include the sender if unblocked
                lambda _: NOT_CONNECTED,  # Use -1 as a placeholder for blocked edges
                operand=None,
            )

        filtered_senders = jax.vmap(__filter_unblocked_senders, in_axes=(0, 0, None))(
            self.graph.senders, self.graph.receivers, blocking_status
        )

        # Remove placeholder values (-1) and return only valid sender values
        unblocked_senders = self.graph.senders[filtered_senders != NOT_CONNECTED]
        unblocked_receivers = self.graph.receivers[filtered_senders != NOT_CONNECTED]
        graph_NX = nx.Graph()
        for i in range(self.graph.n_nodes):
            graph_NX.add_node(i, pos=tuple(self.graph.node_pos[i].tolist()))
        if unblocked_senders.size > 0:
            for i in range(self.graph.n_edges):
                if (
                    blocking_status[unblocked_senders[i], unblocked_receivers[i]]
                    == UNBLOCKED
                ):
                    graph_NX.add_edge(
                        unblocked_senders[i].item(), unblocked_receivers[i].item()
                    )
        solvable = nx.has_path(
            graph_NX, self.graph.origin.item(), self.graph.goal.item()
        )
        self.solvable = solvable
        return jnp.bool_(solvable)
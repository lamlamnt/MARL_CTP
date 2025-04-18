# 2 separate functions: one returns the expected cost and the value of each node in the tree (more nodes than num nodes in the graph)
# The other takes the value of each node and interacts with the environment
# At each node, choose the node with the lowest expected cost + that edge cost (out of the connected nodes)
def AO_Star_Planning_Wrong(
    belief_state: jnp.ndarray, origin: int, goal: int
) -> Tuple[List[int], float]:
    # Returns the expected cost and the Policy
    weights = belief_state[1, 1:, :]
    blocking_prob = belief_state[2, 1:, :]
    origin_optimistic_heuristic = get_optimistic_heuristic(belief_state, origin, goal)
    open_nodes = [(origin, origin_optimistic_heuristic)]  # Open list: (node, cost)
    costs = {origin: origin_optimistic_heuristic}  # Cost to reach each node
    best_paths = {origin: origin}  # Best paths from nodes
    debug = 0

    while open_nodes and debug < 10:
        debug += 1
        print(open_nodes)
        # Sort nodes by cost (ascending)
        open_nodes = sorted(open_nodes, key=lambda x: x[1])
        current_node, current_cost = open_nodes.pop(0)
        if type(current_node) != int:
            current_node = current_node.item()

        # Check if we reached the goal
        if current_node == goal:
            # Reconstruct path
            path = [origin, best_paths[origin]]
            while path[-1] != goal:
                path.append(best_paths[path[-1]])
            return path, costs[origin]

        # Expand current node
        successors = jnp.where(weights[current_node] != CTP_generator.NOT_CONNECTED)[
            0
        ]  # Nodes reachable from current_node
        min_cost = jnp.inf
        best_successor = None

        # This is the OR node
        for successor in successors:
            edge_cost = weights[current_node, successor]
            block_prob = blocking_prob[current_node, successor]

            # Compute expected cost for stochastic edge
            optimistic_heuristic = get_optimistic_heuristic(
                belief_state, successor, goal
            )

            # Compute expected cost
            weighted_traversable_cost = (1 - block_prob) * (
                edge_cost + optimistic_heuristic
            )
            untraversable_belief_state = belief_state.at[
                0, 1 + current_node, successor
            ].set(CTP_generator.BLOCKED)
            untraversable_belief_state = untraversable_belief_state.at[
                0, successor + 1, current_node
            ].set(CTP_generator.BLOCKED)
            untraversable_optimistic_heuristic = get_optimistic_heuristic(
                untraversable_belief_state, current_node, goal
            )
            weighted_untraversable_cost = block_prob * (
                untraversable_optimistic_heuristic
            )
            # This is the AND node
            expected_cost = weighted_traversable_cost + weighted_untraversable_cost

            # Update best path and minimum cost
            if expected_cost < min_cost:
                min_cost = expected_cost
                best_successor = successor

        # Update open list and paths
        if best_successor is not None:
            costs[current_node] = min_cost
            best_paths[current_node] = best_successor.item()
            open_nodes.append((best_successor, min_cost))
        else:
            # No successor - leaf node -> propagate upwards
            pass
        print(best_paths)
        print(costs)

    return [], jnp.inf  # Return failure if no path is found

from typing import Tuple, List
import jax
import jax.numpy as jnp
from Evaluation import optimal_path_length
import sys
from abc import ABC, abstractmethod

sys.path.append("..")
from Environment import CTP_generator
import numpy as np


@jax.jit
def get_optimistic_heuristic(belief_state: jnp.ndarray, node: int, goal: int) -> float:
    # Assume all unknown stochastic edges are not blocked
    belief_state = belief_state.at[0, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.UNKNOWN,
            CTP_generator.UNBLOCKED,
            belief_state[0, 1:, :],
        )
    )
    # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
    belief_state = belief_state.at[1, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.BLOCKED,
            1,
            belief_state[1, 1:, :],
        )
    )
    return optimal_path_length.dijkstra_shortest_path(belief_state, node, goal)


# This can (and often will) return inf
def get_pessimistic_heuristic(belief_state: jnp.ndarray, node: int, goal: int) -> float:
    # Assume all unknown edges are blocked
    belief_state = belief_state.at[0, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.UNKNOWN,
            CTP_generator.BLOCKED,
            belief_state[0, 1:, :],
        )
    )
    # dijkstra expects env_state. Change blocking_prob of known blocked edges to 1.
    belief_state = belief_state.at[1, 1:, :].set(
        jnp.where(
            belief_state[0, 1:, :] == CTP_generator.BLOCKED,
            1,
            belief_state[1, 1:, :],
        )
    )
    return optimal_path_length.dijkstra_shortest_path(belief_state, node, goal)


# Each node in the tree corresponds to a node in the graph and a belief state. But don't need to
# store the belief state.
# Abstract base class
class Node(ABC):
    def __init__(self, graph_node: int, value: float, parent, belief, solved=False):
        self.graph_node = graph_node  # Corresponding node in the graph
        self.value = value  # The current heuristic value of the node
        self.parent = parent
        self.successors = []  # Elements are of type Node
        self.solved = solved
        self.belief = belief  # dimension (num_nodes)x(num_nodes)

    def add_successor(self, successor):
        self.successors.append(successor)

    @abstractmethod
    def expand(self, belief_state, goal) -> List:
        pass

    @abstractmethod
    def solve(self):
        pass


class OR_Node(Node):
    def __init__(
        self,
        graph_node: int,
        value: float,
        parent: Node,
        belief: jnp.ndarray,
        solved=False,
    ):
        super().__init__(graph_node, value, parent, belief, solved)
        self.edge_cost_to_successor = jnp.array([])

    def add_successor(self, successor, edge_cost):
        self.successors.append(successor)
        self.edge_cost_to_successor = jnp.concatenate(
            [self.edge_cost_to_successor, jnp.array([edge_cost])]
        )

    # Successors are all reachable vertices undisambiguated edges (including deterministic edges and recently disambiguated edges from the parent node)
    # Don't add successors if the edge costs between the parent to successor is greater than the risk-free cost from the parent to the goal
    # Also don't add successors if the AND_node's children have the same shortest risk_free path to the goal
    def expand(self, belief_state: jnp.ndarray, goal: int) -> List[Node]:
        # Returns updated fringe list
        belief_state = belief_state.at[0, 1:, :].set(self.belief)
        if not self.successors:
            nodes = jnp.arange(self.belief.shape[0])
            pessimistic_costs = jax.vmap(
                get_pessimistic_heuristic, in_axes=(None, 0, None)
            )(belief_state, nodes, self.graph_node)
            risk_free_cost_to_goal = get_pessimistic_heuristic(
                belief_state, self.graph_node, goal
            )
            # Get list of nodes with pessimistic cost lower than the current node's pessimistic cost
            # Don't want to include current node in the successor list
            pessimistic_costs = pessimistic_costs.at[self.graph_node].set(jnp.inf)
            successors = jnp.where(pessimistic_costs < risk_free_cost_to_goal)[0]
            # Calculate heuristic for each successor
            # Each successor is connected deterministically to the current node
            # Each successor has multiple stochastic edges to other nodes
            successor_origin_list = jnp.array([], dtype=jnp.int8)
            successor_destination_list = jnp.array([], dtype=jnp.int8)
            for successor in successors:
                new_successors = jnp.where(
                    self.belief[successor] == CTP_generator.UNKNOWN
                )[0]
                successor_origin_list = jnp.concatenate(
                    [
                        successor_origin_list,
                        jnp.array([successor for i in range(len(new_successors))]),
                    ]
                )
                successor_destination_list = jnp.concatenate(
                    [successor_destination_list, new_successors]
                )
            if risk_free_cost_to_goal < jnp.inf:
                # Add as a successor
                successor_origin_list = jnp.concatenate(
                    [successor_origin_list, jnp.array([self.graph_node])]
                )
                successor_destination_list = jnp.concatenate(
                    [successor_destination_list, jnp.array([goal])]
                )
            new_fringe_nodes = []
            successor_origin_list = successor_origin_list.astype(jnp.int8)
            # Remove duplicates
            combined = jnp.stack(
                [successor_origin_list, successor_destination_list], axis=-1
            )
            unique_combined = jnp.unique(combined, axis=0)
            successor_origin_list = unique_combined[:, 0]
            successor_destination_list = unique_combined[:, 1]
            # Successor_destination list can contain nodes that are connected to both deterministic and stochastic edges
            for i in range(len(successor_origin_list)):
                # The edge cost is the shortest deterministic distance connecting the current node and the origin node, assuming all
                # unknown edges are blocked!!
                # If terminal node, then it is solved
                pessimistic_path_length = get_pessimistic_heuristic(
                    belief_state, successor_origin_list[i], goal
                )
                if (
                    successor_destination_list[i] == goal
                    and pessimistic_path_length == risk_free_cost_to_goal
                    and pessimistic_path_length < jnp.inf
                ):
                    successor_node = AND_Node(
                        goal, 0, self, 0, self.belief, solved=True
                    )
                    edge_cost = risk_free_cost_to_goal
                else:
                    block_prob = belief_state[
                        2, 1 + successor_origin_list[i], successor_destination_list[i]
                    ]
                    successor_node = AND_Node(
                        successor_origin_list[i],
                        get_optimistic_heuristic(
                            belief_state, successor_origin_list[i], goal
                        ),
                        self,
                        block_prob,
                        self.belief,
                        solved=False,
                    )
                    successor_node.destination_node_of_ambiguated_edge = (
                        successor_destination_list[i]
                    )
                    new_fringe_nodes.append(successor_node)
                    edge_cost = pessimistic_costs[successor_origin_list[i]]
                self.add_successor(successor_node, edge_cost)
        # Sort based on current estimated cost
        values = jnp.zeros(len(self.successors))
        for i in range(len(self.successors)):
            values = values.at[i].set(
                self.successors[i].value + self.edge_cost_to_successor[i]
            )
        sorted_indices = jnp.argsort(values)
        self.successors = [self.successors[i] for i in sorted_indices]
        # Sort edge costs to match the sorted successors
        self.edge_cost_to_successor = self.edge_cost_to_successor[sorted_indices]
        return new_fringe_nodes

    def solve(self, belief_state: jnp.ndarray):
        # Get min of successors'estimated cost+edge cost
        # Assign this as the node's new value
        costs = jnp.array(
            [
                self.successors[i].value + self.edge_cost_to_successor[i]
                for i in range(len(self.successors))
            ]
        )
        self.value = jnp.min(costs)
        best_successor_node = self.successors[jnp.argmin(costs)]
        # If the successor's status is solved, then the node is solved
        if best_successor_node.solved is True:
            self.solved = True


# And Nodes represent edges
class AND_Node(Node):
    def __init__(
        self,
        graph_node: int,
        value: float,
        parent: Node,
        blocking_prob,
        belief,
        solved=False,
    ):
        super().__init__(graph_node, value, parent, belief, solved)
        # The graph node of AND_Node is the origin node of the edge
        self.blocking_prob = blocking_prob
        # Also has attribute destination node. None-expanded AND_Nodes will have this as None
        self.destination_node_of_ambiguated_edge = None

    def expand(self, belief_state: jnp.ndarray, goal: int) -> List[Node]:
        # AND node only has 2 OR_Node successors: traversable and untraversable
        # belief_state with that edge blocked
        destination_node = self.destination_node_of_ambiguated_edge
        untraversable_belief_state = belief_state.at[
            0, 1 + self.graph_node, destination_node
        ].set(CTP_generator.BLOCKED)
        untraversable_belief_state = untraversable_belief_state.at[
            0, 1 + destination_node, self.graph_node
        ].set(CTP_generator.BLOCKED)
        untraversal_value = get_optimistic_heuristic(
            untraversable_belief_state, self.graph_node, goal
        )
        traversable_belief_state = belief_state.at[
            0, 1 + self.graph_node, destination_node
        ].set(CTP_generator.UNBLOCKED)
        traversable_belief_state = traversable_belief_state.at[
            0, 1 + destination_node, self.graph_node
        ].set(CTP_generator.UNBLOCKED)
        traversal_child_node = OR_Node(
            self.graph_node,
            get_optimistic_heuristic(belief_state, self.graph_node, goal),
            self,
            traversable_belief_state[0, 1:, :],
            solved=self.graph_node == goal,
        )
        untraversal_child_node = OR_Node(
            self.graph_node,
            untraversal_value,
            self,
            untraversable_belief_state[0, 1:, :],
        )
        self.add_successor(traversal_child_node)
        self.add_successor(untraversal_child_node)
        return [traversal_child_node, untraversal_child_node]

    def solve(self, belief_state: jnp.ndarray):
        # No cost for disambiguation
        self.value = (1 - self.blocking_prob) * self.successors[
            0
        ].value + self.blocking_prob * self.successors[1].value
        if self.successors[0].solved is True and self.successors[1].solved is True:
            self.solved = True


# Don't pass the whole belief state around - just the relevant parts
def AO_Star_Planning(belief_state: jnp.ndarray, origin: int, goal: int) -> Node:
    # Returns the root node of the policy tree
    # The belief state already disambiguates the edges connected to the origin, so root node is an OR node
    root_node = OR_Node(
        origin,
        get_optimistic_heuristic(belief_state, origin, goal),
        None,
        belief_state[0, 1:, :],
    )
    iteration_num = 0
    max_iterations = 40 * belief_state.shape[2]
    fringe_list = [root_node]
    current_node = root_node
    while (
        root_node.solved == False
        and root_node.value != jnp.inf
        and iteration_num < max_iterations
    ):
        iteration_num += 1
        # Pick the node to expand -> Choose a fringe node with the lowest estimated cost
        # fringe_node_values = jnp.array([node.value for node in fringe_list])
        # current_node = fringe_list[jnp.argmin(fringe_node_values)]
        current_node = fringe_list[0]
        # print("Current node: " + str(current_node.graph_node))
        # print("Current node value: " + str(current_node.value))

        # Expansion (different depending on the type of node)
        fringe_list.remove(current_node)
        fringe_node_values = jnp.array([node.value for node in fringe_list])
        new_fringe_nodes = current_node.expand(belief_state, goal)
        # depth first search like expansion
        new_fringe_node_values = jnp.array([node.value for node in new_fringe_nodes])
        sort_indices = jnp.argsort(new_fringe_node_values)
        sort_indices_np = np.array(sort_indices)
        new_fringe_nodes = [new_fringe_nodes[i] for i in sort_indices_np]
        fringe_list = new_fringe_nodes + fringe_list
        # fringe_list += new_fringe_nodes

        """
        for node in current_node.successors:
            print("Current node's successor: " + str(node.graph_node))
            print("Current node's successor value: " + str(node.value))
            print("Current node's status: " + str(node.solved))
        if type(current_node) is OR_Node:
            print("Edge cost to successor:" + str(current_node.edge_cost_to_successor))
        """
        # Propagate upwards to the root
        while current_node != None:  # not root node
            if current_node.successors != []:
                current_node.solve(belief_state)
            current_node = current_node.parent
    if iteration_num == max_iterations:
        raise Exception("Max iterations reached")
    # Prune children of OR nodes starting from the root node
    # use a recursive function to prune the tree
    _prune(root_node)
    # OR_Nodes' successors and edge_cost_to_successor are now only 1 element each
    return root_node


def _prune(node: OR_Node):
    # Depth first search prune
    if node.successors:
        best_child_node = node.successors[
            0
        ]  # This is an AND_Node (best child of an OR_Node)
        node.successors = best_child_node
        node.edge_cost_to_successor = node.edge_cost_to_successor[0]
        # Not all AND nodes have been expanded
        if best_child_node.successors:
            _prune(best_child_node.successors[0])
            _prune(best_child_node.successors[1])


# Chose to use env_state instead of interacting with the environment because
# this will run faster (but easier to introduce bugs)
def AO_Star_Execute(env_state: jnp.ndarray, root_node: OR_Node, goal: int) -> float:
    # At each AND_node, check whether the edge is traversable or not
    total_length = 0
    current_node = root_node
    # Always end with an AND node that's not fully expanded
    while current_node.graph_node != goal:
        # OR_Node
        print("Current node: " + str(current_node.graph_node))
        total_length += current_node.edge_cost_to_successor
        current_node = current_node.successors  # And_Node
        # Check whether the edge is traversable
        destination_node = current_node.destination_node_of_ambiguated_edge
        # Not all AND nodes are fully expanded.
        if current_node.successors:
            if (
                env_state[0, 1 + current_node.graph_node, destination_node]
                == CTP_generator.UNBLOCKED
            ):
                current_node = current_node.successors[0]
            else:
                current_node = current_node.successors[1]
    return total_length


def AO_Star_Full(
    env_state: jnp.ndarray, belief_state: jnp.ndarray, origin: int, goal: int
) -> float:
    root_node = AO_Star_Planning(belief_state, origin, goal)
    return AO_Star_Execute(env_state, root_node, goal)

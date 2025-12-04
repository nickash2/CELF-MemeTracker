from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Set, Tuple

from .celf import InfluenceGraph, estimate_influence


def degree_centrality(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
    mode: str = "out",
) -> Tuple[List[str], float]:
    """Selects nodes based on degree centrality.

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        costs: Optional node costs (not used by this heuristic).
        mode: 'in' for in-degree, 'out' for out-degree (default).

    Returns:
        Tuple of (selected seeds, total cost).
    """
    if mode not in {"in", "out"}:
        raise ValueError("Mode must be 'in' or 'out'.")

    # Calculate degrees
    degrees: Dict[str, int] = {node: 0 for node in graph.nodes}
    if mode == "out":
        for node in graph.nodes:
            degrees[node] = len(graph.neighbors(node))
    else:  # in-degree
        for node in graph.nodes:
            for neighbor, _ in graph.neighbors(node):
                degrees[neighbor] += 1

    # Sort nodes by degree
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    # Select top k nodes
    seeds = [node for node, _ in sorted_nodes[:budget]]
    total_cost = sum(costs.get(node, 1.0) for node in seeds) if costs else float(budget)

    return seeds, total_cost


def random_selection(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], float]:
    """Selects nodes uniformly at random.

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        costs: Optional node costs (not used by this heuristic).
        rng: Optional random number generator.

    Returns:
        Tuple of (selected seeds, total cost).
    """
    if rng is None:
        rng = random.Random()

    nodes = list(graph.nodes)
    seeds = rng.sample(nodes, k=budget)
    total_cost = sum(costs.get(node, 1.0) for node in seeds) if costs else float(budget)

    return seeds, total_cost


def greedy(
    graph: InfluenceGraph,
    budget: int,
    simulations: int,
    costs: Optional[Mapping[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], float, float]:
    """Standard (non-lazy) greedy algorithm for influence maximization.

    Serves as a baseline to demonstrate CELF's speedup.

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        simulations: Monte Carlo simulations for spread estimation.
        costs: Optional node costs.
        rng: Optional random number generator.

    Returns:
        Tuple of (selected seeds, final spread, total cost).
    """
    if rng is None:
        rng = random.Random()

    selected: List[str] = []
    selected_set: Set[str] = set()
    current_spread = 0.0
    total_cost = 0.0

    for _ in range(budget):
        best_node: Optional[str] = None
        best_marginal_gain = -1.0
        nodes_to_check = graph.nodes - selected_set

        for node in nodes_to_check:
            # This implementation assumes unit costs for simplicity in selection loop

            gain = (
                estimate_influence(graph, selected + [node], simulations, rng)
                - current_spread
            )

            if gain > best_marginal_gain:
                best_marginal_gain = gain
                best_node = node

        if best_node:
            selected.append(best_node)
            selected_set.add(best_node)
            current_spread += best_marginal_gain
            total_cost += costs.get(best_node, 1.0) if costs else 1.0

    return selected, current_spread, total_cost


def pagerank(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[List[str], float]:
    """Accurate PageRank centrality using incoming neighbors for efficiency."""

    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return [], 0.0

    # Initialize scores
    scores = {node: 1.0 / n for node in nodes}

    # Precompute outgoing degree
    out_degree = {node: len(graph.neighbors(node)) for node in nodes}

    # Precompute incoming neighbors for each node
    incoming = {node: [] for node in nodes}
    for node in nodes:
        for neighbor, _ in graph.neighbors(node):
            incoming[neighbor].append(node)

    # Iterative PageRank
    for _ in range(max_iterations):
        new_scores = {}
        diff = 0.0
        for node in nodes:
            rank_sum = sum(
                scores[src] / out_degree[src]
                for src in incoming[node]
                if out_degree[src] > 0
            )
            new_score = (1 - damping) / n + damping * rank_sum
            new_scores[node] = new_score
            diff += abs(new_score - scores[node])

        scores = new_scores
        if diff < tolerance:
            break

    # Select top-k
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:budget]]
    total_cost = sum(costs.get(node, 1.0) for node in seeds) if costs else float(budget)

    return seeds, total_cost


def betweenness_centrality(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
    k_samples: Optional[int] = None,
) -> Tuple[List[str], float]:
    """Selects nodes based on betweenness centrality.

    Uses approximate betweenness via random sampling for efficiency.

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        costs: Optional node costs (not used by this heuristic).
        k_samples: Number of source nodes to sample (None = all nodes).

    Returns:
        Tuple of (selected seeds, total cost).
    """
    nodes = list(graph.nodes)
    if not nodes:
        return [], 0.0

    betweenness = {node: 0.0 for node in nodes}
    sample_nodes = (
        nodes if k_samples is None else random.sample(nodes, min(k_samples, len(nodes)))
    )

    for source in sample_nodes:
        # BFS to find shortest paths
        queue = [source]
        visited = {source}
        distance = {source: 0}
        paths_count = {source: 1}
        predecessors = {node: [] for node in nodes}
        stack = []

        while queue:
            current = queue.pop(0)
            stack.append(current)

            for neighbor, _ in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    distance[neighbor] = distance[current] + 1

                if distance.get(neighbor, float("inf")) == distance[current] + 1:
                    paths_count[neighbor] = (
                        paths_count.get(neighbor, 0) + paths_count[current]
                    )
                    predecessors[neighbor].append(current)

        # Back-propagation of dependencies
        dependency = {node: 0.0 for node in nodes}
        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                dependency[pred] += (
                    paths_count[pred] / paths_count[node] * (1 + dependency[node])
                )
            if node != source:
                betweenness[node] += dependency[node]

    # Normalize
    n = len(nodes)
    if n > 2:
        norm = (n - 1) * (n - 2)
        betweenness = {node: score / norm for node, score in betweenness.items()}

    # Select top k nodes
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:budget]]
    total_cost = sum(costs.get(node, 1.0) for node in seeds) if costs else float(budget)

    return seeds, total_cost


def closeness_centrality(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], float]:
    """Selects nodes based on closeness centrality.

    Closeness is the reciprocal of average distance to all other nodes.

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        costs: Optional node costs (not used by this heuristic).

    Returns:
        Tuple of (selected seeds, total cost).
    """
    nodes = list(graph.nodes)
    if not nodes:
        return [], 0.0

    closeness = {}

    for node in nodes:
        # BFS to find distances
        queue = [node]
        visited = {node}
        distance = {node: 0}

        while queue:
            current = queue.pop(0)
            for neighbor, _ in graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    distance[neighbor] = distance[current] + 1

        # Calculate closeness
        total_distance = sum(distance.values())
        reachable = len(distance)

        if total_distance > 0:
            closeness[node] = (reachable - 1) / total_distance
        else:
            closeness[node] = 0.0

    # Select top k nodes
    sorted_nodes = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
    seeds = [node for node, _ in sorted_nodes[:budget]]
    total_cost = sum(costs.get(node, 1.0) for node in seeds) if costs else float(budget)

    return seeds, total_cost


def high_degree_first(
    graph: InfluenceGraph,
    budget: int,
    costs: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], float]:
    """Alias for out-degree centrality (most commonly used baseline).

    Args:
        graph: The influence graph.
        budget: The number of nodes to select.
        costs: Optional node costs.

    Returns:
        Tuple of (selected seeds, total cost).
    """
    return degree_centrality(graph, budget, costs, mode="out")

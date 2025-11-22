"""
CELF++ implementation for outbreak detection (influence maximization with lazy forward selection).

This version is adapted for exact cascade evaluation (no Monte Carlo), matching the outbreak detection setup.
"""

from typing import Dict, List, Tuple, Callable
import heapq

# Type aliases for clarity
graph_type = any  # Should be InfluenceGraph, but avoid circular import
cascade_dict_type = Dict[str, List[Tuple[str, float]]]


def celfpp_outbreak_detection(
    graph: graph_type,
    cascades: cascade_dict_type,
    budget: int,
    objective: str = "DL",
    T_max: float = 100.0,
    eval_func: Callable = None,
) -> List[str]:
    """
    CELF++ for outbreak detection (exact cascade evaluation, no MC).
    Args:
        graph: InfluenceGraph
        cascades: dict of meme -> cascade
        budget: number of sites to select
        objective: 'DL', 'DT', or 'PA'
        T_max: max time for DT
        eval_func: custom evaluation function (optional)
    Returns:
        List of selected site IDs
    """
    if eval_func is None:
        from .objectives import (
            evaluate_detection_likelihood,
            evaluate_detection_time,
            evaluate_population_affected,
        )

        if objective == "DL":

            def eval_func(events, seeds):
                return evaluate_detection_likelihood(events, seeds)
        elif objective == "DT":

            def eval_func(events, seeds):
                return 1.0 - (evaluate_detection_time(events, seeds, T_max) / T_max)
        elif objective == "PA":
            # For PA, we want reduction, so baseline must be computed outside
            def eval_func(events, seeds, baseline=None):
                return (
                    (baseline - evaluate_population_affected(events, seeds)) / baseline
                    if baseline
                    else 0.0
                )
        else:
            raise ValueError(f"Unknown objective: {objective}")

    # Convert cascades to events (site, time)
    from . import convert_memetracker_cascades_to_events, evaluate_population_affected

    cascade_events = convert_memetracker_cascades_to_events(cascades)
    all_nodes = set(graph.nodes)
    selected: List[str] = []
    gains = []
    last_eval = {}
    baseline_pa = None
    if objective == "PA":
        baseline_pa = evaluate_population_affected(cascade_events, [])

    # Initial marginal gain for all nodes
    for node in all_nodes:
        if objective == "PA":
            gain = eval_func(cascade_events, [node], baseline=baseline_pa)
        else:
            gain = eval_func(cascade_events, [node])
        heapq.heappush(gains, (-gain, node, 0))  # max-heap by gain
        last_eval[node] = 0

    cur_size = 0
    while cur_size < budget and gains:
        neg_gain, node, prev_size = heapq.heappop(gains)
        if last_eval[node] == cur_size:
            selected.append(node)
            cur_size += 1
        else:
            # Recompute marginal gain
            if objective == "PA":
                gain = eval_func(
                    cascade_events, selected + [node], baseline=baseline_pa
                ) - eval_func(cascade_events, selected, baseline=baseline_pa)
            else:
                gain = eval_func(cascade_events, selected + [node]) - eval_func(
                    cascade_events, selected
                )
            heapq.heappush(gains, (-gain, node, cur_size))
            last_eval[node] = cur_size
    return selected

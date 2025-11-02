"""
Data preprocessing utilities for CELF algorithm.

Provides loaders for graphs, node costs, and MemeTracker-specific cascade data.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .celf import InfluenceGraph


def load_graph_from_file(
    path: str,
    default_prob: float = 0.1,
    delimiter: Optional[str] = None,
    skip_header: bool = False,
) -> InfluenceGraph:
    """Builds an InfluenceGraph from an edge list on disk.

    Args:
        path: Path to edge list file
        default_prob: Default edge probability when not specified
        delimiter: Field delimiter (None = whitespace)
        skip_header: Whether to skip first line

    Returns:
        Constructed InfluenceGraph

    File format:
        source target [probability]
    """

    graph = InfluenceGraph(default_prob=default_prob)
    with open(path, "r", encoding="utf-8") as handle:
        if skip_header:
            handle.readline()
        for line_no, raw in enumerate(handle, start=1 if not skip_header else 2):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter is not None else line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Line {line_no}: expected at least 2 columns, found {parts}."
                )
            src, dst = parts[0], parts[1]
            prob: Optional[float] = None
            if len(parts) >= 3:
                try:
                    prob = float(parts[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_no}: invalid probability '{parts[2]}'."
                    ) from exc
            graph.add_edge(src, dst, prob)
    return graph


def load_costs_from_file(path: str) -> Dict[str, float]:
    """Loads node costs from a simple two-column text file.

    Args:
        path: Path to costs file

    Returns:
        Dictionary mapping node -> cost

    File format:
        node cost
    """

    costs: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_no}: expected 'node cost' but found {parts}."
                )
            node, raw_cost = parts
            try:
                costs[node] = float(raw_cost)
            except ValueError as exc:
                raise ValueError(f"Line {line_no}: invalid cost '{raw_cost}'.") from exc
    return costs


def estimate_propagation_probability(
    time_diff_hours: float,
    decay_rate: float = 0.01,
) -> float:
    """Estimates influence probability from temporal cascade data.

    Uses exponential decay model: p(t) = exp(-α * t)

    Args:
        time_diff_hours: Time difference between source and target mention
        decay_rate: Decay parameter α (default: 0.01)

    Returns:
        Propagation probability in [0, 1]
    """

    if time_diff_hours < 0:
        return 0.0
    prob = math.exp(-decay_rate * time_diff_hours)
    return min(1.0, max(0.0, prob))


def build_graph_from_cascades(
    cascades: list[list[tuple[str, float]]],
    decay_rate: float = 0.01,
    min_prob: float = 0.01,
) -> InfluenceGraph:
    """Constructs an influence graph from temporal cascades.

    Each cascade is a list of (node_id, timestamp) pairs showing when
    nodes adopted a meme/information. Creates directed edges from earlier
    to later adopters with probabilities based on time differences.

    Args:
        cascades: List of cascades, each cascade is [(node, timestamp), ...]
        decay_rate: Exponential decay parameter for probability estimation
        min_prob: Minimum probability threshold (edges below are discarded)

    Returns:
        InfluenceGraph with estimated edge probabilities

    Example:
        cascades = [
            [("blog_A", 0.0), ("blog_B", 1.5), ("blog_C", 3.0)],
            [("blog_B", 0.0), ("blog_C", 0.5)],
        ]
        graph = build_graph_from_cascades(cascades)
    """

    graph = InfluenceGraph(default_prob=min_prob)
    edge_counts: Dict[tuple[str, str], list[float]] = {}

    for cascade in cascades:
        sorted_cascade = sorted(cascade, key=lambda x: x[1])

        for i, (source, t1) in enumerate(sorted_cascade):
            for target, t2 in sorted_cascade[i + 1 :]:
                time_diff = t2 - t1
                if time_diff <= 0:
                    continue

                prob = estimate_propagation_probability(time_diff, decay_rate)
                if prob < min_prob:
                    continue

                edge = (source, target)
                edge_counts.setdefault(edge, []).append(prob)

    for (src, dst), probs in edge_counts.items():
        avg_prob = sum(probs) / len(probs)
        graph.add_edge(src, dst, avg_prob)

    return graph

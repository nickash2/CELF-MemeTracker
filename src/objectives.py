"""
Objective functions for outbreak detection and influence blocking.

Implements three penalty reduction metrics for MemeTracker cascades:
- Detection Likelihood (DL): fraction of cascades detected
- Detection Time (DT): time until detection by monitored websites
- Population Affected (PA): number of websites infected before detection

In the MemeTracker context:
- "selected_sites" or "monitors" = websites/blogs we choose to monitor
- "cascade" = a meme spreading across websites over time
- "detection" = when one of our monitored sites participates in the cascade
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

from .celf import InfluenceGraph


@dataclass
class CascadeEvent:
    """Represents a single cascade/outbreak event."""

    cascade_id: int
    source: str  # Origin node
    activated_nodes: List[str]  # Nodes activated in order
    activation_times: List[float]  # Time each node was activated
    total_size: int  # Total size of cascade


@dataclass
class DetectionResult:
    """Results from outbreak detection simulation."""

    cascade_id: int
    detected: bool
    detection_time: Optional[float]
    detection_node: Optional[str]
    population_affected: int  # Size at detection
    total_population: int  # Final size


def simulate_cascade_with_times(
    graph: InfluenceGraph,
    source: str,
    rng: random.Random,
    max_time: float = 100.0,
) -> CascadeEvent:
    """Simulates a single cascade with activation times.

    Uses Independent Cascade model with exponential waiting times.

    Args:
        graph: The influence graph.
        source: Source/seed node of the cascade.
        rng: Random number generator.
        max_time: Maximum simulation time.

    Returns:
        CascadeEvent containing activation sequence and times.
    """
    activated: Set[str] = {source}
    activation_times = {source: 0.0}
    activation_sequence = [source]

    # Priority queue: (activation_time, node)
    frontier: List[Tuple[float, str]] = [(0.0, source)]

    while frontier:
        current_time, node = min(frontier)
        frontier.remove((current_time, node))

        if current_time > max_time:
            break

        for neighbor, edge_prob in graph.neighbors(node):
            if neighbor in activated:
                continue

            prob = edge_prob if edge_prob is not None else graph.default_prob

            # Attempt activation
            if rng.random() < prob:
                # Exponential waiting time
                wait_time = (
                    -1.0
                    * (1.0 / prob)
                    * (rng.random() if rng.random() > 0.001 else 0.001)
                )
                activation_time = current_time + abs(wait_time)

                if activation_time <= max_time:
                    activated.add(neighbor)
                    activation_times[neighbor] = activation_time
                    activation_sequence.append(neighbor)
                    frontier.append((activation_time, neighbor))

    # Sort sequence by activation time
    activation_sequence.sort(key=lambda n: activation_times[n])
    times = [activation_times[n] for n in activation_sequence]

    return CascadeEvent(
        cascade_id=0,
        source=source,
        activated_nodes=activation_sequence,
        activation_times=times,
        total_size=len(activated),
    )


def detect_cascade(
    cascade: CascadeEvent,
    selected_sites: Sequence[str],
    max_time: float = 100.0,
) -> DetectionResult:
    """Determines if and when selected sites (monitors) detect a cascade.

    Args:
        cascade: The cascade event to detect.
        selected_sites: List of selected websites/blogs to monitor.
        max_time: Maximum time horizon.

    Returns:
        DetectionResult with detection information.
    """
    sensor_set = set(selected_sites)

    # Check each activated node in order
    for i, (node, time) in enumerate(
        zip(cascade.activated_nodes, cascade.activation_times)
    ):
        if node in sensor_set:
            # Detected!
            return DetectionResult(
                cascade_id=cascade.cascade_id,
                detected=True,
                detection_time=time,
                detection_node=node,
                population_affected=i
                + 1,  # Nodes affected up to and including detection
                total_population=cascade.total_size,
            )

    # Not detected
    return DetectionResult(
        cascade_id=cascade.cascade_id,
        detected=False,
        detection_time=None,
        detection_node=None,
        population_affected=cascade.total_size,
        total_population=cascade.total_size,
    )


def evaluate_detection_likelihood(
    cascades: List[CascadeEvent],
    selected_sites: Sequence[str],
) -> float:
    """Evaluates Detection Likelihood (DL) objective.

    DL = fraction of cascades detected by monitored websites.
    Penalty: 0 if detected in finite time, 1 otherwise.

    Args:
        cascades: List of cascade events.
        selected_sites: Selected websites to monitor.

    Returns:
        Detection likelihood (fraction detected).
    """
    if not cascades:
        return 0.0

    detected_count = 0
    for cascade in cascades:
        result = detect_cascade(cascade, selected_sites)
        if result.detected:
            detected_count += 1

    return detected_count / len(cascades)


def evaluate_detection_time(
    cascades: List[CascadeEvent],
    selected_sites: Sequence[str],
    max_time: float = 100.0,
) -> float:
    """Evaluates Detection Time (DT) objective.

    DT = average time until detection (or max_time if not detected).
    Penalty: π(t) = min{t, T_max}.

    Args:
        cascades: List of cascade events.
        selected_sites: Selected websites to monitor.
        max_time: Maximum time horizon T_max.

    Returns:
        Average detection time (lower is better).
    """
    if not cascades:
        return max_time

    total_time = 0.0
    for cascade in cascades:
        result = detect_cascade(cascade, selected_sites, max_time)
        if result.detected and result.detection_time is not None:
            total_time += min(result.detection_time, max_time)
        else:
            total_time += max_time

    return total_time / len(cascades)


def evaluate_population_affected(
    cascades: List[CascadeEvent],
    selected_sites: Sequence[str],
) -> float:
    """Evaluates Population Affected (PA) objective.

    PA = average number of websites affected before detection.
    Penalty: π(t) = number of infected websites at time t.

    Args:
        cascades: List of cascade events.
        selected_sites: Selected websites to monitor.

    Returns:
        Average population affected (lower is better).
    """
    if not cascades:
        return 0.0

    total_affected = 0
    for cascade in cascades:
        result = detect_cascade(cascade, selected_sites)
        total_affected += result.population_affected

    return total_affected / len(cascades)


def generate_random_cascades(
    graph: InfluenceGraph,
    num_cascades: int,
    rng: random.Random,
    max_time: float = 100.0,
    source_nodes: Optional[List[str]] = None,
) -> List[CascadeEvent]:
    """Generates random cascade events for evaluation.

    Args:
        graph: The influence graph.
        num_cascades: Number of cascades to generate.
        rng: Random number generator.
        max_time: Maximum simulation time.
        source_nodes: Optional list of source nodes (random if None).

    Returns:
        List of generated cascade events.
    """
    cascades = []
    nodes = list(graph.nodes)

    if not nodes:
        return cascades

    if source_nodes is None:
        source_nodes = [rng.choice(nodes) for _ in range(num_cascades)]

    for i, source in enumerate(source_nodes[:num_cascades]):
        cascade = simulate_cascade_with_times(graph, source, rng, max_time)
        cascade.cascade_id = i
        cascades.append(cascade)

    return cascades


def convert_memetracker_cascades_to_events(
    cascades_dict: dict,
) -> List[CascadeEvent]:
    """Converts MemeTracker cascade data to CascadeEvent objects.

    Args:
        cascades_dict: Dictionary mapping meme -> List[(site, time_hours)]

    Returns:
        List of CascadeEvent objects.
    """
    cascade_events = []

    for cascade_id, (meme, cascade) in enumerate(cascades_dict.items()):
        if len(cascade) < 2:
            continue

        # Sort by time
        sorted_cascade = sorted(cascade, key=lambda x: x[1])

        activated_nodes = [node for node, _ in sorted_cascade]
        activation_times = [time for _, time in sorted_cascade]

        cascade_events.append(
            CascadeEvent(
                cascade_id=cascade_id,
                source=activated_nodes[0],
                activated_nodes=activated_nodes,
                activation_times=activation_times,
                total_size=len(activated_nodes),
            )
        )

    return cascade_events


def compute_penalty_reduction(
    cascades: List[CascadeEvent],
    selected_sites: Sequence[str],
    objective: str = "PA",
    max_time: float = 100.0,
) -> float:
    """Computes penalty reduction for a given objective.

    Args:
        cascades: List of cascade events.
        selected_sites: Selected websites to monitor.
        objective: One of "DL", "DT", or "PA".
        max_time: Maximum time horizon (for DT).

    Returns:
        Penalty reduction value (higher is better).
    """
    if objective == "DL":
        # For DL, higher detection likelihood = lower penalty
        dl = evaluate_detection_likelihood(cascades, selected_sites)
        return dl  # Fraction detected (0 to 1)

    elif objective == "DT":
        # For DT, lower detection time = lower penalty
        avg_time = evaluate_detection_time(cascades, selected_sites, max_time)
        # Convert to reduction: (max_time - avg_time) / max_time
        return 1.0 - (avg_time / max_time)

    elif objective == "PA":
        # For PA, lower population = lower penalty
        # Return as penalty (raw value, not normalized)
        return evaluate_population_affected(cascades, selected_sites)

    else:
        raise ValueError(
            f"Unknown objective: {objective}. Must be 'DL', 'DT', or 'PA'."
        )


def compute_cascade_penalties(
    graph: InfluenceGraph,
    selected_sites: Sequence[str],
    cascade: List[Tuple[str, float]],
    T_max: float = 100.0,
) -> Tuple[float, float, float]:
    """
    Compute all three penalty values for a single cascade.

    Args:
        graph: Influence graph (not used but kept for compatibility)
        selected_sites: Selected websites to monitor
        cascade: Single cascade as list of (site, time) tuples
        T_max: Maximum time horizon

    Returns:
        Tuple of (DL_penalty, DT_penalty, PA_penalty)
        - DL_penalty: 0 if detected, 1 if not
        - DT_penalty: detection time (or T_max if not detected)
        - PA_penalty: number of sites affected before detection
    """
    selected_set = set(selected_sites)

    # Find first detection
    detected = False
    detection_time = T_max
    population_affected = len(cascade)  # Default: all sites affected

    for i, (site, time) in enumerate(cascade):
        if site in selected_set:
            # First detection
            detected = True
            detection_time = time
            population_affected = i + 1  # Sites up to and including detection
            break

    # Compute penalties
    dl_penalty = 0.0 if detected else 1.0
    dt_penalty = detection_time
    pa_penalty = float(population_affected)

    return dl_penalty, dt_penalty, pa_penalty


def evaluate_solution_on_objectives(
    graph: InfluenceGraph,
    selected_sites: Sequence[str],
    num_cascades: int = 1000,
    rng: Optional[random.Random] = None,
    max_time: float = 100.0,
) -> dict:
    """Evaluates a website selection on all three objectives.

    Args:
        graph: The influence graph.
        selected_sites: Selected websites to monitor.
        num_cascades: Number of cascades to simulate for evaluation.
        rng: Random number generator.
        max_time: Maximum time horizon.

    Returns:
        Dictionary with DL, DT, and PA metrics.
    """
    if rng is None:
        rng = random.Random()

    # Generate cascades
    cascades = generate_random_cascades(graph, num_cascades, rng, max_time)

    # Evaluate on all objectives
    dl = evaluate_detection_likelihood(cascades, selected_sites)
    dt = evaluate_detection_time(cascades, selected_sites, max_time)
    pa = evaluate_population_affected(cascades, selected_sites)

    return {
        "detection_likelihood": dl,
        "detection_time": dt,
        "population_affected": pa,
        "num_cascades": len(cascades),
    }

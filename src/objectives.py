from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, Dict

from .celf import InfluenceGraph


@dataclass
class CascadeEvent:
    cascade_id: int
    source: str
    activated_nodes: List[str]
    activation_times: List[float]
    total_size: int


@dataclass
class DetectionResult:
    cascade_id: int
    detected: bool
    detection_time: Optional[float]
    detection_node: Optional[str]
    population_affected: int
    total_population: int


def detect_cascade_fast(
    cascade: CascadeEvent,
    selected_set: Set[str],
    max_time: float = 100.0,
) -> DetectionResult:
    for i, (node, time) in enumerate(
        zip(cascade.activated_nodes, cascade.activation_times)
    ):
        if node in selected_set:
            return DetectionResult(
                cascade_id=cascade.cascade_id,
                detected=True,
                detection_time=time,
                detection_node=node,
                population_affected=i + 1,
                total_population=cascade.total_size,
            )
    return DetectionResult(
        cascade_id=cascade.cascade_id,
        detected=False,
        detection_time=None,
        detection_node=None,
        population_affected=cascade.total_size,
        total_population=cascade.total_size,
    )


def evaluate_detection_likelihood(
    cascades: List[CascadeEvent], selected_sites: Sequence[str]
) -> float:
    selected_set = set(selected_sites)
    if not cascades:
        return 0.0
    detected_count = 0
    for cascade in cascades:
        if any(node in selected_set for node in cascade.activated_nodes):
            detected_count += 1
    return detected_count / len(cascades)


def evaluate_detection_time(
    cascades: List[CascadeEvent], selected_sites: Sequence[str], max_time: float = 100.0
) -> float:
    selected_set = set(selected_sites)
    if not cascades:
        return max_time
    total_time = 0.0
    for cascade in cascades:
        for node, time in zip(cascade.activated_nodes, cascade.activation_times):
            if node in selected_set:
                total_time += min(time, max_time)
                break
        else:
            total_time += max_time
    return total_time / len(cascades)


def evaluate_population_affected(
    cascades: List[CascadeEvent], selected_sites: Sequence[str]
) -> float:
    selected_set = set(selected_sites)
    if not cascades:
        return 0.0
    total_affected = 0
    for cascade in cascades:
        for i, node in enumerate(cascade.activated_nodes):
            if node in selected_set:
                total_affected += i + 1
                break
        else:
            total_affected += cascade.total_size
    return total_affected / len(cascades)


def convert_memetracker_cascades_to_events(
    cascades_dict: Dict[str, List[Tuple[str, float]]],
) -> List[CascadeEvent]:
    cascade_events: List[CascadeEvent] = []
    for cascade_id, (meme, cascade) in enumerate(cascades_dict.items()):
        if len(cascade) < 2:
            continue
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


def compute_cascade_penalties(
    cascade: List[Tuple[str, float]],
    selected_sites: Sequence[str],
    T_max: float = 100.0,
) -> Tuple[float, float, float]:
    selected_set = set(selected_sites)
    detected = False
    detection_time = T_max
    population_affected = len(cascade)

    for i, (site, time) in enumerate(cascade):
        if site in selected_set:
            detected = True
            detection_time = time
            population_affected = i + 1
            break

    dl_penalty = 0.0 if detected else 1.0
    dt_penalty = detection_time
    pa_penalty = float(population_affected)
    return dl_penalty, dt_penalty, pa_penalty


def compute_penalty_reduction(
    cascades: List[CascadeEvent],
    selected_sites: Sequence[str],
    objective: str = "PA",
    max_time: float = 100.0,
) -> float:
    if objective == "DL":
        return evaluate_detection_likelihood(cascades, selected_sites)
    elif objective == "DT":
        avg_time = evaluate_detection_time(cascades, selected_sites, max_time)
        return 1.0 - (avg_time / max_time)
    elif objective == "PA":
        return evaluate_population_affected(cascades, selected_sites)
    else:
        raise ValueError(
            f"Unknown objective: {objective}. Must be 'DL', 'DT', or 'PA'."
        )


def evaluate_solution_on_objectives(
    graph: InfluenceGraph,
    selected_sites: Sequence[str],
    num_cascades: int = 1000,
    rng: Optional[random.Random] = None,
    max_time: float = 100.0,
) -> dict:
    if rng is None:
        rng = random.Random()
    from .objectives import generate_random_cascades

    cascades = generate_random_cascades(graph, num_cascades, rng, max_time)
    dl = evaluate_detection_likelihood(cascades, selected_sites)
    dt = evaluate_detection_time(cascades, selected_sites, max_time)
    pa = evaluate_population_affected(cascades, selected_sites)
    return {
        "detection_likelihood": dl,
        "detection_time": dt,
        "population_affected": pa,
        "num_cascades": len(cascades),
    }

"""
Faithful implementation of the Cost-Effective Lazy Forward (CELF) algorithm.

The code mirrors Algorithms 1 (LazyForward / CELF) and 2 (GetBound) from
Leskovec et al., KDD'07. It targets the independent cascade (IC) diffusion
model, supports optional sensor costs, and exposes both UC and CB variants
along with the online upper bound R^.
"""

from __future__ import annotations

import collections
import dataclasses
import heapq
import random
from typing import (
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)


@dataclasses.dataclass
class CELFEntry:
    """Tracks the marginal gain, cost, and evaluation stage for a candidate."""

    node: str
    gain: float
    cost: float
    last_eval: int


class InfluenceGraph:
    """Minimal adjacency container for IC-style influence propagation."""

    def __init__(self, default_prob: float = 0.1) -> None:
        if not 0.0 < default_prob <= 1.0:
            raise ValueError("default_prob must lie in (0, 1].")
        self._adjacency: MutableMapping[str, List[Tuple[str, Optional[float]]]] = {}
        self._nodes: Set[str] = set()
        self.default_prob = default_prob

    @property
    def nodes(self) -> Set[str]:
        return self._nodes

    def add_edge(self, src: str, dst: str, prob: Optional[float] = None) -> None:
        if prob is not None and not 0.0 <= prob <= 1.0:
            raise ValueError("Edge probability must lie in [0, 1].")
        self._adjacency.setdefault(src, []).append((dst, prob))
        self._nodes.add(src)
        self._nodes.add(dst)
        self._adjacency.setdefault(dst, self._adjacency.get(dst, []))

    def neighbors(self, node: str) -> Sequence[Tuple[str, Optional[float]]]:
        return self._adjacency.get(node, ())


def simulate_independent_cascade(
    graph: InfluenceGraph,
    seeds: Sequence[str],
    rng: random.Random,
) -> int:
    """Runs one IC simulation and returns the number of activated nodes."""

    activated: Set[str] = set(seeds)
    frontier: Deque[str] = collections.deque(seeds)

    while frontier:
        node = frontier.popleft()
        for neighbor, edge_prob in graph.neighbors(node):
            if neighbor in activated:
                continue
            prob = edge_prob if edge_prob is not None else graph.default_prob
            if rng.random() < prob:
                activated.add(neighbor)
                frontier.append(neighbor)

    return len(activated)


def estimate_influence(
    graph: InfluenceGraph,
    seeds: Sequence[str],
    simulations: int,
    rng: random.Random,
) -> float:
    """Monte Carlo estimate of the expected influence spread for IC."""

    unique_seeds = list(dict.fromkeys(seeds))
    if not unique_seeds:
        return 0.0
    if simulations <= 0:
        raise ValueError("simulations must be a positive integer.")

    spread = 0.0
    for _ in range(simulations):
        spread += simulate_independent_cascade(graph, unique_seeds, rng)
    return spread / simulations


def _priority(entry: CELFEntry, mode: str) -> float:
    """Computes the priority key used in the lazy queues."""

    if mode == "UC":
        return entry.gain
    if entry.cost <= 0:
        return float("inf")
    return entry.gain / entry.cost


def _has_feasible_extension(
    entries: Mapping[str, CELFEntry],
    selected: Set[str],
    current_cost: float,
    budget: float,
) -> bool:
    """Checks for at least one node that can still be added under the budget."""

    for entry in entries.values():
        if entry.node in selected:
            continue
        if current_cost + entry.cost <= budget:
            return True
    return False


def _lazy_forward(
    graph: InfluenceGraph,
    budget: float,
    simulations: int,
    costs: Mapping[str, float],
    rng: random.Random,
    mode: str,
) -> Tuple[List[str], float, float]:
    """Implements LazyForward as described in Algorithm 1 of the paper."""

    if mode not in {"UC", "CB"}:
        raise ValueError("mode must be either 'UC' or 'CB'.")

    selected: List[str] = []
    selected_set: Set[str] = set()
    metadata: Dict[str, CELFEntry] = {}
    heap: List[Tuple[float, str, int]] = []

    for node in graph.nodes:
        cost = costs.get(node, 1.0)
        entry = CELFEntry(node=node, gain=float("inf"), cost=cost, last_eval=-1)
        metadata[node] = entry
        heapq.heappush(heap, (-_priority(entry, mode), node, entry.last_eval))

    current_spread = 0.0
    current_cost = 0.0

    while _has_feasible_extension(metadata, selected_set, current_cost, budget):
        while heap:
            neg_priority, node, snapshot = heapq.heappop(heap)
            entry = metadata[node]
            if node in selected_set:
                continue
            if snapshot != entry.last_eval:
                continue
            if current_cost + entry.cost > budget:
                continue

            if entry.last_eval == len(selected):
                selected.append(node)
                selected_set.add(node)
                current_cost += entry.cost
                current_spread += entry.gain
                break

            extended_spread = estimate_influence(
                graph, selected + [node], simulations, rng
            )
            entry.gain = extended_spread - current_spread
            entry.last_eval = len(selected)
            heapq.heappush(heap, (-_priority(entry, mode), node, entry.last_eval))
        else:
            break

    return selected, current_spread, current_cost


def run_celf(
    graph: InfluenceGraph,
    budget: float,
    simulations: int,
    costs: Optional[Mapping[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], float, float, str]:
    """Runs CELF by comparing the UC and CB variants from Algorithm 1."""

    if budget <= 0:
        raise ValueError("budget must be a positive value.")
    if simulations <= 0:
        raise ValueError("simulations must be a positive integer.")

    rng = rng or random.Random()
    base_costs = costs or {}

    state = rng.getstate()
    uc_seeds, uc_spread, uc_cost = _lazy_forward(
        graph, budget, simulations, base_costs, rng, mode="UC"
    )

    rng.setstate(state)
    cb_seeds, cb_spread, cb_cost = _lazy_forward(
        graph, budget, simulations, base_costs, rng, mode="CB"
    )

    if cb_spread > uc_spread:
        return cb_seeds, cb_spread, cb_cost, "CB"
    return uc_seeds, uc_spread, uc_cost, "UC"


def compute_online_bound(
    graph: InfluenceGraph,
    budget: float,
    simulations: int,
    costs: Optional[Mapping[str, float]] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Implements Algorithm 2 (GetBound) for the online bound R^."""

    if budget <= 0:
        raise ValueError("budget must be a positive value.")
    if simulations <= 0:
        raise ValueError("simulations must be a positive integer.")

    base_costs = costs or {}
    driver = rng or random.Random()
    work_rng = random.Random()
    work_rng.setstate(driver.getstate())

    deltas: Dict[str, float] = {}
    ratios: Dict[str, float] = {}

    for node in graph.nodes:
        cost = base_costs.get(node, 1.0)
        spread = estimate_influence(graph, [node], simulations, work_rng)
        deltas[node] = spread
        ratios[node] = spread / cost if cost > 0 else float("inf")

    total_cost = 0.0
    r_hat = 0.0
    available: Set[str] = set(graph.nodes)

    while True:
        feasible = [
            node
            for node in available
            if total_cost + base_costs.get(node, 1.0) <= budget
        ]
        if not feasible:
            break
        best = max(feasible, key=lambda node: ratios[node])
        total_cost += base_costs.get(best, 1.0)
        r_hat += deltas[best]
        available.remove(best)

    remaining = budget - total_cost
    if remaining > 0 and available:
        positive = [node for node in available if base_costs.get(node, 1.0) > 0]
        if positive:
            best = max(positive, key=lambda node: ratios[node])
            cost_best = base_costs.get(best, 1.0)
            if cost_best > 0:
                lam = min(1.0, remaining / cost_best)
                r_hat += lam * deltas[best]

    return r_hat

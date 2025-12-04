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


def fast_greedy_outbreak_detection(
    graph,
    cascades: dict,
    budget: int,
    objective: str = "DL",
    T_max: float = 100.0,
    rng: Optional[random.Random] = None,
    sample_size: int = 10000,
) -> list:
    """
    Approximate greedy outbreak detection for large datasets.

    Args:
        graph: InfluenceGraph
        cascades: meme_id -> [(site, time), ...]
        budget: number of sensors
        objective: 'DL', 'DT', or 'PA'
        T_max: max time for DT
        rng: optional random number generator
        sample_size: number of cascades to sample for evaluation

    Returns:
        List of selected nodes
    """
    from .objectives import compute_cascade_penalties

    rng = rng or random.Random()
    nodes = list(graph.nodes)
    selected = []
    selected_set = set()

    # Sample a subset of cascades for speed
    all_cascades = list(cascades.values())
    if sample_size > 0 and len(all_cascades) > sample_size:
        sampled_cascades = rng.sample(all_cascades, sample_size)
    else:
        sampled_cascades = all_cascades

    # Precompute per-node earliest detection for DL/DT/PA if possible
    def evaluate_candidate(candidate_node):
        total = 0.0
        for cascade in sampled_cascades:
            dl, dt, pa = compute_cascade_penalties(
                cascade, selected + [candidate_node], T_max=T_max
            )
            if objective == "DL":
                total += dl
            elif objective == "DT":
                total += dt
            elif objective == "PA":
                total += pa
        # Return average reduction (DL: fewer=better, DT/PA: smaller=better)
        return total / len(sampled_cascades)

    # Initialize current penalty with empty selection
    current_penalty = 0.0
    for cascade in sampled_cascades:
        dl, dt, pa = compute_cascade_penalties(cascade, [], T_max=T_max)
        if objective == "DL":
            current_penalty += dl
        elif objective == "DT":
            current_penalty += dt
        elif objective == "PA":
            current_penalty += pa
    current_penalty /= len(sampled_cascades)

    for _ in range(budget):
        best_gain = -float("inf")
        best_node = None

        for node in nodes:
            if node in selected_set:
                continue
            candidate_penalty = evaluate_candidate(node)
            gain = current_penalty - candidate_penalty
            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)
        selected_set.add(best_node)
        current_penalty -= best_gain

    return selected


def greedy_outbreak_detection(
    graph: InfluenceGraph,
    cascades: Dict[str, List[Tuple[str, float]]],
    budget: int,
    objective: str = "PA",
    T_max: float = 100.0,
    costs: Optional[Mapping[str, float]] = None,
) -> List[str]:
    """
    Greedy algorithm for outbreak detection (no lazy evaluation).

    Same as CELF but recomputes marginal gain for ALL nodes every iteration.
    """
    from .objectives import compute_cascade_penalties

    base_costs = costs or {}
    selected: List[str] = []
    selected_set: Set[str] = set()
    current_cost = 0.0
    budget_cost = float(budget)

    def evaluate_penalty(sites: List[str]) -> float:
        if not sites:
            return float("inf")
        penalties = []
        for cascade in cascades.values():
            # Pass T_max as a keyword argument
            dl, dt, pa = compute_cascade_penalties(cascade, sites, T_max=T_max)
            if objective == "DL":
                penalties.append(dl)
            elif objective == "DT":
                penalties.append(dt)
            elif objective == "PA":
                penalties.append(pa)
        return sum(penalties) / len(penalties) if penalties else float("inf")

    # Start with a baseline penalty using empty selection
    current_penalty = evaluate_penalty([])

    # Greedy selection: recompute gain for every candidate each iteration
    while current_cost < budget_cost:
        best_node = None
        best_gain = -float("inf")

        for node in graph.nodes:
            if node in selected_set:
                continue
            cost = base_costs.get(node, 1.0)
            if current_cost + cost > budget_cost:
                continue

            # Compute marginal gain
            new_penalty = evaluate_penalty(selected + [node])
            gain = current_penalty - new_penalty

            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None:
            break

        selected.append(best_node)
        selected_set.add(best_node)
        current_cost += base_costs.get(best_node, 1.0)
        current_penalty -= best_gain

    return selected


def celf_outbreak_detection(
    graph: InfluenceGraph,
    cascades: Dict[str, List[Tuple[str, float]]],
    budget: int,
    objective: str = "PA",  # "DL", "DT", or "PA"
    T_max: float = 100.0,
    costs: Optional[Mapping[str, float]] = None,
    sample_size: int = 0,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    CELF for outbreak detection using observed cascades.
    No Monte Carlo - evaluates exactly on all cascades.

    Args:
        graph: Influence graph
        cascades: Dict of meme_id -> [(site, time), ...]
        budget: Number of monitors to select
        objective: "DL" (detection likelihood), "DT" (detection time), or "PA" (population affected)
        T_max: Maximum time horizon
        costs: Optional node costs (default: 1.0 per node)

    Returns:
        List of selected monitoring sites
    """
    from .objectives import (
        compute_cascade_penalties,
    )  # Import here to avoid circular dependency

    base_costs = costs or {}
    selected: List[str] = []
    selected_set: Set[str] = set()

    # Sample a subset of cascades if requested
    all_cascades = list(cascades.values())
    if sample_size > 0 and len(all_cascades) > sample_size:
        rng = rng or random.Random()
        cascades_to_use = rng.sample(all_cascades, sample_size)
    else:
        cascades_to_use = all_cascades

    # Helper: compute average penalty on all cascades
    def evaluate_penalty(sites: List[str]) -> float:
        if not sites:
            return float("inf")

        penalties = []
        for cascade in cascades_to_use:
            dl, dt, pa = compute_cascade_penalties(cascade, sites, T_max=T_max)
            if objective == "DL":
                penalties.append(dl)
            elif objective == "DT":
                penalties.append(dt)
            elif objective == "PA":
                penalties.append(pa)
        return sum(penalties) / len(penalties) if penalties else float("inf")

    # Initialize: compute marginal gain for each node
    metadata: Dict[str, CELFEntry] = {}
    heap: List[Tuple[float, str, int]] = []

    baseline_penalty = evaluate_penalty(list(graph.nodes)[:1]) if graph.nodes else 0.0

    for node in graph.nodes:
        cost = base_costs.get(node, 1.0)
        entry = CELFEntry(node=node, gain=float("inf"), cost=cost, last_eval=-1)
        metadata[node] = entry
        heapq.heappush(heap, (-entry.gain, node, entry.last_eval))

    current_penalty = baseline_penalty
    current_cost = 0.0
    budget_cost = float(budget)

    while current_cost < budget_cost and heap:
        while heap:
            neg_gain, node, snapshot = heapq.heappop(heap)
            entry = metadata[node]

            if node in selected_set:
                continue
            if snapshot != entry.last_eval:
                continue
            if current_cost + entry.cost > budget_cost:
                continue

            # Check if this gain is current
            if entry.last_eval == len(selected):
                # This gain is fresh - select this node
                selected.append(node)
                selected_set.add(node)
                current_cost += entry.cost
                current_penalty -= entry.gain
                break

            # Recompute marginal gain (lazy evaluation)
            new_penalty = evaluate_penalty(selected + [node])
            entry.gain = current_penalty - new_penalty  # Reduction in penalty
            entry.last_eval = len(selected)
            heapq.heappush(heap, (-entry.gain, node, entry.last_eval))
        else:
            break

    return selected

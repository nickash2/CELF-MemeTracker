#!/usr/bin/env python3
"""
Comprehensive heuristics comparison for outbreak detection.

Compares CELF against multiple baseline algorithms:
- Random selection
- Greedy (standard)
- Degree centrality (in/out)
- PageRank
- Betweenness centrality
- Closeness centrality

Evaluates on three objectives:
- Detection Likelihood (DL): fraction of cascades detected
- Detection Time (DT): average time until detection
- Population Affected (PA): average population size at detection
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    InfluenceGraph,
    betweenness_centrality,
    closeness_centrality,
    degree_centrality,
    evaluate_solution_on_objectives,
    generate_random_cascades,
    greedy,
    load_graph_from_file,
    pagerank,
    plot_multi_objective_comparison,
    plot_normalized_comparison,
    plot_penalty_reduction_comparison,
    random_selection,
    run_celf,
)


def run_all_heuristics(
    graph: InfluenceGraph,
    budget: int,
    simulations: int,
    rng: random.Random,
) -> Dict[str, Tuple[List[str], float]]:
    """Run all heuristic algorithms and return their results.

    Args:
        graph: The influence graph.
        budget: Number of sensors to select.
        simulations: Monte Carlo simulations for spread estimation.
        rng: Random number generator.

    Returns:
        Dict mapping algorithm name to (selected_seeds, runtime_seconds).
    """
    results = {}

    print("\nRunning heuristics...")

    # 1. CELF (Cost-Effective Lazy Forward)
    print("  [1/8] CELF...")
    start = time.time()
    celf_seeds, _, _, _ = run_celf(
        graph, budget=float(budget), simulations=simulations, rng=rng
    )
    results["CELF"] = (celf_seeds, time.time() - start)

    # 2. Standard Greedy
    print("  [2/8] Greedy...")
    start = time.time()
    greedy_seeds, _, _ = greedy(graph, budget, simulations, rng=rng)
    results["Greedy"] = (greedy_seeds, time.time() - start)

    # 3. Out-Degree (High Degree First)
    print("  [3/8] Out-Degree...")
    start = time.time()
    degree_out_seeds, _ = degree_centrality(graph, budget, mode="out")
    results["Out-Degree"] = (degree_out_seeds, time.time() - start)

    # 4. In-Degree
    print("  [4/8] In-Degree...")
    start = time.time()
    degree_in_seeds, _ = degree_centrality(graph, budget, mode="in")
    results["In-Degree"] = (degree_in_seeds, time.time() - start)

    # 5. PageRank
    print("  [5/8] PageRank...")
    start = time.time()
    pagerank_seeds, _ = pagerank(graph, budget)
    results["PageRank"] = (pagerank_seeds, time.time() - start)

    # 6. Betweenness Centrality
    print("  [6/8] Betweenness...")
    start = time.time()
    betweenness_seeds, _ = betweenness_centrality(graph, budget, k_samples=100)
    results["Betweenness"] = (betweenness_seeds, time.time() - start)

    # 7. Closeness Centrality
    print("  [7/8] Closeness...")
    start = time.time()
    closeness_seeds, _ = closeness_centrality(graph, budget)
    results["Closeness"] = (closeness_seeds, time.time() - start)

    # 8. Random Selection
    print("  [8/8] Random...")
    start = time.time()
    random_seeds, _ = random_selection(graph, budget, rng=rng)
    results["Random"] = (random_seeds, time.time() - start)

    return results


def evaluate_on_objectives(
    graph: InfluenceGraph,
    heuristic_results: Dict[str, Tuple[List[str], float]],
    num_cascades: int,
    rng: random.Random,
    max_time: float = 100.0,
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """Evaluate all heuristics on all three objectives.

    Args:
        graph: The influence graph.
        heuristic_results: Results from run_all_heuristics.
        num_cascades: Number of cascades for evaluation.
        rng: Random number generator.
        max_time: Maximum time horizon for simulations.

    Returns:
        Nested dict: {algorithm: {objective: [(num_sensors, value)]}}
    """
    print("\nEvaluating on objectives...")

    # Generate cascades once for consistent evaluation
    cascades = generate_random_cascades(graph, num_cascades, rng, max_time)
    print(f"  Generated {len(cascades)} cascades")

    results: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for alg_name, (sensors, _) in heuristic_results.items():
        print(f"  Evaluating {alg_name}...")

        num_sensors = len(sensors)

        # Evaluate on all three objectives
        metrics = evaluate_solution_on_objectives(
            graph, sensors, num_cascades=num_cascades, rng=rng, max_time=max_time
        )

        # Store results
        results[alg_name] = {
            "DL": [(num_sensors, metrics["detection_likelihood"])],
            "DT": [
                (
                    num_sensors,
                    1.0 - (metrics["detection_time"] / max_time),
                )
            ],  # Convert to reduction
            "PA": [(num_sensors, metrics["population_affected"])],
        }

    return results


def run_budget_sweep(
    graph: InfluenceGraph,
    budgets: List[int],
    simulations: int,
    num_cascades: int,
    rng: random.Random,
    max_time: float = 100.0,
) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """Run all heuristics across multiple budgets.

    Args:
        graph: The influence graph.
        budgets: List of budget values (number of sensors).
        simulations: Monte Carlo simulations for spread estimation.
        num_cascades: Number of cascades for evaluation.
        rng: Random number generator.
        max_time: Maximum time horizon.

    Returns:
        Nested dict: {algorithm: {objective: [(num_sensors, value)]}}
    """
    # Generate cascades once for all evaluations
    print(f"\nGenerating {num_cascades} cascades for evaluation...")
    cascades = generate_random_cascades(graph, num_cascades, rng, max_time)
    print(f"  Generated {len(cascades)} valid cascades")

    results: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for budget in budgets:
        print(f"\n{'=' * 70}")
        print(f"Budget = {budget} sensors")
        print(f"{'=' * 70}")

        # Run all heuristics
        heuristic_results = run_all_heuristics(graph, budget, simulations, rng)

        # Evaluate each algorithm on objectives
        for alg_name, (sensors, runtime) in heuristic_results.items():
            if alg_name not in results:
                results[alg_name] = {"DL": [], "DT": [], "PA": []}

            # Evaluate on all objectives using the pre-generated cascades
            metrics = evaluate_solution_on_objectives(
                graph, sensors, num_cascades=len(cascades), rng=rng, max_time=max_time
            )

            # Store results
            results[alg_name]["DL"].append((budget, metrics["detection_likelihood"]))
            results[alg_name]["DT"].append(
                (budget, 1.0 - (metrics["detection_time"] / max_time))
            )
            results[alg_name]["PA"].append((budget, metrics["population_affected"]))

            print(
                f"  {alg_name:15s} - DL: {metrics['detection_likelihood']:.3f}, "
                f"DT: {metrics['detection_time']:.2f}, PA: {metrics['population_affected']:.2f}, "
                f"Time: {runtime:.3f}s"
            )

    return results


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(
        description="Compare CELF against baseline heuristics on outbreak detection objectives."
    )
    parser.add_argument("--graph", required=True, help="Path to edge list file")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="List of budget values (number of sensors)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Monte Carlo simulations for spread estimation (default: 1000)",
    )
    parser.add_argument(
        "--num-cascades",
        type=int,
        default=500,
        help="Number of cascades for objective evaluation (default: 500)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=100.0,
        help="Maximum time horizon for simulations (default: 100.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/heuristics_comparison",
        help="Output directory for plots (default: results/heuristics_comparison)",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Field delimiter (default: whitespace). Use '\\t' for tabs.",
    )

    args = parser.parse_args()

    # Parse delimiter
    delimiter = (
        bytes(args.delimiter, "utf-8").decode("unicode_escape")
        if args.delimiter
        else None
    )

    # Initialize RNG
    rng = random.Random(args.seed)

    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = load_graph_from_file(args.graph, delimiter=delimiter)
    print(f"  Loaded {len(graph.nodes)} nodes")

    if len(graph.nodes) == 0:
        print("Error: Graph is empty!")
        return

    # Run budget sweep
    results = run_budget_sweep(
        graph,
        args.budgets,
        args.simulations,
        args.num_cascades,
        rng,
        args.max_time,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("Generating visualizations...")
    print(f"{'=' * 70}")

    # Plot 1: Multi-objective comparison (all three in one figure)
    plot_multi_objective_comparison(
        results, output_path=str(output_dir / "multi_objective_comparison.png")
    )

    # Plot 2-4: Individual objectives
    for obj_key, obj_name, ylabel in [
        ("DL", "Detection Likelihood", "Detection Rate"),
        ("DT", "Detection Time Reduction", "Time Reduction (normalized)"),
        ("PA", "Population Affected", "Average Population Size"),
    ]:
        obj_data = {alg: data[obj_key] for alg, data in results.items()}

        plot_penalty_reduction_comparison(
            obj_data,
            output_path=str(output_dir / f"{obj_key.lower()}_comparison.png"),
            title=f"{obj_name} vs Number of Sensors",
            ylabel=ylabel,
            objective_name=obj_name,
        )

    # Plot 5-7: Normalized comparisons
    for obj_key, obj_name in [
        ("DL", "Detection Likelihood"),
        ("DT", "Detection Time"),
        ("PA", "Population Affected"),
    ]:
        obj_data = {alg: data[obj_key] for alg, data in results.items()}

        plot_normalized_comparison(
            obj_data,
            output_path=str(output_dir / f"{obj_key.lower()}_normalized.png"),
            title=f"Normalized {obj_name} Comparison",
            objective_name=obj_name,
        )

    print(f"\nAll plots saved to {output_dir}/")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

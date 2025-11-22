#!/usr/bin/env python3
"""
Quick demo of heuristics comparison on toy dataset.

Shows how to:
1. Load a graph
2. Run multiple heuristics
3. Evaluate on detection objectives
4. Generate comparison plots
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    betweenness_centrality,
    degree_centrality,
    evaluate_solution_on_objectives,
    generate_random_cascades,
    load_graph_from_file,
    pagerank,
    plot_multi_objective_comparison,
    random_selection,
    run_celf,
)


def main():
    """Run quick comparison on toy graph."""
    print("=" * 70)
    print("HEURISTICS COMPARISON DEMO")
    print("=" * 70)

    # Load toy graph
    print("\nLoading toy graph...")
    graph = load_graph_from_file("data/toy_edges.txt", delimiter="\t")
    print(f"  Nodes: {len(graph.nodes)}")

    # Setup
    budget = 3
    simulations = 1000
    num_cascades = 200
    rng = random.Random(42)

    print("\nConfiguration:")
    print(f"  Budget (websites to monitor): {budget}")
    print(f"  Simulations: {simulations}")
    print(f"  Cascades: {num_cascades}")

    # Run heuristics
    print(f"\n{'=' * 70}")
    print(f"Running heuristics with budget = {budget}...")
    print(f"{'=' * 70}")

    algorithms = []

    # CELF
    print("\n[1/6] CELF...")
    celf_seeds, _, _, _ = run_celf(
        graph, budget=float(budget), simulations=simulations, rng=rng
    )
    algorithms.append(("CELF", celf_seeds))
    print(f"  Seeds: {celf_seeds}")

    # Out-Degree
    print("\n[2/6] Out-Degree...")
    out_deg_seeds, _ = degree_centrality(graph, budget, mode="out")
    algorithms.append(("Out-Degree", out_deg_seeds))
    print(f"  Seeds: {out_deg_seeds}")

    # In-Degree
    print("\n[3/6] In-Degree...")
    in_deg_seeds, _ = degree_centrality(graph, budget, mode="in")
    algorithms.append(("In-Degree", in_deg_seeds))
    print(f"  Seeds: {in_deg_seeds}")

    # PageRank
    print("\n[4/6] PageRank...")
    pr_seeds, _ = pagerank(graph, budget)
    algorithms.append(("PageRank", pr_seeds))
    print(f"  Seeds: {pr_seeds}")

    # Betweenness
    print("\n[5/6] Betweenness...")
    btw_seeds, _ = betweenness_centrality(graph, budget, k_samples=50)
    algorithms.append(("Betweenness", btw_seeds))
    print(f"  Seeds: {btw_seeds}")

    # Random
    print("\n[6/6] Random...")
    rand_seeds, _ = random_selection(graph, budget, rng=rng)
    algorithms.append(("Random", rand_seeds))
    print(f"  Seeds: {rand_seeds}")

    # Generate cascades
    print(f"\n{'=' * 70}")
    print(f"Generating {num_cascades} cascades for evaluation...")
    print(f"{'=' * 70}")
    cascades = generate_random_cascades(graph, num_cascades, rng)
    print(f"  Generated {len(cascades)} valid cascades")

    # Evaluate all algorithms
    print(f"\n{'=' * 70}")
    print("Evaluating on objectives...")
    print(f"{'=' * 70}")

    results = {}
    for alg_name, seeds in algorithms:
        print(f"\n{alg_name}:")
        metrics = evaluate_solution_on_objectives(graph, seeds, num_cascades, rng)

        print(f"  Detection Likelihood: {metrics['detection_likelihood']:.3f}")
        print(f"  Detection Time:       {metrics['detection_time']:.2f}")
        print(f"  Population Affected:  {metrics['population_affected']:.2f}")

        results[alg_name] = {
            "DL": [(budget, metrics["detection_likelihood"])],
            "DT": [(budget, 1.0 - (metrics["detection_time"] / 100.0))],
            "PA": [(budget, metrics["population_affected"])],
        }

    # Generate visualization
    print(f"\n{'=' * 70}")
    print("Generating comparison plot...")
    print(f"{'=' * 70}")

    output_path = "results/figures/demo_heuristics_comparison.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plot_multi_objective_comparison(results, output_path=output_path)

    print(f"\n{'=' * 70}")
    print("DEMO COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nComparison plot saved to: {output_path}")
    print("\nTo run a full budget sweep, use:")
    print(
        "  python examples/heuristics_comparison.py --graph data/toy_edges.txt --budgets 1 2 3 4 5"
    )


if __name__ == "__main__":
    main()

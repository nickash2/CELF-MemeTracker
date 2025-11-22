#!/usr/bin/env python3
"""
Quick demo of heuristics comparison on MemeTracker dataset.

Uses small subset of data for fast demonstration.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    betweenness_centrality,
    build_graph_from_memetracker,
    convert_memetracker_cascades_to_events,
    degree_centrality,
    evaluate_detection_likelihood,
    evaluate_detection_time,
    evaluate_population_affected,
    pagerank,
    plot_multi_objective_comparison,
    random_selection,
    run_celf,
)


def main():
    """Run quick comparison on MemeTracker data."""
    print("=" * 70)
    print("MEMETRACKER HEURISTICS DEMO")
    print("=" * 70)

    # Load MemeTracker data (small subset)
    print("\nLoading MemeTracker data...")
    graph, cascades_dict = build_graph_from_memetracker(
        path="data/quotes_2008-08.txt",
        # Use the same defaults as the full pipeline so we extract valid cascades
        # (memetracker_pipeline uses top_memes=50 and larger max_documents).
        top_memes=50,
        min_prob=0.01,
        max_documents=10000,
    )
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Cascades: {len(cascades_dict)}")

    if not graph.nodes or not cascades_dict:
        print("\nError: Could not load data. Check file path.")
        return

    # Convert to events
    print("\nConverting cascades to events...")
    cascade_events = convert_memetracker_cascades_to_events(cascades_dict)
    print(f"  Valid events: {len(cascade_events)}")

    if not cascade_events:
        print("\nError: No valid cascade events.")
        return

    # Setup
    budget = 3
    simulations = 500
    rng = random.Random(42)

    print("\nConfiguration:")
    print(f"  Budget (websites to monitor): {budget}")
    print(f"  Simulations: {simulations}")
    print(f"  Cascades for eval: {len(cascade_events)}")

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

    # Evaluate all algorithms
    print(f"\n{'=' * 70}")
    print("Evaluating on objectives...")
    print(f"{'=' * 70}")

    max_time = 100.0
    results = {}

    for alg_name, seeds in algorithms:
        print(f"\n{alg_name}:")

        # Evaluate on objectives
        dl = evaluate_detection_likelihood(cascade_events, seeds)
        dt = evaluate_detection_time(cascade_events, seeds, max_time)
        pa = evaluate_population_affected(cascade_events, seeds)

        print(f"  Detection Likelihood: {dl:.3f}")
        print(f"  Detection Time:       {dt:.2f}")
        print(f"  Population Affected:  {pa:.2f}")

        results[alg_name] = {
            "DL": [(budget, dl)],
            "DT": [(budget, 1.0 - (dt / max_time))],
            "PA": [(budget, pa)],
        }

    # Generate visualization
    print(f"\n{'=' * 70}")
    print("Generating comparison plot...")
    print(f"{'=' * 70}")

    output_path = "results/figures/demo_memetracker_comparison.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plot_multi_objective_comparison(results, output_path=output_path)

    print(f"\n{'=' * 70}")
    print("DEMO COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nComparison plot saved to: {output_path}")
    print("\nTo run a full budget sweep, use:")
    print("  python examples/memetracker_heuristics_comparison.py \\")
    print("    --input data/quotes_2008-08.txt \\")
    print("    --budgets 1 2 3 4 5 \\")
    print("    --max-docs 10000")


if __name__ == "__main__":
    main()

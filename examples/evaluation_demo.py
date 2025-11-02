#!/usr/bin/env python3
"""
Example script demonstrating CELF evaluation and visualization capabilities.

This script shows how to:
1. Run CELF with performance tracking
2. Compare against online bounds
3. Generate all visualization types
4. Create summary reports
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    CELFResult,
    HeuristicComparison,
    PerformanceTracker,
    compute_online_bound,
    create_summary_report,
    load_graph_from_file,
    plot_bounds_comparison,
    plot_marginal_gains,
    plot_runtime_comparison,
    plot_spread_comparison,
    plot_spread_vs_budget,
    run_celf,
    save_results,
)


def main():
    """Run comprehensive evaluation on toy graph."""
    print("CELF Evaluation Demo")
    print("=" * 80)

    # Load toy graph
    graph = load_graph_from_file(
        path="data/toy_edges.txt",
        default_prob=0.1,
        delimiter="\t",
    )
    print(f"\nLoaded graph with {len(graph.nodes)} nodes")

    rng = random.Random(42)
    simulations = 5000

    # ========================================================================
    # 1. Sweep budget and track results
    # ========================================================================
    print("\n1. Running CELF with different budgets...")
    budgets = [1, 2, 3, 4, 5]
    results = []

    for budget in budgets:
        tracker = PerformanceTracker()
        tracker.start()

        seeds, spread, total_cost, mode = run_celf(
            graph, budget=float(budget), simulations=simulations, rng=rng
        )

        tracker.stop()

        # Compute online bound
        online_bound = compute_online_bound(
            graph, budget=float(budget), simulations=simulations, rng=rng
        )

        result = CELFResult(
            seeds=seeds,
            spread=spread,
            total_cost=total_cost,
            mode=mode,
            budget=float(budget),
            simulations=simulations,
            runtime_seconds=tracker.elapsed(),
            num_nodes=len(graph.nodes),
            online_bound=online_bound,
        )
        results.append(result)

        print(
            f"  Budget={budget}: spread={spread:.2f}, bound={online_bound:.2f}, "
            f"ratio={result.approximation_ratio():.3f}, time={tracker.elapsed():.3f}s"
        )

    # ========================================================================
    # 2. Generate visualizations
    # ========================================================================
    print("\n2. Generating visualizations...")

    # Plot spread vs budget with bounds
    plot_spread_vs_budget(
        results,
        output_path="results/figures/demo_spread_vs_budget.png",
        title="CELF Performance Across Budgets",
    )

    # Plot bounds comparison for middle budget
    mid_result = results[len(results) // 2]
    plot_bounds_comparison(
        mid_result,
        output_path="results/figures/demo_bounds_comparison.png",
        title=f"CELF vs Bounds (Budget={mid_result.budget})",
    )

    # ========================================================================
    # 3. Simulate heuristic comparisons (placeholder for future work)
    # ========================================================================
    print("\n3. Simulating heuristic comparisons...")

    # In reality, you would implement different algorithms here
    # For now, we'll create mock comparisons
    comparisons = [
        HeuristicComparison(
            algorithm_name="CELF",
            seeds=mid_result.seeds,
            spread=mid_result.spread,
            runtime_seconds=mid_result.runtime_seconds,
            total_cost=mid_result.total_cost,
            budget=mid_result.budget,
        ),
        HeuristicComparison(
            algorithm_name="Greedy (simulated)",
            seeds=mid_result.seeds[::-1],  # Mock: reversed order
            spread=mid_result.spread * 0.85,  # Mock: 85% of CELF
            runtime_seconds=mid_result.runtime_seconds * 1.5,  # Mock: slower
            total_cost=mid_result.total_cost,
            budget=mid_result.budget,
        ),
        HeuristicComparison(
            algorithm_name="Degree-based (simulated)",
            seeds=["B", "D"],  # Mock: high-degree nodes
            spread=mid_result.spread * 0.70,  # Mock: 70% of CELF
            runtime_seconds=mid_result.runtime_seconds * 0.1,  # Mock: much faster
            total_cost=mid_result.total_cost,
            budget=mid_result.budget,
        ),
    ]

    plot_runtime_comparison(
        comparisons,
        output_path="results/figures/demo_runtime_comparison.png",
        title="Algorithm Runtime Comparison",
    )

    plot_spread_comparison(
        comparisons,
        output_path="results/figures/demo_spread_comparison.png",
        title="Influence Spread vs Runtime",
    )

    # ========================================================================
    # 4. Plot marginal gains for best result
    # ========================================================================
    print("\n4. Analyzing marginal gains...")

    best_result = max(results, key=lambda r: r.spread)

    # Recompute to get per-seed marginal gains
    tracker = PerformanceTracker()
    tracker.start()
    marginal_spreads = []
    selected = []

    for seed in best_result.seeds:
        selected.append(seed)
        spread_with = compute_online_bound(
            graph,
            budget=len(selected),
            simulations=1000,
            costs={n: 1.0 for n in graph.nodes},
            rng=rng,
        )
        spread_without = (
            compute_online_bound(
                graph,
                budget=len(selected) - 1,
                simulations=1000,
                costs={n: 1.0 for n in graph.nodes},
                rng=rng,
            )
            if len(selected) > 1
            else 0.0
        )
        marginal_spreads.append(spread_with - spread_without)

    plot_marginal_gains(
        best_result.seeds,
        marginal_spreads,
        output_path="results/figures/demo_marginal_gains.png",
        title="Marginal Influence Gains",
    )

    # ========================================================================
    # 5. Save results and create summary
    # ========================================================================
    print("\n5. Saving results...")

    for i, result in enumerate(results):
        save_results(result, f"results/demo_result_budget{result.budget:.0f}.json")

    create_summary_report(
        results,
        comparisons=comparisons,
        output_path="results/demo_summary_report.txt",
    )

    print("\n" + "=" * 80)
    print("Demo complete! Check results/ directory for outputs.")
    print("=" * 80)


if __name__ == "__main__":
    main()

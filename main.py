"""
Main entry point for running CELF influence maximization.

Supports loading graphs from edge lists, optional cost files, and computing
both optimal seed sets and online bounds with performance tracking.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Optional, Sequence

from src import (
    CELFResult,
    PerformanceTracker,
    compute_online_bound,
    load_costs_from_file,
    load_graph_from_file,
    plot_bounds_comparison,
    run_celf,
    save_results,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CELF for influence maximization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Unit-cost seed selection (k=5)
  python main.py --graph data/toy_edges.txt --k 5 --simulations 5000 --seed 42

  # Budget-constrained with costs
  python main.py --graph data/edges.txt --costs data/costs.txt --budget 100 --simulations 10000

  # Compute online bound
  python main.py --graph data/edges.txt --k 10 --compute-bound --simulations 5000
        """,
    )
    parser.add_argument(
        "--graph", required=True, help="Path to directed edge list file."
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="Total cost budget B (defaults to --k when omitted).",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Seed budget (interpreted as unit-cost budget when --budget not provided).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Monte Carlo samples per marginal gain estimate (default: 1000).",
    )
    parser.add_argument(
        "--default-prob",
        type=float,
        default=0.1,
        help="Fallback edge probability when not specified (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Field delimiter (default: whitespace). Use '\\t' for tabs.",
    )
    parser.add_argument(
        "--skip-header",
        action="store_true",
        help="Skip first line of edge list file.",
    )
    parser.add_argument(
        "--costs",
        help="Optional two-column file 'node cost' for sensor costs c(s).",
    )
    parser.add_argument(
        "--compute-bound",
        action="store_true",
        help="Compute online upper bound R^ using Algorithm 2.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostic information.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Parse delimiter escapes
    delimiter = (
        bytes(args.delimiter, "utf-8").decode("unicode_escape")
        if args.delimiter
        else None
    )

    # Initialize RNG
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    # Validate budget arguments
    if args.budget is not None and args.k is not None:
        raise ValueError("Provide either --budget or --k, not both.")
    if args.budget is not None:
        budget = float(args.budget)
    elif args.k is not None:
        budget = float(args.k)
    else:
        raise ValueError("A budget must be specified via --budget or --k.")

    # Load graph
    if args.verbose:
        print(f"Loading graph from {args.graph}...")

    graph = load_graph_from_file(
        path=args.graph,
        default_prob=args.default_prob,
        delimiter=delimiter,
        skip_header=args.skip_header,
    )

    if not graph.nodes:
        raise ValueError("The graph is empty; provide at least one edge.")

    if args.verbose:
        print(f"  Loaded {len(graph.nodes)} nodes")

    # Load costs if provided
    costs = load_costs_from_file(args.costs) if args.costs else {}
    if args.verbose and costs:
        print(f"  Loaded costs for {len(costs)} nodes")

    # Track performance
    tracker = PerformanceTracker()
    tracker.start()

    # Run CELF
    if args.verbose:
        print(f"\nRunning CELF with budget={budget}, simulations={args.simulations}...")

    seeds, spread, total_cost, mode = run_celf(
        graph,
        budget=budget,
        simulations=args.simulations,
        costs=costs,
        rng=rng,
    )

    tracker.stop()

    # Print results
    print("\n" + "=" * 60)
    print("CELF RESULTS")
    print("=" * 60)
    print(f"Selected seeds ({mode}):", ", ".join(seeds))
    print(f"Total cost: {total_cost:.3f} / {budget:.3f}")
    print(f"Estimated spread: {spread:.3f}")
    print(f"Runtime: {tracker.elapsed():.3f}s")

    # Compute bound if requested
    online_bound = None
    if args.compute_bound:
        if args.verbose:
            print("\nComputing online bound...")

        online_bound = compute_online_bound(
            graph,
            budget=budget,
            simulations=args.simulations,
            costs=costs,
            rng=rng,
        )
        print(f"Online bound R^: {online_bound:.3f}")
        if spread > 0 and online_bound > 0:
            print(f"Approximation ratio: {spread / online_bound:.3f}")

    print("=" * 60)

    # Save results
    result = CELFResult(
        seeds=seeds,
        spread=spread,
        total_cost=total_cost,
        mode=mode,
        budget=budget,
        simulations=args.simulations,
        runtime_seconds=tracker.elapsed(),
        num_nodes=len(graph.nodes),
        online_bound=online_bound,
    )

    # Generate outputs
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(result, f"results/celf_result_{timestamp}.json")

    if args.compute_bound and online_bound is not None:
        plot_bounds_comparison(
            result,
            output_path=f"results/figures/bounds_comparison_{timestamp}.png",
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Example script for processing MemeTracker data and running CELF.

This demonstrates how to:
1. Parse MemeTracker formatted files (.txt or .txt.gz)
2. Extract meme cascades from the documents
3. Build an influence graph from cascades
4. Run CELF to find optimal seed nodes
5. Visualize results
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    CELFResult,
    PerformanceTracker,
    build_graph_from_memetracker,
    compute_online_bound,
    plot_bounds_comparison,
    run_celf,
    save_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Process MemeTracker data and run CELF algorithm"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to MemeTracker file (quotes_YYYY-MM.txt or .txt.gz)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=10000,
        help="Maximum documents to parse (default: 10000)",
    )
    parser.add_argument(
        "--top-memes",
        type=int,
        default=50,
        help="Number of top memes to track (default: 50)",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of seed nodes to select",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="Monte Carlo simulations (default: 1000)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=0.01,
        help="Minimum edge probability threshold (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-prefix",
        default="memetracker",
        help="Prefix for output files (default: memetracker)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MEMETRACKER CELF PIPELINE")
    print("=" * 80)

    # Build graph from MemeTracker data
    print(f"\n1. Building influence graph from {args.input}")
    print(f"   Max documents: {args.max_docs}")
    print(f"   Top memes: {args.top_memes}")
    print(f"   Min probability: {args.min_prob}\n")

    graph, cascades = build_graph_from_memetracker(
        path=args.input,
        top_memes=args.top_memes,
        min_prob=args.min_prob,
        max_documents=args.max_docs,
    )

    print(f"\n2. Graph statistics:")
    print(f"   Nodes (sites): {len(graph.nodes)}")
    print(f"   Cascades extracted: {len(cascades)}")
    print(f"   Top 5 memes:")
    for i, (meme, cascade) in enumerate(list(cascades.items())[:5], 1):
        print(f"     {i}. '{meme[:50]}...' - {len(cascade)} sites")

    if not graph.nodes:
        print("\nError: Empty graph. Try increasing --max-docs or --top-memes")
        return

    # Run CELF
    print(f"\n3. Running CELF (k={args.k}, simulations={args.simulations})")
    tracker = PerformanceTracker()
    tracker.start()

    rng = random.Random(args.seed)
    seeds, spread, total_cost, mode = run_celf(
        graph,
        budget=float(args.k),
        simulations=args.simulations,
        rng=rng,
    )

    tracker.stop()

    print(f"\n4. CELF Results:")
    print(f"   Variant: {mode}")
    print(f"   Selected sites: {', '.join(seeds)}")
    print(f"   Expected spread: {spread:.3f}")
    print(f"   Runtime: {tracker.elapsed():.3f}s")

    # Compute online bound
    print(f"\n5. Computing online bound...")
    online_bound = compute_online_bound(
        graph,
        budget=float(args.k),
        simulations=args.simulations,
        rng=rng,
    )

    ratio = spread / online_bound if online_bound > 0 else 0.0
    print(f"   Online bound R^: {online_bound:.3f}")
    print(f"   Approximation ratio: {ratio:.3f}")

    # Save results
    print(f"\n6. Saving results...")
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")

    result = CELFResult(
        seeds=seeds,
        spread=spread,
        total_cost=total_cost,
        mode=mode,
        budget=float(args.k),
        simulations=args.simulations,
        runtime_seconds=tracker.elapsed(),
        num_nodes=len(graph.nodes),
        online_bound=online_bound,
    )

    output_json = f"results/{args.output_prefix}_result_{timestamp}.json"
    save_results(result, output_json)

    output_fig = f"results/figures/{args.output_prefix}_bounds_{timestamp}.png"
    plot_bounds_comparison(
        result,
        output_path=output_fig,
        title=f"CELF on MemeTracker ({Path(args.input).name})",
    )

    print(f"\n7. Top cascades for selected seeds:")
    for seed in seeds[:3]:  # Show top 3
        seed_memes = [
            m for m, cas in cascades.items() if any(s == seed for s, _ in cas)
        ]
        if seed_memes:
            print(f"   {seed}: {len(seed_memes)} memes")
            print(f"     Example: '{seed_memes[0][:60]}...'")

    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

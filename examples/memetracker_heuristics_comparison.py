#!/usr/bin/env python3
"""
Comprehensive heuristics comparison on MemeTracker dataset.

Compares CELF with baseline heuristics on real cascade data:
- Random selection
- Degree centrality (in/out)
- PageRank
- Betweenness centrality
- Greedy

Evaluates solutions on three objectives:
- Detection Likelihood (DL): fraction of cascades detected
- Detection Time (DT): average time until detection
- Population Affected (PA): average nodes infected before detection

Generates comparison plots showing penalty reduction vs. number of sensors.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    betweenness_centrality,
    build_graph_from_memetracker,
    celf_outbreak_detection,
    convert_memetracker_cascades_to_events,
    degree_centrality,
    evaluate_detection_likelihood,
    evaluate_detection_time,
    evaluate_population_affected,
    greedy_outbreak_detection,
    pagerank,
    plot_multi_objective_comparison,
    random_selection,
)
from src.celfpp import celfpp_outbreak_detection


def main():
    parser = argparse.ArgumentParser(
        description="Compare heuristics on MemeTracker data"
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
        help="Maximum documents to parse (default: 10000, 0 or -1 for no limit)",
    )
    parser.add_argument(
        "--top-memes",
        type=int,
        default=50,
        help="Number of top memes to track (default: 50, 0 or -1 for no limit)",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Budgets (number of sensors) to test (default: 1 2 3 4 5)",
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
        "--output",
        default="results/figures/memetracker_comparison.png",
        help="Output plot path (default: results/figures/memetracker_comparison.png)",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=100.0,
        help="Maximum time horizon for DT objective (default: 100.0)",
    )
    parser.add_argument(
        "--objective",
        choices=["DL", "DT", "PA"],
        default="DL",
        help="Objective for CELF detection (default: DL)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MEMETRACKER HEURISTICS COMPARISON")
    print("=" * 80)

    # Build graph from MemeTracker data
    print(f"\n1. Building influence graph from {args.input}")
    max_docs_display = "ALL" if args.max_docs in (0, -1) else str(args.max_docs)
    print(f"   Max documents: {max_docs_display}")
    top_memes_display = "ALL" if args.top_memes in (0, -1) else str(args.top_memes)
    print(f"   Top memes: {top_memes_display}")
    print(f"   Min probability: {args.min_prob}\n")

    # Interpret 0 or -1 as no limit
    max_docs = None if args.max_docs in (0, -1) else args.max_docs
    # Interpret 0 or -1 as no limit
    top_memes = None if args.top_memes in (0, -1) else args.top_memes
    graph, cascades_dict = build_graph_from_memetracker(
        path=args.input,
        top_memes=top_memes,
        min_prob=args.min_prob,
        max_documents=max_docs,
    )

    print("\n2. Graph statistics:")
    print(f"   Nodes (sites): {len(graph.nodes)}")
    print(f"   Cascades: {len(cascades_dict)}")

    if not graph.nodes:
        print("\nError: Empty graph. Try increasing --max-docs or --top-memes")
        return

    if not cascades_dict:
        print("\nError: No cascades extracted. Check data format.")
        return

    # Convert cascades to events
    print("\n3. Converting cascades to events...")
    cascade_events = convert_memetracker_cascades_to_events(cascades_dict)
    print(f"   Valid cascade events: {len(cascade_events)}")

    if not cascade_events:
        print("\nError: No valid cascade events. Need cascades with >= 2 nodes.")
        return

    # Setup
    rng = random.Random(args.seed)
    budgets = sorted(args.budgets)

    # Compute baseline PA (no monitors = worst case)
    baseline_pa = evaluate_population_affected(cascade_events, [])

    print("\n4. Configuration:")
    print(f"   Budgets: {budgets}")
    print(f"   Max time: {args.max_time}")
    print(f"   Baseline PA (no monitors): {baseline_pa:.1f}")

    # Algorithms to compare
    algorithms = {
        # Outbreak detection algorithms (use exact cascade evaluation)
        "CELF": lambda g, k: celf_outbreak_detection(
            g, cascades_dict, k, objective=args.objective, T_max=args.max_time
        ),
        "CELF++": lambda g, k: celfpp_outbreak_detection(
            g, cascades_dict, k, objective=args.objective, T_max=args.max_time
        ),
        "Greedy": lambda g, k: greedy_outbreak_detection(
            g, cascades_dict, k, objective=args.objective, T_max=args.max_time
        ),
        # Heuristic algorithms (structure-based, no cascades needed)
        "Out-Degree": lambda g, k: degree_centrality(g, k, mode="out")[0],
        "In-Degree": lambda g, k: degree_centrality(g, k, mode="in")[0],
        "PageRank": lambda g, k: pagerank(g, k)[0],
        "Betweenness": lambda g, k: betweenness_centrality(g, k, k_samples=100)[0],
        "Random": lambda g, k: random_selection(g, k, rng=rng)[0],
    }

    # Run comparison
    print(f"\n{'=' * 80}")
    print("Running heuristics comparison...")
    print(f"{'=' * 80}\n")

    results = {}

    for alg_name, alg_func in algorithms.items():
        print(f"\n{alg_name}:")
        print("-" * 40)

        results[alg_name] = {"DL": [], "DT": [], "PA": []}

        for budget in budgets:
            print(f"  Budget = {budget}: ", end="", flush=True)

            try:
                # Run algorithm
                seeds = alg_func(graph, budget)
                print(f"seeds={seeds[:3]}{'...' if len(seeds) > 3 else ''}", end=" ")

                # Evaluate on objectives
                dl = evaluate_detection_likelihood(cascade_events, seeds)
                dt = evaluate_detection_time(cascade_events, seeds, args.max_time)
                pa = evaluate_population_affected(cascade_events, seeds)

                # Store results
                results[alg_name]["DL"].append((budget, dl))
                # Normalize DT to reduction (0 to 1)
                dt_reduction = 1.0 - (dt / args.max_time)
                results[alg_name]["DT"].append((budget, dt_reduction))
                # Compute PA reduction: (baseline - current) / baseline
                pa_reduction = (
                    (baseline_pa - pa) / baseline_pa if baseline_pa > 0 else 0.0
                )
                results[alg_name]["PA"].append((budget, pa_reduction))

                print(f"DL={dl:.3f} DT={dt:.1f} PA={pa:.1f}")

            except Exception as e:
                print(f"ERROR: {e}")
                # Fill with zeros to maintain plot structure
                results[alg_name]["DL"].append((budget, 0.0))
                results[alg_name]["DT"].append((budget, 0.0))
                results[alg_name]["PA"].append((budget, 0.0))

    # Generate comparison plot
    print(f"\n{'=' * 80}")
    print("Generating comparison plots...")
    print(f"{'=' * 80}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_multi_objective_comparison(results, output_path=str(output_path))

    print(f"\n{'=' * 80}")
    print("COMPARISON COMPLETE!")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print("-" * 40)

    for alg_name in algorithms.keys():
        max_budget = budgets[-1]
        dl_final = results[alg_name]["DL"][-1][1]
        dt_final = results[alg_name]["DT"][-1][1]
        pa_final = results[alg_name]["PA"][-1][1]

        print(
            f"{alg_name:15s} (k={max_budget}): "
            f"DL={dl_final:.3f} DT_red={dt_final:.3f} PA={pa_final:.1f}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

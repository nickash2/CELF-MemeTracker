"""
Evaluation utilities for CELF algorithm.

Provides runtime tracking, bound comparisons, and visualization tools for
analyzing algorithm performance and comparing different heuristics.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CELFResult:
    """Container for CELF algorithm results and metadata."""

    seeds: List[str]
    spread: float
    total_cost: float
    mode: str  # UC or CB
    budget: float
    simulations: int
    runtime_seconds: float
    num_nodes: int
    num_evaluations: int = 0
    online_bound: Optional[float] = None
    offline_bound: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def approximation_ratio(self) -> Optional[float]:
        """Compute approximation ratio relative to online bound."""
        if self.online_bound and self.online_bound > 0:
            return self.spread / self.online_bound
        return None


@dataclass
class HeuristicComparison:
    """Container for comparing multiple algorithm variants."""

    algorithm_name: str
    seeds: List[str]
    spread: float
    runtime_seconds: float
    total_cost: float
    budget: float


def save_results(result: CELFResult, output_path: str) -> None:
    """Save CELF results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Results saved to {output_path}")


def load_results(input_path: str) -> CELFResult:
    """Load CELF results from JSON file."""
    with open(input_path, "r") as f:
        data = json.load(f)
    return CELFResult(**data)


def plot_spread_vs_budget(
    results: List[CELFResult],
    output_path: str = "results/figures/spread_vs_budget.png",
    title: str = "Influence Spread vs Budget",
) -> None:
    """Plot spread achieved at different budget levels."""
    budgets = [r.budget for r in results]
    spreads = [r.spread for r in results]
    online_bounds = [r.online_bound for r in results if r.online_bound is not None]

    plt.figure(figsize=(10, 6))
    plt.plot(budgets, spreads, "o-", label="CELF Solution", linewidth=2, markersize=8)

    if len(online_bounds) == len(budgets):
        plt.plot(
            budgets,
            online_bounds,
            "s--",
            label="Online Bound",
            linewidth=2,
            markersize=6,
            alpha=0.7,
        )

    plt.xlabel("Budget", fontsize=12)
    plt.ylabel("Expected Influence Spread", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_runtime_comparison(
    comparisons: List[HeuristicComparison],
    output_path: str = "results/figures/runtime_comparison.png",
    title: str = "Algorithm Runtime Comparison",
) -> None:
    """Compare runtime of different algorithms/heuristics."""
    algorithms = [c.algorithm_name for c in comparisons]
    runtimes = [c.runtime_seconds for c in comparisons]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, runtimes, color="steelblue", alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xlabel("Algorithm", fontsize=12)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_spread_comparison(
    comparisons: List[HeuristicComparison],
    output_path: str = "results/figures/spread_comparison.png",
    title: str = "Influence Spread Comparison",
) -> None:
    """Compare spread achieved by different algorithms/heuristics."""
    algorithms = [c.algorithm_name for c in comparisons]
    spreads = [c.spread for c in comparisons]
    runtimes = [c.runtime_seconds for c in comparisons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Spread comparison
    bars1 = ax1.bar(algorithms, spreads, color="forestgreen", alpha=0.8)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax1.set_xlabel("Algorithm", fontsize=12)
    ax1.set_ylabel("Expected Influence Spread", fontsize=12)
    ax1.set_title("Spread Achieved", fontsize=13)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, axis="y", alpha=0.3)

    # Runtime comparison
    bars2 = ax2.bar(algorithms, runtimes, color="steelblue", alpha=0.8)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax2.set_xlabel("Algorithm", fontsize=12)
    ax2.set_ylabel("Runtime (seconds)", fontsize=12)
    ax2.set_title("Computation Time", fontsize=13)
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_bounds_comparison(
    result: CELFResult,
    output_path: str = "results/figures/bounds_comparison.png",
    title: str = "CELF Solution vs Bounds",
) -> None:
    """Visualize CELF solution quality relative to online/offline bounds."""
    metrics = ["CELF Solution"]
    values = [result.spread]

    if result.online_bound is not None:
        metrics.append("Online Bound")
        values.append(result.online_bound)

    if result.offline_bound is not None:
        metrics.append("Offline Bound")
        values.append(result.offline_bound)

    colors = ["forestgreen", "orange", "red"]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=colors[: len(metrics)], alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    # Add approximation ratio if available
    if result.online_bound and result.online_bound > 0:
        ratio = result.spread / result.online_bound
        plt.text(
            0.5,
            max(values) * 0.9,
            f"Approximation Ratio: {ratio:.3f}",
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.ylabel("Expected Influence Spread", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_marginal_gains(
    seeds: List[str],
    marginal_spreads: List[float],
    output_path: str = "results/figures/marginal_gains.png",
    title: str = "Marginal Influence Gains",
) -> None:
    """Plot diminishing marginal gains as seeds are added."""
    iterations = list(range(1, len(seeds) + 1))
    cumulative = np.cumsum(marginal_spreads)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Marginal gains
    ax1.plot(iterations, marginal_spreads, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Iteration (seed added)", fontsize=12)
    ax1.set_ylabel("Marginal Spread", fontsize=12)
    ax1.set_title("Marginal Gains per Iteration", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Cumulative spread
    ax2.plot(
        iterations, cumulative, "s-", color="forestgreen", linewidth=2, markersize=8
    )
    ax2.set_xlabel("Number of Seeds", fontsize=12)
    ax2.set_ylabel("Total Expected Spread", fontsize=12)
    ax2.set_title("Cumulative Influence Spread", fontsize=13)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


class PerformanceTracker:
    """Track algorithm performance metrics during execution."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.num_evaluations: int = 0
        self.marginal_gains: List[float] = []

    def start(self) -> None:
        """Start timing."""
        self.start_time = time.time()

    def stop(self) -> None:
        """Stop timing."""
        self.end_time = time.time()

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def record_evaluation(self, marginal_gain: Optional[float] = None) -> None:
        """Record a function evaluation."""
        self.num_evaluations += 1
        if marginal_gain is not None:
            self.marginal_gains.append(marginal_gain)


def create_summary_report(
    results: List[CELFResult],
    comparisons: Optional[List[HeuristicComparison]] = None,
    output_path: str = "results/summary_report.txt",
) -> None:
    """Generate a text summary report of all results."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CELF ALGORITHM PERFORMANCE SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("CELF Results:\n")
        f.write("-" * 80 + "\n")
        for i, result in enumerate(results, 1):
            f.write(f"\nRun {i}:\n")
            f.write(f"  Budget: {result.budget:.2f}\n")
            f.write(f"  Seeds Selected ({result.mode}): {len(result.seeds)}\n")
            f.write(f"  Seeds: {', '.join(result.seeds)}\n")
            f.write(f"  Expected Spread: {result.spread:.3f}\n")
            f.write(f"  Total Cost: {result.total_cost:.3f}\n")
            f.write(f"  Runtime: {result.runtime_seconds:.3f}s\n")
            f.write(f"  Simulations: {result.simulations}\n")
            if result.online_bound:
                f.write(f"  Online Bound: {result.online_bound:.3f}\n")
                f.write(f"  Approximation Ratio: {result.approximation_ratio():.3f}\n")
            if result.offline_bound:
                f.write(f"  Offline Bound: {result.offline_bound:.3f}\n")

        if comparisons:
            f.write("\n\nHeuristic Comparisons:\n")
            f.write("-" * 80 + "\n")
            for comp in comparisons:
                f.write(f"\n{comp.algorithm_name}:\n")
                f.write(f"  Spread: {comp.spread:.3f}\n")
                f.write(f"  Runtime: {comp.runtime_seconds:.3f}s\n")
                f.write(f"  Cost: {comp.total_cost:.3f} / {comp.budget:.3f}\n")
                f.write(f"  Seeds: {', '.join(comp.seeds)}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to {output_path}")

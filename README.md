# CELF-MemeTracker

Faithful implementation of the **Cost-Effective Lazy Forward (CELF)** algorithm from:

> Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. (2007).  
> **Cost-effective outbreak detection in networks.**  
> *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'07)*.

## Overview

This implementation mirrors Algorithms 1 and 2 from the paper:
- **Algorithm 1 (CELF)**: Lazy greedy selection with Unit Cost (UC) and Cost-Benefit (CB) variants
- **Algorithm 2 (GetBound)**: Online upper bound computation for approximation guarantees

The code targets the **Independent Cascade (IC)** diffusion model and is designed for influence maximization on networks, including MemeTracker-style cascade data.

## Project Structure

```
CELF-MemeTracker/
├── src/
│   ├── __init__.py           # Package exports
│   ├── celf.py               # Core CELF algorithms (Alg 1 & 2)
│   ├── preprocessing.py      # Data loaders & MemeTracker utilities
│   └── evaluation.py         # Performance tracking, plotting, comparisons
├── data/
│   ├── toy_edges.txt         # Example graph (7 edges)
│   └── toy_costs.txt         # Example node costs
├── results/
│   ├── figures/              # Generated plots
│   ├── *.json                # Run results with metadata
│   └── *.txt                 # Summary reports
├── examples/
│   └── evaluation_demo.py    # Full evaluation workflow demo
├── main.py                   # CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/nickash2/CELF-MemeTracker.git
cd CELF-MemeTracker

# Install optional dependencies (for visualization)
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Run CELF with unit costs (k=2 seeds)
python main.py --graph data/toy_edges.txt --k 2 --simulations 5000 --delimiter '\t' --seed 42

# Run with custom budget and costs
python main.py --graph data/toy_edges.txt --costs data/toy_costs.txt --budget 2.5 --simulations 5000 --delimiter '\t'

# Compute online bound and generate plots
python main.py --graph data/toy_edges.txt --k 3 --compute-bound --simulations 5000 --seed 42
```

### Output

```
============================================================
CELF RESULTS
============================================================
Selected seeds (UC): B, A
Total cost: 2.000 / 2.000
Estimated spread: 3.173
Runtime: 1.234s
Online bound R^: 3.450
Approximation ratio: 0.920
============================================================

Results saved to results/celf_result_20251102_123045.json
Plot saved to results/figures/bounds_comparison_20251102_123045.png
```

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--graph` | str | **(Required)** Path to edge list file (format: `src dst [prob]`) |
| `--k` | int | Seed budget (unit-cost mode) |
| `--budget` | float | Total cost budget (general mode) |
| `--costs` | str | Optional node cost file (format: `node cost`) |
| `--simulations` | int | Monte Carlo samples per evaluation (default: 1000) |
| `--default-prob` | float | Edge probability when not specified (default: 0.1) |
| `--seed` | int | Random seed for reproducibility |
| `--delimiter` | str | Field delimiter (default: whitespace) |
| `--skip-header` | flag | Skip first line of edge list |
| `--compute-bound` | flag | Compute online upper bound R^ |
| `--verbose` | flag | Print diagnostic information |

## Algorithm Details

### CELF (Algorithm 1)

The implementation runs both UC and CB variants and returns the better result:

- **UC (Unit Cost)**: Maximizes marginal gain directly
- **CB (Cost-Benefit)**: Maximizes marginal gain per unit cost

Key features:
- Lazy evaluation: avoid redundant influence spread computations
- Priority queue with validity checking (snapshot tracking)
- Submodularity exploitation for efficiency

### Online Bound (Algorithm 2)

Computes an upper bound on the optimal solution using fractional relaxation:
- Greedily selects nodes by cost-benefit ratio
- Allows fractional allocation of remaining budget
- Provides approximation guarantee: `CELF ≥ (1 - 1/e) × OPT ≈ 0.632 × OPT`

## Evaluation & Visualization

The `src/evaluation.py` module provides:

### Performance Tracking
```python
from src import PerformanceTracker

tracker = PerformanceTracker()
tracker.start()
# ... run algorithm ...
tracker.stop()
print(f"Runtime: {tracker.elapsed():.3f}s")
```

### Result Persistence
```python
from src import CELFResult, save_results, load_results

result = CELFResult(seeds=..., spread=..., runtime_seconds=...)
save_results(result, "results/my_run.json")
loaded = load_results("results/my_run.json")
```

### Visualization Functions
- `plot_bounds_comparison()`: CELF vs online/offline bounds
- `plot_spread_vs_budget()`: Influence spread at different budgets
- `plot_runtime_comparison()`: Algorithm speed comparison
- `plot_spread_comparison()`: Spread vs runtime tradeoff
- `plot_marginal_gains()`: Diminishing returns analysis
- `create_summary_report()`: Text-based summary

### Example: Full Evaluation

```bash
python examples/evaluation_demo.py
```

This script demonstrates:
1. Budget sweeps with performance tracking
2. Bound computation and approximation ratios
3. Heuristic comparisons (placeholders for future algorithms)
4. Marginal gains analysis
5. Automated report generation

## MemeTracker Application

The CELF algorithm can be applied to MemeTracker cascades:

1. **Build influence graph** from blog post cascades:
   ```python
   from src import build_graph_from_cascades, estimate_propagation_probability
   
   cascades = [
       [("blog1", 1000), ("blog2", 1020), ("blog3", 1045)],
       [("blog2", 2000), ("blog4", 2030)],
   ]
   graph = build_graph_from_cascades(cascades)
   ```

2. **Estimate edge probabilities** using temporal decay models
3. **Run CELF** to find optimal seed blogs for viral propagation

See `src/preprocessing.py` for MemeTracker-specific utilities.

## Contributing

Future enhancements could include:
- Additional heuristics (degree-based, PageRank-based, random)
- Scalability optimizations (inverted index from Section 4.1)
- Alternative diffusion models (Linear Threshold, Weighted Cascade)
- Parallel Monte Carlo simulation
- Real MemeTracker dataset integration

## References

1. Leskovec et al. (2007). Cost-effective outbreak detection in networks. KDD.
2. Kempe et al. (2003). Maximizing the spread of influence through a social network. KDD.
3. Leskovec et al. (2009). MemeTracker: Tracking quotes across the web. WWW.

## License

[Add your license here]

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{leskovec2007cost,
  title={Cost-effective outbreak detection in networks},
  author={Leskovec, Jure and Krause, Andreas and Guestrin, Carlos and Faloutsos, Christos and VanBriesen, Jeanne and Glance, Natalie},
  booktitle={Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining},
  pages={420--429},
  year={2007}
}
```

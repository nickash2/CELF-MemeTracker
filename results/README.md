# Results Directory

This directory contains experimental results, performance metrics, and visualizations from CELF algorithm runs.

## Structure

```
results/
├── figures/              # Generated plots and visualizations
│   ├── bounds_comparison_*.png
│   ├── spread_vs_budget.png
│   ├── runtime_comparison.png
│   ├── spread_comparison.png
│   └── marginal_gains.png
├── celf_result_*.json   # Individual run results with metadata
└── summary_report.txt   # Aggregated summary across multiple runs
```

## Output Files

### JSON Results (`celf_result_YYYYMMDD_HHMMSS.json`)
Each run generates a timestamped JSON file containing:
- Selected seed nodes
- Expected influence spread
- Total cost and budget
- Algorithm variant used (UC/CB)
- Runtime in seconds
- Number of simulations
- Online/offline bounds (if computed)
- Approximation ratio

### Figures

**`bounds_comparison_*.png`**
- Compares CELF solution quality against online and offline bounds
- Shows approximation ratio
- Generated when `--compute-bound` flag is used

**`spread_vs_budget.png`**
- Shows influence spread achieved at different budget levels
- Useful for understanding budget allocation efficiency

**`runtime_comparison.png`**
- Compares execution time across different algorithms/heuristics
- Bar chart with runtime in seconds

**`spread_comparison.png`**
- Side-by-side comparison of spread and runtime for multiple algorithms
- Useful for evaluating speed-quality tradeoffs

**`marginal_gains.png`**
- Visualizes diminishing returns as seeds are added incrementally
- Shows both marginal and cumulative influence spread

### Summary Report (`summary_report.txt`)
Text-based summary containing:
- All CELF runs with parameters and results
- Heuristic comparisons (when multiple algorithms tested)
- Aggregated statistics

## Example Usage

```bash
# Run CELF with bound computation (automatically saves to results/)
python main.py --graph data/toy_edges.txt --k 5 --compute-bound --seed 42

# View generated files
ls results/
ls results/figures/

# Load and analyze results programmatically
python -c "
from src import load_results
result = load_results('results/celf_result_20251102_120530.json')
print(f'Spread: {result.spread:.2f}')
print(f'Runtime: {result.runtime_seconds:.3f}s')
print(f'Approximation ratio: {result.approximation_ratio():.3f}')
"
```

## Future Enhancements

This directory is designed to support:
- Comparison plots for multiple heuristics (greedy, degree-based, etc.)
- Statistical analysis across parameter sweeps
- Scalability experiments (runtime vs. graph size)
- MemeTracker-specific cascade analysis

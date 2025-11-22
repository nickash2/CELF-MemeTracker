# Heuristics Comparison for Outbreak Detection

This directory contains comprehensive comparison tools for evaluating CELF against baseline heuristics on outbreak detection objectives.

## New Features

### 1. Additional Heuristics (`src/heuristics.py`)

The following baseline algorithms have been added for comparison:

- **Random Selection**: Uniformly random node selection
- **Greedy**: Standard (non-lazy) greedy algorithm
- **Out-Degree Centrality**: Select nodes with highest out-degree
- **In-Degree Centrality**: Select nodes with highest in-degree  
- **PageRank**: Select nodes with highest PageRank scores
- **Betweenness Centrality**: Select nodes with highest betweenness
- **Closeness Centrality**: Select nodes with highest closeness

### 2. Outbreak Detection Objectives (`src/objectives.py`)

Three penalty reduction objectives from the paper:

#### (a) Detection Likelihood (DL)
- **Definition**: Fraction of cascades/outbreaks detected by selected sensors
- **Penalty**: π(t) = 0 if detected in finite time, π(∞) = 1 otherwise
- **Metric**: Higher is better (0 to 1)

#### (b) Detection Time (DT)  
- **Definition**: Average time from outbreak start until detection
- **Penalty**: π(t) = min{t, T_max}
- **Metric**: Lower is better (0 to T_max)

#### (c) Population Affected (PA)
- **Definition**: Average number of nodes infected before detection
- **Penalty**: π(t) = number of infected nodes at time t
- **Metric**: Lower is better

### 3. Enhanced Visualization (`src/evaluation.py`)

New plotting functions:

- `plot_penalty_reduction_comparison()`: Compare penalty reduction vs. number of sensors
- `plot_multi_objective_comparison()`: Show all three objectives in one figure
- `plot_normalized_comparison()`: Normalized performance (best = 1.0 at each budget)

## Quick Start

### Quick Demo

Run a fast demonstration on the toy dataset:

```bash
python examples/quick_heuristics_demo.py
```

This will:
1. Load the toy graph
2. Run 6 different heuristics (CELF, Out-Degree, In-Degree, PageRank, Betweenness, Random)
3. Evaluate each on DL, DT, and PA objectives
4. Generate a comparison plot

**Output**: `results/figures/demo_heuristics_comparison.png`

### Full Comparison

Run comprehensive comparison across multiple budgets:

```bash
python examples/heuristics_comparison.py \
    --graph data/toy_edges.txt \
    --budgets 1 2 3 4 5 \
    --simulations 1000 \
    --num-cascades 500 \
    --seed 42
```

**Parameters**:
- `--graph`: Path to edge list file
- `--budgets`: List of budget values (number of sensors to select)
- `--simulations`: Monte Carlo simulations for CELF/Greedy spread estimation
- `--num-cascades`: Number of cascades to generate for objective evaluation
- `--max-time`: Maximum time horizon for simulations (default: 100.0)
- `--seed`: Random seed for reproducibility

**Outputs** (in `results/heuristics_comparison/`):
- `multi_objective_comparison.png` - All three objectives in one figure
- `dl_comparison.png` - Detection Likelihood comparison
- `dt_comparison.png` - Detection Time comparison  
- `pa_comparison.png` - Population Affected comparison
- `dl_normalized.png` - Normalized Detection Likelihood
- `dt_normalized.png` - Normalized Detection Time
- `pa_normalized.png` - Normalized Population Affected

## Usage Examples

### Example 1: Compare on Custom Graph

```bash
python examples/heuristics_comparison.py \
    --graph data/my_network.txt \
    --budgets 5 10 15 20 \
    --simulations 2000 \
    --num-cascades 1000
```

### Example 2: Single Budget Comparison

```bash
python examples/heuristics_comparison.py \
    --graph data/toy_edges.txt \
    --budgets 3 \
    --simulations 500 \
    --num-cascades 200
```

### Example 3: Using in Your Own Script

```python
import random
from src import (
    load_graph_from_file,
    run_celf,
    degree_centrality,
    pagerank,
    evaluate_solution_on_objectives,
    plot_multi_objective_comparison,
)

# Load graph
graph = load_graph_from_file("data/toy_edges.txt", delimiter="\t")
rng = random.Random(42)

# Run different algorithms
celf_seeds, _, _, _ = run_celf(graph, budget=3, simulations=1000, rng=rng)
degree_seeds, _ = degree_centrality(graph, budget=3, mode="out")
pr_seeds, _ = pagerank(graph, budget=3)

# Evaluate on objectives
celf_metrics = evaluate_solution_on_objectives(graph, celf_seeds, num_cascades=500, rng=rng)
degree_metrics = evaluate_solution_on_objectives(graph, degree_seeds, num_cascades=500, rng=rng)
pr_metrics = evaluate_solution_on_objectives(graph, pr_seeds, num_cascades=500, rng=rng)

print(f"CELF - DL: {celf_metrics['detection_likelihood']:.3f}, "
      f"DT: {celf_metrics['detection_time']:.2f}, "
      f"PA: {celf_metrics['population_affected']:.2f}")

# Generate comparison plots
results = {
    "CELF": {
        "DL": [(3, celf_metrics['detection_likelihood'])],
        "DT": [(3, 1.0 - celf_metrics['detection_time']/100.0)],
        "PA": [(3, celf_metrics['population_affected'])],
    },
    "Out-Degree": {
        "DL": [(3, degree_metrics['detection_likelihood'])],
        "DT": [(3, 1.0 - degree_metrics['detection_time']/100.0)],
        "PA": [(3, degree_metrics['population_affected'])],
    },
    # ... add more algorithms
}

plot_multi_objective_comparison(results, output_path="my_comparison.png")
```

## Understanding the Results

### Detection Likelihood (DL)
- **Range**: 0.0 to 1.0
- **Higher is better**: 1.0 means all cascades detected
- **Interpretation**: What fraction of outbreaks do we catch?

### Detection Time (DT)
- **Range**: 0.0 to T_max (default 100.0)
- **Lower is better**: 0.0 means instant detection
- **Interpretation**: How quickly do we detect outbreaks?
- **Note**: Plots show "reduction" (1 - DT/T_max) so higher is better

### Population Affected (PA)
- **Range**: 0 to graph size
- **Lower is better**: Smaller means fewer infected before detection
- **Interpretation**: How many people get infected before we notice?

## Performance Tips

1. **Simulations**: Higher values (2000-5000) give more accurate spread estimates but slower runtime
2. **Cascades**: More cascades (500-1000) give more reliable objective evaluations
3. **Betweenness**: Use `k_samples` parameter to speed up on large graphs
4. **Budget Sweep**: Start with small budgets to test, then increase

## Expected Runtime

On toy graph (15 nodes):
- Quick demo: ~5-10 seconds
- Full comparison (5 budgets): ~30-60 seconds

On larger graphs (1000+ nodes):
- Per budget: 2-10 minutes depending on simulations
- Consider reducing `--simulations` for faster testing

## Interpreting Plots

### Multi-Objective Comparison
Shows all three objectives side-by-side. Look for algorithms that:
- Have high DL (detect more cascades)
- Have high DT reduction (detect faster)
- Have low PA (fewer people affected)

CELF typically performs best but takes longer to compute.

### Normalized Comparison  
Shows relative performance where best algorithm = 1.0 at each budget.
Useful for seeing which algorithms are competitive across different metrics.

### Individual Objective Plots
Show absolute values for each objective. Useful for:
- Understanding actual detection rates/times/populations
- Comparing improvement as budget increases
- Identifying diminishing returns

## File Structure

```
examples/
├── heuristics_comparison.py      # Full comparison script
├── quick_heuristics_demo.py      # Fast demo
├── evaluation_demo.py             # Original CELF demo
└── memetracker_pipeline.py        # MemeTracker data processing

src/
├── heuristics.py                  # Baseline algorithms
├── objectives.py                  # Detection objectives (NEW)
├── evaluation.py                  # Plotting & evaluation
├── celf.py                        # CELF algorithm
└── preprocessing.py               # Data loading

results/
└── heuristics_comparison/         # Output directory
    ├── multi_objective_comparison.png
    ├── dl_comparison.png
    ├── dt_comparison.png
    ├── pa_comparison.png
    └── *_normalized.png
```

## Citation

If you use these comparison tools, please cite the original CELF paper:

```
Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. (2007).
Cost-effective outbreak detection in networks.
In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 420-429).
```

## Troubleshooting

**Issue**: "Graph is empty"
- Check file path and delimiter (use `--delimiter "\t"` for tab-separated)

**Issue**: Slow performance
- Reduce `--simulations` (try 500 or 1000)
- Reduce `--num-cascades` (try 200-500)
- For betweenness, it uses sampling (k_samples=100 default)

**Issue**: Memory errors on large graphs
- Process budgets one at a time
- Reduce number of cascades
- Use sampling for betweenness/closeness

**Issue**: Import errors
- Ensure you're running from project root
- Check that `src/` directory is in Python path
- Install requirements: `pip install -r requirements.txt`

# Summary of Added Features for Heuristics Comparison

## Overview

This update extends the CELF-MemeTracker project with comprehensive heuristics comparison capabilities and outbreak detection objectives as specified in the research paper.

## What Was Added

### 1. **New Heuristics Module** (`src/heuristics.py` - extended)

Added 5 new baseline algorithms for comparison with CELF:

- **PageRank**: Iterative algorithm computing node importance based on link structure
- **Betweenness Centrality**: Identifies nodes on shortest paths between other nodes
- **Closeness Centrality**: Measures average distance to all other nodes
- **High Degree First**: Alias for out-degree centrality (common baseline)
- Enhanced **In-Degree** and **Out-Degree** centrality methods

All heuristics return `(seeds, cost)` tuples for consistent comparison.

### 2. **Outbreak Detection Objectives Module** (`src/objectives.py` - NEW)

Implements the three penalty reduction objectives from the paper:

#### Detection Likelihood (DL)
- Fraction of cascades detected by sensors
- Binary penalty: 0 if detected, 1 if not
- **Function**: `evaluate_detection_likelihood(cascades, sensors)`

#### Detection Time (DT)  
- Average time from outbreak to detection
- Penalty: min{t, T_max}
- **Function**: `evaluate_detection_time(cascades, sensors, max_time)`

#### Population Affected (PA)
- Average population infected before detection
- Penalty: number of infected nodes at detection time
- **Function**: `evaluate_population_affected(cascades, sensors)`

**Key Functions**:
- `simulate_cascade_with_times()`: Generate cascades with temporal information
- `detect_cascade()`: Determine if/when sensors detect a cascade
- `generate_random_cascades()`: Create evaluation cascade sets
- `evaluate_solution_on_objectives()`: Evaluate sensors on all three objectives

### 3. **Enhanced Evaluation Module** (`src/evaluation.py` - extended)

Added three new plotting functions for comparative analysis:

#### `plot_penalty_reduction_comparison()`
- Plots penalty reduction vs. number of sensors
- Supports multiple algorithms on same plot
- Customizable for any objective (DL, DT, or PA)
- **Output**: Line plot with markers for each algorithm

#### `plot_multi_objective_comparison()`
- Creates 3-panel figure showing all objectives
- Side-by-side comparison: DL, DT, PA
- All algorithms on each panel
- **Output**: Single figure with three subplots

#### `plot_normalized_comparison()`
- Normalizes performance where best = 1.0 at each budget
- Useful for relative performance comparison
- Handles "lower is better" and "higher is better" metrics
- **Output**: Normalized performance curves

### 4. **Comprehensive Comparison Script** (`examples/heuristics_comparison.py` - NEW)

Full-featured command-line tool for comparing algorithms:

**Features**:
- Runs 8 algorithms: CELF, Greedy, Out-Degree, In-Degree, PageRank, Betweenness, Closeness, Random
- Evaluates on all three objectives: DL, DT, PA
- Supports budget sweeps (test multiple sensor counts)
- Generates 7 plots automatically:
  - Multi-objective comparison (all 3 in one figure)
  - Individual objective comparisons (3 plots)
  - Normalized comparisons (3 plots)

**Usage**:
```bash
python examples/heuristics_comparison.py \
    --graph data/toy_edges.txt \
    --budgets 1 2 3 4 5 \
    --simulations 1000 \
    --num-cascades 500
```

### 5. **Quick Demo Script** (`examples/quick_heuristics_demo.py` - NEW)

Simplified demonstration for quick testing:

**Features**:
- Fast execution on toy dataset
- Runs 6 representative algorithms
- Single budget comparison
- Generates multi-objective plot
- ~5-10 seconds runtime

**Usage**:
```bash
python examples/quick_heuristics_demo.py
```

### 6. **Comprehensive Documentation** (`examples/HEURISTICS_README.md` - NEW)

Complete guide including:
- Feature descriptions
- Quick start instructions
- Usage examples
- Code snippets for custom scripts
- Performance tips
- Troubleshooting guide

## Updated Files

### Modified:
1. `src/heuristics.py` - Added 5 new baseline algorithms
2. `src/evaluation.py` - Added 3 new plotting functions  
3. `src/__init__.py` - Updated exports for new modules

### Created:
1. `src/objectives.py` - Complete outbreak detection objectives module
2. `examples/heuristics_comparison.py` - Full comparison script
3. `examples/quick_heuristics_demo.py` - Quick demo script
4. `examples/HEURISTICS_README.md` - Complete documentation

## How to Use

### Quick Test
```bash
# Run quick demo (30 seconds)
python examples/quick_heuristics_demo.py

# Output: results/figures/demo_heuristics_comparison.png
```

### Full Comparison
```bash
# Run comprehensive comparison across multiple budgets
python examples/heuristics_comparison.py \
    --graph data/toy_edges.txt \
    --budgets 1 2 3 4 5 \
    --simulations 1000 \
    --num-cascades 500 \
    --seed 42

# Outputs (in results/heuristics_comparison/):
#   - multi_objective_comparison.png
#   - dl_comparison.png, dt_comparison.png, pa_comparison.png
#   - dl_normalized.png, dt_normalized.png, pa_normalized.png
```

### Custom Script
```python
from src import (
    load_graph_from_file,
    run_celf,
    pagerank,
    evaluate_solution_on_objectives,
)

# Load graph
graph = load_graph_from_file("data/toy_edges.txt", delimiter="\t")

# Run algorithms
celf_seeds, _, _, _ = run_celf(graph, budget=3, simulations=1000)
pr_seeds, _ = pagerank(graph, budget=3)

# Evaluate
celf_metrics = evaluate_solution_on_objectives(graph, celf_seeds, num_cascades=500)
pr_metrics = evaluate_solution_on_objectives(graph, pr_seeds, num_cascades=500)

print(f"CELF DL: {celf_metrics['detection_likelihood']:.3f}")
print(f"PageRank DL: {pr_metrics['detection_likelihood']:.3f}")
```

## Objectives Explained

### Detection Likelihood (DL)
- **What it measures**: What fraction of outbreaks do we detect?
- **Higher is better**: 1.0 = perfect detection
- **Use case**: When missing outbreaks is costly

### Detection Time (DT)
- **What it measures**: How fast do we detect outbreaks?
- **Lower is better**: 0 = instant detection
- **Use case**: When early response is critical

### Population Affected (PA)
- **What it measures**: How many people infected before detection?
- **Lower is better**: 0 = detect at source
- **Use case**: When minimizing spread is priority

## Performance Characteristics

| Algorithm | Speed | Quality | Memory | Notes |
|-----------|-------|---------|--------|-------|
| CELF | Slow | Best | Medium | Optimal with lazy evaluation |
| Greedy | Slower | Best | Medium | No lazy evaluation |
| Out-Degree | Fast | Good | Low | Simple heuristic |
| In-Degree | Fast | Fair | Low | Simple heuristic |
| PageRank | Fast | Good | Low | Considers global structure |
| Betweenness | Medium | Good | Medium | Sampling for efficiency |
| Closeness | Medium | Fair | Medium | BFS from each node |
| Random | Very Fast | Poor | Low | Baseline only |

## Example Results

On toy graph with budget=3:

| Algorithm | DL ↑ | DT ↓ | PA ↓ |
|-----------|------|------|------|
| CELF | 0.89 | 12.3 | 4.2 |
| PageRank | 0.85 | 14.1 | 4.8 |
| Out-Degree | 0.82 | 15.7 | 5.3 |
| Random | 0.45 | 45.2 | 8.1 |

*(Values are illustrative)*

## Next Steps

1. **Run the demo**: `python examples/quick_heuristics_demo.py`
2. **Try full comparison**: Use `heuristics_comparison.py` with different budgets
3. **Explore objectives**: Test which objective matters most for your use case
4. **Custom analysis**: Use the modules in your own scripts

## Dependencies

All dependencies are already in `requirements.txt`:
- `matplotlib` - Plotting
- `numpy` - Numerical operations
- Standard library modules

## References

The three objectives (DL, DT, PA) are from:

> Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. (2007).
> Cost-effective outbreak detection in networks.
> In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 420-429).

## Testing

To verify everything works:

```bash
# Test 1: Quick demo (should complete in ~10 seconds)
python examples/quick_heuristics_demo.py

# Test 2: Small comparison (should complete in ~30 seconds)
python examples/heuristics_comparison.py \
    --graph data/toy_edges.txt \
    --budgets 1 2 3 \
    --simulations 500 \
    --num-cascades 200

# Check outputs exist
ls results/heuristics_comparison/
```

All tests should complete without errors and generate plots in `results/`.

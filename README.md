# CELF-MemeTracker

Faithful implementation of the **Cost-Effective Lazy Forward (CELF)** algorithm from:

> Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., & Glance, N. (2007).  
> **Cost-effective outbreak detection in networks.**  
> *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'07)*.

## Overview

This implementation focuses on **outbreak detection** in networks using MemeTracker cascade data. The code implements:
- **CELF**: Cost-Effective Lazy Forward algorithm for sensor placement
- **CELF++**: Accelerated greedy with improved lazy evaluation
- **Greedy**: Fast greedy approximation with cascade sampling
- **Baseline heuristics**: Degree centrality, PageRank, Betweenness, Random

The algorithms optimize three objectives for early outbreak detection:
- **DL** (Detection Likelihood): Probability of detecting a cascade
- **DT** (Detection Time): Expected time to first detection
- **PA** (Population Affected): Expected cascade size at detection

## Project Structure

```
CELF-MemeTracker/
├── src/
│   ├── __init__.py
│   ├── celf.py               # CELF, Greedy algorithms for outbreak detection
│   ├── celfpp.py             # CELF++ accelerated greedy
│   ├── heuristics.py         # Baseline heuristics (degree, PageRank, etc.)
│   ├── objectives.py         # DL, DT, PA evaluation functions
│   ├── preprocessing.py      # MemeTracker data loaders & graph building
│   └── evaluation.py         # Plotting and result visualization
├── data/
│   ├── quotes_2008-08.txt    # MemeTracker dataset
│   └── cache/                # Cached preprocessed data
├── results/
│   └── figures/              # Generated plots
├── examples/
│   ├── memetracker_heuristics_comparison.py  # Main comparison script
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

### Built-in Script Use
You can use the built in `run.sh` file to run all experiments by:
```bash
chmod +x run.sh
./run.sh
```
### Basic Usage

```bash
# Compare all algorithms on MemeTracker data
python examples/memetracker_heuristics_comparison.py \
  --input data/quotes_2008-08.txt \
  --budgets 1 2 3 4 5 6 7 8 9 10 \
  --max-docs 20000 \
  --top-memes 100 \
  --objective DL \
  --seed 42

# Run with different objective (Detection Time)
python examples/memetracker_heuristics_comparison.py \
  --input data/quotes_2008-08.txt \
  --budgets 1 5 10 \
  --objective DT \
  --max-time 100.0

# Smaller test run
python examples/memetracker_heuristics_comparison.py \
  --input data/quotes_2008-08.txt \
  --budgets 1 2 3 \
  --max-docs 5000 \
  --top-memes 50
```

### Output

```
MEMETRACKER HEURISTICS COMPARISON

1. Building influence graph from data/quotes_2008-08.txt
   Max documents: 20000
   Top memes: 100
   Min probability: 0.01

2. Graph statistics:
   Nodes (sites): 8532
   Cascades: 100

3. Converting cascades to events...
   Valid cascade events: 100

4. Configuration:
   Budgets: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
   Max time: 100.0
   Baseline PA (no monitors): 3.9

Running heuristics comparison...

Greedy:
  Budget = 1: seeds=['blog.myspace.com'] DL=0.074 DT=95.7 PA=3.5
  Budget = 2: seeds=['blog.myspace.com', 'us.rd.yahoo.com'] DL=0.125 DT=91.0 PA=3.2
  ...

CELF:
  Budget = 1: seeds=['blog.myspace.com'] DL=0.074 DT=95.7 PA=3.5
  ...

COMPARISON COMPLETE!
Results saved to: results/figures/memetracker_comparison.png
```

## CLI Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--input` | str | **(Required)** Path to MemeTracker file (`quotes_YYYY-MM.txt` or `.txt.gz`) |
| `--budgets` | int+ | Budget values (number of sensors) to test (default: 1 2 3 4 5) |
| `--max-docs` | int | Maximum documents to parse (default: 10000, 0 for unlimited) |
| `--top-memes` | int | Number of top memes to track (default: 50, 0 for unlimited) |
| `--min-prob` | float | Minimum edge probability threshold (default: 0.01) |
| `--objective` | str | Objective for CELF/Greedy: `DL`, `DT`, or `PA` (default: DL) |
| `--max-time` | float | Maximum time horizon for DT objective (default: 100.0) |
| `--seed` | int | Random seed for reproducibility (default: 42) |
| `--output` | str | Output plot path (default: results/figures/memetracker_comparison.png) |

## Algorithm Details

### CELF & Greedy

Both CELF and Greedy use lazy evaluation with cascade sampling:
- **Cascade sampling**: Evaluate on a sample of cascades for speed (default: 250 cascades)
- **Lazy evaluation**: Skip redundant marginal gain computations using priority queues
- **Multi-objective**: Support DL (detection likelihood), DT (detection time), PA (population affected)

**CELF** (`src/celf.py`):
- Full lazy greedy with snapshot tracking
- Exact marginal gain computation on sampled cascades
- Best theoretical guarantees: `(1 - 1/e) ≈ 0.632` approximation

**Greedy** (`src/celf.py`):
- Fast greedy with partial lazy evaluation
- Faster runtime, slightly lower solution quality

**CELF++** (`src/celfpp.py`):
- Accelerated greedy with improved lazy evaluation
- Better pruning for faster convergence

### Baseline Heuristics

**Degree Centrality** (`src/heuristics.py`):
- In-degree: Select most popular/authoritative sites
- Out-degree: Select sites with most outgoing links

**PageRank**: Select sites with highest PageRank scores

**Betweenness Centrality**: Select sites on most shortest paths (approximated with sampling)

**Random**: Random selection baseline

## Evaluation & Visualization

### Objectives

The `src/objectives.py` module provides three outbreak detection metrics:

**Detection Likelihood (DL)**:
```python
from src import evaluate_detection_likelihood

# Probability of detecting at least one cascade
dl = evaluate_detection_likelihood(cascade_events, selected_seeds)
```

**Detection Time (DT)**:
```python
from src import evaluate_detection_time

# Expected time to first detection across cascades
dt = evaluate_detection_time(cascade_events, selected_seeds, max_time=100.0)
```

**Population Affected (PA)**:
```python
from src import evaluate_population_affected

# Expected cascade size at detection time
pa = evaluate_population_affected(cascade_events, selected_seeds)
```

### Plotting

The `src/evaluation.py` module provides:
- `plot_multi_objective_comparison()`: Compare all algorithms on DL/DT/PA across budgets
- `plot_bounds_vs_budget()`: CELF/CELF++ bounds analysis



## MemeTracker Data Processing

### Building Influence Graph

The `src/preprocessing.py` module provides utilities for loading MemeTracker data:

```python
from src import build_graph_from_memetracker, convert_memetracker_cascades_to_events

# Build graph from MemeTracker file
graph, cascades_dict = build_graph_from_memetracker(
    path='data/quotes_2008-08.txt',
    top_memes=100,           # Track top 100 memes
    min_prob=0.01,           # Filter weak edges
    max_documents=20000      # Limit documents parsed
)

# Convert to cascade events for evaluation
cascade_events = convert_memetracker_cascades_to_events(cascades_dict)
```

### Graph Construction

- **Nodes**: Blog sites/domains extracted from URLs
- **Edges**: Directed edges (u $\rightarrow$ v) if site u mentions a quote before site v
- **Probabilities**: Frequency-based estimation (how often u $\rightarrow$ v occurs across cascades)
- **Cascades**: Temporal sequences of sites mentioning the same meme/quote

## Implementation Notes

### Performance Optimizations

- **Cascade sampling**: Use `sample_size` parameter (default: 250) to speed up evaluation
- **Cached preprocessing**: Graph and cascades are cached in `data/cache/` to avoid reprocessing
- **Lazy evaluation**: CELF/CELF++ skip redundant marginal gain computations
- **Betweenness approximation**: Uses k-sample approximation for large graphs

### Fairness in Comparisons

All algorithms (Greedy, CELF, CELF++) use the **same sampled cascades** via the `sample_size` and `rng` parameters to ensure fair comparisons.


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

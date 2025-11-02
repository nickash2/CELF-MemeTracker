# Data Directory

This directory contains graph data and MemeTracker dataset files for CELF experiments.

## Example Files

### `toy_edges.txt`
Small test graph with 7 edges:
```
A	B	0.3
A	C	0.1
B	C	0.4
B	D	0.2
C	D	0.5
C	E	0.3
D	E	0.6
```

Format: `source\ttarget\tprobability`

### `toy_costs.txt`
Node costs for budget-constrained experiments:
```
A 1.0
B 1.5
C 1.0
D 0.8
E 0.5
```

Format: `node cost`

## MemeTracker Data

The MemeTracker dataset tracks meme propagation across 96M+ blog posts and news articles from Aug 2008 - Apr 2009.

### Download

Official dataset: http://snap.stanford.edu/data/memetracker9.html

```bash
# Download a monthly file (example: Aug 2008)
cd data/
wget http://snap.stanford.edu/data/quotes_2008-08.txt.gz

# Or download all months (warning: ~50GB compressed)
for month in 08 09 10 11 12; do
    wget http://snap.stanford.edu/data/quotes_2008-$month.txt.gz
done

for month in 01 02 03 04; do
    wget http://snap.stanford.edu/data/quotes_2009-$month.txt.gz
done
```

### MemeTracker File Format

Each document is separated by blank lines:
```
P       http://blogs.abcnews.com/politicalpunch/2008/09/post.html
T       2008-09-09 22:35:24
Q       that's not change
Q       you can put lipstick on a pig
Q       what's the difference between a hockey mom and a pit bull lipstick
L       http://reuters.com/article/politicsnews/idusn2944356420080901
L       http://cbn.com/cbnnews/436448.aspx

P       http://another-blog.com/article2.html
T       2008-09-09 23:15:00
Q       lipstick on a pig
L       http://blogs.abcnews.com/politicalpunch/2008/09/post.html
```

**Line types:**
- `P`: Document URL (blog post or news article)
- `T`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `Q`: Phrase/quote extracted from document text
- `L`: Hyperlink to another document

**Notes:**
- Some documents have zero phrases or zero links
- Files are plain text or gzip compressed (.txt.gz)
- Total dataset: 96.6M documents, 211M meme mentions, 418M links

## Processing MemeTracker

### Quick Start (Small Sample)

```bash
# Process first 10,000 documents from Aug 2008
python examples/memetracker_pipeline.py \
    --input data/quotes_2008-08.txt.gz \
    --max-docs 10000 \
    --top-memes 50 \
    --k 10 \
    --simulations 1000
```

### Full Month Processing

```bash
# Process entire month (may take hours)
python examples/memetracker_pipeline.py \
    --input data/quotes_2008-08.txt.gz \
    --max-docs 1000000 \
    --top-memes 200 \
    --k 50 \
    --simulations 5000 \
    --output-prefix aug2008
```

### Extract Specific Meme Cascade

```python
from src import parse_memetracker_file, build_memetracker_cascade

# Parse documents
docs = parse_memetracker_file("data/quotes_2008-08.txt.gz", max_documents=50000)

# Extract cascade for specific phrase
cascade = build_memetracker_cascade(docs, "lipstick on a pig")
print(f"Cascade length: {len(cascade)} sites")
print("Sites:", [site for site, _ in cascade[:5]])
```

## Custom Datasets

### Edge List Format

Create your own graph files:
```
# Comments start with #
# Format: source target [probability]
blog1	blog2	0.3
blog1	blog3	0.1
blog2	blog3	0.5
```

Run CELF:
```bash
python main.py --graph data/my_graph.txt --k 5 --simulations 1000
```

### Cost File Format

Specify node costs:
```
# Format: node cost
blog1 10.5
blog2 5.0
blog3 2.5
```

Run with budget constraint:
```bash
python main.py --graph data/my_graph.txt \
               --costs data/my_costs.txt \
               --budget 20.0 \
               --simulations 1000
```

## Dataset Statistics

| Dataset | Nodes | Documents | Time Range | Size (compressed) |
|---------|-------|-----------|------------|-------------------|
| toy_edges.txt | 5 | - | - | <1KB |
| quotes_2008-08.txt.gz | ~500K | ~10M | Aug 2008 | ~5GB |
| quotes_2008-09.txt.gz | ~500K | ~11M | Sep 2008 | ~5.5GB |
| Full dataset | ~1M | 96.6M | Aug 2008 - Apr 2009 | ~50GB |

## Citation

If using MemeTracker data, cite:
```bibtex
@inproceedings{leskovec2009memetracker,
  title={Meme-tracking and the Dynamics of the News Cycle},
  author={Leskovec, Jure and Backstrom, Lars and Kleinberg, Jon},
  booktitle={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2009}
}
```

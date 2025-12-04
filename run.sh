#!/bin/zsh

source venv/bin/activate

# Install requirements (uncomment if needed)
pip install -r requirements.txt

INPUT="data/quotes_2008-08.txt"
RESULTS_DIR="results/figures"
mkdir -p "$RESULTS_DIR"

for OBJ in DL DT PA; do
  echo "Generating comparison for objective: $OBJ"
  OUT="$RESULTS_DIR/memetracker_comparison_${OBJ}.png"
  python examples/memetracker_heuristics_comparison.py \
    --input "$INPUT" \
    --budgets 1 2 3 4 5 6 7 8 9 10 \
    --max-docs -1 \
    --top-memes -1 \
    --objective $OBJ \
    --output "$OUT"
  echo "Results saved to: $OUT"
done
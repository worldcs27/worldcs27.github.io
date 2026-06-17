#!/usr/bin/env bash
# Zero-Shot Controllability (Option A: code mapping). model3 (III)→IV, model5 (IV)→III.
# Generate → map to target code space → downstream eval (Acc, F1, AUPRC). Results in output/ for Table 3.
# Use your own env (e.g. conda activate sft_lab) before running.

set -e
cd "$(dirname "$0")"

python3 run_zeroshot.py

echo "Done. Check output/zeroshot_table3.csv and output/eval_model3_to_iv/ output/eval_model5_to_iii/"

#!/bin/bash
# Run tail-code prediction for all 6 models × 2 datasets (12 jobs).
# Results in output/tail_predict_summary.csv and output/tail_predict_all.jsonl
cd "$(dirname "$0")"
python3 run_tail_predict.py --dataset all --model all

#!/usr/bin/env bash
# 前 500 尾码、GPU：在选好的环境中执行本脚本即可（例如先 conda activate xxx）
# 结果写入当前目录下 output/tail_predict_summary.csv 与 output/tail_predict_all.jsonl

set -e
cd "$(dirname "$0")"

python3 run_tail_predict.py --dataset all --model all --max_tail 500

echo "Done. Check output/tail_predict_summary.csv and output/tail_predict_all.jsonl"

#!/usr/bin/env bash
# 全部尾码、GPU：在选好的环境中执行本脚本即可（例如先 conda activate xxx）
# 结果写入当前目录下 output/，文件名带 _full 以区分前 500 尾码结果：
#   tail_predict_summary_full.csv
#   tail_predict_all_full.jsonl

set -e
cd "$(dirname "$0")"

python3 run_tail_predict.py --dataset all --model all --out_suffix full

echo "Done. Check output/tail_predict_summary_full.csv and output/tail_predict_all_full.jsonl"

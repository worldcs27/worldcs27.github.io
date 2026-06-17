#!/usr/bin/env bash
# 仅跑 model3 seed3 的下游评估（合成数据已生成：save_anneal/seed3/datasets/haloDataset.pkl）
# 使用 --skip_real 减少显存占用，只得到 MyModel2（AdaPCLA seed3）的 acc/f1。
# 若仍 OOM，可先执行：export CUDA_VISIBLE_DEVICES="" 再运行本脚本（用 CPU 跑评估，较慢但不会爆显存）。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python EVAL_PY \
  --base_data_dir DATA_MIMICIII \
  --mymodel2_path "$SCRIPT_DIR/save_anneal/seed3/datasets/haloDataset.pkl" \
  --save_dir "$SCRIPT_DIR/evaluate_anneal/seed3" \
  --sources MyModel2 \
  --skip_real

echo "===== 完成 ====="
echo "结果 CSV: $SCRIPT_DIR/evaluate_anneal/seed3/compare_real_halo_mymodel2.csv"
echo "查看 MyModel2 行的 mean Accuracy / F1 即为 seed3 的 acc/f1。"

#!/usr/bin/env bash
# AdaPCLA (model3, MIMIC-III): 仅跑 seed3，得到 save_anneal/seed3 下的 haloDataset.pkl 及下游评估（含 Real 对比）。
# 带 --eval：生成后跑下游 25 类诊断分类（含 Real 训练→真实测试集），在真实测试集上得到 acc/f1。
# 使用前请激活含 torch 等依赖的 conda/venv 环境。

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== model3 AdaPCLA seed3 ====="
python run_pcla_fixed_bias_anneal.py \
  --seed 3 \
  --save_dir "$SCRIPT_DIR/save_anneal/seed3" \
  --eval

echo "===== 完成 ====="
echo "haloDataset.pkl 位于: $SCRIPT_DIR/save_anneal/seed3/datasets/haloDataset.pkl"

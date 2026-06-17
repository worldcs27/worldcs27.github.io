#!/usr/bin/env bash
# AdaPCLA (model5, MIMIC-IV)：跑 seed2、seed3，得到 save_anneal_mimiciv/seed2、seed3 下的 haloDataset.pkl 及下游评估。
# 训练/生成使用指定 GPU：通过 CUDA_VISIBLE_DEVICES 指定（脚本内设为 2 和 3，你可改为 4/6/7）。
# 使用前请激活含 torch 等依赖的 conda/venv 环境。
export CUDA_VISIBLE_DEVICES=2,3,4,6,7
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 可选：只使用你指定的显卡 2,3,4,6,7 中的某一块，避免占用其他卡。这里 seed2 用 2，seed3 用 3；可改成 4,6,7。
export CUDA_VISIBLE_DEVICES=2
echo "===== model5 AdaPCLA seed2 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ====="
python run_pcla_fixed_bias_anneal_mimiciv.py \
  --seed 2 \
  --save_dir "$SCRIPT_DIR/save_anneal_mimiciv/seed2" \
  --eval

export CUDA_VISIBLE_DEVICES=3
echo "===== model5 AdaPCLA seed3 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) ====="
python run_pcla_fixed_bias_anneal_mimiciv.py \
  --seed 3 \
  --save_dir "$SCRIPT_DIR/save_anneal_mimiciv/seed3" \
  --eval

echo "===== 完成 ====="
echo "haloDataset.pkl 位于:"
echo "  $SCRIPT_DIR/save_anneal_mimiciv/seed2/datasets/haloDataset.pkl"
echo "  $SCRIPT_DIR/save_anneal_mimiciv/seed3/datasets/haloDataset.pkl"
echo "下游评估 CSV:"
echo "  $SCRIPT_DIR/evaluate_anneal_mimiciv/seed2/compare_real_halo_mymodel2.csv"
echo "  $SCRIPT_DIR/evaluate_anneal_mimiciv/seed3/compare_real_halo_mymodel2.csv"

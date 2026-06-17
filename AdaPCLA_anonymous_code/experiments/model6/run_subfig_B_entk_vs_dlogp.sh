#!/usr/bin/env bash
# =============================================================================
# 子图 B：eNTK 相似度 vs Δlog p（主文 Fig.X(b)）
# 日期: 2026-01-29
# 创建: 2026-01-29
# 用途: 生成 FIG/anneal_entk_vs_dlogp.png，供 KDD main.tex 引用
# 依赖: micro_probe_logs/seed1/micro_probe_ckpt_*.csv 已存在
# =============================================================================

set -e

PROJECT_ROOT=EXPERIMENTS_ROOT
MODEL6_DIR="${PROJECT_ROOT}/model6"
KDD_FIG_DIR="${PROJECT_ROOT}/KDD/FIG"

echo "=============================================="
echo " 子图 B：eNTK vs Δlog p"
echo " 执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "${MODEL6_DIR}"

# Step 1: 分析 eNTK 相似度与 Δlog p 的关系，输出 entk_vs_dlogp.csv
echo "[Step 1/3] Running analyze_entk_vs_dlogp.py ..."
python analyze_entk_vs_dlogp.py \
    --logs_dir "${MODEL6_DIR}/micro_probe_logs/seed1" \
    --out_csv "${MODEL6_DIR}/entk_vs_dlogp.csv"

# Step 2: 绘制散点图，输出 fig_entk_vs_dlogp.png
echo "[Step 2/3] Running plot_entk_vs_dlogp.py ..."
python plot_entk_vs_dlogp.py

# Step 3: 复制到 KDD/FIG，供主文引用
mkdir -p "${KDD_FIG_DIR}"
cp -v "${MODEL6_DIR}/fig_entk_vs_dlogp.png" "${KDD_FIG_DIR}/anneal_entk_vs_dlogp.png"

echo ""
echo "=============================================="
echo " 完成: ${KDD_FIG_DIR}/anneal_entk_vs_dlogp.png"
echo " 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

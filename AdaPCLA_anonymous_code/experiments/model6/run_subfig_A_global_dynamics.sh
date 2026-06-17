#!/usr/bin/env bash
# =============================================================================
# 子图 A：全局概率演化（30 个 probing contexts 平均 Δlog p 随退火步数）
# 创建: 2026-01-29
# 用途: 生成 FIG/anneal_global_dynamics.png，供 KDD main.tex Fig.X(a) 引用
# 依赖: micro_probe_logs/seed1/micro_probe_ckpt_*.csv 与 micro_probe_oracle.csv
# =============================================================================

set -e

PROJECT_ROOT=EXPERIMENTS_ROOT
MODEL6_DIR="${PROJECT_ROOT}/model6"
KDD_FIG_DIR="${PROJECT_ROOT}/KDD/FIG"

echo "=============================================="
echo " 子图 A：Global dynamics (avg Δlog p, 30 contexts)"
echo " 执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "${MODEL6_DIR}"

python plot_global_dynamics.py \
    --logs_dir "${MODEL6_DIR}/micro_probe_logs/seed1" \
    --oracle_csv "${MODEL6_DIR}/micro_probe_logs/seed1/micro_probe_oracle.csv" \
    --out "${MODEL6_DIR}/fig_global_dynamics.png"

mkdir -p "${KDD_FIG_DIR}"
cp -v "${MODEL6_DIR}/fig_global_dynamics.png" "${KDD_FIG_DIR}/anneal_global_dynamics.png"

echo ""
echo "=============================================="
echo " 完成: ${KDD_FIG_DIR}/anneal_global_dynamics.png"
echo " 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

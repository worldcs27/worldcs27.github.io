#!/usr/bin/env bash
# =============================================================================
# 子图 A（100 contexts）：全局概率演化 — 100 个 probing contexts 上三类（Related / Unrelated / Wrong）
# 的平均 Δlog p 随退火步数变化。
# =============================================================================

set -e

PROJECT_ROOT=EXPERIMENTS_ROOT
MODEL6_DIR="${PROJECT_ROOT}/model6"
KDD_FIG_DIR="${PROJECT_ROOT}/KDD/FIG"

echo "=============================================="
echo " 子图 A (100ctx)：Global dynamics (avg Δlog p, 100 contexts)"
echo " 执行时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

cd "${MODEL6_DIR}"

python plot_global_dynamics.py \
    --logs_dir "${MODEL6_DIR}/micro_probe_logs_100ctx/seed1" \
    --oracle_csv "${MODEL6_DIR}/micro_probe_logs/seed1/micro_probe_oracle.csv" \
    --out "${MODEL6_DIR}/fig_global_dynamics_100ctx.png"

mkdir -p "${KDD_FIG_DIR}"
cp -v "${MODEL6_DIR}/fig_global_dynamics_100ctx.png" "${KDD_FIG_DIR}/anneal_global_dynamics_100ctx.png"

echo ""
echo "=============================================="
echo " 完成: ${KDD_FIG_DIR}/anneal_global_dynamics_100ctx.png"
echo " 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="


#!/usr/bin/env bash

# model6：MIMIC-IV data2 + 固定 Bias 退火 + 微观探测 + 概率轨迹记录（100 个 probing contexts）
# 说明：与 run_micro_probing_fixed_bias_anneal_mimiciv.sh 相同，只是使用 100-context 配置与单独日志目录。

set -e

PROJECT_ROOT=EXPERIMENTS_ROOT
MODEL5_DIR="${PROJECT_ROOT}/model5"
MODEL6_DIR="${PROJECT_ROOT}/model6"

DATA_DIR=DATA_MIMICIV    # MIMIC-IV data2

# 微观探测集配置（100 个 context；保证包含原 30 个 context）
MICRO_PROBE_CONFIG="${MODEL6_DIR}/micro_probe_configs/mimiciv_long_tail_triplets_seed1_100ctx.csv"

MICRO_PROBE_CKPT_PER_EPOCH=10
EPOCHS=10

SEED=1
SAVE_DIR="${MODEL6_DIR}/save_micro_probe_mimiciv_100ctx/seed${SEED}"
LOG_DIR="${MODEL6_DIR}/micro_probe_logs_100ctx/seed${SEED}"

echo "[model6-100ctx] Micro-probing fixed-bias-anneal on MIMIC-IV (seed=${SEED})"
echo "  DATA_DIR       = ${DATA_DIR}"
echo "  SAVE_DIR       = ${SAVE_DIR}"
echo "  MICRO_PROBECFG = ${MICRO_PROBE_CONFIG}"
echo "  LOG_DIR        = ${LOG_DIR}"
echo "  CKPT/epoch     = ${MICRO_PROBE_CKPT_PER_EPOCH} (=> ${EPOCHS} * 10 = 100 probability checkpoints)"

mkdir -p "${SAVE_DIR}"
mkdir -p "$(dirname "${MICRO_PROBE_CONFIG}")"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${MICRO_PROBE_CONFIG}" ]]; then
  echo "[model6-100ctx] ERROR: Micro-probe config not found: ${MICRO_PROBE_CONFIG}"
  echo "  Generate it with:"
  echo "    cd ${MODEL6_DIR}"
  echo "    python gen_micro_probe_config.py --out mimiciv_long_tail_triplets_seed1_100ctx.csv --n_ctx 100 --seed 1 --with_entk --base_config mimiciv_long_tail_triplets_seed1.csv"
  exit 1
fi
N_LINES=$(wc -l < "${MICRO_PROBE_CONFIG}")
if [[ "${N_LINES}" -lt 2 ]]; then
  echo "[model6-100ctx] ERROR: Config has no data rows (only ${N_LINES} line(s))."
  exit 1
fi
echo "[model6-100ctx] Micro-probe config OK: ${MICRO_PROBE_CONFIG} (${N_LINES} lines)"

python "${MODEL5_DIR}/run_pcla_fixed_bias_anneal_mimiciv.py" \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --eval \
  --save_micro_probe_ckpts \
  --log_micro_probe \
  --micro_probe_config "${MICRO_PROBE_CONFIG}" \
  --micro_probe_ckpt_per_epoch "${MICRO_PROBE_CKPT_PER_EPOCH}" \
  --micro_probe_out_dir "${LOG_DIR}"


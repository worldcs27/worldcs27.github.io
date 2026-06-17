#!/usr/bin/env bash

# model6：MIMIC-IV data2 + 固定 Bias 退火 + 微观探测 + 概率轨迹记录
# 说明：对应 EXPERIMENT_micro_probing_fixed_bias_anneal_mimiciv.md。
# Python 已支持 --log_micro_probe / --micro_probe_config / --micro_probe_ckpt_per_epoch / --micro_probe_out_dir。

set -e

PROJECT_ROOT=EXPERIMENTS_ROOT
MODEL5_DIR="${PROJECT_ROOT}/model5"
MODEL6_DIR="${PROJECT_ROOT}/model6"

DATA_DIR=DATA_MIMICIV    # MIMIC-IV data2

# 微观探测集配置（CSV：context_id, disease_id, type；可由 gen_micro_probe_config.py 生成）
MICRO_PROBE_CONFIG="${MODEL6_DIR}/micro_probe_configs/mimiciv_long_tail_triplets_seed1.csv"

# 每个 epoch 内记录微观探测的 checkpoint 次数：
# - 这里设置为 10，即每 0.1 个 epoch 记录一次，总共 10 epoch -> 100 个 checkpoint。
MICRO_PROBE_CKPT_PER_EPOCH=10

# 训练 epoch 总数（与 model5 保持一致）
EPOCHS=10

# ========== Seed 1 ==========
SEED=1
SAVE_DIR="${MODEL6_DIR}/save_micro_probe_mimiciv/seed${SEED}"

echo "[model6] Micro-probing fixed-bias-anneal on MIMIC-IV (seed=${SEED})"
echo "  DATA_DIR       = ${DATA_DIR}"
echo "  SAVE_DIR       = ${SAVE_DIR}"
echo "  MICRO_PROBECFG = ${MICRO_PROBE_CONFIG}"
echo "  CKPT/epoch     = ${MICRO_PROBE_CKPT_PER_EPOCH} (=> ${EPOCHS} * 10 = 100 probability checkpoints)"
echo "  SAVE_MODEL     = 每 0.1 epoch 存一次 => 100 个 model checkpoint (epoch_ckpts/micro_probe_ckpt_0000..0099.pt)"

mkdir -p "${SAVE_DIR}"
mkdir -p "$(dirname "${MICRO_PROBE_CONFIG}")"

# 开跑前自检：无配置 CSV 时提示并退出，避免整晚白跑
if [[ ! -f "${MICRO_PROBE_CONFIG}" ]]; then
  echo "[model6] ERROR: Micro-probe config not found: ${MICRO_PROBE_CONFIG}"
  echo "  Generate it with: python ${MODEL6_DIR}/gen_micro_probe_config.py --out mimiciv_long_tail_triplets_seed1.csv"
  exit 1
fi
N_LINES=$(wc -l < "${MICRO_PROBE_CONFIG}")
if [[ "${N_LINES}" -lt 2 ]]; then
  echo "[model6] ERROR: Config has no data rows (only ${N_LINES} line(s))."
  exit 1
fi
echo "[model6] Micro-probe config OK: ${MICRO_PROBE_CONFIG} (${N_LINES} lines)"

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
  --micro_probe_out_dir "${MODEL6_DIR}/micro_probe_logs/seed${SEED}"

# 运行完成后，预期在：
#   - ${SAVE_DIR} 下有固定 bias + 退火的主 checkpoint 与合成数据
#   - ${MODEL6_DIR}/micro_probe_logs/seed${SEED} 下有：
#       * 每个 checkpoint 的 logit / prob / log-prob 轨迹（例如 CSV/NPY）
#       * 方便后续在 notebook / KDD main.tex 中画出：
#           - 单个 context 的三轨迹 + Oracle 水平线
#           - 各 type 的平均轨迹
#           - eNTK 相似度 vs. Δlog p 散点图等


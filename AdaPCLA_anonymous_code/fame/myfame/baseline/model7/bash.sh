#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TORCHRUN_BIN="${TORCHRUN:-}"
if [[ -z "${TORCHRUN_BIN}" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    TORCHRUN_BIN="torchrun"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/torchrun" ]]; then
    TORCHRUN_BIN="${CONDA_PREFIX}/bin/torchrun"
  elif [[ -x torchrun ]]; then
    TORCHRUN_BIN=torchrun
  else
    echo "torchrun not found. Activate your conda env or set TORCHRUN=/path/to/torchrun" >&2
    exit 127
  fi
fi

# -------- DDP / GPU settings --------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  IFS=',' read -r -a _gpu_arr <<<"${CUDA_VISIBLE_DEVICES}"
  NUM_GPUS="${#_gpu_arr[@]}"
fi
MASTER_PORT="${MASTER_PORT:-29540}"

# -------- Paths --------
DATA_DIR="${DATA_DIR:-DATA_MIMICIII}"
SAVE_DIR="${SAVE_DIR:-MODEL7_DIR/save}"

# -------- Train hyperparams (override via env vars) --------
SEED="${SEED:-4}"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-48}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POS_LOSS_WEIGHT="${POS_LOSS_WEIGHT:-}" # empty => None

LOGIT_ADJUST_TAU="${LOGIT_ADJUST_TAU:-1.0}"
LOGIT_ADJUST_CLIP="${LOGIT_ADJUST_CLIP:-10.0}"
APPLY_LOGIT_ADJUST_IN_SAMPLING="${APPLY_LOGIT_ADJUST_IN_SAMPLING:-1}" # 1=true, 0=false
RESUME="${RESUME:-1}" # 1=true, 0=false
INIT_CKPT_PATH="${INIT_CKPT_PATH:-HALO_MIMICIII_CKPT}"

# -------- Generation settings --------
TOTAL_SAMPLES="${TOTAL_SAMPLES:-33494}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [train|test|all]

Env overrides:
  CUDA_VISIBLE_DEVICES, NUM_GPUS, MASTER_PORT
  DATA_DIR, SAVE_DIR
  SEED
  LR, EPOCHS, BATCH_SIZE, SAMPLE_BATCH_SIZE, NUM_WORKERS
  POS_LOSS_WEIGHT
  LOGIT_ADJUST_TAU, LOGIT_ADJUST_CLIP
  APPLY_LOGIT_ADJUST_IN_SAMPLING=1|0
  RESUME=1|0
  INIT_CKPT_PATH
  TOTAL_SAMPLES
EOF
}

MODE="${1:-all}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi
EXTRA_ARGS=("${@:2}")

COMMON_ARGS=(--data_dir "${DATA_DIR}" --save_dir "${SAVE_DIR}" --num_workers "${NUM_WORKERS}")

TRAIN_ARGS=(
  --seed "${SEED}"
  --lr "${LR}"
  --epoch "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --sample_batch_size "${SAMPLE_BATCH_SIZE}"
  --pos_loss_weight "${POS_LOSS_WEIGHT}"
  --logit_adjust_tau "${LOGIT_ADJUST_TAU}"
  --logit_adjust_clip "${LOGIT_ADJUST_CLIP}"
)
if [[ -n "${INIT_CKPT_PATH}" ]]; then
  TRAIN_ARGS+=(--init_ckpt_path "${INIT_CKPT_PATH}")
fi
if [[ "${APPLY_LOGIT_ADJUST_IN_SAMPLING}" == "0" ]]; then
  TRAIN_ARGS+=(--no-apply_logit_adjust_in_sampling)
fi
if [[ "${RESUME}" == "0" ]]; then
  TRAIN_ARGS+=(--no-resume)
fi

TEST_ARGS=(
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --sample_batch_size "${SAMPLE_BATCH_SIZE}"
  --total_samples "${TOTAL_SAMPLES}"
)

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPUS=${NUM_GPUS} MASTER_PORT=${MASTER_PORT}"
echo "DATA_DIR=${DATA_DIR}"
echo "SAVE_DIR=${SAVE_DIR}"
echo "SEED=${SEED}"
echo "TORCHRUN=${TORCHRUN_BIN}"

run_train() {
  echo "Starting Model7 DDP training..."
  "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train.py" "${COMMON_ARGS[@]}" "${TRAIN_ARGS[@]}" "${EXTRA_ARGS[@]}"
}

run_test() {
  echo "Starting Model7 DDP test + generation..."
  "${TORCHRUN_BIN}" --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/test.py" "${COMMON_ARGS[@]}" "${TEST_ARGS[@]}" "${EXTRA_ARGS[@]}"
}

case "${MODE}" in
  train) run_train ;;
  test) run_test ;;
  all) run_train; run_test ;;
  *)
    echo "Unknown mode: ${MODE}"
    usage
    exit 2
    ;;
esac

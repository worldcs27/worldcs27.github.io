#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PYTHON="${PYTHON:-sys.executable}"

# DDP config (override via env)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-7}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-29560}"

# Tuning grid (override via env)
TAUS="${TAUS:-0.2,0.5,1.0}"
CLIPS="${CLIPS:-5,10,15}"

# Paths (override via env)
DATA_DIR="${DATA_DIR:-DATA_MIMICIII}"
HALO_PATH="${HALO_PATH:-FAME_ROOT"
EVAL_SCRIPT="${EVAL_SCRIPT:-$REPO_ROOT/fame/myfame/evaluate/evaluate_synthetic_training.py}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/save/tune_logit_adjust_$(date +%Y%m%d_%H%M%S)}"

echo "OUT_DIR=$OUT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS=$NUM_GPUS"
echo "MASTER_PORT_BASE=$MASTER_PORT_BASE"
echo "TAUS=$TAUS"
echo "CLIPS=$CLIPS"
echo "DATA_DIR=$DATA_DIR"
echo "HALO_PATH=$HALO_PATH"
echo "EVAL_SCRIPT=$EVAL_SCRIPT"

"$PYTHON" "$SCRIPT_DIR/tune_logit_adjust.py" \
  --out_dir "$OUT_DIR" \
  --python "$PYTHON" \
  --eval_script "$EVAL_SCRIPT" \
  --data_dir "$DATA_DIR" \
  --halo_path "$HALO_PATH" \
  --num_gpus "$NUM_GPUS" \
  --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
  --master_port_base "$MASTER_PORT_BASE" \
  --taus "$TAUS" \
  --clips "$CLIPS" \
  "$@"


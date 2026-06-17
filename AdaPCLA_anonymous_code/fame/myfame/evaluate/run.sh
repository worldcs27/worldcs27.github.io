#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env vars)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29505}

echo "Running evaluation with ${NUM_GPUS} GPUs (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})..."

# Run evaluation with Torchrun (DDP). Pass through any extra args to the script.
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/evaluate_synthetic_training.py" "$@"

echo "Evaluation finished."

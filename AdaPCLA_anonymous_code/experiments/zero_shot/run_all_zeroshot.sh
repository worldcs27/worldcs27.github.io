#!/usr/bin/env bash
# Run full zero-shot suite: AdaPCLA + baselines (GPT, LSTM, EVA, SynTEG, HALO).
# Output: mywork/zero-shot/output/zeroshot_baselines_table.csv
#         mywork/output/zero-shot/output/zeroshot_table3.csv (AdaPCLA)
# Use: conda activate your_env; export NUM_GPUS=1; ./run_all_zeroshot.sh

set -e
ZERO_SHOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ZERO_SHOT_DIR"

echo "=== 1. AdaPCLA zero-shot (III→IV, IV→III) ==="
if [[ -f "$ZERO_SHOT_DIR/../output/zero-shot/run_zeroshot.sh" ]]; then
  bash "$ZERO_SHOT_DIR/../output/zero-shot/run_zeroshot.sh"
else
  echo "Skip AdaPCLA: run_zeroshot.sh not found at ../output/zero-shot/"
fi

echo "=== 2. Baselines zero-shot (GPT, LSTM, EVA, SynTEG, HALO) ==="
python3 run_baselines_zeroshot.py --baselines gpt,lstm,eva,synteg,halo --directions iii_to_iv,iv_to_iii

echo "Done. AdaPCLA: ../output/zero-shot/output/zeroshot_table3.csv"
echo "      Baselines: $ZERO_SHOT_DIR/output/zeroshot_baselines_table.csv"

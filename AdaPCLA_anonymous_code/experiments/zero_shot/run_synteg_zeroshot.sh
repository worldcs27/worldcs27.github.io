#!/usr/bin/env bash
# Zero-shot for SynTEG only (III→IV and IV→III).
set -e
cd "$(dirname "$0")"
python3 run_baselines_zeroshot.py --baselines synteg --directions iii_to_iv,iv_to_iii
echo "Done. See output/zeroshot_baselines_table.csv"

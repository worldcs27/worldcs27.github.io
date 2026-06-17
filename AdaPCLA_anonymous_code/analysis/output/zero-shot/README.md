# Zero-Shot Controllability (Table 3, Option A: code mapping)

- **model3** (III) + π_target(IV) → generate (III indices) → **map III→IV code space** → downstream eval on real MIMIC-IV.
- **model5** (IV) + π_target(III) → generate (IV indices) → **map IV→III code space** → downstream eval on real MIMIC-III.

Baseline numbers from main table (理解 B); only Ours (zero-shot) are produced here.

## Run

```bash
cd EXPERIMENTS_ROOT/output/zero-shot
# optional: conda activate your_env
./run_zeroshot.sh
```

## Output

- `output/model3_to_iv_syn.pkl`, `output/model5_to_iii_syn.pkl` — raw generated data (source-domain indices).
- `output/model3_to_iv_syn_mapped.pkl`, `output/model5_to_iii_syn_mapped.pkl` — mapped to target-domain code space (used for eval).
- `output/eval_model3_to_iv/compare_real_halo_mymodel2.csv` — per-label Acc/F1/AUPRC (model3→IV).
- `output/eval_model5_to_iii/compare_real_halo_mymodel2.csv` — per-label (model5→III).
- `output/zeroshot_table3.csv` — mean Acc, F1, AUPRC for Table 3 (Ours rows; baseline from main table).

## Paths

See `paths.py`. Requires model3 checkpoint `model3/save_anneal/seed1/model_anneal.pt` and model5 checkpoint `model5/save_anneal_mimiciv/seed1/model_anneal_mimiciv.pt`.

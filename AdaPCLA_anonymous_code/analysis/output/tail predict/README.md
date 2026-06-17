# Tail-Code Prediction (Task ii)

Given non-tail codes in a visit, predict tail code occurrence. Train downstream classifier on **synthetic** data (one generator per run), evaluate on **real** test set.

## Metrics

- **AUPRC (macro)**: Average precision (area under PR curve) per tail code, then averaged.
- **Top-K recall**: For each test visit with at least one tail code, recall = |true ∩ pred_topK| / |true|; averaged over visits. Default K=10.

## Usage

```bash
# Run all 6 models × 2 datasets (12 jobs). Results in output/
python run_tail_predict.py

# Single dataset / model
python run_tail_predict.py --dataset mimic4 --model AdaPCLA
python run_tail_predict.py --dataset mimic3 --model HALO

# Quick test (subsample tail codes, fewer epochs)
python run_tail_predict.py --dataset mimic4 --model AdaPCLA --max_tail 100 --epochs 20

# Quick run for all 12 jobs, write to *_quick.csv (does not overwrite main run)
python run_tail_predict.py --dataset all --model all --max_tail 50 --epochs 10 --out_suffix quick
```

## Options

- `--dataset`: mimic3 | mimic4 | all (default: all)
- `--model`: GPT-style | LSTM | EVA | SynTEG | HALO | AdaPCLA | all (default: all)
- `--epochs`: MLP max_iter (default: 30)
- `--batch_size`: default 256
- `--hidden`: MLP hidden size (default: 256)
- `--max_tail`: Cap number of tail codes (default: all). Use for quick tests.
- `--out_suffix`: Write to `tail_predict_summary_<suffix>.csv` and `tail_predict_all_<suffix>.jsonl` so a quick run does not overwrite the main run.

## Output

- `output/tail_predict_summary.csv`: dataset, model, auprc_macro, top10_recall (successful runs only). **Appended after each job** when running `--dataset all --model all`, so you can see progress.
- `output/tail_predict_all.jsonl`: All runs including errors (one dict per line). Also appended per job.

## Why it can be very slow

Each (dataset, model) job trains **one MLP per tail code** (MIMIC-III ~1289 tail codes, MIMIC-IV ~1736). So one full job can take many hours. For a quick sanity check, use **`--max_tail 50 --epochs 10`** (or similar) so each job finishes in minutes.

## Paths

Paths are in `config.py` (relative to PCLA repo root). MIMIC-III uses `fame/myfame/data/` and baseline `save/`; MIMIC-IV uses `fame/myfame/data2/` and baseline `save_mimiciv_seed1/`. AdaPCLA: `mywork/model3/save_anneal/seed1` (MIMIC-III), `mywork/model5/save_anneal_mimiciv/seed1` (MIMIC-IV).

## Runtime

With **full tail codes** (no `--max_tail`), each job trains hundreds/thousands of MLPs; one job can take **several hours to tens of hours**. All 12 runs can take days. Use `--max_tail 50` or `100` for faster experiments (minutes per job).

#!/usr/bin/env python3
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
"""
Paths for zero-shot baseline runs: GPT, LSTM, EVA, SynTEG, HALO.
- III = MIMIC-III (HALO/save), IV = MIMIC-IV (data2).
- Each baseline: save/ = trained on III, save_mimiciv_seed1/ = trained on IV.
"""
from pathlib import Path

ZERO_SHOT_DIR = Path(__file__).resolve().parent
MYWORK = ZERO_SHOT_DIR.parent
PCLA_ROOT = MYWORK.parent
FAME = PCLA_ROOT / "fame" / "myfame"

OUT_DIR = ZERO_SHOT_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Data: source/target for zero-shot
DATA_III = FAME / "baseline" / "HALO" / "save"
DATA_IV = FAME / "data2"

# Baseline dirs (fame/myfame/baseline)
BASELINE = FAME / "baseline"

# Per-baseline: (data_dir for vocab when generating, model_dir for ckpt, output dataset filename)
# III-trained: data_dir=III, model_dir=baseline/save  → generate in III index space → map to IV for III→IV
# IV-trained:  data_dir=IV,  model_dir=baseline/save_mimiciv_seed1 → generate in IV index space → map to III for IV→III
BASELINES = {
    "gpt": {
        "script_dir": BASELINE / "gpt",
        "test_script": "test_gpt.py",
        "ckpt_name_iii": "gpt_model.pt",
        "ckpt_name_iv": "gpt_model.pt",
        "dataset_name": "gptDataset.pkl",
        "save_subdir_iii": "save",
        "save_subdir_iv": "save_mimiciv_seed1",
    },
    "lstm": {
        "script_dir": BASELINE / "lstm",
        "test_script": "test_lstm.py",
        "ckpt_name_iii": "lstm_model.pt",
        "ckpt_name_iv": "lstm_model.pt",
        "dataset_name": "lstmDataset.pkl",
        "save_subdir_iii": "save",
        "save_subdir_iv": "save_mimiciv_seed1",
    },
    "eva": {
        "script_dir": BASELINE / "eva",
        "test_script": "test_eva.py",
        "ckpt_name_iii": "eva_model.pt",
        "ckpt_name_iv": "eva_model.pt",
        "dataset_name": "evaDataset.pkl",
        "save_subdir_iii": "save",
        "save_subdir_iv": "save_mimiciv_seed1",
    },
    "synteg": {
        "script_dir": BASELINE / "synteg",
        "test_script": "test_synteg.py",
        "ckpt_name_iii": None,  # uses dep + gan
        "ckpt_name_iv": None,
        "dataset_name": "syntegDataset.pkl",
        "save_subdir_iii": "save",
        "save_subdir_iv": "save_mimiciv_seed1",
    },
    "halo": {
        "script_dir": BASELINE / "HALO",
        "test_script": "test.py",
        "ckpt_name_iii": "halo_model_ddp.pt",
        "ckpt_name_iv": "halo_model_ddp.pt",
        "dataset_name": "haloDataset.pkl",
        "save_subdir_iii": "save",
        "save_subdir_iv": "save_mimiciv_seed1",
    },
}

# Downstream evaluation script
EVAL_PY = str(FAME / "evaluate" / "evaluate_synthetic_training.py")

TOTAL_SAMPLES = 50000
SEED = 4

#!/usr/bin/env python3
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
"""
Paths for Zero-Shot Controllability experiment.
- model3: trained on MIMIC-III (HALO/save); zero-shot target = MIMIC-IV.
- model5: trained on MIMIC-IV (data2); zero-shot target = MIMIC-III.
Results (Acc, F1, AUPRC) go to output/ for Table 3.
"""
from pathlib import Path

ZERO_SHOT_DIR = Path(__file__).resolve().parent
# zero-shot 在 mywork/output/zero-shot，故 parent.parent = mywork
MYWORK = ZERO_SHOT_DIR.parent.parent
PCLA_ROOT = MYWORK.parent
FAME = PCLA_ROOT / "fame" / "myfame"

OUT_DIR = ZERO_SHOT_DIR / "output"

# MIMIC-III data (HALO save: train/val/test, codeToIndex, idToLabel)
DATA_III = FAME / "baseline" / "HALO" / "save"
# MIMIC-IV data (data2)
DATA_IV = FAME / "data2"

# Model3: trained on MIMIC-III (AdaPCLA save_anneal seed1)
MODEL3_CKPT = MYWORK / "model3" / "save_anneal" / "seed1" / "model_anneal.pt"
MODEL3_DIR = str(MYWORK / "model3")  # for sys.path; model3 uses MODEL7_DIR
MODEL7_DIR = str(PCLA_ROOT / "fame" / "myfame" / "baseline" / "model7")

# Model5: trained on MIMIC-IV
MODEL5_CKPT = MYWORK / "model5" / "save_anneal_mimiciv" / "seed1" / "model_anneal_mimiciv.pt"
MODEL5_SCRIPT_DIR = str(MYWORK / "model5")
MODEL8_DIR = str(PCLA_ROOT / "fame" / "myfame" / "baseline" / "model8")

# Downstream evaluation script (Acc, F1, AUPRC)
EVAL_PY = str(FAME / "evaluate" / "evaluate_synthetic_training.py")

# Logit adjustment (same as model3/model5)
LOGIT_ADJUST_TAU = 0.2
LOGIT_ADJUST_EPS = 1e-8
LOGIT_ADJUST_CLIP = 15.0

# Generation
SAMPLE_BATCH_SIZE = 256
N_CTX = 48
TOTAL_SAMPLES = 50000  # match typical eval size

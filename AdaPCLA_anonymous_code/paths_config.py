"""Central path configuration for the AdaPCLA anonymous code release.

Set ADAPCLA_ROOT to the unpacked repository root, or rely on auto-detection.
MIMIC data and pretrained checkpoints are not included in this archive; set the
environment variables below before running full experiments.
"""
from __future__ import annotations

import os


ROOT = os.environ.get("ADAPCLA_ROOT", os.path.dirname(os.path.abspath(__file__)))
FAME_ROOT = os.path.join(ROOT, "fame", "myfame")
EXPERIMENTS_ROOT = os.path.join(ROOT, "experiments")
ANALYSIS_ROOT = os.path.join(ROOT, "analysis")

# Preprocessed EHR task data. Users must obtain MIMIC access separately and
# prepare these task files following the expected HALO-style schema.
DATA_MIMICIII = os.environ.get(
    "DATA_MIMICIII", os.path.join(FAME_ROOT, "baseline", "HALO", "save")
)
DATA_MIMICIV = os.environ.get("DATA_MIMICIV", os.path.join(FAME_ROOT, "data2"))

# Baseline model code included in this release.
MODEL7_DIR = os.path.join(FAME_ROOT, "baseline", "model7")
MODEL8_DIR = os.path.join(FAME_ROOT, "baseline", "model8")
EVAL_DIR = os.path.join(FAME_ROOT, "evaluate")
EVAL_PY = os.path.join(EVAL_DIR, "evaluate_synthetic_training.py")

# HALO initialization checkpoints. They are intentionally excluded from this
# archive and should be supplied by the user when reproducing full training.
HALO_MIMICIII_CKPT = os.environ.get(
    "HALO_MIMICIII_CKPT",
    os.path.join(FAME_ROOT, "baseline", "model2", "save", "model2_halo_logit.pt"),
)
HALO_MIMICIV_CKPT = os.environ.get(
    "HALO_MIMICIV_CKPT",
    os.path.join(
        FAME_ROOT,
        "baseline",
        "model8",
        "save_mimiciv_seed1_best",
        "best_ckpt",
        "model8.pt",
    ),
)

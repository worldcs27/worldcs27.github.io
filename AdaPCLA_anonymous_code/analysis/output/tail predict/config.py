#!/usr/bin/env python3
"""
Config for tail-code prediction (task ii): paths for MIMIC-III and MIMIC-IV,
real train/test and synthetic data per model (GPT-style, LSTM, EVA, SynTEG, HALO, AdaPCLA), seed1.
All paths relative to PCLA repo root (ADAPCLA_ROOT).
"""
from pathlib import Path

# PCLA repo root (parent of fame/ and mywork/)
PCLA_ROOT = Path(__file__).resolve().parents[3]  # tail predict -> output -> mywork -> PCLA
FAME = PCLA_ROOT / "fame" / "myfame"
MYWORK = PCLA_ROOT / "mywork"
OUT_DIR = Path(__file__).resolve().parent / "output"

DATASETS = {
    "mimic3": {
        "real_train": FAME / "data" / "trainDataset.pkl",
        "real_test": FAME / "data" / "testDataset.pkl",
        "bucket_csv": FAME / "output" / "长尾分布问题分析" / "mimiciii_code_buckets.csv",
    },
    "mimic4": {
        "real_train": FAME / "data2" / "trainDataset.pkl",
        "real_test": FAME / "data2" / "testDataset.pkl",
        "bucket_csv": FAME / "output" / "长尾分布问题分析" / "mimiciv_code_buckets.csv",
    },
}

# Synthetic data paths: seed1. GPT/EVA use single merged .pkl when present.
MODELS = {
    "GPT-style": {
        "mimic3": FAME / "baseline" / "gpt" / "save" / "gen_seed1_20260109_133326" / "datasets" / "gptDataset.pkl",
        "mimic4": FAME / "baseline" / "gpt" / "save_mimiciv_seed1" / "datasets" / "gptDataset.pkl",
    },
    "LSTM": {
        "mimic3": FAME / "baseline" / "lstm" / "save" / "gen_seed1_20260109_133326" / "datasets" / "lstmDataset.pkl",
        "mimic4": FAME / "baseline" / "lstm" / "save_mimiciv_seed1" / "datasets" / "lstmDataset.pkl",
    },
    "EVA": {
        "mimic3": FAME / "baseline" / "eva" / "save" / "gen_seed1_20260109_133326" / "datasets" / "evaDataset.pkl",
        "mimic4": FAME / "baseline" / "eva" / "save_mimiciv_seed1" / "datasets" / "evaDataset.pkl",
    },
    "SynTEG": {
        "mimic3": FAME / "baseline" / "synteg" / "save" / "gen_seed1_20260109_133326" / "datasets" / "syntegDataset.pkl",
        "mimic4": FAME / "baseline" / "synteg" / "save_mimiciv_seed1" / "datasets" / "syntegDataset.pkl",
    },
    "HALO": {
        "mimic3": FAME / "baseline" / "HALO" / "save" / "datasets" / "haloDataset.pkl",
        "mimic4": FAME / "baseline" / "HALO2" / "save_mimiciv_seed1" / "datasets" / "haloDataset.pkl",
    },
    "AdaPCLA": {
        "mimic3": MYWORK / "model3" / "save_anneal" / "seed1" / "datasets" / "haloDataset.pkl",
        "mimic4": MYWORK / "model5" / "save_anneal_mimiciv" / "seed1" / "datasets" / "haloDataset.pkl",
    },
}

TOP_K = 10  # for Top-K recall


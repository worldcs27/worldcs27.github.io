#!/usr/bin/env python3
"""
Frequency–rank plot (Option A): Real vs HALO vs LSTM vs GPT vs AdaPCLA.
Supports MIMIC-III or MIMIC-IV via first arg (default: mimic4).
Output: extend/freq_rank_models/output/freq_rank_models_{mimic3|mimic4}.png
"""
import sys
from pathlib import Path
from collections import Counter
import pickle
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
MYWORK = SCRIPT_DIR.parent.parent
PCLA_ROOT = MYWORK.parent
FAME = PCLA_ROOT / "fame" / "myfame"
OUT_DIR = SCRIPT_DIR / "output"
MAX_RANK = None  # no truncation; show full distribution including tail

PATHS = {
    "mimic3": {
        "real": FAME / "data" / "trainDataset.pkl",
        "HALO": FAME / "baseline" / "HALO" / "save" / "datasets" / "haloDataset.pkl",
        "LSTM": FAME / "baseline" / "lstm" / "save" / "gen_seed1_20260109_133326" / "datasets" / "lstmDataset.pkl",
        "GPT": FAME / "baseline" / "gpt" / "save" / "gen_seed1_20260109_133326" / "datasets" / "gptDataset.pkl",
        "AdaPCLA": MYWORK / "model3" / "save_anneal" / "seed1" / "datasets" / "haloDataset.pkl",
    },
    "mimic4": {
        "real": FAME / "data2" / "trainDataset.pkl",
        "HALO": FAME / "baseline" / "HALO2" / "save_mimiciv_seed1" / "datasets" / "haloDataset.pkl",
        "LSTM": FAME / "baseline" / "lstm" / "save_mimiciv_seed1" / "datasets" / "lstmDataset.pkl",
        "GPT": FAME / "baseline" / "gpt" / "save_mimiciv_seed1" / "datasets" / "gptDataset.pkl",
        "AdaPCLA": MYWORK / "model5" / "save_anneal_mimiciv" / "seed1" / "datasets" / "haloDataset.pkl",
    },
}


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def get_freq_rank(data: list) -> tuple[np.ndarray, np.ndarray]:
    cnt = Counter()
    for p in data:
        for v in p.get("visits", []):
            for c in v:
                cnt[int(c)] += 1
    freqs = np.array(sorted(cnt.values(), reverse=True))
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    if MAX_RANK is not None and len(ranks) > MAX_RANK:
        ranks = ranks[:MAX_RANK]
        freqs = freqs[:MAX_RANK]
    return ranks, freqs


def main():
    dataset = (sys.argv[1] if len(sys.argv) > 1 else "mimic4").lower()
    if dataset not in PATHS:
        dataset = "mimic4"
    paths = PATHS[dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = []
    print(f"Loading {dataset.upper()} Real train...")
    d_real = load_pkl(paths["real"]) if paths["real"].exists() else []
    r, f = get_freq_rank(d_real)
    curves.append(("Real", r, f, "C0", 2.0))
    for name in ["HALO", "LSTM", "GPT", "AdaPCLA"]:
        path = paths[name]
        if path.exists():
            print(f"Loading {name}...")
            d = load_pkl(path)
            r, f = get_freq_rank(d)
            curves.append((name, r, f, None, 1.2))
        else:
            print(f"Skip {name} (not found)")
    plt.figure(figsize=(7, 5))
    colors = ["C0", "C1", "C2", "C3", "C4"]
    for i, (name, ranks, freqs, c, lw) in enumerate(curves):
        plt.plot(ranks, np.log1p(freqs), label=name, linewidth=lw, alpha=0.9, color=colors[i % len(colors)] if c is None else c)
    plt.xscale("log")
    plt.xlabel("Code rank (1 = most frequent, log scale)")
    plt.ylabel("log(1 + frequency)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / f"freq_rank_models_{dataset}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

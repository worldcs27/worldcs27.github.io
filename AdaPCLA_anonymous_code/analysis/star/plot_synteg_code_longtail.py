#!/usr/bin/env python3
"""
SynTEG 生成数据的 code 频率 vs rank 图（log-log）。
横轴：rank（按频率降序排序）
纵轴：frequency
输出：mywork/star/synteg_code_longtail.png
"""
import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PCLA_ROOT = SCRIPT_DIR.parent.parent
FAME = PCLA_ROOT / "fame" / "myfame"

# SynTEG MIMIC-IV seed1 生成数据
SYNTEG_PATH = FAME / "baseline" / "synteg" / "save_mimiciv_seed1" / "datasets" / "syntegDataset.pkl"
VOCAB_PATH = FAME / "data2" / "codeToIndex.pkl"
OUT_PATH = SCRIPT_DIR / "synteg_code_longtail.png"


def main():
    with open(SYNTEG_PATH, "rb") as f:
        data = pickle.load(f)
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    n_codes = len(vocab)
    counts = Counter()
    for p in data:
        for v in p.get("visits", []):
            counts.update(v)

    freqs = np.array([counts.get(i, 0) for i in range(n_codes)])
    freqs = np.sort(freqs)[::-1]
    rank = np.arange(1, len(freqs) + 1, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rank, freqs, linewidth=1.5, color="#E74C3C")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (sorted by frequency)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax.set_title("SynTEG (MIMIC-IV) — Code long tail")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

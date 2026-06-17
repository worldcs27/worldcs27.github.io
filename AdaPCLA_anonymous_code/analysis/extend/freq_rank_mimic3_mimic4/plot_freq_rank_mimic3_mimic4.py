#!/usr/bin/env python3
"""
Frequency–rank plot: MIMIC-III vs MIMIC-IV Real train data.
X: code rank (1 = most frequent); Y: log(1 + frequency).
Output: extend/freq_rank_mimic3_mimic4/output/freq_rank_mimic3_mimic4.png
"""
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

MIMIC3_TRAIN = FAME / "data" / "trainDataset.pkl"
MIMIC4_TRAIN = FAME / "data2" / "trainDataset.pkl"
MAX_RANK = None  # no truncation; show full distribution including tail


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def get_freq_rank(data: list) -> tuple[np.ndarray, np.ndarray]:
    """Return (ranks, frequencies) sorted by frequency desc. ranks 1..N."""
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading MIMIC-III train...")
    d3 = load_pkl(MIMIC3_TRAIN) if MIMIC3_TRAIN.exists() else []
    print("Loading MIMIC-IV train...")
    d4 = load_pkl(MIMIC4_TRAIN) if MIMIC4_TRAIN.exists() else []
    r3, f3 = get_freq_rank(d3)
    r4, f4 = get_freq_rank(d4)
    plt.figure(figsize=(6, 4))
    plt.plot(r3, np.log1p(f3), label="MIMIC-III (Real)", linewidth=1.5, alpha=0.9)
    plt.plot(r4, np.log1p(f4), label="MIMIC-IV (Real)", linewidth=1.5, alpha=0.9)
    plt.xscale("log")
    plt.xlabel("Code rank (1 = most frequent, log scale)")
    plt.ylabel("log(1 + frequency)")
    plt.title("Frequency–rank: MIMIC-III vs MIMIC-IV (Real train)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_DIR / "freq_rank_mimic3_mimic4.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

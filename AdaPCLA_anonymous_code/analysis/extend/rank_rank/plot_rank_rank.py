#!/usr/bin/env python3
"""
方案 D: Rank–rank plot. Per-code: x = real rank, y = generated rank (1 = most frequent).
One subplot per model. Supports MIMIC-III or MIMIC-IV via first arg (default: mimic4).
Output: extend/rank_rank/output/rank_rank_{mimic3|mimic4}.png
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


def get_code_freq(data: list) -> dict:
    cnt = Counter()
    for p in data:
        for v in p.get("visits", []):
            for c in v:
                cnt[int(c)] += 1
    return dict(cnt)


def freq_to_rank(code2freq: dict) -> dict:
    """Return dict: code_id -> rank (1-based, 1 = most frequent)."""
    sorted_codes = sorted(code2freq.keys(), key=lambda c: -code2freq[c])
    return {c: r for r, c in enumerate(sorted_codes, start=1)}


def main():
    dataset = (sys.argv[1] if len(sys.argv) > 1 else "mimic4").lower()
    if dataset not in PATHS:
        dataset = "mimic4"
    paths = PATHS[dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset.upper()} Real...")
    d_real = load_pkl(paths["real"]) if paths["real"].exists() else []
    real_freq = get_code_freq(d_real)
    real_rank = freq_to_rank(real_freq)
    codes = sorted(real_freq.keys())
    x = np.array([real_rank[c] for c in codes], dtype=float)

    model_data = []
    for name in ["HALO", "LSTM", "GPT", "AdaPCLA"]:
        path = paths[name]
        if path.exists():
            print(f"Loading {name}...")
            d = load_pkl(path)
            gen_freq = get_code_freq(d)
            gen_rank = freq_to_rank(gen_freq)
            # For codes not in generated, assign a rank beyond max (e.g. max_rank+1)
            max_gen_rank = len(gen_rank) if gen_rank else 1
            y = np.array([gen_rank.get(c, max_gen_rank + 1) for c in codes], dtype=float)
            model_data.append((name, y))
        else:
            print(f"Skip {name} (not found)")

    n_models = len(model_data)
    if n_models == 0:
        print("No model data.")
        return
    ncol = 2
    nrow = (n_models + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 4 * nrow), squeeze=False)
    axes = axes.flatten()
    max_rank = np.max(x)
    for i, (name, y) in enumerate(model_data):
        ax = axes[i]
        ax.scatter(x, y, s=8, alpha=0.4, c="C0")
        ax.plot([1, max_rank], [1, max_rank], "k--", alpha=0.5, label="y=x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Real rank (1 = most frequent)")
        ax.set_ylabel(f"{name} rank")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Rank–rank: Real vs {name}")
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    out_path = OUT_DIR / f"rank_rank_{dataset}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

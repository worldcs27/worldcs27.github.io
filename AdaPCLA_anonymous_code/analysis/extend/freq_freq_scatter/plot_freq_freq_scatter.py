#!/usr/bin/env python3
"""
方案 C: Real vs Generated frequency–frequency scatter plot.
Per-code: x = real frequency, y = generated frequency (one subplot per model).
Supports MIMIC-III or MIMIC-IV via first arg (default: mimic4).
Output: extend/freq_freq_scatter/output/freq_freq_scatter_{mimic3|mimic4}.png
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
    """Return dict: code_id (int) -> count."""
    cnt = Counter()
    for p in data:
        for v in p.get("visits", []):
            for c in v:
                cnt[int(c)] += 1
    return dict(cnt)


def main():
    dataset = (sys.argv[1] if len(sys.argv) > 1 else "mimic4").lower()
    if dataset not in PATHS:
        dataset = "mimic4"
    paths = PATHS[dataset]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset.upper()} Real...")
    d_real = load_pkl(paths["real"]) if paths["real"].exists() else []
    real_freq = get_code_freq(d_real)
    codes = sorted(real_freq.keys())
    x = np.array([real_freq[c] for c in codes], dtype=float)

    model_data = []
    for name in ["HALO", "LSTM", "GPT", "AdaPCLA"]:
        path = paths[name]
        if path.exists():
            print(f"Loading {name}...")
            d = load_pkl(path)
            gen_freq = get_code_freq(d)
            y = np.array([gen_freq.get(c, 0) for c in codes], dtype=float)
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
    for i, (name, y) in enumerate(model_data):
        ax = axes[i]
        ax.scatter(x, y, s=8, alpha=0.4, c="C0")
        max_val = max(np.max(x), np.max(y), 1)
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="y=x")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlabel("Real frequency")
        ax.set_ylabel(f"{name} frequency")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Real vs {name}")
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    out_path = OUT_DIR / f"freq_freq_scatter_{dataset}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot bias magnitude |b_stat(c)| vs code frequency (rank).
b_stat(c) = tau * log((1 - pi + eps) / (pi + eps))
Supports Sec. 4.3.2 (selective scaffold) in main.tex.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PCLA_ROOT = SCRIPT_DIR.parents[2]
FAME_DATA = PCLA_ROOT / "fame" / "myfame" / "data2" / "trainDataset.pkl"
OUT_PNG = SCRIPT_DIR / "bias_magnitude_vs_frequency.png"
OUT_CSV = SCRIPT_DIR / "bias_magnitude_by_region.csv"

TAU = 1.0
EPS = 1e-6


def main():
    if not FAME_DATA.exists():
        print(f"Data not found: {FAME_DATA}, using illustrative values")
        _plot_illustrative()
        return

    train_data = pickle.load(open(FAME_DATA, "rb"))
    code_vocab = 0
    for p in train_data:
        for v in p.get("visits", []):
            if v:
                code_vocab = max(code_vocab, max(c for c in v if isinstance(c, int)))
    code_vocab += 1

    visit_counts = np.zeros(code_vocab, dtype=np.int64)
    total_visits = 0
    for p in train_data:
        for v in p.get("visits", []):
            if not v:
                continue
            total_visits += 1
            for c in set(v):
                if 0 <= c < code_vocab:
                    visit_counts[c] += 1

    pi = visit_counts.astype(np.float64) / max(total_visits, 1)
    b_stat = np.log((1.0 - pi + EPS) / (pi + EPS)) * TAU
    b_stat = np.where(visit_counts > 0, b_stat, 0.0)
    abs_b = np.abs(b_stat)

    # Rank by frequency (descending): rank 1 = most frequent
    order = np.argsort(-visit_counts)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order)) + 1

    # Plot: x=rank, y=|b_stat|
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(4, 2.2))
    valid = visit_counts > 0
    ax.scatter(ranks[valid], abs_b[valid], s=2, alpha=0.4, c="#3498DB")
    ax.set_xlabel("Code rank (1 = most frequent)")
    ax.set_ylabel(r"$|b_{\mathrm{stat}}(c)|$")
    ax.set_xlim(0, np.max(ranks[valid]) * 1.02)
    ax.set_ylim(0, np.max(abs_b[valid]) * 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=np.median(abs_b[valid]), color="gray", linestyle="--", alpha=0.7, label="Median")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")
    plt.close()

    # Table: Head (top 100) / Mid (100-2000) / Tail (2000+)
    n = np.sum(valid)
    head_end = min(100, n)
    mid_start, mid_end = 100, min(2000, n)
    tail_start = min(2000, n)
    head_ranks = ranks[valid][:head_end]
    mid_ranks = ranks[valid][mid_start:mid_end]
    tail_ranks = ranks[valid][tail_start:]
    head_pi = pi[valid][:head_end]
    mid_pi = pi[valid][mid_start:mid_end]
    tail_pi = pi[valid][tail_start:]
    head_b = abs_b[valid][:head_end]
    mid_b = abs_b[valid][mid_start:mid_end]
    tail_b = abs_b[valid][tail_start:]

    with open(OUT_CSV, "w") as f:
        f.write("region,rank_range,mean_pi,mean_abs_b\n")
        f.write(f"Head,1-{head_end},{np.mean(head_pi):.4f},{np.mean(head_b):.2f}\n")
        f.write(f"Mid,{mid_start}-{mid_end},{np.mean(mid_pi):.6f},{np.mean(mid_b):.2f}\n")
        f.write(f"Tail,{tail_start}-{n},{np.mean(tail_pi):.6f},{np.mean(tail_b):.2f}\n")
    print(f"Saved: {OUT_CSV}")


def _plot_illustrative():
    """Fallback: illustrative plot when data not found."""
    plt.rcParams["font.family"] = "serif"
    ranks = np.arange(1, 6000)
    pi_approx = 0.1 / (ranks ** 0.8)
    pi_approx = np.clip(pi_approx, 1e-6, 0.99)
    b_stat = np.log((1 - pi_approx + EPS) / (pi_approx + EPS)) * TAU
    abs_b = np.abs(b_stat)
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.plot(ranks, abs_b, "-", color="#3498DB", linewidth=1, alpha=0.8)
    ax.set_xlabel("Code rank (1 = most frequent)")
    ax.set_ylabel(r"$|b_{\mathrm{stat}}(c)|$")
    ax.set_xlim(0, 6000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved (illustrative): {OUT_PNG}")
    plt.close()


if __name__ == "__main__":
    main()

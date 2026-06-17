#!/usr/bin/env python3
"""
Annealing coefficient α(e) curve: warmup + linear decay.
Output: mywork/star/annealing_schedule.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "annealing_schedule.png"

plt.rcParams["font.family"] = "serif"


def alpha(e: np.ndarray, E_w: int, E_max: int) -> np.ndarray:
    """α(e): warmup α=1 for e ≤ E_w, then linear decay to 0 at E_max."""
    out = np.ones_like(e, dtype=float)
    mask = e > E_w
    out[mask] = 1.0 - (e[mask] - E_w) / (E_max - E_w)
    return np.clip(out, 0, 1)


def main():
    E_max = 100
    E_w = int(0.1 * E_max)  # warmup 10% of total steps
    e = np.linspace(0, E_max, 501)
    a = alpha(e, E_w, E_max)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(e, a, color="#2878B5", linewidth=2)
    ax.axvline(E_w, color="gray", linestyle="--", alpha=0.6, label=f"$E_w={E_w}$")
    ax.set_xlabel("Training step $e$", fontsize=14, fontweight="bold")
    ax.set_ylabel("Annealing coefficient $\\alpha(e)$", fontsize=14, fontweight="bold")
    ax.set_xlim(0, E_max)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()

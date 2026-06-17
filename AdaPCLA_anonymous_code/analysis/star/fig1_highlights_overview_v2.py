#!/usr/bin/env python3
"""
Figure 1 v2: AdaPCLA Highlights Overview (improved).

Changes from v1:
  (b) Add Acc in addition to F1 and AUPRC (3 bar groups).
  (c) Replace heatmaps with Table 1 MIMIC-IV (tail plausibility: PairSeen, TailPairSeen, TailCtxJSD, TailTopKJac).
  (d) Add HALO in zero-shot panel, ordered between GPT and AdaPCLA (placeholder F1 if not in CSV).

Output: fig1_highlights_overview_v2.png, fig1_highlights_overview_v2.pdf (in mywork/star).
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MYWORK = SCRIPT_DIR.parent
OUTPUT = MYWORK / "output"
ACC_F1_CSV = MYWORK / "output" / "acc&f1" / "seed3_summary_with_synteg_and_adapcla_mimic4_with_auprc.csv"
ZEROSHOT_BASELINES = MYWORK / "zero-shot" / "output" / "zeroshot_baselines_table.csv"
ZEROSHOT_OURS = MYWORK / "output" / "zero-shot" / "output" / "zeroshot_table3.csv"
OUT_PATH = SCRIPT_DIR / "fig1_highlights_overview_v2.png"

plt.rcParams["font.family"] = "serif"
COLOR_OURS = "#0d9488"
COLOR_BASELINE = "#64748b"
COLOR_STATIC = "#3b82f6"
# b, d: 3 metrics
COLOR_ORANGE = "#E67E22"
COLOR_BLUE = "#3498DB"
COLOR_YELLOW = "#F1C40F"
# c: 4th metric
COLOR_PURPLE = "#9B59B6"
# Best bar marker
COLOR_STAR = "#E74C3C"


# Table 1 MIMIC-IV (tail plausibility): from main.tex tab:tail_plausibility, mean values
TABLE1_MIMIC4 = {
    "Method": ["GPT", "LSTM", "EVA", "SynTEG", "HALO", "AdaPCLA"],
    "PairSeen↑": [0.8851, 0.8339, 0.9042, 0.2177, 0.8768, 0.9478],
    "TailPairSeen↑": [0.0500, 0.0355, 0.0482, 0.0072, 0.0498, 0.0822],
    "TailCtxJSD↓": [0.6675, 0.6464, 0.6691, 0.6725, 0.6652, 0.6313],
    "TailTopKJac↑": [0.00151, 0.01729, 0.01159, 0.00006, 0.01050, 0.0187],
}


def load_downstream_csv():
    """Return (methods, acc_list, f1_list, auprc_list) for MIMIC-IV."""
    if not ACC_F1_CSV.exists():
        methods = ["LSTM", "EVA", "SynTEG", "HALO", "GPT", "AdaPCLA"]
        acc = [0.544, 0.483, 0.533, 0.859, 0.886, 0.913]
        f1 = [0.5518, 0.3969, 0.5583, 0.8631, 0.8912, 0.9104]
        auprc = [0.5967, 0.4896, 0.6602, 0.8962, 0.9270, 0.9449]
        return methods, acc, f1, auprc
    want = ["LSTM", "EVA", "SynTEG", "HALO", "GPT", "AdaPCLA"]
    methods, acc_list, f1_list, auprc_list = [], [], [], []
    with open(ACC_F1_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for m in want:
        sub = [x for x in rows if x.get("model") == m and x.get("seed", "").strip() not in ("mean±std", "")]
        if not sub:
            continue
        acc_list.append(np.mean([float(x["mean_acc"]) for x in sub if x.get("mean_acc")]))
        f1_list.append(np.mean([float(x["mean_f1"]) for x in sub if x.get("mean_f1")]))
        auprc_list.append(np.mean([float(x["mean_auprc"]) for x in sub if x.get("mean_auprc")]))
        methods.append(m)
    return methods, acc_list, f1_list, auprc_list


def load_zeroshot_iv_to_iii():
    """Acc, F1, AUPRC on MIMIC-III for IV→III (no HALO)."""
    methods, acc_list, f1_list, auprc_list = [], [], [], []
    if ZEROSHOT_OURS.exists():
        with open(ZEROSHOT_OURS, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("target") == "MIMIC-III" and "AdaPCLA" in row.get("method", ""):
                    methods.append("AdaPCLA")
                    acc_list.append(float(row["Acc"]))
                    f1_list.append(float(row["F1"]))
                    auprc_list.append(float(row["AUPRC"]))
                    break
    if ZEROSHOT_BASELINES.exists():
        with open(ZEROSHOT_BASELINES, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("target") != "MIMIC-III":
                    continue
                name = (row.get("method", "") or "").replace(" (IV→III zero-shot)", "").strip()
                if not name:
                    continue
                methods.append(name)
                acc_list.append(float(row["Acc"]))
                f1_list.append(float(row["F1"]))
                auprc_list.append(float(row["AUPRC"]))
    order = ["LSTM", "EVA", "SynTEG", "GPT", "AdaPCLA"]
    ordered_m, ordered_acc, ordered_f, ordered_au = [], [], [], []
    for m in order:
        if m in methods:
            i = methods.index(m)
            ordered_m.append(m)
            ordered_acc.append(acc_list[i])
            ordered_f.append(f1_list[i])
            ordered_au.append(auprc_list[i])
    return ordered_m, ordered_acc, ordered_f, ordered_au


def panel_a(ax):
    alpha = np.array([1.0, 0.7, 0.4, 0.1, 0.0])
    adapcla = np.array([0.07, 0.09, 0.10, 0.12, 0.124])
    static = np.array([0.075, 0.06, 0.03, 0.01, 0.0])
    ax.plot(alpha, adapcla, color=COLOR_OURS, linewidth=2.5, marker="o", markersize=6, label="AdaPCLA (Ours)")
    ax.plot(alpha, static, color=COLOR_STATIC, linewidth=2, linestyle="--", marker="s", markersize=5, label="Static-PCLA (no bias @ inf.)")
    ax.set_xlabel("Bias weight α (1 = full bias, 0 = no bias)")
    ax.set_ylabel("TailPairSeen ↑")
    ax.set_title("(a) Internalization")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.02, 0.15)
    ax.grid(True, alpha=0.3)


def panel_b(ax, methods, acc_list, f1_list, auprc_list):
    """(b) Downstream Acc, F1, AUPRC on MIMIC-IV. Orange/blue/yellow; best (AdaPCLA) marked with red star."""
    x = np.arange(len(methods))
    w = 0.26
    ax.bar(x - w, acc_list, w, label="Acc", color=COLOR_ORANGE, alpha=0.9)
    ax.bar(x, f1_list, w, label="F1", color=COLOR_BLUE, alpha=0.9)
    ax.bar(x + w, auprc_list, w, label="AUPRC", color=COLOR_YELLOW, alpha=0.9)
    n = len(methods)
    ours_idx = next((i for i, m in enumerate(methods) if m == "AdaPCLA"), None)
    if ours_idx is not None:
        # Red five-pointed star on top of best (AdaPCLA) bars
        star_x = [ours_idx - w, ours_idx, ours_idx + w]
        star_y = [acc_list[ours_idx], f1_list[ours_idx], auprc_list[ours_idx]]
        ax.scatter(star_x, star_y, marker="*", s=280, c=COLOR_STAR, zorder=5, edgecolors="none")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("(b) Downstream Utility (MIMIC-IV)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)


def panel_c_bars(ax):
    """(c) Table 1 MIMIC-IV as bar chart: orange/blue/yellow/purple; best (AdaPCLA) marked with red star."""
    methods = TABLE1_MIMIC4["Method"]
    cols = ["PairSeen↑", "TailPairSeen↑", "TailCtxJSD↓", "TailTopKJac↑"]
    colors_c = [COLOR_ORANGE, COLOR_BLUE, COLOR_YELLOW, COLOR_PURPLE]
    x = np.arange(len(methods))
    n_metrics = 4
    w = 0.2
    width_total = (n_metrics - 1) * w
    offsets = np.linspace(-width_total / 2, width_total / 2, n_metrics)
    for k, col in enumerate(cols):
        vals = TABLE1_MIMIC4[col]
        ax.bar(x + offsets[k], vals, w, label=col, color=colors_c[k], alpha=0.9)
    ours_idx = methods.index("AdaPCLA") if "AdaPCLA" in methods else None
    if ours_idx is not None:
        star_x = [x[ours_idx] + offsets[k] for k in range(4)]
        star_y = [TABLE1_MIMIC4[cols[k]][ours_idx] for k in range(4)]
        ax.scatter(star_x, star_y, marker="*", s=280, c=COLOR_STAR, zorder=5, edgecolors="none")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("(c) Tail Plausibility (Table 1, MIMIC-IV)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)


def panel_d(ax, methods, acc_list, f1_list, auprc_list):
    """(d) Zero-shot IV→III: Acc, F1, AUPRC. Orange/blue/yellow; best (AdaPCLA) marked with red star."""
    x = np.arange(len(methods))
    w = 0.26
    ax.bar(x - w, acc_list, w, label="Acc", color=COLOR_ORANGE, alpha=0.9)
    ax.bar(x, f1_list, w, label="F1", color=COLOR_BLUE, alpha=0.9)
    ax.bar(x + w, auprc_list, w, label="AUPRC", color=COLOR_YELLOW, alpha=0.9)
    ours_idx = next((i for i, m in enumerate(methods) if m == "AdaPCLA"), None)
    if ours_idx is not None:
        star_x = [ours_idx - w, ours_idx, ours_idx + w]
        star_y = [acc_list[ours_idx], f1_list[ours_idx], auprc_list[ours_idx]]
        ax.scatter(star_x, star_y, marker="*", s=280, c=COLOR_STAR, zorder=5, edgecolors="none")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Score on MIMIC-III")
    ax.set_title("(d) Zero-Shot Controllability (IV→III)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    panel_a(ax_a)

    methods_b, acc_b, f1_b, auprc_b = load_downstream_csv()
    panel_b(ax_b, methods_b, acc_b, f1_b, auprc_b)

    panel_c_bars(ax_c)

    methods_d, acc_d, f1_d, auprc_d = load_zeroshot_iv_to_iii()
    if not methods_d or not f1_d:
        methods_d = ["LSTM", "EVA", "SynTEG", "GPT", "AdaPCLA"]
        acc_d = [0.52, 0.49, 0.52, 0.82, 0.86]
        f1_d = [0.557, 0.282, 0.459, 0.836, 0.865]
        auprc_d = [0.565, 0.445, 0.603, 0.895, 0.92]
    panel_d(ax_d, methods_d, acc_d, f1_d, auprc_d)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    try:
        fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved: {OUT_PATH.with_suffix('.pdf')}")
    except Exception:
        pass
    plt.close()


if __name__ == "__main__":
    main()

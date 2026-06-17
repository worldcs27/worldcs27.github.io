#!/usr/bin/env python3
"""
Downstream Utility 1×3 figure: Acc, F1, AUPRC.
Each subplot: methods with MIMIC-III (red) and MIMIC-IV (blue) bars.
Output (mywork/star/):
  - downstream_utility_1x3.png: 三子图合并图
  - downstream_utility_acc.png: Acc 准确率单独图
  - downstream_utility_f1.png: F1 分数单独图
  - downstream_utility_auprc.png: AUPRC 曲线下面积单独图
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
MYWORK = SCRIPT_DIR.parent
ACC_F1_M3 = MYWORK / "output" / "acc&f1" / "seed3_summary_with_synteg_and_adapcla_mimic3_with_auprc.csv"
ACC_F1_M4 = MYWORK / "output" / "acc&f1" / "seed3_summary_with_synteg_and_adapcla_mimic4_with_auprc.csv"
OUT_PATH = SCRIPT_DIR / "downstream_utility_1x3.png"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 12
COLOR_MIMIC3 = "#E74C3C"   # 红 (MIMIC-III)
COLOR_MIMIC4 = "#3498DB"   # 蓝 (MIMIC-IV)


def load_downstream_both():
    """Return (methods, acc_m3, f1_m3, auprc_m3, acc_m4, f1_m4, auprc_m4)."""
    want = ["EVA", "SynTEG", "LSTM", "HALO", "GPT", "AdaPCLA"]  # EVA 放第一项

    # Fallback from main.tex Table (tab:downstream), order: EVA, SynTEG, LSTM, ...
    fallback = {
        "methods": want,
        "acc_m3": [0.5243, 0.5095, 0.5145, 0.8761, 0.8256, 0.9040],
        "f1_m3": [0.5167, 0.4016, 0.5017, 0.8779, 0.8304, 0.9051],
        "auprc_m3": [0.5607, 0.5721, 0.5563, 0.9295, 0.8902, 0.9514],
        "acc_m4": [0.4829, 0.5333, 0.5441, 0.8592, 0.8856, 0.9130],
        "f1_m4": [0.3969, 0.5583, 0.5518, 0.8631, 0.8912, 0.9104],
        "auprc_m4": [0.4896, 0.6602, 0.5967, 0.8962, 0.9270, 0.9449],
    }

    def parse_csv(path, want_list):
        methods, acc, f1, auprc = [], [], [], []
        if not path.exists():
            return None
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for m in want_list:
            sub = [x for x in rows if x.get("model") == m and x.get("seed", "").strip() not in ("mean±std", "")]
            if not sub:
                continue
            acc.append(np.mean([float(x["mean_acc"]) for x in sub if x.get("mean_acc")]))
            f1.append(np.mean([float(x["mean_f1"]) for x in sub if x.get("mean_f1")]))
            auprc.append(np.mean([float(x["mean_auprc"]) for x in sub if x.get("mean_auprc")]))
            methods.append(m)
        return methods, acc, f1, auprc

    r3 = parse_csv(ACC_F1_M3, want)
    r4 = parse_csv(ACC_F1_M4, want)
    if r3 and r4 and len(r3[0]) == len(r4[0]) == len(want):
        m3, a3, f3, u3 = r3
        m4, a4, f4, u4 = r4
        return m3, a3, f3, u3, a4, f4, u4

    return (
        fallback["methods"],
        fallback["acc_m3"],
        fallback["f1_m3"],
        fallback["auprc_m3"],
        fallback["acc_m4"],
        fallback["f1_m4"],
        fallback["auprc_m4"],
    )


def plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w):
    """画单个指标的柱状图：MIMIC-III（红）+ MIMIC-IV（蓝）双柱，第二条指示线画在 GPT 的 MIMIC-IV 柱上。"""
    ax.bar(x - w / 2, vals_m3, w, label="MIMIC-III", color=COLOR_MIMIC3, alpha=0.9)
    ax.bar(x + w / 2, vals_m4, w, label="MIMIC-IV", color=COLOR_MIMIC4, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # 第一条指示线：画在 EVA 的 MIMIC-IV 柱上，从最小值到最大值，百分比=(最大值-最小值)/最小值×100，放柱子右侧
    vmin_m4, vmax_m4 = min(vals_m4), max(vals_m4)
    diff_pct_eva = (vmax_m4 - vmin_m4) / vmin_m4 * 100 if vmin_m4 > 0 else 0
    eva_idx = methods.index("EVA") if "EVA" in methods else 0
    x_line_eva = x[eva_idx] + w / 2
    ax.annotate(
        "",
        xy=(x_line_eva, vmax_m4),
        xytext=(x_line_eva, vmin_m4),
        arrowprops=dict(arrowstyle="|-|", color="#00CC00", lw=1.5, mutation_scale=8),
    )
    ax.text(
        x_line_eva + 0.08,
        (vmin_m4 + vmax_m4) / 2,
        f"{{+{diff_pct_eva:.0f}%}}",
        ha="left",
        va="center",
        fontsize=20,
        color="#00AA00",
    )

    # 第二条指示线：画在 GPT 的 MIMIC-IV 柱上，从次高值到最高值
    sorted_m4 = sorted(vals_m4)
    second_max = sorted_m4[-2] if len(sorted_m4) >= 2 else vmax_m4
    diff_pct = (vmax_m4 - second_max) / second_max * 100 if second_max > 0 else 0
    right_method = "GPT"
    right_idx = methods.index(right_method) if right_method in methods else 4
    x_line2 = x[right_idx] + w / 2  # GPT 的 MIMIC-IV 柱中心
    ax.annotate(
        "",
        xy=(x_line2, vmax_m4),
        xytext=(x_line2, second_max),
        arrowprops=dict(arrowstyle="|-|", color="#00CC00", lw=1.5, mutation_scale=8),
    )
    ax.text(
        x_line2,
        vmax_m4 + 0.02,
        f"{{+{diff_pct:.1f}%}}",
        ha="center",
        va="bottom",
        fontsize=20,
        color="#00AA00",
    )

    # 从靠左第一条纵向指示线（EVA）顶端到 AdaPCLA 的 MIMIC-IV 柱子顶点的水平连线
    adapcla_idx = methods.index("AdaPCLA") if "AdaPCLA" in methods else -1
    if adapcla_idx >= 0:
        x_adapcla = x[adapcla_idx] + w / 2
        ax.plot([x_line_eva, x_adapcla], [vmax_m4, vals_m4[adapcla_idx]], color="#00CC00", lw=1.5, zorder=5)


def main():
    methods, acc_m3, f1_m3, auprc_m3, acc_m4, f1_m4, auprc_m4 = load_downstream_both()

    x = np.arange(len(methods))
    w = 0.35 * 0.75  # 柱宽

    metrics = [
        ("Acc ↑", acc_m3, acc_m4),
        ("F1 ↑", f1_m3, f1_m4),
        ("AUPRC ↑", auprc_m3, auprc_m4),
    ]

    out_names = [
        "downstream_utility_acc",
        "downstream_utility_f1",
        "downstream_utility_auprc",
    ]

    # 1. 分别生成三张单图
    for (ylabel, vals_m3, vals_m4), name in zip(metrics, out_names):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w)
        plt.tight_layout()
        out_png = SCRIPT_DIR / f"{name}.png"
        out_pdf = SCRIPT_DIR / f"{name}.pdf"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_png}")
        try:
            fig.savefig(out_pdf, bbox_inches="tight")
            print(f"Saved: {out_pdf}")
        except Exception:
            pass
        plt.close(fig)

    # 2. 1×3 合并图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (ylabel, vals_m3, vals_m4) in zip(axes, metrics):
        plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w)
    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    try:
        fig.savefig(OUT_PATH.with_suffix(".pdf"), bbox_inches="tight")
        print(f"Saved: {OUT_PATH.with_suffix('.pdf')}")
    except Exception:
        pass
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tail clinical plausibility figure: PairSeen, TailPairSeen, TailTopKJac.
数据来源: main.tex Table tab:app_tail_plausibility_full (mean values).
每个指标: methods × MIMIC-III(红) / MIMIC-IV(蓝).

Output:
  - tail_plausibility_1x3.png: 1×3 合并图
  - tail_plausibility_pairseen.png: 仅 PairSeen↑
  - tail_plausibility_tailpairseen.png: 仅 TailPairSeen↑
  - tail_plausibility_tailtopkjac.png: 仅 TailTopKJac↑
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "tail_plausibility_1x3.png"

plt.rcParams["font.family"] = "serif"
COLOR_BAR = "#AA66FF"  # 亮紫


def load_tail_plausibility():
    """从 main.tex tab:app_tail_plausibility_full 提取 mean 值，顺序 SynTEG, EVA, LSTM, HALO, GPT, AdaPCLA。"""
    methods = ["SynTEG", "EVA", "LSTM", "HALO", "GPT", "AdaPCLA"]

    # PairSeen↑ (mean, 不含 std)
    pairseen_m3 = [0.0373, 0.0260, 0.7272, 0.8263, 0.7977, 0.7983]
    pairseen_m4 = [0.2177, 0.9042, 0.8339, 0.8768, 0.8851, 0.9478]

    # TailPairSeen↑
    tailpairseen_m3 = [0.0019, 0.0023, 0.0297, 0.0520, 0.0384, 0.1114]
    tailpairseen_m4 = [0.0072, 0.0482, 0.0355, 0.0498, 0.0500, 0.0822]

    # TailTopKJac↑
    tailtopkjac_m3 = [0.00002, 0.00083, 0.04905, 0.01967, 0.01131, 0.0345]
    tailtopkjac_m4 = [0.00006, 0.01159, 0.01729, 0.01050, 0.00151, 0.0187]

    return (
        methods,
        pairseen_m3, pairseen_m4,
        tailpairseen_m3, tailpairseen_m4,
        tailtopkjac_m3, tailtopkjac_m4,
    )


def plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w, ylim=None, left_method="SynTEG", right_method="GPT"):
    """画单个指标的柱状图及两条纵向指示线。left_method: 靠左指示条; right_method: 靠右指示条。"""
    ax.bar(x, vals_m4, w, label="MIMIC-IV", color=COLOR_BAR, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        vmin, vmax = min(vals_m3 + vals_m4), max(vals_m3 + vals_m4)
        ax.set_ylim(max(0, vmin - 0.02), vmax * 1.15 if vmax > 0 else 0.1)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    vmin, vmax = min(vals_m4), max(vals_m4)
    diff_pct = (vmax - vmin) / vmin * 100 if vmin > 0 else 0

    # 靠左指示条：最高值与最低值的差距，放在 left_method 柱上
    left_idx = methods.index(left_method) if left_method in methods else 0
    x_line = x[left_idx]
    ax.annotate(
        "",
        xy=(x_line, vmax),
        xytext=(x_line, vmin),
        arrowprops=dict(arrowstyle="|-|", color="#00CC00", lw=1.5, mutation_scale=8),
    )
    ax.text(
        x_line + 0.08,
        vmin + 0.75 * (vmax - vmin),
        f"{{+{diff_pct:.0f}%}}",
        ha="left",
        va="center",
        fontsize=8,
        color="#00AA00",
    )

    # 靠右指示条：最高值与次高值的差距，放在 right_method 柱上
    sorted_m4 = sorted(vals_m4)
    second_max = sorted_m4[-2] if len(sorted_m4) >= 2 else vmin
    diff_pct2 = (vmax - second_max) / second_max * 100 if second_max > 0 else 0

    right_idx = methods.index(right_method) if right_method in methods else 4
    x_line2 = x[right_idx]
    ax.annotate(
        "",
        xy=(x_line2, vmax),
        xytext=(x_line2, second_max),
        arrowprops=dict(arrowstyle="|-|", color="#00CC00", lw=1.5, mutation_scale=8),
    )
    ax.text(
        x_line2 - 0.08,
        (second_max + vmax) / 2,
        f"{{+{diff_pct2:.1f}%}}",
        ha="right",
        va="center",
        fontsize=8,
        color="#00AA00",
    )


def main():
    (
        methods,
        pairseen_m3, pairseen_m4,
        tailpairseen_m3, tailpairseen_m4,
        tailtopkjac_m3, tailtopkjac_m4,
    ) = load_tail_plausibility()

    x = np.arange(len(methods))
    w = 0.35 * 0.75

    # 子图1 PairSeen: 左 SynTEG 右 EVA；子图2 TailPairSeen: 左 SynTEG 右 GPT；子图3 TailTopKJac: 左 SynTEG 右 LSTM
    metrics = [
        ("PairSeen ↑", pairseen_m3, pairseen_m4, (0, 1.05), "SynTEG", "EVA"),
        ("TailPairSeen ↑", tailpairseen_m3, tailpairseen_m4, (0, 0.14), "SynTEG", "GPT"),
        ("TailTopKJac ↑", tailtopkjac_m3, tailtopkjac_m4, (0, 0.06), "SynTEG", "LSTM"),
    ]

    out_names = [
        "tail_plausibility_pairseen",
        "tail_plausibility_tailpairseen",
        "tail_plausibility_tailtopkjac",
    ]

    # 1. 分别生成三张单图
    for (ylabel, vals_m3, vals_m4, ylim, left_m, right_m), name in zip(metrics, out_names):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w, ylim=ylim, left_method=left_m, right_method=right_m)
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
    for ax, (ylabel, vals_m3, vals_m4, ylim, left_m, right_m) in zip(axes, metrics):
        plot_single_metric(ax, methods, vals_m3, vals_m4, ylabel, x, w, ylim=ylim, left_method=left_m, right_method=right_m)
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

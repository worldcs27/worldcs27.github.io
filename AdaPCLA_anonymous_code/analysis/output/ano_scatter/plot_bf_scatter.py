#!/usr/bin/env python3
"""
BF scatter plots (B_*^+ vs F_0^*) for Real vs generators.

目的：
- 利用论文同一套数据和模型，在 code 级别上可视化：
  - B_*^+(c): 真实数据中 code c 的 visit-level 频率（每个 visit 至多计一次）；
  - F_* (c): 某生成模型中 code c 的 visit-level 频率；
  - F_0^*(c) = (F_*(c) - B_*(c))^+：模型对 code c 的“过度生成量”（只看超出真实的部分）。
- 对于每个模型 (HALO / LSTM / GPT / AdaPCLA)，画一张图：
  x 轴 = B_*^+(c)，y 轴 = F_0^*(c)，并在对数坐标下划分四个区域，直观展示：
    - 低频但被明显过度生成的 code（tail 过拟合 / 噪声）；
    - 高频且被明显放大的 code；
    - 等。

数据：
- MIMIC-III 或 MIMIC-IV，与其他脚本保持路径一致。

输出：
- output/ano_scatter/bf_scatter_{dataset}_{model}.png（全量 code）
- output/ano_scatter/bf_scatter_{dataset}_{model}_tail.png（仅 real rank ≥ 5000 的 tail code）
  其中 dataset ∈ {mimic3, mimic4}，model ∈ {HALO, LSTM, GPT, AdaPCLA}（若存在）。
  图中标出 b1, b2（横轴阈值）与 f1, f2（纵轴阈值）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches


SCRIPT_DIR = Path(__file__).resolve().parent
MYWORK = SCRIPT_DIR.parent.parent
PCLA_ROOT = MYWORK.parent
FAME = PCLA_ROOT / "fame" / "myfame"
OUT_DIR = SCRIPT_DIR
TAIL_START = 5000  # real rank >= this 视为 tail code


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


def get_visit_freq(data: list[dict]) -> Counter[int]:
    """
    统计 visit-level 频率：每个 visit 内对 code 去重后计数一次。
    返回 Counter: code_id -> #visits containing this code.
    """
    cnt: Counter[int] = Counter()
    for p in data:
        for v in p.get("visits", []):
            if not v:
                continue
            codes = {int(c) for c in v}
            for c in codes:
                cnt[c] += 1
    return cnt


def freq_to_rank(code2freq: Counter[int]) -> dict[int, int]:
    """code_id -> rank (1-based, 1 = most frequent)."""
    sorted_codes = sorted(code2freq.keys(), key=lambda c: -code2freq[c])
    return {c: r for r, c in enumerate(sorted_codes, start=1)}


def compute_b_f0_for_model(
    real_freq: Counter[int],
    model_freq: Counter[int],
    codes_subset: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对在 Real 中出现过的 code 计算：
      B(c) = freq_real(c)
      F0(c) = max(0, freq_model(c) - freq_real(c))
    若 codes_subset 给定，只保留该子集（且 B>0）；否则保留所有 B>0 的 code。
    """
    if codes_subset is not None:
        codes = [c for c in codes_subset if real_freq.get(c, 0) > 0]
    else:
        codes = [c for c, f in real_freq.items() if f > 0]
    codes.sort()
    B = np.array([real_freq[c] for c in codes], dtype=float)
    F0 = np.array(
        [max(0.0, float(model_freq.get(c, 0)) - float(real_freq[c])) for c in codes],
        dtype=float,
    )
    return B, F0


def plot_marginal_curve(ax, data: np.ndarray, orientation: str, color: str):
    """在对数刻度下画边缘分布曲线（折线 + 填充），用于顶部/右侧子图。"""
    data = data[data > 0]
    if data.size == 0:
        return
    bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 60)
    counts, bin_edges = np.histogram(data, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    if orientation == "vertical":
        ax.plot(centers, counts, color=color, lw=1)
        ax.fill_between(centers, 0, counts, color=color, alpha=0.3)
        ax.set_xscale("log")
    else:
        ax.plot(counts, centers, color=color, lw=1)
        ax.fill_betweenx(centers, 0, counts, color=color, alpha=0.3)
        ax.set_yscale("log")


def plot_bf_scatter(
    B: np.ndarray,
    F0: np.ndarray,
    model_name: str,
    dataset: str,
    out_path: Path,
    title_suffix: str = "",
):
    """
    仿照用户提供的布局：主散点 + 上侧/右侧边缘分布 + 四个区域。
    x 轴: B_*^+ = visit-level freq (Real)
    y 轴: F_0^* = max(0, freq_model - freq_real)
    """
    # 只保留 B 和 F0 都正的点（避免 log 0 问题），但 F0=0 的点用很小值替代以可视化
    mask = B > 0
    B = B[mask]
    F0 = F0[mask]

    # 为了在对数坐标下显示 F0=0 的点，给一个很小的 epsilon
    eps = 1e-3
    F0_plot = np.where(F0 <= 0, eps, F0)

    # 自动阈值：用分位数大致划出区域
    b1 = np.percentile(B, 50)   # median
    b2 = np.percentile(B, 90)   # high-frequency threshold
    # 只在 F0>0 的点上取分位数
    F_pos = F0[F0 > 0]
    if F_pos.size > 0:
        f1 = np.percentile(F_pos, 75)
        f2 = np.percentile(F_pos, 95)
    else:
        f1 = f2 = 1.0

    fig = plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(
        4,
        4,
        width_ratios=[4, 4, 4, 1],
        height_ratios=[1, 4, 4, 4],
    )
    gs.update(wspace=0.05, hspace=0.05)

    ax_main = plt.subplot(gs[1:4, 0:3])
    ax_top = plt.subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = plt.subplot(gs[1:4, 3], sharey=ax_main)

    # 边缘分布
    plot_marginal_curve(ax_top, B, orientation="vertical", color="blue")
    plot_marginal_curve(ax_right, F0_plot, orientation="horizontal", color="red")

    # 主散点
    ax_main.scatter(B, F0_plot, s=4, c="navy", alpha=0.4, edgecolors="none")
    ax_main.set_xscale("log")
    ax_main.set_yscale("log")

    # 设定可视范围
    xmin, xmax = B.min(), B.max()
    ymin, ymax = F0_plot.min(), F0_plot.max()
    ax_main.set_xlim(xmin, xmax * 1.2)
    ax_main.set_ylim(ymin, ymax * 1.2)

    # 画 b1, b2, f1, f2 的辅助线，并统一 f1/f2 的纵坐标（用于标注）
    f1_line = max(f1, eps)
    f2_line = max(f2, f1_line * 1.01)
    ax_main.axvline(b1, color="black", linestyle="--", lw=1.0)
    ax_main.axvline(b2, color="black", linestyle="--", lw=1.0)
    ax_main.axhline(f1_line, color="black", linestyle="--", lw=1.0)
    ax_main.axhline(f2_line, color="black", linestyle="--", lw=1.0)

    # 分区背景（大致示意）
    # Part I: low B, high F0 (tail codes severely over-generated)
    p1 = patches.Rectangle(
        (xmin, f1_line),
        max(b1 - xmin, xmin * 0.01),
        ymax - f1_line,
        color="lightblue",
        alpha=0.2,
    )
    ax_main.add_patch(p1)
    ax_main.text(
        xmin * 1.1,
        f1_line * 1.5,
        "Part I\n(tail over-gen.)",
        fontsize=9,
        fontweight="bold",
    )

    # Part II: high B, high F0
    p2 = patches.Rectangle(
        (b2, f1_line),
        xmax - b2,
        ymax - f1_line,
        color="lightgrey",
        alpha=0.25,
    )
    ax_main.add_patch(p2)
    ax_main.text(
        b2 * 1.1,
        f1_line * 1.5,
        "Part II\n(head over-gen.)",
        fontsize=9,
        fontweight="bold",
    )

    # Part III: high B, low F0 (well-aligned head codes)
    p3 = patches.Rectangle(
        (b2, ymin),
        xmax - b2,
        f1_line - ymin,
        color="lightgreen",
        alpha=0.15,
    )
    ax_main.add_patch(p3)
    ax_main.text(
        b2 * 1.1,
        ymin * (10 if ymin > 0 else 1.1),
        "Part III\n(head, small F_0^*)",
        fontsize=9,
        fontweight="bold",
    )

    # Part IV: low B, low F0 (tail codes without large over-gen)
    ax_main.text(
        xmin * 1.1,
        ymin * (10 if ymin > 0 else 1.1),
        "Part IV\n(tail, small F_0^*)",
        fontsize=9,
        fontweight="bold",
        color="green",
    )

    # 轴标签
    ax_main.set_xlabel(r"$B_*^+$: visit-level frequency in Real", fontsize=11)
    ax_main.set_ylabel(
        r"$F_0^* = (F_* - B_*)^+$: excess visit frequency in model",
        fontsize=11,
    )
    ax_main.set_title(
        f"{dataset.upper()} BF scatter for {model_name}{title_suffix}",
        fontsize=12,
    )

    # 顶部标注 b1, b2
    ax_top.text(b1, ax_top.get_ylim()[1] * 0.6, r"$b_1$", color="red", fontsize=10, ha="center")
    ax_top.text(b2, ax_top.get_ylim()[1] * 0.6, r"$b_2$", color="red", fontsize=10, ha="center")

    # 右侧标注 f1, f2（在主图右侧边缘，与水平线对齐）
    ax_main.text(
        xmax * 0.98,
        f1_line,
        r"$f_1$",
        color="red",
        fontsize=10,
        ha="right",
        va="center",
    )
    ax_main.text(
        xmax * 0.98,
        f2_line,
        r"$f_2$",
        color="red",
        fontsize=10,
        ha="right",
        va="center",
    )

    ax_top.axis("off")
    ax_right.axis("off")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    dataset = (sys.argv[1] if len(sys.argv) > 1 else "mimic4").lower()
    if dataset not in PATHS:
        dataset = "mimic4"
    paths = PATHS[dataset]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dataset.upper()} Real...")
    d_real = load_pkl(paths["real"]) if paths["real"].exists() else []
    real_freq = get_visit_freq(d_real)
    real_rank = freq_to_rank(real_freq)
    print(f"  Real codes with freq>0: {len(real_freq)}")

    tail_codes = [c for c, r in real_rank.items() if r >= TAIL_START]
    print(f"  Tail codes (real rank >= {TAIL_START}): {len(tail_codes)}")

    for model_name in ["HALO", "LSTM", "GPT", "AdaPCLA"]:
        path = paths[model_name]
        if not path.exists():
            print(f"Skip {model_name}: {path} not found.")
            continue
        print(f"Loading {model_name}...")
        d_model = load_pkl(path)
        model_freq = get_visit_freq(d_model)

        # 全量
        B, F0 = compute_b_f0_for_model(real_freq, model_freq)
        print(f"  {model_name}: full {len(B)} codes.")
        out_png = OUT_DIR / f"bf_scatter_{dataset}_{model_name.lower()}.png"
        plot_bf_scatter(B, F0, model_name, dataset, out_png)
        print(f"  Saved {out_png}")

        # tail-only (real rank >= TAIL_START)
        B_tail, F0_tail = compute_b_f0_for_model(
            real_freq, model_freq, codes_subset=tail_codes
        )
        if len(B_tail) == 0:
            print(f"  {model_name}: no tail codes, skip tail plot.")
            continue
        print(f"  {model_name}: tail {len(B_tail)} codes.")
        out_tail = OUT_DIR / f"bf_scatter_{dataset}_{model_name.lower()}_tail.png"
        plot_bf_scatter(
            B_tail,
            F0_tail,
            model_name,
            dataset,
            out_tail,
            title_suffix=f" (tail codes, rank ≥ {TAIL_START})",
        )
        print(f"  Saved {out_tail}")


if __name__ == "__main__":
    main()


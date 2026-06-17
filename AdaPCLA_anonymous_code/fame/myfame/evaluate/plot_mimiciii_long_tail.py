#!/usr/bin/env python3
"""
Plot long-tail statistics of medical codes in the (processed) MIMIC-III train split.

By default, we compute visit-level code counts: for each visit, a code is counted at most once.
Outputs a PNG figure containing:
  (1) rank-frequency curve (log-log)
  (2) cumulative mass coverage vs rank
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import os
import pickle
from pathlib import Path

import numpy as np


DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/long_tail_mimiciii"


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _compute_visit_counts(train_data, *, code_vocab_size: int) -> np.ndarray:
    counts = np.zeros((code_vocab_size,), dtype=np.int64)
    for p in train_data:
        for v in p.get("visits", []):
            if not v:
                continue
            for c in set(v):
                ci = int(c)
                if 0 <= ci < code_vocab_size:
                    counts[ci] += 1
    return counts


def _chapter(code: str) -> str:
    c = str(code).strip().upper()
    if c.startswith("E"):
        return "E (external causes)"
    if c.startswith("V"):
        return "V (supplementary)"
    digits = "".join(ch for ch in c if ch.isdigit())
    if len(digits) < 3:
        return "Other"
    try:
        n = int(digits[:3])
    except Exception:
        return "Other"
    if 1 <= n <= 139:
        return "001-139 Infectious"
    if 140 <= n <= 239:
        return "140-239 Neoplasms"
    if 240 <= n <= 279:
        return "240-279 Endocrine/metabolic"
    if 280 <= n <= 289:
        return "280-289 Blood"
    if 290 <= n <= 319:
        return "290-319 Mental"
    if 320 <= n <= 389:
        return "320-389 Nervous/sense"
    if 390 <= n <= 459:
        return "390-459 Circulatory"
    if 460 <= n <= 519:
        return "460-519 Respiratory"
    if 520 <= n <= 579:
        return "520-579 Digestive"
    if 580 <= n <= 629:
        return "580-629 Genitourinary"
    if 630 <= n <= 679:
        return "630-679 Pregnancy"
    if 680 <= n <= 709:
        return "680-709 Skin"
    if 710 <= n <= 739:
        return "710-739 Musculoskeletal"
    if 740 <= n <= 759:
        return "740-759 Congenital"
    if 760 <= n <= 779:
        return "760-779 Perinatal"
    if 780 <= n <= 799:
        return "780-799 Symptoms"
    if 800 <= n <= 999:
        return "800-999 Injury/poison"
    return "Other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Directory containing trainDataset.pkl and codeToIndex.pkl")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--count_mode",
        choices=["visit"],
        default="visit",
        help="Counting mode (currently only visit-level, de-duplicated within visit).",
    )
    ap.add_argument(
        "--paper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a paper-friendly style and export both PNG and PDF.",
    )
    ap.add_argument(
        "--pie",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also export a pie chart summarizing head mass concentration (top 1/5/10%% vs rest).",
    )
    ap.add_argument(
        "--export_lists",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export CSV lists of top 1% codes and remaining 90% codes (with counts and coarse chapter labels).",
    )
    args = ap.parse_args()

    data_dir = str(args.data_dir)
    out_dir = str(args.out_dir)
    _safe_mkdir(out_dir)

    code_to_index = _load_pkl(os.path.join(data_dir, "codeToIndex.pkl"))
    train_data = _load_pkl(os.path.join(data_dir, "trainDataset.pkl"))
    code_vocab_size = int(len(code_to_index))

    if args.count_mode != "visit":
        raise ValueError(f"unsupported count_mode: {args.count_mode}")
    counts = _compute_visit_counts(train_data, code_vocab_size=code_vocab_size)

    freq = np.asarray([int(x) for x in counts.tolist() if int(x) > 0], dtype=np.int64)
    freq.sort()
    freq = freq[::-1]  # descending

    ranks = np.arange(1, int(freq.size) + 1, dtype=np.int64)
    total_mass = float(freq.sum())
    cum_mass = np.cumsum(freq, dtype=np.float64)
    cum_frac = cum_mass / max(1.0, total_mass)

    # Summary for paper text.
    def _top_frac(p: float) -> float:
        k = max(1, int(round(p * freq.size)))
        return float(cum_frac[min(k - 1, cum_frac.size - 1)])

    top_1pct = _top_frac(0.01)
    top_5pct = _top_frac(0.05)
    top_10pct = _top_frac(0.10)

    if bool(args.export_lists):
        # Build rank list with code strings.
        idx_to_code = [None] * code_vocab_size
        for code, idx in code_to_index.items():
            if 0 <= int(idx) < code_vocab_size:
                idx_to_code[int(idx)] = str(code)
        present = np.where(counts > 0)[0]
        order = present[np.argsort(counts[present])[::-1]]
        n_present = int(order.size)
        k1 = max(1, int(round(0.01 * n_present)))
        k10 = max(1, int(round(0.10 * n_present)))
        top1 = order[:k1]
        rest90 = order[k10:]

        import csv

        top1_csv = os.path.join(out_dir, "mimiciii_train_top1pct_codes.csv")
        with open(top1_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "code", "visit_count", "chapter"])
            w.writeheader()
            for r, idx in enumerate(top1.tolist(), start=1):
                code = idx_to_code[int(idx)] or str(int(idx))
                w.writerow({"rank": int(r), "code": code, "visit_count": int(counts[int(idx)]), "chapter": _chapter(code)})

        rest_csv = os.path.join(out_dir, "mimiciii_train_remaining90pct_codes.csv")
        with open(rest_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["rank_global", "code", "visit_count", "chapter"])
            w.writeheader()
            for r0, idx in enumerate(rest90.tolist(), start=k10 + 1):
                code = idx_to_code[int(idx)] or str(int(idx))
                w.writerow({"rank_global": int(r0), "code": code, "visit_count": int(counts[int(idx)]), "chapter": _chapter(code)})

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if bool(args.paper):
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.dpi": 200,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), dpi=200)

    ax = axes[0]
    ax.plot(ranks, freq, linewidth=1.2, color="#1B3A57")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Code rank (log)")
    ax.set_ylabel("Visit count (log)")
    ax.set_title("Rank-frequency")
    if not bool(args.paper):
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)

    ax = axes[1]
    ax.plot(ranks, cum_frac, linewidth=1.2, color="#1B3A57")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Top-k codes (log)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative coverage")
    if not bool(args.paper):
        ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.text(
        0.03,
        0.10,
        f"Top 1% cover {top_1pct:.1%}\\nTop 5% cover {top_5pct:.1%}\\nTop 10% cover {top_10pct:.1%}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, linewidth=0.5),
    )

    fig.tight_layout()

    stem = "mimiciii_train_visit_long_tail_paper" if bool(args.paper) else "mimiciii_train_visit_long_tail"
    out_png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out_png, bbox_inches="tight")
    out_pdf = None
    if bool(args.paper):
        out_pdf = os.path.join(out_dir, f"{stem}.pdf")
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    pie_png = None
    pie_pdf = None
    if bool(args.pie):
        labels = ["Top 1%", "Next 4% (1–5%)", "Next 5% (5–10%)", "Remaining 90%"]
        vals = [
            float(top_1pct),
            float(top_5pct - top_1pct),
            float(top_10pct - top_5pct),
            float(1.0 - top_10pct),
        ]
        vals = np.clip(np.asarray(vals, dtype=np.float64), 0.0, 1.0)
        vals = vals / max(1e-12, float(vals.sum()))

        fig2, ax2 = plt.subplots(1, 1, figsize=(5.2, 3.2), dpi=200)
        colors = ["#1B3A57", "#2F6F9E", "#4DAA9A", "#D9D9D9"]
        wedges, _texts, _autotexts = ax2.pie(
            vals,
            colors=colors,
            startangle=90,
            autopct=lambda p: f"{p:.1f}%",
            pctdistance=0.75,
            wedgeprops=dict(linewidth=0.8, edgecolor="white"),
        )
        ax2.axis("equal")
        ax2.set_title("MIMIC-III head mass concentration (visit-level)")
        ax2.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig2.tight_layout()

        pie_stem = "mimiciii_train_visit_long_tail_pie_paper" if bool(args.paper) else "mimiciii_train_visit_long_tail_pie"
        pie_png = os.path.join(out_dir, f"{pie_stem}.png")
        fig2.savefig(pie_png, bbox_inches="tight")
        if bool(args.paper):
            pie_pdf = os.path.join(out_dir, f"{pie_stem}.pdf")
            fig2.savefig(pie_pdf, bbox_inches="tight")
        plt.close(fig2)

    stats_path = os.path.join(out_dir, "mimiciii_train_visit_long_tail_stats.txt")
    Path(stats_path).write_text(
        "\n".join(
            [
                f"data_dir={data_dir}",
                f"code_vocab_size={code_vocab_size}",
                f"codes_with_positive_count={int(freq.size)}",
                f"total_visit_code_mass={int(total_mass)}",
                f"top_1pct_mass_frac={top_1pct:.6f}",
                f"top_5pct_mass_frac={top_5pct:.6f}",
                f"top_10pct_mass_frac={top_10pct:.6f}",
                f"wrote_png={out_png}",
                f"wrote_pdf={out_pdf or ''}",
                f"wrote_pie_png={pie_png or ''}",
                f"wrote_pie_pdf={pie_pdf or ''}",
                f"wrote_top1pct_csv={os.path.join(out_dir, 'mimiciii_train_top1pct_codes.csv') if bool(args.export_lists) else ''}",
                f"wrote_remaining90pct_csv={os.path.join(out_dir, 'mimiciii_train_remaining90pct_codes.csv') if bool(args.export_lists) else ''}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved figure: {out_png}")
    if out_pdf:
        print(f"Saved figure: {out_pdf}")
    if pie_png:
        print(f"Saved figure: {pie_png}")
    if pie_pdf:
        print(f"Saved figure: {pie_pdf}")
    print(f"Saved stats:  {stats_path}")


if __name__ == "__main__":
    main()

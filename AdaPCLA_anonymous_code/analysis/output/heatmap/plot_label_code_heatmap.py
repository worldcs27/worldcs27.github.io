#!/usr/bin/env python3
"""
Label-Code Association Heatmap (MIMIC-IV model5).

绘制 25 labels × ~30 codes 的分桶热力图：head / mid / tail 各一张。
使用 P(code | label) = 在 label 为正的患者的 visit 中，包含 code 的 visit 比例。
真实数据 + HALO + LSTM + AdaPCLA 对比。横轴使用 D_ICD_DIAGNOSES 的 SHORT_TITLE。

数据来源:
- 真实: fame/myfame/data2/trainDataset.pkl
- HALO: fame/myfame/baseline/HALO2/save_mimiciv_seed1/datasets/haloDataset.pkl
- LSTM: fame/myfame/baseline/lstm/save_mimiciv_seed1/datasets/lstmDataset.pkl
- AdaPCLA: model5/save_anneal_mimiciv/seed1/datasets/haloDataset.pkl
- 分桶: fame/myfame/output/长尾分布问题分析/mimiciv_code_buckets.csv
- ICD名称: D_ICD_DIAGNOSES.csv (SHORT_TITLE)
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import csv
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PCLA_ROOT = SCRIPT_DIR.parents[2]  # heatmap -> output -> mywork -> PCLA
FAME_ROOT = PCLA_ROOT / "fame" / "myfame"
MYWORK_ROOT = PCLA_ROOT / "mywork"

REAL_PATH = FAME_ROOT / "data2" / "trainDataset.pkl"
HALO_PATH = FAME_ROOT / "baseline" / "HALO2" / "save_mimiciv_seed1" / "datasets" / "haloDataset.pkl"
LSTM_PATH = FAME_ROOT / "baseline" / "lstm" / "save_mimiciv_seed1" / "datasets" / "lstmDataset.pkl"
ADAPCLA_PATH = MYWORK_ROOT / "model5" / "save_anneal_mimiciv" / "seed1" / "datasets" / "haloDataset.pkl"
BUCKET_CSV = FAME_ROOT / "output" / "长尾分布问题分析" / "mimiciv_code_buckets.csv"
ID_TO_LABEL_PATH = FAME_ROOT / "data2" / "idToLabel.pkl"
D_ICD_PATH = SCRIPT_DIR / "D_ICD_DIAGNOSES.csv"

N_LABELS = 25
N_CODES_PER_BUCKET = 30
SHORT_TITLE_MAX_LEN = 16
OUT_DIR = SCRIPT_DIR


def load_icd_short_titles(path: Path) -> dict[str, str]:
    """Build mapping: ICD code string -> SHORT_TITLE. Tries ICD9_CODE and ICD10_CODE."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = str(row.get("SHORT_TITLE", "") or "").strip().strip('"')
            for col in ("ICD9_CODE", "ICD10_CODE"):
                code = (row.get(col, "") or row.get(col.replace("_", ""), "")).strip().strip('"')
                if code and title and code not in out:
                    out[code] = title
                    break
    return out


def load_buckets() -> dict[str, list[tuple[int, str, int]]]:
    """Return {bucket: [(code_id, code_str, train_visit_count), ...]}."""
    buckets: dict[str, list[tuple[int, str, int]]] = {"head": [], "mid": [], "tail": []}
    with open(BUCKET_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            b = row["bucket"].strip().lower()
            if b not in buckets:
                continue
            code_id = int(row["code_id"])
            code_str = row["code"]
            cnt = int(row["train_visit_count"])
            buckets[b].append((code_id, code_str, cnt))
    for b in buckets:
        # Sort by train_visit_count desc, take top N_CODES_PER_BUCKET
        buckets[b].sort(key=lambda x: x[2], reverse=True)
        buckets[b] = buckets[b][:N_CODES_PER_BUCKET]
    return buckets


def compute_p_code_given_label(
    data: list[dict],
    code_ids: list[int],
    n_labels: int = N_LABELS,
) -> np.ndarray:
    """
    M[label_idx, col_idx] = P(code in visit | label positive).
    For label L: among visits from patients with labels[L]==1, fraction that contain code C.
    """
    code_set = set(code_ids)
    M = np.zeros((n_labels, len(code_ids)), dtype=np.float64)
    code_to_col = {c: i for i, c in enumerate(code_ids)}

    for label_idx in range(n_labels):
        total_visits = 0
        count_per_code = np.zeros(len(code_ids), dtype=np.int64)

        for p in data:
            if not (p["labels"][label_idx] > 0.5):
                continue
            visits = p.get("visits", [])
            for v in visits:
                if not v:
                    continue
                total_visits += 1
                codes_in_visit = set(v) & code_set
                for c in codes_in_visit:
                    count_per_code[code_to_col[c]] += 1

        if total_visits > 0:
            M[label_idx, :] = count_per_code.astype(np.float64) / total_visits
    return M


def load_data(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def code_strs_to_labels(code_strs: list[str], icd_map: dict[str, str]) -> list[str]:
    """Convert ICD code strings to SHORT_TITLE labels (truncated)."""
    labels = []
    for s in code_strs:
        title = icd_map.get(s, s)
        if len(title) > SHORT_TITLE_MAX_LEN:
            title = title[: SHORT_TITLE_MAX_LEN - 1] + "…"
        labels.append(title)
    return labels


def plot_heatmap(
    M: np.ndarray,
    label_names: list[str],
    code_labels: list[str],
    title: str,
    out_path: Path,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "YlOrRd",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    if vmin is None:
        vmin = float(np.nanmin(M)) if np.any(~np.isnan(M)) else 0
    if vmax is None:
        vmax = float(np.nanmax(M)) if np.any(~np.isnan(M)) else 1

    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(code_labels)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(code_labels, fontsize=6, rotation=90, ha="right")
    ax.set_yticklabels(label_names, fontsize=7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Code (ICD)", fontsize=9)
    ax.set_ylabel("Label", fontsize=9)
    plt.colorbar(im, ax=ax, label="P(code | label)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("1. Loading buckets...")
    buckets = load_buckets()

    print("2. Loading ICD SHORT_TITLE mapping...")
    icd_map = load_icd_short_titles(D_ICD_PATH) if D_ICD_PATH.exists() else {}
    print(f"   Loaded {len(icd_map)} ICD code mappings")

    print("3. Loading label names...")
    with open(ID_TO_LABEL_PATH, "rb") as f:
        id_to_label = pickle.load(f)
    if isinstance(id_to_label, list):
        label_names = [str(id_to_label[i])[:30] if i < len(id_to_label) else f"L{i}" for i in range(N_LABELS)]
    else:
        label_names = [str(id_to_label.get(i, f"L{i}"))[:30] for i in range(N_LABELS)]

    print("4. Loading datasets...")
    datasets: dict[str, list[dict]] = {}
    datasets["Real"] = load_data(REAL_PATH)
    datasets["HALO"] = load_data(HALO_PATH) if HALO_PATH.exists() else []
    datasets["LSTM"] = load_data(LSTM_PATH) if LSTM_PATH.exists() else []
    datasets["AdaPCLA"] = load_data(ADAPCLA_PATH) if ADAPCLA_PATH.exists() else datasets["Real"]
    for k, v in datasets.items():
        print(f"   {k}: {len(v)} patients")

    sources = ["Real", "HALO", "LSTM", "AdaPCLA"]

    for bucket_name in ["head", "mid", "tail"]:
        entries = buckets[bucket_name]
        if not entries:
            print(f"   Skip {bucket_name}: no codes")
            continue
        code_ids = [e[0] for e in entries]
        code_strs = [e[1] for e in entries]
        code_labels = code_strs_to_labels(code_strs, icd_map)

        print(f"5. Computing P(code|label) for {bucket_name} ({len(code_ids)} codes)...")
        Ms: dict[str, np.ndarray] = {}
        for src in sources:
            if datasets[src]:
                Ms[src] = compute_p_code_given_label(datasets[src], code_ids)
            else:
                Ms[src] = np.zeros((N_LABELS, len(code_ids)), dtype=np.float64)

        vmin = min(np.nanmin(Ms[s]) for s in sources if Ms[s].size > 0)
        vmax = max(np.nanmax(Ms[s]) for s in sources if Ms[s].size > 0)
        if vmax <= vmin:
            vmax = vmin + 0.01

        for src in sources:
            out_path = OUT_DIR / f"heatmap_{src.lower()}_{bucket_name}.png"
            plot_heatmap(
                Ms[src], label_names, code_labels,
                title=f"{src} ({bucket_name.capitalize()} Codes)",
                out_path=out_path,
                vmin=vmin, vmax=vmax,
            )
            print(f"   Saved {out_path.name}")

    # Combined figure: 3 rows (head/mid/tail) x 4 cols (Real, HALO, LSTM, AdaPCLA)
    # Per-row normalization so mid/tail rows have visible structure; one color bar = relative (0-1) per row.
    print("6. Creating combined figure (3x4) with per-row scale and shared colorbar...")

    combined_Ms: dict[tuple[str, str], np.ndarray] = {}
    combined_code_labels: dict[str, list[str]] = {}

    for bucket_name in ["head", "mid", "tail"]:
        entries = buckets[bucket_name]
        if not entries:
            continue
        code_ids = [e[0] for e in entries]
        code_strs = [e[1] for e in entries]
        code_labels = code_strs_to_labels(code_strs, icd_map)
        combined_code_labels[bucket_name] = code_labels

        for src in sources:
            if datasets[src]:
                M = compute_p_code_given_label(datasets[src], code_ids)
            else:
                M = np.zeros((N_LABELS, len(code_ids)), dtype=np.float64)
            combined_Ms[(bucket_name, src)] = M

    fig, axes = plt.subplots(3, 4, figsize=(18, 14), constrained_layout=True)
    last_im = None
    bucket_order = ["head", "mid", "tail"]

    for row, bucket_name in enumerate(bucket_order):
        entries = buckets[bucket_name]
        if not entries:
            continue
        code_labels = combined_code_labels[bucket_name]

        # Per-row vmin/vmax: use only this row's four matrices so all three rows have visible gradient.
        row_mats = [combined_Ms.get((bucket_name, src)) for src in sources]
        row_mats = [m for m in row_mats if m is not None and m.size > 0]
        if row_mats:
            rmin = min(np.nanmin(m) for m in row_mats)
            rmax = max(np.nanmax(m) for m in row_mats)
            if rmax <= rmin:
                rmax = rmin + 1e-9
        else:
            rmin, rmax = 0.0, 1.0

        for col, src in enumerate(sources):
            ax = axes[row, col]
            M = combined_Ms.get((bucket_name, src))
            if M is None:
                M = np.zeros((N_LABELS, len(code_labels)), dtype=np.float64)

            # Normalize this row's data to [0, 1] so color bar is "relative per row".
            M_show = (M.astype(np.float64) - rmin) / (rmax - rmin)
            M_show = np.clip(M_show, 0.0, 1.0)

            im = ax.imshow(
                M_show,
                aspect="auto",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            last_im = im

            # X-axis: each row has its own code set, so keep ticks but use small fonts.
            ax.set_xticks(np.arange(len(code_labels)))
            ax.set_xticklabels(code_labels, fontsize=5, rotation=90, ha="right")

            # Y-axis: only show labels on the first column to avoid repetition.
            ax.set_yticks(np.arange(len(label_names)))
            if col == 0:
                ax.set_yticklabels(label_names, fontsize=6)
                # Optional row label to the left, indicating bucket.
                ax.set_ylabel(f"{bucket_name.capitalize()} labels", fontsize=9)
            else:
                ax.set_yticklabels([])

            # Column titles only on the top row: model names.
            if row == 0:
                ax.set_title(src, fontsize=12)

            # X-axis label only on the bottom row.
            if row == len(bucket_order) - 1:
                ax.set_xlabel("Codes (ICD)", fontsize=9)

    # Single colorbar: scale is 0--1 "relative per row" (each row normalized to its own min/max).
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, location="right", shrink=0.8, pad=0.02)
        cbar.set_label("P(code | label) (relative per row)", fontsize=10)

    plt.savefig(OUT_DIR / "heatmap_combined_3x4.png", dpi=150)
    plt.close()
    print("   Saved heatmap_combined_3x4.png")

    # Save matrices to CSV for reference
    for bucket_name in ["head", "mid", "tail"]:
        entries = buckets[bucket_name]
        if not entries:
            continue
        code_ids = [e[0] for e in entries]
        code_strs = [e[1] for e in entries]
        for src in sources:
            if datasets[src]:
                M = compute_p_code_given_label(datasets[src], code_ids)
            else:
                M = np.zeros((N_LABELS, len(code_ids)))
            out_csv = OUT_DIR / f"matrix_{src.lower()}_{bucket_name}.csv"
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["label"] + code_strs)
                for i, ln in enumerate(label_names):
                    w.writerow([ln] + [f"{M[i, j]:.6f}" for j in range(M.shape[1])])
            print(f"   Saved {out_csv.name}")

    print("Done.")


if __name__ == "__main__":
    main()

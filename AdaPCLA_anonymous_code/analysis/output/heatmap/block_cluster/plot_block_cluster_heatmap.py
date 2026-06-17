#!/usr/bin/env python3
"""
Code co-occurrence block heatmaps with Real-based hierarchical clustering.

目标:
- 从真实数据 (MIMIC-IV) 中选取频率最高的 Top-200 codes。
- 用真实数据的 code–code 共现矩阵做层次聚类 (hierarchical clustering)，
  得到一个固定的 code 排序索引。
- 将该排序同时应用到 Real / AdaPCLA / HALO 的共现矩阵，画成一行三张子图:
  [Real | AdaPCLA | HALO]，对比谁更好地保留 block-diagonal 结构。

说明:
- 共现定义为: 同一 visit 内共同出现的 code 计为一次共现（对角线是 code 出现于 visit 的次数）。
- 矩阵变换采用 PPMI (Positive Pointwise Mutual Information) 增强对比度。
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

from pathlib import Path
from collections import Counter
import csv
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list


SCRIPT_DIR = Path(__file__).resolve().parent
# block_cluster -> heatmap -> output -> mywork -> PCLA
PCLA_ROOT = SCRIPT_DIR.parents[3]
FAME_ROOT = PCLA_ROOT / "fame" / "myfame"
MYWORK_ROOT = PCLA_ROOT / "mywork"

# 数据路径 (MIMIC-IV)
REAL_PATH = FAME_ROOT / "data2" / "trainDataset.pkl"
HALO_PATH = FAME_ROOT / "baseline" / "HALO2" / "save_mimiciv_seed1" / "datasets" / "haloDataset.pkl"
LSTM_PATH = FAME_ROOT / "baseline" / "lstm" / "save_mimiciv_seed1" / "datasets" / "lstmDataset.pkl"
ADAPCLA_PATH = MYWORK_ROOT / "model5" / "save_anneal_mimiciv" / "seed1" / "datasets" / "haloDataset.pkl"

OUT_DIR = SCRIPT_DIR
TOP_K_CODES = 200

# head/mid/tail bucket 信息
BUCKET_CSV = FAME_ROOT / "output" / "长尾分布问题分析" / "mimiciv_code_buckets.csv"
BUCKET_NAMES = ["head", "mid", "tail"]
N_CODES_PER_BUCKET = 40  # 每个 bucket 选取的 code 数量


def load_data(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_code_counts(data: list[dict]) -> Counter[int]:
    cnt: Counter[int] = Counter()
    for p in data:
        for v in p.get("visits", []):
            for c in v:
                cnt[int(c)] += 1
    return cnt


def load_bucket_top_codes() -> dict[str, list[int]]:
    """
    从 mimiciv_code_buckets.csv 读取 head/mid/tail 的 code，
    按 train_visit_count 降序选取每个 bucket 的 Top-N_CODES_PER_BUCKET code_id。
    返回: {bucket_name: [code_id, ...]}
    """
    buckets: dict[str, list[tuple[int, int]]] = {b: [] for b in BUCKET_NAMES}
    if not BUCKET_CSV.exists():
        return {b: [] for b in BUCKET_NAMES}

    with open(BUCKET_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            b = (row.get("bucket") or "").strip().lower()
            if b not in buckets:
                continue
            try:
                code_id = int(row["code_id"])
                cnt = int(row.get("train_visit_count", "0"))
            except Exception:
                continue
            buckets[b].append((code_id, cnt))

    code_ids_per_bucket: dict[str, list[int]] = {}
    for b in BUCKET_NAMES:
        entries = buckets.get(b, [])
        entries.sort(key=lambda x: x[1], reverse=True)
        entries = entries[:N_CODES_PER_BUCKET]
        code_ids_per_bucket[b] = [cid for cid, _ in entries]
    return code_ids_per_bucket


def build_coocc_matrix(data: list[dict], codes: list[int]) -> np.ndarray:
    """
    返回 |codes| x |codes| 的共现矩阵:
      M[i, j] = 在同一 visit 中 i 与 j 共同出现的次数 (对称，含对角线)。
    """
    n = len(codes)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    M = np.zeros((n, n), dtype=np.float64)

    code_set = set(codes)
    for p in data:
        for v in p.get("visits", []):
            if not v:
                continue
            vs = sorted({int(c) for c in v if int(c) in code_set})
            if not vs:
                continue
            idxs = [code_to_idx[c] for c in vs]
            # 对角线: code 出现在该 visit
            for i in idxs:
                M[i, i] += 1.0
            # 非对角线: 成对共现
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    M[a, b] += 1.0
                    M[b, a] += 1.0
    return M


def ppmi_transform(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对共现矩阵做 PPMI 变换:
      PMI(i,j) = log2( P(i,j) / (P(i) P(j)) ), PPMI = max(PMI, 0)
    """
    M = M.astype(np.float64)
    total = M.sum()
    if total <= 0:
        return np.zeros_like(M)

    P_ij = M / total
    P_i = P_ij.sum(axis=1, keepdims=True)
    P_j = P_ij.sum(axis=0, keepdims=True)

    denom = P_i @ P_j  # outer product
    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log2((P_ij + eps) / (denom + eps))
    PMI[np.isneginf(PMI)] = 0.0
    PMI[np.isnan(PMI)] = 0.0
    PPMI = np.maximum(PMI, 0.0)
    return PPMI


def get_real_based_order(M_real_ppmi: np.ndarray) -> np.ndarray:
    """
    使用真实数据的 PPMI 矩阵做层次聚类，返回 code 的重排索引。

    这里把每一行 (code 的共现向量) 当作一个样本，使用欧氏距离 + average linkage。
    scipy.cluster.hierarchy.leaves_list 返回树上叶节点的顺序，用于重排矩阵行列。
    """
    # linkage 接受 (n_samples, n_features) 的矩阵
    Z = linkage(M_real_ppmi, method="average", metric="euclidean")
    order = leaves_list(Z)  # shape (n_codes,)
    return order


def reorder_matrix(M: np.ndarray, order: np.ndarray) -> np.ndarray:
    return M[np.ix_(order, order)]


def plot_block_row_3(
    M_real: np.ndarray,
    M_halo: np.ndarray,
    M_adapcla: np.ndarray,
    codes: list[int],
    out_path: Path,
) -> None:
    """单行 3 列: Real | HALO | AdaPCLA，无标题。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    mats = [M_real, M_halo, M_adapcla]

    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    last_im = None
    for ax, M in zip(axes, mats):
        im = ax.imshow(
            M,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        last_im = im
        ax.set_xlabel("Codes (clustered index)")
        ax.set_ylabel("Codes (clustered index)")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("PPMI of co-occurrence", fontsize=10)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_block_single(
    M: np.ndarray, out_path: Path, vmin: float, vmax: float, show_colorbar: bool = True
) -> None:
    """单张热力图，无标题。show_colorbar=False 时去掉 colorbar，保持 heatmap 尺寸不变。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    # 增强对比：使用 plasma 色图（块状更醒目），vmax 限制使高值饱和、块块更突出
    im = ax.imshow(
        M,
        cmap="plasma",  # plasma 比 viridis 在块状区对比更明显
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xlabel("Codes (clustered index)")
    ax.set_ylabel("Codes (clustered index)")

    # 统一 axes 位置，保持 heatmap 尺寸一致
    fig.subplots_adjust(left=0.12, right=0.85, bottom=0.12, top=0.95)
    if show_colorbar:
        cbar_ax = fig.add_axes([0.87, 0.12, 0.03, 0.76])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("PPMI of co-occurrence", fontsize=10)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_block_row_4(
    M_real: np.ndarray,
    M_halo: np.ndarray,
    M_lstm: np.ndarray,
    M_adapcla: np.ndarray,
    codes: list[int],
    out_path: Path,
) -> None:
    """单行 4 列: Real | HALO | LSTM | AdaPCLA。"""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    mats = [M_real, M_halo, M_lstm, M_adapcla]
    titles = ["Real", "HALO", "LSTM", "AdaPCLA"]

    # 统一 color scale 以便比较
    vmin = min(m.min() for m in mats)
    vmax = max(m.max() for m in mats)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    last_im = None
    for ax, M, title in zip(axes, mats, titles):
        im = ax.imshow(
            M,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        last_im = im
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Codes (clustered index)")
        ax.set_ylabel("Codes (clustered index)")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("PPMI of co-occurrence", fontsize=10)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_block_3x4(
    row_mats: dict[tuple[str, str], np.ndarray],
    codes_per_bucket: dict[str, list[int]],
    out_path: Path,
) -> None:
    """
    3×4 总图:
      行: head / mid / tail
      列: Real / HALO / LSTM / AdaPCLA
    row_mats[(bucket, model)] = PPMI 矩阵（已按 Real-based clustering 排序）
    """
    models = ["Real", "HALO", "LSTM", "AdaPCLA"]
    fig, axes = plt.subplots(3, 4, figsize=(20, 16), constrained_layout=True)

    for row, bucket in enumerate(BUCKET_NAMES):
        codes = codes_per_bucket.get(bucket, [])
        if not codes:
            # 清空该行
            for col in range(4):
                axes[row, col].axis("off")
            continue

        # 该行的 4 个矩阵，用于确定 vmin/vmax（行内共享色标）
        mats_row = [row_mats[(bucket, m)] for m in models]
        vmin = min(m.min() for m in mats_row)
        vmax = max(m.max() for m in mats_row)
        if vmax <= vmin:
            vmax = vmin + 1e-6

        for col, model in enumerate(models):
            ax = axes[row, col]
            M = row_mats[(bucket, model)]
            im = ax.imshow(
                M,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )

            # 第一行显示列标题（模型名）
            if row == 0:
                ax.set_title(model, fontsize=12)
            # 第一列显示 bucket 名称
            if col == 0:
                ax.set_ylabel(f"{bucket.capitalize()} codes", fontsize=11)
            else:
                ax.set_yticklabels([])
            ax.set_xticklabels([])

    # 共享 colorbar（用第一行第一个 axes 的 im）
    first_im = axes[0, 0].images[0] if axes[0, 0].images else None
    if first_im is not None:
        cbar = fig.colorbar(first_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("PPMI of co-occurrence", fontsize=10)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading datasets (MIMIC-IV)...")
    real_data = load_data(REAL_PATH)
    halo_data = load_data(HALO_PATH) if HALO_PATH.exists() else []
    lstm_data = load_data(LSTM_PATH) if LSTM_PATH.exists() else []
    adapcla_data = load_data(ADAPCLA_PATH) if ADAPCLA_PATH.exists() else []
    print(f"   Real:     {len(real_data)} patients")
    print(f"   HALO:     {len(halo_data)} patients")
    print(f"   LSTM:     {len(lstm_data)} patients")
    print(f"   AdaPCLA:  {len(adapcla_data)} patients")

    # --------------------------------------------------------------
    # A. 单一 Top-K code (整体频率最高) 的 1×4 行图
    # --------------------------------------------------------------
    print(f"2. Selecting Top-{TOP_K_CODES} codes from Real by frequency...")
    cnt_real = get_code_counts(real_data)
    top_codes = [c for c, _ in cnt_real.most_common(TOP_K_CODES)]
    print(f"   Got {len(top_codes)} codes.")

    print("3. Building co-occurrence matrices for Top-K codes...")
    M_real = build_coocc_matrix(real_data, top_codes)
    M_halo = build_coocc_matrix(halo_data, top_codes) if halo_data else np.zeros_like(M_real)
    M_lstm = build_coocc_matrix(lstm_data, top_codes) if lstm_data else np.zeros_like(M_real)
    M_adapcla = build_coocc_matrix(adapcla_data, top_codes) if adapcla_data else np.zeros_like(M_real)

    print("4. Applying PPMI transform (Top-K)...")
    M_real_ppmi = ppmi_transform(M_real)
    M_halo_ppmi = ppmi_transform(M_halo)
    M_lstm_ppmi = ppmi_transform(M_lstm)
    M_adapcla_ppmi = ppmi_transform(M_adapcla)

    print("5. Hierarchical clustering on Real PPMI (Top-K) to get code order...")
    order = get_real_based_order(M_real_ppmi)
    print("   Order length:", len(order))

    print("6. Reordering all matrices with Real-based order (Top-K)...")
    M_real_ord = reorder_matrix(M_real_ppmi, order)
    M_halo_ord = reorder_matrix(M_halo_ppmi, order)
    M_lstm_ord = reorder_matrix(M_lstm_ppmi, order)
    M_adapcla_ord = reorder_matrix(M_adapcla_ppmi, order)

    print("7. Plotting block-structured heatmaps (Top-K, Real | HALO | LSTM | AdaPCLA)...")
    out_png_row = OUT_DIR / "code_coocc_block_cluster_mimic4_topk_row.png"
    plot_block_row_4(M_real_ord, M_halo_ord, M_lstm_ord, M_adapcla_ord, top_codes, out_png_row)
    print(f"   Saved {out_png_row}")

    print("7b. Plotting 3-panel heatmap (Real | HALO | AdaPCLA, no titles)...")
    out_png_3 = OUT_DIR / "code_coocc_block_cluster_mimic4_real_halo_adapcla.png"
    plot_block_row_3(M_real_ord, M_halo_ord, M_adapcla_ord, top_codes, out_png_3)
    print(f"   Saved {out_png_3}")

    print("7c. Plotting 3 separate heatmaps (Real, HALO, AdaPCLA)...")
    # 限制 colorbar 区间以增强对比：vmin=0，vmax 取 92 分位数使高值饱和，块状更明显
    all_vals = np.concatenate([M_real_ord.ravel(), M_halo_ord.ravel(), M_adapcla_ord.ravel()])
    vmin = 0.0
    vmax = float(np.percentile(all_vals[all_vals > 0], 92)) if np.any(all_vals > 0) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    plot_block_single(M_real_ord, OUT_DIR / "code_coocc_block_cluster_mimic4_real.png", vmin, vmax, show_colorbar=False)
    plot_block_single(M_halo_ord, OUT_DIR / "code_coocc_block_cluster_mimic4_halo.png", vmin, vmax, show_colorbar=False)
    plot_block_single(M_adapcla_ord, OUT_DIR / "code_coocc_block_cluster_mimic4_adapcla.png", vmin, vmax, show_colorbar=True)
    print(f"   Saved Real, HALO, AdaPCLA separate images")

    # --------------------------------------------------------------
    # B. head / mid / tail 三个 code 组别的 3×4 总图
    # --------------------------------------------------------------
    print("8. Loading head/mid/tail buckets and selecting top codes per bucket...")
    codes_per_bucket = load_bucket_top_codes()
    for b in BUCKET_NAMES:
        print(f"   {b}: {len(codes_per_bucket.get(b, []))} codes")

    row_mats: dict[tuple[str, str], np.ndarray] = {}

    for bucket in BUCKET_NAMES:
        codes_b = codes_per_bucket.get(bucket, [])
        if not codes_b:
            continue
        print(f"9. Building co-occurrence matrices for bucket '{bucket}'...")
        M_real_b = build_coocc_matrix(real_data, codes_b)
        M_halo_b = build_coocc_matrix(halo_data, codes_b) if halo_data else np.zeros_like(M_real_b)
        M_lstm_b = build_coocc_matrix(lstm_data, codes_b) if lstm_data else np.zeros_like(M_real_b)
        M_adapcla_b = build_coocc_matrix(adapcla_data, codes_b) if adapcla_data else np.zeros_like(M_real_b)

        print(f"   Applying PPMI transform for bucket '{bucket}'...")
        M_real_b_ppmi = ppmi_transform(M_real_b)
        M_halo_b_ppmi = ppmi_transform(M_halo_b)
        M_lstm_b_ppmi = ppmi_transform(M_lstm_b)
        M_adapcla_b_ppmi = ppmi_transform(M_adapcla_b)

        print(f"   Hierarchical clustering on Real PPMI for bucket '{bucket}'...")
        order_b = get_real_based_order(M_real_b_ppmi)

        print(f"   Reordering all matrices for bucket '{bucket}'...")
        row_mats[(bucket, "Real")] = reorder_matrix(M_real_b_ppmi, order_b)
        row_mats[(bucket, "HALO")] = reorder_matrix(M_halo_b_ppmi, order_b)
        row_mats[(bucket, "LSTM")] = reorder_matrix(M_lstm_b_ppmi, order_b)
        row_mats[(bucket, "AdaPCLA")] = reorder_matrix(M_adapcla_b_ppmi, order_b)

    print("10. Plotting 3x4 block-structured heatmaps (head/mid/tail × Real/HALO/LSTM/AdaPCLA)...")
    out_png_3x4 = OUT_DIR / "code_coocc_block_cluster_mimic4_3x4.png"
    plot_block_3x4(row_mats, codes_per_bucket, out_png_3x4)
    print(f"   Saved {out_png_3x4}")


if __name__ == "__main__":
    main()


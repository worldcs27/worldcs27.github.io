#!/usr/bin/env python3
"""
Compute λ_min(K_x) vs related_rare gain/AUC for internalization bound validation.

Goal: Use λ_min(K_x) as a proxy for kernel strength and relate it to
related_rare gain/AUC from trajectory data. The bound suggests larger
λ_min(K) should correlate with better internalization.

K_x is the label-wise eNTK Gram matrix on context x's related_rare labels:
  K_ij = <∇_θ z_i(x), ∇_θ z_j(x)> for i,j in related_rare.

We compute λ_min(K) at init and optionally at final checkpoint. Main analysis
uses init (theory aligns kernel strength with early/local dynamics).

Usage
-----
  python model6/compute_lambda_min_vs_gain.py \\
    --trajectory_csv model6/anneal_tail_trajectory_base.csv \\
    --probe_config model6/micro_probe_configs/mimiciv_long_tail_triplets_seed1.csv \\
    --ckpt_path_init model6/save_micro_probe_mimiciv/seed1/epoch_ckpts/micro_probe_ckpt_0000.pt \\
    [--ckpt_path_final model6/save_micro_probe_mimiciv/seed1/epoch_ckpts/micro_probe_ckpt_0099.pt] \\
    --out_csv model6/lambda_min_vs_gain.csv \\
    [--out_fig model6/lambda_min_vs_gain.png]
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL5_DIR = os.path.join(PROJECT_ROOT, "model5")
MODEL6_DIR = os.path.join(PROJECT_ROOT, "model6")
DEFAULT_DATA_DIR = DATA_MIMICIV


@dataclass(frozen=True)
class Pair:
    context_id: int
    disease_id: int


def load_related_rare_pairs(config_path: str) -> List[Pair]:
    pairs: List[Pair] = []
    with open(config_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("type") or "").strip().lower()
            if "related_rare" not in t:
                continue
            try:
                ctx = int(row["context_id"])
                did = int(row["disease_id"])
            except (KeyError, ValueError):
                continue
            pairs.append(Pair(ctx, did))
    seen = set()
    return [p for p in pairs if p not in seen and not seen.add(p)]


def context_to_related_rare_diseases(pairs: List[Pair]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = defaultdict(list)
    for p in pairs:
        out[p.context_id].append(p.disease_id)
    return dict(out)


def load_trajectory(path: str, focus_pairs: List[Pair]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"checkpoint_idx", "context_id", "disease_id", "mean_log_prob"}
    if required - set(df.columns):
        raise ValueError(f"Trajectory CSV missing columns: {required - set(df.columns)}")
    focus_set = {(p.context_id, p.disease_id) for p in focus_pairs}
    mask = df.apply(
        lambda r: (int(r["context_id"]), int(r["disease_id"])) in focus_set,
        axis=1,
    )
    return df.loc[mask].copy()


def compute_gain_and_auc_one_pair(sub: pd.DataFrame) -> Tuple[float, float]:
    if sub.empty:
        return float("nan"), float("nan")
    sub = sub.sort_values("checkpoint_idx")
    logs = sub["mean_log_prob"].to_numpy(dtype=np.float64)
    if logs.shape[0] < 2:
        return float("nan"), float("nan")
    log0, logT = logs[0], logs[-1]
    gain = float(logT - log0)
    y = logs - log0
    y_pos = np.maximum(y, 0.0)
    auc = float(np.sum(y_pos[1:]))  # Riemann sum Δt=1
    return gain, auc


def aggregate_trajectory_metrics(
    traj_df: pd.DataFrame, related_pairs: List[Pair]
) -> pd.DataFrame:
    """Per-context mean of gain and AUC over related_rare pairs."""
    focus_set = {(p.context_id, p.disease_id) for p in related_pairs}
    results: List[dict] = []
    for (ctx, did), sub in traj_df.groupby(["context_id", "disease_id"]):
        if (int(ctx), int(did)) not in focus_set:
            continue
        gain, auc = compute_gain_and_auc_one_pair(sub)
        results.append({"context_id": int(ctx), "disease_id": int(did), "gain": gain, "auc": auc})
    if not results:
        return pd.DataFrame(columns=["context_id", "disease_id", "gain", "auc"])
    df = pd.DataFrame(results)
    agg = df.groupby("context_id").agg({"gain": "mean", "auc": "mean"}).reset_index()
    agg = agg.rename(columns={"gain": "related_rare_final_gain_mean", "auc": "related_rare_AUC_mean"})
    return agg


def build_model_and_dataset(data_dir: str, device: str):
    import sys

    if MODEL8_DIR not in sys.path:
        sys.path.insert(0, MODEL8_DIR)
    if MODEL5_DIR not in sys.path:
        sys.path.insert(0, MODEL5_DIR)

    import pickle

    from config import Model2Config  # type: ignore
    from model import HALOModel  # type: ignore
    from run_pcla_fixed_bias_anneal_mimiciv import MIMICDataset  # type: ignore

    def load_pkl(name: str):
        with open(os.path.join(data_dir, name), "rb") as f:
            return pickle.load(f)

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")

    cfg = Model2Config()
    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    if not hasattr(cfg, "special_vocab_size"):
        cfg.special_vocab_size = 4
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    model = HALOModel(cfg).to(torch.device(device))
    train_ds = MIMICDataset(train_data, cfg)
    return model, cfg, train_ds


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    if next(iter(state.keys()), "").startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def flatten_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad.view(-1))
    if not grads:
        raise RuntimeError("No gradients found.")
    return torch.cat(grads)


def compute_grad_for_label(
    model: torch.nn.Module,
    ehr_np: np.ndarray,
    label_id: int,
    device: torch.device,
    code_vocab_size: int,
) -> torch.Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)

    ehr_tensor = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    probs = model(
        ehr_tensor,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        pos_loss_weight=None,
        logit_adjust=None,
    )
    if probs.dim() != 3 or label_id >= probs.size(-1):
        raise RuntimeError(f"Invalid probs shape or label_id {label_id} out of range.")
    label_probs = probs[:, :, label_id]
    scalar = label_probs.mean()
    scalar.backward()
    return flatten_grad_vector(model).detach()


def compute_lambda_min_for_context(
    model: torch.nn.Module,
    train_ds,
    context_id: int,
    disease_ids: List[int],
    ckpt_path: str,
    device: torch.device,
    cfg,
) -> float:
    """Compute λ_min(K) for context x, where K is Gram matrix over related_rare labels."""
    load_checkpoint_into_model(model, ckpt_path, device)
    ehr_np, _ = train_ds[context_id]

    grads: List[torch.Tensor] = []
    for did in disease_ids:
        if did >= cfg.code_vocab_size:
            continue
        g = compute_grad_for_label(model, ehr_np, did, device, cfg.code_vocab_size)
        grads.append(g.cpu().numpy())

    if len(grads) < 2:
        return float("nan")

    n = len(grads)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            K[i, j] = float(np.dot(grads[i], grads[j]))

    evals = np.linalg.eigvalsh(K)
    return float(np.min(evals))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute λ_min(K_x) vs related_rare AUC/gain for internalization bound."
    )
    ap.add_argument("--trajectory_csv", type=str, required=True, help="Trajectory CSV (base schedule).")
    ap.add_argument("--probe_config", type=str, required=True, help="Micro-probe config with type.")
    ap.add_argument("--ckpt_path_init", type=str, required=True, help="Init checkpoint for λ_min(K).")
    ap.add_argument(
        "--ckpt_path_final",
        type=str,
        default="",
        help="Optional final checkpoint for λ_min(K) sanity check.",
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="MIMIC-IV data dir.",
    )
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV path.")
    ap.add_argument("--out_fig", type=str, default="", help="Output scatter figure path.")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for gradient computation.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    traj_path = os.path.abspath(args.trajectory_csv)
    probe_path = os.path.abspath(args.probe_config)
    ckpt_init = os.path.abspath(args.ckpt_path_init)
    ckpt_final = os.path.abspath(args.ckpt_path_final) if args.ckpt_path_final else None

    out_csv = args.out_csv
    if not out_csv:
        out_csv = os.path.join(MODEL6_DIR, "lambda_min_vs_gain.csv")
    out_csv = os.path.abspath(out_csv)

    out_fig = args.out_fig or os.path.join(MODEL6_DIR, "lambda_min_vs_gain.png")
    out_fig = os.path.abspath(out_fig)

    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    if not os.path.exists(probe_path):
        raise FileNotFoundError(f"Probe config not found: {probe_path}")
    if not os.path.exists(ckpt_init):
        raise FileNotFoundError(f"Init checkpoint not found: {ckpt_init}")
    if ckpt_final and not os.path.exists(ckpt_final):
        raise FileNotFoundError(f"Final checkpoint not found: {ckpt_final}")

    related_pairs = load_related_rare_pairs(probe_path)
    ctx_to_diseases = context_to_related_rare_diseases(related_pairs)
    contexts_with_2plus = [c for c, ds in ctx_to_diseases.items() if len(ds) >= 2]
    if not contexts_with_2plus:
        raise ValueError("No context has >= 2 related_rare labels.")

    traj_df = load_trajectory(traj_path, related_pairs)
    metrics_df = aggregate_trajectory_metrics(traj_df, related_pairs)

    model, cfg, train_ds = build_model_and_dataset(args.data_dir, args.device)
    device_t = torch.device(args.device)

    rows: List[dict] = []

    for i, context_id in enumerate(sorted(contexts_with_2plus)):
        disease_ids = ctx_to_diseases[context_id]
        lm_init = compute_lambda_min_for_context(
            model, train_ds, context_id, disease_ids, ckpt_init, device_t, cfg
        )
        lm_final: Optional[float] = None
        if ckpt_final:
            lm_final = compute_lambda_min_for_context(
                model, train_ds, context_id, disease_ids, ckpt_final, device_t, cfg
            )

        m = metrics_df[metrics_df["context_id"] == context_id]
        auc_mean = m["related_rare_AUC_mean"].iloc[0] if not m.empty else float("nan")
        gain_mean = m["related_rare_final_gain_mean"].iloc[0] if not m.empty else float("nan")

        row = {
            "context_id": context_id,
            "lambda_min_K_init": lm_init,
            "related_rare_AUC_mean": auc_mean,
            "related_rare_final_gain_mean": gain_mean,
        }
        if lm_final is not None:
            row["lambda_min_K_final"] = lm_final
        rows.append(row)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[lambda_min] Processed {i + 1}/{len(contexts_with_2plus)} contexts")

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[lambda_min] Wrote {out_csv}")

    # Scatter and Spearman: λ_min_K_init vs AUC (main), vs gain
    valid = df.dropna(subset=["lambda_min_K_init", "related_rare_AUC_mean"])
    if len(valid) >= 3:
        from scipy import stats

        r_auc, p_auc = stats.spearmanr(valid["lambda_min_K_init"], valid["related_rare_AUC_mean"])
        r_gain, p_gain = stats.spearmanr(
            valid["lambda_min_K_init"], valid["related_rare_final_gain_mean"]
        )
        print(f"[lambda_min] Spearman(λ_min_K_init, AUC_mean):  r={r_auc:.4f}, p={p_auc:.4f}")
        print(f"[lambda_min] Spearman(λ_min_K_init, gain_mean): r={r_gain:.4f}, p={p_gain:.4f}")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(valid["lambda_min_K_init"], valid["related_rare_AUC_mean"], alpha=0.7)
        axes[0].set_xlabel("λ_min(K_init)")
        axes[0].set_ylabel("related_rare_AUC_mean")
        axes[0].set_title(f"r={r_auc:.3f}, p={p_auc:.3f}")

        axes[1].scatter(valid["lambda_min_K_init"], valid["related_rare_final_gain_mean"], alpha=0.7)
        axes[1].set_xlabel("λ_min(K_init)")
        axes[1].set_ylabel("related_rare_final_gain_mean")
        axes[1].set_title(f"r={r_gain:.3f}, p={p_gain:.3f}")

        plt.tight_layout()
        plt.savefig(out_fig, dpi=200)
        print(f"[lambda_min] Saved figure {out_fig}")
        plt.close()
    else:
        print("[lambda_min] Too few valid points for Spearman/plot.")


if __name__ == "__main__":
    main()

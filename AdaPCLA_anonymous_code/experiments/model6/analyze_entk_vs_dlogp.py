#!/usr/bin/env python3
"""
Analyze relationship between empirical NTK similarity and Δ log p
for micro-probe items.

For each (context_id, disease_id, type):
  - Δ log p: using micro_probe_ckpt_*.csv (last - first mean_log_prob)
  - eNTK similarity: cosine similarity between
        grad_theta f_{disease}(x)
    and grad_theta f_{context-bundle}(x),
    where the context-bundle gradient is the average gradient over
    all codes that actually appear in this patient's visits.

This gives a CSV suitable for plotting:
  X = entk_sim
  Y = delta_log_prob
  color = type (related_rare / unrelated_rare / wrong)
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL5_DIR = os.path.join(PROJECT_ROOT, "model5")
MODEL6_DIR = os.path.join(PROJECT_ROOT, "model6")

# model8 (MIMIC-IV HALO baseline) directory, same as in model5 script


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute eNTK similarity vs Δ log p for micro-probe items."
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=DATA_MIMICIV,
        help="MIMIC-IV data2 directory (codeToIndex.pkl, idToLabel.pkl, trainDataset.pkl).",
    )
    ap.add_argument(
        "--logs_dir",
        type=str,
        default=os.path.join(MODEL6_DIR, "micro_probe_logs", "seed1"),
        help="Directory containing micro_probe_ckpt_*.csv files.",
    )
    ap.add_argument(
        "--init_ckpt_path",
        type=str,
        default=(
            HALO_MIMICIV_CKPT
        ),
        help="HALO checkpoint path used as early-time θ (approximate t≈0) for NTK.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device for gradient computation (e.g. cuda:1 or cpu).",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default=os.path.join(MODEL6_DIR, "entk_vs_dlogp.csv"),
        help="Output CSV for per-item entk_sim and delta_log_prob.",
    )
    ap.add_argument(
        "--ckpt_start",
        type=int,
        default=0,
        help="Global ckpt index to use as start (for Δ log p).",
    )
    ap.add_argument(
        "--ckpt_end",
        type=int,
        default=99,
        help="Global ckpt index to use as end (for Δ log p).",
    )
    return ap.parse_args()


def load_delta_logp(
    logs_dir: str,
    ckpt_start: int,
    ckpt_end: int,
) -> Dict[Tuple[int, int, str], float]:
    """
    From micro_probe_ckpt_*.csv, compute Δ log p for each
    (context_id, disease_id, type) between ckpt_start and ckpt_end.
    """
    pattern = os.path.join(logs_dir, "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    # (ctx, did, type) -> {gidx -> mean_log_prob}
    values: Dict[Tuple[int, int, str], Dict[int, float]] = defaultdict(dict)

    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gidx = int(row["global_ckpt_idx"])
                except Exception:
                    continue
                if gidx < ckpt_start or gidx > ckpt_end:
                    continue
                try:
                    ctx = int(row["context_id"])
                    did = int(row["disease_id"])
                    tag = row.get("type", "")
                    mlp = float(row["mean_log_prob"])
                except Exception:
                    continue
                key = (ctx, did, tag)
                # keep the first and last within [ckpt_start, ckpt_end]
                values[key][gidx] = mlp

    delta: Dict[Tuple[int, int, str], float] = {}
    for key, series in values.items():
        if not series:
            continue
        t_min = min(series.keys())
        t_max = max(series.keys())
        delta[key] = series[t_max] - series[t_min]
    if not delta:
        raise RuntimeError("No Δ log p could be computed; check ckpt_start/ckpt_end.")
    return delta


def build_model_and_dataset(
    data_dir: str,
    init_ckpt_path: str,
    device: str,
):
    """
    Reuse Model2Config, HALOModel, and MIMICDataset from existing code.
    """
    import sys

    if MODEL8_DIR not in sys.path:
        sys.path.insert(0, MODEL8_DIR)
    if MODEL5_DIR not in sys.path:
        sys.path.insert(0, MODEL5_DIR)

    from config import Model2Config  # type: ignore
    from model import HALOModel  # type: ignore
    from run_pcla_fixed_bias_anneal_mimiciv import MIMICDataset  # type: ignore

    # Load data2 pickles
    import pickle

    def load_pkl(name: str):
        path = os.path.join(data_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")

    cfg = Model2Config()
    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    print(
        f"[data2] train size = {len(train_data)}, "
        f"code_vocab_size = {cfg.code_vocab_size}, labels = {cfg.label_vocab_size}"
    )

    device_t = torch.device(device)
    model = HALOModel(cfg).to(device_t)

    if init_ckpt_path and os.path.exists(init_ckpt_path):
        ckpt = torch.load(init_ckpt_path, map_location=device_t, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            if next(iter(state.keys()), "").startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)
        print(f"[eNTK] Loaded init checkpoint from: {init_ckpt_path}")
    else:
        print(f"[eNTK] WARNING: init checkpoint not found, using random init: {init_ckpt_path}")

    train_ds = MIMICDataset(train_data, cfg)
    return model, cfg, train_ds


def flatten_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    """Concatenate all parameter gradients into a single 1D tensor."""
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad.view(-1))
    if not grads:
        raise RuntimeError("No gradients found on model parameters.")
    return torch.cat(grads)


def compute_entk_for_item(
    model: torch.nn.Module,
    cfg,
    ehr_np: np.ndarray,
    disease_id: int,
    device: torch.device,
) -> Tuple[float, float]:
    """
    For a single (context, disease):
      - grad for disease_id
      - grad for context-bundle (average over all codes in visits)

    Returns:
      (||g_disease||, cosine(g_disease, g_bundle))
    """
    model.eval()

    ehr_tensor = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    # identify all codes that appear in this context (head + tail, but limited to code_vocab)
    context_codes: List[int] = []
    for j in range(ehr_np.shape[0]):
        visit = ehr_np[j]
        codes = np.nonzero(visit[: cfg.code_vocab_size])[0]
        for c in codes:
            if int(c) not in context_codes:
                context_codes.append(int(c))
    if not context_codes:
        return float("nan"), float("nan")

    # Gradient for target disease
    model.zero_grad(set_to_none=True)
    probs = model(
        ehr_tensor,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        pos_loss_weight=None,
        logit_adjust=None,
    )
    if probs.dim() != 3 or disease_id >= probs.size(-1):
        return float("nan"), float("nan")
    disease_probs = probs[:, :, disease_id]  # (1, n_ctx-1)
    target_scalar = disease_probs.mean()
    target_scalar.backward(retain_graph=True)
    g_disease = flatten_grad_vector(model).detach()

    # Gradient for context-bundle: average gradient over all context codes
    g_bundle_accum: torch.Tensor | None = None
    count = 0
    for c in context_codes:
        if c >= cfg.code_vocab_size:
            continue
        model.zero_grad(set_to_none=True)
        c_probs = probs[:, :, c]
        c_scalar = c_probs.mean()
        c_scalar.backward(retain_graph=True)
        g_c = flatten_grad_vector(model).detach()
        if g_bundle_accum is None:
            g_bundle_accum = g_c
        else:
            g_bundle_accum = g_bundle_accum + g_c
        count += 1
    if g_bundle_accum is None or count == 0:
        return float("nan"), float("nan")
    g_bundle = g_bundle_accum / float(count)

    # cosine similarity
    norm_d = torch.norm(g_disease)
    norm_b = torch.norm(g_bundle)
    if norm_d.item() == 0.0 or norm_b.item() == 0.0:
        cos_sim = float("nan")
    else:
        cos_sim = float(torch.dot(g_disease, g_bundle) / (norm_d * norm_b))
    return float(norm_d.item()), cos_sim


def main() -> None:
    args = parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    print(f"[entk] Loading Δ log p from logs_dir={logs_dir}")
    delta_logp = load_delta_logp(
        logs_dir=logs_dir,
        ckpt_start=args.ckpt_start,
        ckpt_end=args.ckpt_end,
    )
    print(f"[entk] Got Δ log p for {len(delta_logp)} (context,disease,type) items.")

    print(f"[entk] Building model and dataset from data_dir={args.data_dir}")
    model, cfg, train_ds = build_model_and_dataset(
        data_dir=os.path.abspath(args.data_dir),
        init_ckpt_path=args.init_ckpt_path,
        device=args.device,
    )
    device_t = torch.device(args.device)

    # Precompute gradients per (context_id, disease_id) to reuse between types if needed
    grad_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}

    out_rows: List[Dict[str, object]] = []
    for (ctx, did, tag), dlp in delta_logp.items():
        key_cd = (ctx, did)
        if key_cd not in grad_cache:
            ehr_np, _ = train_ds[ctx]
            norm_d, cos_sim = compute_entk_for_item(
                model=model,
                cfg=cfg,
                ehr_np=ehr_np,
                disease_id=did,
                device=device_t,
            )
            grad_cache[key_cd] = (norm_d, cos_sim)
        else:
            norm_d, cos_sim = grad_cache[key_cd]

        out_rows.append(
            {
                "context_id": ctx,
                "disease_id": did,
                "type": tag,
                "delta_log_prob": dlp,
                "grad_norm_disease": norm_d,
                "entk_sim_ctx_bundle": cos_sim,
            }
        )

    out_path = os.path.abspath(args.out_csv)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "context_id",
                "disease_id",
                "type",
                "delta_log_prob",
                "grad_norm_disease",
                "entk_sim_ctx_bundle",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[entk] Wrote {len(out_rows)} rows to {out_path}")
    print("You can now make scatter plots: X=entk_sim_ctx_bundle, Y=delta_log_prob, color=type.")


if __name__ == "__main__":
    main()


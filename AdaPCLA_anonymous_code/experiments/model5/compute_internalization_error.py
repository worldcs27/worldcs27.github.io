#!/usr/bin/env python3
"""
Compute internalization error focusing on training-induced logit changes dz = z_T - z_0,
instead of the large static bias b_stat.

Main metrics:
  - E_l2(x):        || z_T(x) - z_0(x) ||_2          (full code vocab, no b_stat)
  - E_tail_l2(x):   || (z_T(x) - z_0(x))_tail ||_2   (restricted to tail codes)
  - E_mae(x):       mean_i | z_T(x) - z_0(x) |_i     (full code vocab, L1 / MAE)

We still compute:
  - ||b_stat||_2 and ||z_T - z_0 - b_stat||_2 for sanity checks, but they are not
    the main internalization error used for scaling analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import List, Tuple

import numpy as np
import torch

# Import from training script
from run_pcla_fixed_bias_anneal_mimiciv import (  # type: ignore
    Model2Config,
    HALOModel,
    MIMICDataset,
    compute_logit_adjust,
    DEFAULT_DATA_DIR,
)


def _load_train_data(data_dir: str):
    import pickle

    with open(os.path.join(data_dir, "trainDataset.pkl"), "rb") as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_dir, "codeToIndex.pkl"), "rb") as f:
        code_to_index = pickle.load(f)
    return train_data, code_to_index


def _build_model_and_dataset(data_dir: str, device: torch.device):
    train_data, code_to_index = _load_train_data(data_dir)
    cfg = Model2Config()
    cfg.code_vocab_size = len(code_to_index)
    # label_vocab_size / total_vocab_size 等在 Model2Config 内部已有默认逻辑
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    model0 = HALOModel(cfg).to(device)
    modelT = HALOModel(cfg).to(device)

    train_ds = MIMICDataset(train_data, cfg)
    # compute b_stat (fixed bias) on training data
    adj, _stats = compute_logit_adjust(train_data, config=cfg)
    b_stat_code = adj[: cfg.code_vocab_size].astype("float32")
    return cfg, model0, modelT, train_ds, b_stat_code


def _load_state_dict(path: str, model: torch.nn.Module, device: torch.device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state
    # 去掉可能的 "module." 前缀
    if sd and next(iter(sd.keys())).startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()


def _forward_last_code_logits(
    model: HALOModel,
    cfg: Model2Config,
    ehr_np: np.ndarray,
    mask_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run model on a single EHR and return last valid visit's code logits as 1D np.array of shape (V,).
    """
    with torch.no_grad():
        ehr = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T, D)
        mask = torch.from_numpy(mask_np).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, T-1, 1) in MIMICDataset
        # training forward uses ehr_labels=batch_ehr, ehr_masks=batch_mask
        loss, logits, _ = model(
            ehr,
            position_ids=None,
            ehr_labels=ehr,
            ehr_masks=mask,
            pos_loss_weight=cfg.pos_loss_weight,
            logit_adjust=None,  # we want intrinsic logits here
        )
        # logits: (1, T-1, total_vocab_size) (follow training convention)
        logits = logits.detach().cpu().numpy()[0]  # (T-1, D)
    # mask_np shape: (T-1, 1); find last valid index
    valid_idx = np.where(mask_np[:, 0] > 0.0)[0]
    if len(valid_idx) == 0:
        # fallback: use last time step
        t_idx = logits.shape[0] - 1
    else:
        t_idx = int(valid_idx[-1])
    z_t = logits[t_idx]  # (D,)
    z_code = z_t[: int(cfg.code_vocab_size)]  # (V,)
    return z_code.astype("float32")


def _load_context_and_tail_info(micro_probe_config: str):
    """
    从 micro_probe_config 中同时读取:
      - 所有出现过的 context_id 列表
      - 每个 context_id 下 tail-code 索引列表（type 含 'rare' 的 disease_id）
    """
    ctx_ids = set()
    ctx_to_tail: dict[int, List[int]] = {}
    with open(micro_probe_config, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(row.get("context_id", -1))
                did = int(row.get("disease_id", -1))
            except Exception:
                continue
            if cid < 0:
                continue
            ctx_ids.add(cid)
            t = (row.get("type") or "").lower()
            if did >= 0 and ("rare" in t):
                ctx_to_tail.setdefault(cid, []).append(did)
    ctx_list = sorted(ctx_ids)
    if not ctx_list:
        raise ValueError(f"No valid context_id found in {micro_probe_config}")
    return ctx_list, ctx_to_tail


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute internalization error E(x) for given checkpoints.")
    ap.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="MIMIC-IV data2 dir (trainDataset.pkl, codeToIndex.pkl).",
    )
    ap.add_argument(
        "--ckpt_init",
        type=str,
        required=True,
        help="Initial checkpoint path (e.g., micro_probe_ckpt_0000.pt).",
    )
    ap.add_argument(
        "--ckpt_final",
        type=str,
        required=True,
        help="Final checkpoint path (e.g., micro_probe_ckpt_0099.pt or model_anneal_mimiciv.pt).",
    )
    ap.add_argument(
        "--micro_probe_config",
        type=str,
        required=True,
        help="CSV with columns context_id,disease_id,type (only context_id is used here).",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for E(x) values.",
    )
    ap.add_argument(
        "--schedule",
        type=str,
        default="base",
        help="Label for schedule / anneal_mode used (for logging).",
    )
    ap.add_argument(
        "--config_variant",
        type=str,
        default="base",
        help="Label for config variant (e.g., kernel_strong / kernel_weak).",
    )
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate label (for logging only).")
    ap.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay label (for logging only).")
    ap.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference (e.g., cuda:0, cuda:1, cpu).",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg, model0, modelT, train_ds, b_stat_code = _build_model_and_dataset(args.data_dir, device=device)

    # load checkpoints
    _load_state_dict(args.ckpt_init, model0, device=device)
    _load_state_dict(args.ckpt_final, modelT, device=device)

    context_ids, ctx_to_tail = _load_context_and_tail_info(args.micro_probe_config)
    print(f"[E(x)] computing for {len(context_ids)} contexts from {args.micro_probe_config}")

    rows: List[dict] = []
    b_vec = torch.from_numpy(b_stat_code).to(device=device, dtype=torch.float32)
    b_norm = torch.norm(b_vec, p=2).item()

    for ctx_id in context_ids:
        ehr_np, mask_np = train_ds[ctx_id]  # MIMICDataset __getitem__
        z0 = _forward_last_code_logits(model0, cfg, ehr_np, mask_np, device=device)
        zT = _forward_last_code_logits(modelT, cfg, ehr_np, mask_np, device=device)
        z0_t = torch.from_numpy(z0).to(device=device, dtype=torch.float32)
        zT_t = torch.from_numpy(zT).to(device=device, dtype=torch.float32)
        dz = zT_t - z0_t                      # (V,)
        diff = dz - b_vec                    # (V,)  # diagnostics only

        # Full-code metrics: use dz only (no b_stat)
        E_l2 = torch.norm(dz, p=2).item()
        E_mae = torch.mean(torch.abs(dz)).item()
        dz_l2 = E_l2

        # Tail-code metrics: 仅在 tail-code 子空间上计算 dz 的 L2
        tail_idxs = [i for i in ctx_to_tail.get(ctx_id, []) if 0 <= i < int(cfg.code_vocab_size)]
        if tail_idxs:
            dz_tail = dz[tail_idxs]
            E_tail_l2 = torch.norm(dz_tail, p=2).item()
        else:
            E_tail_l2 = float("nan")

        rows.append(
            {
                "schedule": args.schedule,
                "config_variant": args.config_variant,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "context_id": ctx_id,
                "E_l2": E_l2,
                "E_tail_l2": E_tail_l2,
                "E_mae": E_mae,
                "norm_dz_l2": dz_l2,
                "norm_b_l2": b_norm,
                "norm_dz_minus_b_l2": torch.norm(diff, p=2).item(),
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "schedule",
                "config_variant",
                "lr",
                "weight_decay",
                "context_id",
                "E_l2",
                "E_tail_l2",
                "E_mae",
                "norm_dz_l2",
                "norm_b_l2",
                "norm_dz_minus_b_l2",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[E(x)] wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()

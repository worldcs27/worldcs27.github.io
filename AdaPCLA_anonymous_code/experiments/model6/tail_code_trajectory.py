#!/usr/bin/env python3
"""
对任意 (context_id, disease_id) 对，在 100 个 micro-probe checkpoint 上跑前向，
输出每个 checkpoint 的「最后一步」概率，并可选画成轨迹图。

输入：CSV，列至少包含 context_id, disease_id（可含 type 等额外列，会被忽略）。
输出：
  - 表：checkpoint_idx, context_id, disease_id, mean_prob, mean_log_prob
  - 可选图：每条 (context_id, disease_id) 一条曲线，x=checkpoint_idx, y=mean_log_prob
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
from typing import List, Tuple

import numpy as np
import torch

PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL5_DIR = os.path.join(PROJECT_ROOT, "model5")
DEFAULT_DATA_DIR = DATA_MIMICIV
EPOCH_CKPT_DIR = os.path.join(PROJECT_ROOT, "model6", "save_micro_probe_mimiciv", "seed1", "epoch_ckpts")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tail code probability trajectory over 100 checkpoints.")
    ap.add_argument("--input_csv", type=str, required=True, help="CSV with columns context_id, disease_id.")
    ap.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="MIMIC-IV data2 dir.")
    ap.add_argument("--ckpt_dir", type=str, default=EPOCH_CKPT_DIR, help="Dir containing micro_probe_ckpt_*.pt")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--out_csv", type=str, default="tail_code_trajectory.csv", help="Output table path.")
    ap.add_argument("--out_fig", type=str, default="", help="If set, save trajectory figure to this path.")
    return ap.parse_args()


def resolve_device(s: str) -> str:
    if s:
        return s
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    return "cpu"


def load_pairs(csv_path: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ctx = int(row["context_id"])
                did = int(row["disease_id"])
            except (KeyError, ValueError):
                continue
            pairs.append((ctx, did))
    return pairs


def get_ckpt_paths(ckpt_dir: str) -> List[str]:
    pattern = os.path.join(ckpt_dir, "micro_probe_ckpt_*.pt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints: {pattern}")
    return paths


def build_model_and_dataset(data_dir: str, device: str):
    import sys
    import pickle

    if MODEL8_DIR not in sys.path:
        sys.path.insert(0, MODEL8_DIR)
    if MODEL5_DIR not in sys.path:
        sys.path.insert(0, MODEL5_DIR)

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
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    device_t = torch.device(device)
    model = HALOModel(cfg).to(device_t)
    train_ds = MIMICDataset(train_data, cfg)
    return model, cfg, train_ds


def load_ckpt(model: torch.nn.Module, path: str, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    if next(iter(state.keys()), "").startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)


def last_valid_step_index(ehr_np: np.ndarray, cfg) -> int:
    n_steps = ehr_np.shape[0] - 1  # model outputs 1..n_ctx-1
    for t_idx in range(n_steps - 1, -1, -1):
        row = ehr_np[t_idx + 1]
        if row[: int(cfg.code_vocab_size)].sum() > 0 or row[cfg.end_record_token] > 0:
            return t_idx
    return n_steps - 1


def run_one_checkpoint(
    model: torch.nn.Module,
    cfg,
    train_ds,
    pairs: List[Tuple[int, int]],
    device: torch.device,
) -> List[Tuple[int, int, float, float]]:
    """Returns list of (context_id, disease_id, mean_prob, mean_log_prob)."""
    model.eval()
    results = []
    with torch.no_grad():
        for ctx_id, disease_id in pairs:
            if disease_id < 0 or disease_id >= cfg.code_vocab_size:
                results.append((ctx_id, disease_id, float("nan"), float("nan")))
                continue
            ehr_np, _ = train_ds[ctx_id]
            ehr_t = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)
            probs = model(
                ehr_t,
                position_ids=None,
                ehr_labels=None,
                ehr_masks=None,
                pos_loss_weight=None,
                logit_adjust=None,
            )
            probs = probs.squeeze(0)  # (n_ctx-1, V)
            if probs.dim() != 2 or disease_id >= probs.size(-1):
                results.append((ctx_id, disease_id, float("nan"), float("nan")))
                continue
            last_idx = last_valid_step_index(ehr_np, cfg)
            p = float(probs[last_idx, disease_id].item())
            log_p = float(torch.log(torch.tensor(max(1e-8, min(1 - 1e-8, p)), device=device)).item())
            results.append((ctx_id, disease_id, p, log_p))
    return results


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    device_t = torch.device(device)

    pairs = load_pairs(args.input_csv)
    if not pairs:
        raise SystemExit("No (context_id, disease_id) pairs found in input CSV.")

    ckpt_paths = get_ckpt_paths(args.ckpt_dir)
    print(f"[tail-code] Loaded {len(pairs)} pairs from {args.input_csv}, {len(ckpt_paths)} checkpoints.")

    model, cfg, train_ds = build_model_and_dataset(os.path.abspath(args.data_dir), device)

    rows: List[dict] = []
    for ckpt_idx, path in enumerate(ckpt_paths):
        load_ckpt(model, path, device_t)
        one_results = run_one_checkpoint(model, cfg, train_ds, pairs, device_t)
        for (ctx_id, disease_id, mean_prob, mean_log_prob) in one_results:
            rows.append({
                "checkpoint_idx": ckpt_idx,
                "context_id": ctx_id,
                "disease_id": disease_id,
                "mean_prob": mean_prob,
                "mean_log_prob": mean_log_prob,
            })
        if (ckpt_idx + 1) % 20 == 0 or ckpt_idx == 0:
            print(f"  checkpoint {ckpt_idx + 1}/{len(ckpt_paths)} done.")

    out_csv = os.path.abspath(args.out_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["checkpoint_idx", "context_id", "disease_id", "mean_prob", "mean_log_prob"])
        w.writeheader()
        w.writerows(rows)
    print(f"[tail-code] Wrote {len(rows)} rows -> {out_csv}")

    if args.out_fig:
        import matplotlib.pyplot as plt
        # One curve per (context_id, disease_id)
        pairs_uniq = list(dict.fromkeys(pairs))
        plt.figure(figsize=(7, 4))
        for i, (ctx_id, disease_id) in enumerate(pairs_uniq):
            sub = [r for r in rows if r["context_id"] == ctx_id and r["disease_id"] == disease_id]
            sub = sorted(sub, key=lambda x: x["checkpoint_idx"])
            xs = [r["checkpoint_idx"] for r in sub]
            ys = [r["mean_log_prob"] for r in sub]
            if xs and ys:
                plt.plot(xs, ys, label=f"ctx={ctx_id} code={disease_id}", alpha=0.8)
        plt.xlabel("checkpoint_idx (0..99)")
        plt.ylabel("mean_log_prob (last-step)")
        plt.title("Tail code probability trajectory over checkpoints")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.abspath(args.out_fig), dpi=200)
        print(f"[tail-code] Saved figure -> {args.out_fig}")


if __name__ == "__main__":
    main()

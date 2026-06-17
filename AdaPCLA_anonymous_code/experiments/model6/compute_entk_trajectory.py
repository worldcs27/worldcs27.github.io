#!/usr/bin/env python3
"""
Compute and plot empirical NTK K_t(y1, y2; x) across checkpoints.

For a fixed context x (context_id) and two label indices y1, y2 (disease_id),
this script:
  - loads a sequence of model checkpoints θ_t
  - for each θ_t, computes gradients:
        g1_t = ∇_θ f_{y1}(x; θ_t)
        g2_t = ∇_θ f_{y2}(x; θ_t)
  - computes:
        K_t      = g1_t^T g2_t        (inner product)
        cos_sim  = cos(g1_t, g2_t)    (normalized NTK similarity)
  - saves a CSV with K_t and cos_sim over t
  - optionally saves a PNG curve for cos_sim vs. t
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import os
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL5_DIR = os.path.join(PROJECT_ROOT, "model5")

# model8 (MIMIC-IV HALO baseline) directory, same as in model5 script

DEFAULT_DATA_DIR = DATA_MIMICIV
DEFAULT_INIT_CKPT = (
    HALO_MIMICIV_CKPT
)
DEFAULT_FINAL_CKPT = os.path.join(
    PROJECT_ROOT,
    "model6",
    "save_micro_probe_mimiciv",
    "seed1",
    "model_anneal_mimiciv.pt",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute empirical NTK trajectory K_t(y1,y2;x) across checkpoints."
    )
    ap.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="MIMIC-IV data2 directory (codeToIndex.pkl, idToLabel.pkl, trainDataset.pkl).",
    )
    ap.add_argument(
        "--context_id",
        type=int,
        required=True,
        help="Context id (index into trainDataset / MIMICDataset).",
    )
    ap.add_argument(
        "--disease1_id",
        type=int,
        required=True,
        help="First label index y1 (typically rare complication A).",
    )
    ap.add_argument(
        "--disease2_id",
        type=int,
        required=True,
        help="Second label index y2 (e.g., base disease B or another code).",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        action="append",
        default=None,
        help=(
            "Checkpoint path to include in trajectory. "
            "If omitted, uses [DEFAULT_INIT_CKPT, DEFAULT_FINAL_CKPT]. "
            "Can be specified multiple times to add more time points."
        ),
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
        default="entk_trajectory.csv",
        help="Output CSV filename for K_t and cos_sim vs t.",
    )
    ap.add_argument(
        "--out_fig",
        type=str,
        default="fig_entk_trajectory.png",
        help="Output PNG filename for cos_sim vs t.",
    )
    return ap.parse_args()


def ensure_ckpts(arg_ckpts: Iterable[str] | None) -> List[str]:
    """
    Resolve which checkpoints to use for the trajectory.

    Priority:
      1) If --ckpt is provided, use those paths (in given order).
      2) Otherwise, if 100 micro-probe ckpts exist under
         model6/save_micro_probe_mimiciv/seed1/epoch_ckpts/micro_probe_ckpt_*.pt,
         use them (sorted) to obtain a 100-point trajectory.
      3) Fallback to [DEFAULT_INIT_CKPT, DEFAULT_FINAL_CKPT].
    """
    if arg_ckpts:
        ckpts = [os.path.abspath(p) for p in arg_ckpts]
    else:
        # Try to auto-detect 100 micro-probe checkpoints
        epoch_ckpt_dir = os.path.join(
            PROJECT_ROOT,
            "model6",
            "save_micro_probe_mimiciv",
            "seed1",
            "epoch_ckpts",
        )
        pattern = os.path.join(epoch_ckpt_dir, "micro_probe_ckpt_*.pt")
        import glob as _glob

        micro_ckpts = sorted(_glob.glob(pattern))
        if micro_ckpts:
            ckpts = micro_ckpts
        else:
            ckpts = [DEFAULT_INIT_CKPT, DEFAULT_FINAL_CKPT]

    for p in ckpts:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return ckpts


def build_model_and_dataset(data_dir: str, device: str):
    """
    Reuse Model2Config, HALOModel, and MIMICDataset from existing code.
    """
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

    train_ds = MIMICDataset(train_data, cfg)
    return model, cfg, train_ds


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        if next(iter(state.keys()), "").startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)


def flatten_grad_vector(model: torch.nn.Module) -> torch.Tensor:
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            continue
        grads.append(p.grad.view(-1))
    if not grads:
        raise RuntimeError("No gradients found on model parameters.")
    return torch.cat(grads)


def compute_grad_for_label(
    model: torch.nn.Module,
    ehr_np: np.ndarray,
    label_id: int,
    device: torch.device,
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
        raise RuntimeError(
            f"Invalid probs shape {tuple(probs.shape)} or label_id {label_id} out of range."
        )
    label_probs = probs[:, :, label_id]  # (1, n_ctx-1)
    scalar = label_probs.mean()
    scalar.backward()
    g = flatten_grad_vector(model).detach()
    return g


def main() -> None:
    args = parse_args()

    ckpts = ensure_ckpts(args.ckpt)
    print("[entk-traj] Using checkpoints (t increasing in this order):")
    for i, p in enumerate(ckpts):
        print(f"  t={i}: {p}")

    model, cfg, train_ds = build_model_and_dataset(
        data_dir=os.path.abspath(args.data_dir),
        device=args.device,
    )
    device_t = torch.device(args.device)

    ehr_np, _ = train_ds[args.context_id]
    if args.disease1_id >= cfg.code_vocab_size or args.disease2_id >= cfg.code_vocab_size:
        raise ValueError(
            f"disease ids must be < code_vocab_size={cfg.code_vocab_size}, "
            f"got {args.disease1_id}, {args.disease2_id}"
        )

    rows: List[dict] = []
    K_vals: List[float] = []
    cos_vals: List[float] = []

    for t_idx, ckpt_path in enumerate(ckpts):
        print(f"[entk-traj] t={t_idx}: loading {ckpt_path}")
        load_checkpoint_into_model(model, ckpt_path, device_t)

        # compute gradients for y1 and y2
        g1 = compute_grad_for_label(model, ehr_np, args.disease1_id, device_t)
        g2 = compute_grad_for_label(model, ehr_np, args.disease2_id, device_t)

        inner = float(torch.dot(g1, g2).item())
        n1 = float(torch.norm(g1).item())
        n2 = float(torch.norm(g2).item())
        if n1 == 0.0 or n2 == 0.0:
            cos = float("nan")
        else:
            cos = float(inner / (n1 * n2))

        K_vals.append(inner)
        cos_vals.append(cos)

        rows.append(
            {
                "t_index": t_idx,
                "ckpt_path": ckpt_path,
                "context_id": args.context_id,
                "disease1_id": args.disease1_id,
                "disease2_id": args.disease2_id,
                "K_inner": inner,
                "grad_norm_y1": n1,
                "grad_norm_y2": n2,
                "cos_sim": cos,
            }
        )

    # Save CSV
    import csv

    out_csv_path = os.path.abspath(args.out_csv)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_index",
                "ckpt_path",
                "context_id",
                "disease1_id",
                "disease2_id",
                "K_inner",
                "grad_norm_y1",
                "grad_norm_y2",
                "cos_sim",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[entk-traj] Saved trajectory CSV to {out_csv_path}")

    # Save simple figure: cos_sim vs t_index
    t_axis = list(range(len(cos_vals)))
    plt.figure(figsize=(5, 4))
    plt.plot(t_axis, cos_vals, marker="o")
    plt.xlabel("Checkpoint index t")
    plt.ylabel("cos_sim(K_t(y1,y2;x))")
    plt.title(
        f"eNTK trajectory (ctx={args.context_id}, y1={args.disease1_id}, y2={args.disease2_id})"
    )
    plt.tight_layout()
    out_fig_path = os.path.abspath(args.out_fig)
    plt.savefig(out_fig_path, dpi=200)
    print(f"[entk-traj] Saved trajectory figure to {out_fig_path}")


if __name__ == "__main__":
    main()


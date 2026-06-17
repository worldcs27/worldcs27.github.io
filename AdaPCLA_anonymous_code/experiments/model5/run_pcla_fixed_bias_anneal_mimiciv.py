#!/usr/bin/env python3
"""
MIMIC-IV data2 + 固定 bias + 退火 (Fixed Bias + Annealing)。

- 数据：fame/myfame/data2（MIMIC-IV 预处理）
- 初始化：model8 在 MIMIC-IV 上训好的 HALO checkpoint（非 MIMIC-III）
- 训练：out_logits = logits + α(epoch) * logit_adjust，α 从 1.0 分段线性降到 0。
- 采样/生成：不加 bias（α=0），即 logit_adjust=None。

所有结果与脚本放在 mywork/model5。
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import json
import os
import pickle
import random
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# MIMIC-IV data2 路径
DEFAULT_DATA_DIR = DATA_MIMICIV
DEFAULT_SAVE_DIR = os.path.join(THIS_DIR, "save_anneal_mimiciv")
DEFAULT_SEED = 1
OUTPUT_ROOT = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 使用 model8（MIMIC-IV HALO/PCLA 基线），导入 config 与 model

# MIMIC-IV 上训好的 HALO checkpoint（model8 best）
DEFAULT_INIT_CKPT = (
    HALO_MIMICIV_CKPT
)

import sys
if MODEL8_DIR not in sys.path:
    sys.path.insert(0, MODEL8_DIR)
from config import Model2Config  # type: ignore
from model import HALOModel  # type: ignore


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_logit_adjust(train_data, *, config: Model2Config) -> Tuple[np.ndarray, dict]:
    """与 model8/train.py 中 _compute_logit_adjust 一致，得到固定 bias 向量。"""
    code_vocab = int(config.code_vocab_size)
    total_visits = 0
    visit_counts = np.zeros((code_vocab,), dtype=np.int64)
    for p in train_data:
        visits = p.get("visits", [])
        total_visits += int(len(visits))
        for v in visits:
            if not v:
                continue
            for c in set(v):
                ci = int(c)
                if 0 <= ci < code_vocab:
                    visit_counts[ci] += 1

    eps = float(config.logit_adjust_eps)
    if total_visits <= 0:
        return np.zeros((config.total_vocab_size,), dtype=np.float32), {"total_visits": 0}

    pi = visit_counts.astype(np.float64) / float(total_visits)
    b = np.log((1.0 - pi + eps) / (pi + eps)) * float(config.logit_adjust_tau)
    if config.logit_adjust_clip is not None:
        b = np.clip(b, -float(config.logit_adjust_clip), float(config.logit_adjust_clip))
    b = np.where(visit_counts > 0, b, 0.0)

    adj = np.zeros((config.total_vocab_size,), dtype=np.float32)
    adj[:code_vocab] = b.astype(np.float32)
    stats = {
        "total_visits": int(total_visits),
        "codes_with_pos": int(np.sum(visit_counts > 0)),
        "tau": float(config.logit_adjust_tau),
        "clip": float(config.logit_adjust_clip) if config.logit_adjust_clip is not None else None,
        "adj_min": float(adj[:code_vocab].min()) if code_vocab else 0.0,
        "adj_max": float(adj[:code_vocab].max()) if code_vocab else 0.0,
    }
    return adj, stats


def alpha_anneal(epoch: int, total_epochs: int, mode: str = "base") -> float:
    """
    退火系数 α(epoch)：支持多种分段线性 schedule。
    - base: 原始 30% warm / 40% linear / 30% cold
    - fast: 10% warm / 40% linear / 50% cold（更快退火）
    - slow: 50% warm / 40% linear / 10% cold（更慢退火）
    """
    if total_epochs <= 0:
        return 1.0
    t = epoch / max(total_epochs - 1, 1)

    if mode == "fast":
        # 0.0-0.1: α=1; 0.1-0.5: linear 1->0; 0.5-1.0: α=0
        if t <= 0.1:
            return 1.0
        if t <= 0.5:
            return 1.0 - (t - 0.1) / 0.4
        return 0.0
    if mode == "slow":
        # 0.0-0.5: α=1; 0.5-0.9: linear 1->0; 0.9-1.0: α=0
        if t <= 0.5:
            return 1.0
        if t <= 0.9:
            return 1.0 - (t - 0.5) / 0.4
        return 0.0

    if mode == "cosine":
        # CosineAnnealing-style from 1 -> 0 over the whole training
        # α(t) = 0.5 * (1 + cos(pi * t)), t in [0,1]
        return 0.5 * (1.0 + math.cos(math.pi * t))

    # base: 原始 30% warm / 40% linear / 30% cold
    if t <= 0.3:
        return 1.0
    if t <= 0.7:
        return 1.0 - (t - 0.3) / 0.4
    return 0.0


class MIMICDataset(Dataset):
    def __init__(self, data, config: Model2Config):
        self.data = data
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        p = self.data[idx]
        visits = p["visits"]
        sample_ehr = np.zeros((self.config.n_ctx, self.config.total_vocab_size), dtype=np.float32)
        sample_mask = np.zeros((self.config.n_ctx, 1), dtype=np.float32)
        for j, v in enumerate(visits):
            if j + 2 < self.config.n_ctx:
                sample_ehr[j + 2][v] = 1
                sample_mask[j + 2] = 1
        sample_ehr[
            1, self.config.code_vocab_size : self.config.code_vocab_size + self.config.label_vocab_size
        ] = np.array(p["labels"])
        if len(visits) + 1 < self.config.n_ctx:
            sample_ehr[len(visits) + 1, self.config.end_record_token] = 1
        if len(visits) + 2 < self.config.n_ctx:
            sample_ehr[len(visits) + 2 :, self.config.pad_visit_token] = 1
        sample_mask[1] = 1
        sample_ehr[0, self.config.start_record_token] = 1
        sample_mask = sample_mask[1:, :]
        return sample_ehr, sample_mask


def sample_sequence(
    model: HALOModel,
    length: int,
    start_token: np.ndarray,
    batch_size: int,
    config: Model2Config,
    device: torch.device,
    logit_adjust=None,
) -> np.ndarray:
    """采样时传入 logit_adjust=None 表示生成阶段不加 bias。"""
    empty = torch.zeros((1, 1, config.total_vocab_size), device=device, dtype=torch.float32).repeat(
        batch_size, 1, 1
    )
    context = torch.tensor(start_token, device=device, dtype=torch.float32).unsqueeze(0).repeat(
        batch_size, 1
    )
    prev = context.unsqueeze(1)
    model.eval()
    with torch.no_grad():
        for _ in range(length - 1):
            prev = model.sample(torch.cat((prev, empty), dim=1), random=True, logit_adjust=logit_adjust)
            end_mask = prev[:, :, config.end_record_token].sum(dim=1) > 0
            if bool(end_mask.all()):
                break
    return prev.cpu().detach().numpy()


def convert_ehr(ehrs: np.ndarray, config: Model2Config):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        labels_output = ehr[1][
            config.code_vocab_size : config.code_vocab_size + config.label_vocab_size
        ]
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_output = []
            indices = np.nonzero(visit)[0]
            end = False
            for idx in indices:
                if idx < config.code_vocab_size:
                    visit_output.append(int(idx))
                elif idx == config.end_record_token:
                    end = True
            if visit_output:
                ehr_output.append(visit_output)
            if end:
                break
        ehr_outputs.append({"visits": ehr_output, "labels": labels_output})
    return ehr_outputs


@dataclass
class Args:
    data_dir: str
    save_dir: str
    seed: int
    epochs: int
    batch_size: int
    sample_batch_size: int
    num_workers: int
    lr: float
    pos_loss_weight: float | None
    total_samples: int
    device: str
    do_eval: bool
    init_ckpt_path: str | None
    # micro-probing options (for model6 experiments)
    log_micro_probe: bool
    micro_probe_config: str | None
    micro_probe_ckpt_per_epoch: int
    micro_probe_out_dir: str | None
    # extra checkpoint saving (for eNTK trajectory)
    save_epoch_ckpts: int
    save_micro_probe_ckpts: bool
    # annealing schedule mode
    anneal_mode: str
    # config variant for kernel strength (kernel_strong / kernel_weak)
    config_variant: str


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _parse_mean_metrics(csv_path: str, *, source: str = "MyModel2") -> Tuple[float, float]:
    accs: List[float] = []
    f1s: List[float] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") != source:
                continue
            a, f1 = row.get("Accuracy"), row.get("F1 Score")
            if a not in (None, ""):
                accs.append(float(a))
            if f1 not in (None, ""):
                f1s.append(float(f1))
    if not accs or not f1s:
        raise ValueError(f"No rows for source={source} in {csv_path}")
    return _mean(accs), _mean(f1s)


def _load_micro_probe_items(
    *,
    config: Model2Config,
    train_ds: MIMICDataset,
    config_path: str,
) -> List[dict]:
    """
    Load micro-probing items from a CSV-like config.
    Expected columns: context_id,disease_id,type
    - context_id: index into train_dataset (0-based)
    - disease_id: code index in [0, code_vocab_size)
    - type: string tag, e.g. related_rare / unrelated_rare / wrong
    """
    if not os.path.exists(config_path):
        print(f"[micro-probe] Config path not found: {config_path}")
        return []

    items: List[dict] = []
    with open(config_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ctx_id = int(row.get("context_id", -1))
                disease_id = int(row.get("disease_id", -1))
                tag = row.get("type", "")
            except Exception:
                continue
            if ctx_id < 0 or ctx_id >= len(train_ds):
                continue
            if disease_id < 0 or disease_id >= int(config.code_vocab_size):
                continue
            # Reuse dataset preprocessing to obtain EHR tensor
            ehr_np, _ = train_ds[ctx_id]
            items.append(
                {
                    "context_id": ctx_id,
                    "disease_id": disease_id,
                    "type": tag,
                    "ehr": ehr_np.astype(np.float32),
                }
            )
    return items


def _log_micro_probe(
    *,
    model: HALOModel,
    config: Model2Config,
    device: torch.device,
    items: List[dict],
    epoch: int,
    step_in_epoch: int,
    alpha: float,
    global_ckpt_idx: int,
    out_dir: str,
    logit_adjust: torch.Tensor | None = None,
) -> None:
    """
    For each (context, disease) micro-probe item, compute the current
    probability at a specific training snapshot and append to CSV.

    - By default, uses logit_adjust=None (unbiased model).
    - If logit_adjust is provided, passes it into the model (e.g. for Oracle baseline).
    - For each sequence, we use the **last valid time-step** (closest to end-of-record),
      instead of averaging over all time-steps, to better reflect the probability
      after seeing the full context.
    """
    if not items:
        return

    was_training = model.training
    model.eval()

    # Special case: global_ckpt_idx < 0 (e.g. Oracle baseline) uses a fixed filename.
    if global_ckpt_idx < 0:
        out_path = os.path.join(out_dir, "micro_probe_oracle.csv")
    else:
        out_path = os.path.join(out_dir, f"micro_probe_ckpt_{global_ckpt_idx:04d}.csv")
    write_header = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "epoch",
                    "step_in_epoch",
                    "alpha",
                    "global_ckpt_idx",
                    "context_id",
                    "disease_id",
                    "type",
                    "mean_prob",
                    "mean_log_prob",
                ]
            )

        with torch.no_grad():
            for it in items:
                ehr_np = it["ehr"]  # (n_ctx, total_vocab_size)
                ctx_id = it["context_id"]
                disease_id = int(it["disease_id"])
                tag = it.get("type", "")

                ehr_tensor = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)
                # Forward: allow optional logit_adjust (e.g. Oracle with fixed_adj)
                probs = model(
                    ehr_tensor,
                    position_ids=None,
                    ehr_labels=None,
                    ehr_masks=None,
                    logit_adjust=logit_adjust,
                )
                # probs shape: (1, n_ctx-1, total_vocab_size)
                probs = probs.squeeze(0)  # (n_ctx-1, V)
                if probs.dim() != 2 or disease_id >= probs.size(-1):
                    continue

                # 选择最后一个“有效”时间步（非 padding），更贴合“看完完整上下文”的预测
                # 有效步：该 row 在 code_vocab 区间或 end_record_token 上有非零激活。
                last_valid_idx = None
                n_steps = probs.size(0)
                for t_idx in range(n_steps - 1, -1, -1):
                    row = ehr_np[t_idx + 1]  # shift by 1: model outputs for positions 1..n_ctx-1
                    codes_slice = row[: int(config.code_vocab_size)]
                    if codes_slice.sum() > 0 or row[config.end_record_token] > 0:
                        last_valid_idx = t_idx
                        break
                if last_valid_idx is None:
                    # fallback: use the final time-step in probs
                    last_valid_idx = n_steps - 1

                disease_probs = probs[last_valid_idx, disease_id].unsqueeze(0)  # shape (1,)
                mean_prob = float(disease_probs.item())
                # 避免 log(0)
                clipped = disease_probs.clamp(min=1e-8, max=1 - 1e-8)
                mean_log_prob = float(torch.log(clipped).item())

                writer.writerow(
                    [
                        epoch,
                        step_in_epoch,
                        f"{alpha:.6f}",
                        global_ckpt_idx,
                        ctx_id,
                        disease_id,
                        tag,
                        f"{mean_prob:.8f}",
                        f"{mean_log_prob:.8f}",
                    ]
                )

    if was_training:
        model.train()



def run_single_seed(args: Args) -> None:
    device = torch.device(args.device)
    print(f"Using device: {device}")
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "datasets"), exist_ok=True)

    def load_pkl(name: str):
        path = os.path.join(args.data_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")
    val_data = load_pkl("valDataset.pkl")

    cfg = Model2Config()
    cfg.lr = float(args.lr)
    cfg.epoch = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.sample_batch_size = int(args.sample_batch_size)
    cfg.pos_loss_weight = args.pos_loss_weight
    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    print(f"[MIMIC-IV data2] Vocab size (codes) = {cfg.code_vocab_size}, labels = {cfg.label_vocab_size}")

    # 固定 bias（只算一次，训练中按 α(epoch) 缩放）
    adj_np, stats = compute_logit_adjust(train_data, config=cfg)
    print(f"Logit adjust stats: {stats}")
    fixed_adj = torch.from_numpy(adj_np).to(device=device, dtype=torch.float32)

    model = HALOModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if args.init_ckpt_path and os.path.exists(args.init_ckpt_path):
        ckpt = torch.load(args.init_ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            if next(iter(state.keys()), "").startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            print(f"Loaded init from (MIMIC-IV HALO): {args.init_ckpt_path}")
        else:
            model.load_state_dict(ckpt, strict=True)
            print(f"Loaded init from (MIMIC-IV HALO): {args.init_ckpt_path}")
    else:
        print(f"Warning: init_ckpt_path not found, training from scratch: {args.init_ckpt_path}")

    # Apply kernel strength variant: scale loaded weights to change effective λ_min(K)
    if args.config_variant == "kernel_strong":
        scale = 1.2
        for p in model.parameters():
            p.data.mul_(scale)
        print(f"[config_variant] kernel_strong: scaled all params by {scale}")
    elif args.config_variant == "kernel_weak":
        scale = 0.8
        for p in model.parameters():
            p.data.mul_(scale)
        print(f"[config_variant] kernel_weak: scaled all params by {scale}")

    train_ds = MIMICDataset(train_data, cfg)
    val_ds = MIMICDataset(val_data, cfg)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Micro-probing setup (optional)
    micro_probe_items: List[dict] | None = None
    if args.log_micro_probe and args.micro_probe_config:
        micro_probe_items = _load_micro_probe_items(
            config=cfg,
            train_ds=train_ds,
            config_path=args.micro_probe_config,
        )
        if micro_probe_items:
            print(f"[micro-probe] Loaded {len(micro_probe_items)} items from {args.micro_probe_config}")
        else:
            print(f"[micro-probe] WARNING: no valid items loaded from {args.micro_probe_config}")

    micro_probe_out_dir = None
    if args.log_micro_probe and micro_probe_items:
        micro_probe_out_dir = args.micro_probe_out_dir or os.path.join(
            os.path.dirname(THIS_DIR), "model6", "micro_probe_logs", f"seed{args.seed}"
        )
        os.makedirs(micro_probe_out_dir, exist_ok=True)
        print(f"[micro-probe] Logging enabled. Output dir: {micro_probe_out_dir}")
        # Oracle baseline: 使用初始参数 θ0 + 固定 bias (fixed_adj) 计算“理想目标”概率
        _log_micro_probe(
            model=model,
            config=cfg,
            device=device,
            items=micro_probe_items,
            epoch=0,
            step_in_epoch=0,
            alpha=1.0,
            global_ckpt_idx=-1,
            out_dir=micro_probe_out_dir,
            logit_adjust=fixed_adj.to(device=device, dtype=torch.float32),
        )

    best_val = float("inf")
    ckpt_path = os.path.join(args.save_dir, "model_anneal_mimiciv.pt")
    epoch_ckpt_dir = os.path.join(args.save_dir, "epoch_ckpts")
    if args.save_epoch_ckpts > 0:
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
    if args.save_micro_probe_ckpts:
        os.makedirs(epoch_ckpt_dir, exist_ok=True)

    total_steps_per_epoch = len(train_loader)
    for epoch in range(cfg.epoch):
        alpha = alpha_anneal(epoch, cfg.epoch, mode=args.anneal_mode)
        adj_epoch = (fixed_adj * alpha).to(device=device, dtype=torch.float32) if alpha > 0 else None

        model.train()
        train_loss_accum = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} alpha={alpha:.3f}")
        # 每个 epoch 内的 micro-probe checkpoint 间隔（步数）
        steps_per_probe = 0
        if args.log_micro_probe and micro_probe_items and args.micro_probe_ckpt_per_epoch > 0:
            steps_per_probe = max(total_steps_per_epoch // args.micro_probe_ckpt_per_epoch, 1)

        for step_idx, (batch_ehr, batch_mask) in enumerate(pbar, start=1):
            batch_ehr = batch_ehr.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(
                batch_ehr,
                position_ids=None,
                ehr_labels=batch_ehr,
                ehr_masks=batch_mask,
                pos_loss_weight=cfg.pos_loss_weight,
                logit_adjust=adj_epoch,
            )
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.item())
            train_steps += 1
            pbar.set_postfix({"loss": float(loss.item())})

            # Micro-probing: 在每个 epoch 内按照指定频率记录一次概率轨迹
            if (
                steps_per_probe > 0
                and (step_idx % steps_per_probe == 0)
                and micro_probe_items
                and micro_probe_out_dir is not None
            ):
                ckpt_idx_in_epoch = min(step_idx // steps_per_probe, args.micro_probe_ckpt_per_epoch)
                global_ckpt_idx = epoch * args.micro_probe_ckpt_per_epoch + (ckpt_idx_in_epoch - 1)
                _log_micro_probe(
                    model=model,
                    config=cfg,
                    device=device,
                    items=micro_probe_items,
                    epoch=epoch,
                    step_in_epoch=step_idx,
                    alpha=alpha,
                    global_ckpt_idx=global_ckpt_idx,
                    out_dir=micro_probe_out_dir,
                )
                # 每 0.1 epoch 存一次模型权重，供 eNTK 轨迹画 100 点
                if args.save_micro_probe_ckpts:
                    micro_ckpt_path = os.path.join(
                        epoch_ckpt_dir, f"micro_probe_ckpt_{global_ckpt_idx:04d}.pt"
                    )
                    torch.save(
                        {"model": model.state_dict(), "global_ckpt_idx": global_ckpt_idx},
                        micro_ckpt_path,
                    )
                    if global_ckpt_idx % 10 == 0 or global_ckpt_idx >= 99:
                        print(f"[micro-probe-ckpt] Saved {micro_ckpt_path}")

        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_ehr, batch_mask in val_loader:
                batch_ehr = batch_ehr.to(device)
                batch_mask = batch_mask.to(device)
                vloss, _, _ = model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=cfg.pos_loss_weight,
                    logit_adjust=adj_epoch,
                )
                if vloss.dim() > 0:
                    vloss = vloss.mean()
                val_loss_accum += float(vloss.item())
                val_steps += 1

        avg_train = train_loss_accum / max(train_steps, 1)
        avg_val = val_loss_accum / max(val_steps, 1)
        print(f"Epoch {epoch} | alpha={alpha:.3f} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                    "logit_adjust": fixed_adj.cpu(),
                    "anneal_schedule": "piecewise_linear_30_70",
                    "dataset": "MIMIC-IV_data2",
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to: {ckpt_path}")

        # optional: save extra checkpoint at fixed epoch intervals for eNTK trajectory
        if args.save_epoch_ckpts > 0 and ((epoch % args.save_epoch_ckpts) == 0 or epoch == cfg.epoch - 1):
            extra_path = os.path.join(epoch_ckpt_dir, f"model_epoch{epoch}_anneal_mimiciv.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val,
                    "logit_adjust": fixed_adj.cpu(),
                    "anneal_schedule": "piecewise_linear_30_70",
                    "dataset": "MIMIC-IV_data2",
                },
                extra_path,
            )
            print(f"[epoch-ckpt] Saved epoch checkpoint to: {extra_path}")

    # 采样：不加 bias（logit_adjust=None）
    print("Start synthetic generation (no bias, alpha=0 at sampling) ...")
    model.eval()
    total_samples = int(args.total_samples)
    generated: List[dict] = []
    stoken = np.zeros(cfg.total_vocab_size, dtype=np.float32)
    stoken[cfg.start_record_token] = 1.0

    for i in tqdm(range(0, total_samples, cfg.sample_batch_size), desc="Generating"):
        bs = min(total_samples - i, cfg.sample_batch_size)
        batch_seq = sample_sequence(
            model, length=cfg.n_ctx, start_token=stoken, batch_size=bs,
            config=cfg, device=device, logit_adjust=None,
        )
        generated += convert_ehr(batch_seq, cfg)

    ds_dir = os.path.join(args.save_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, "haloDataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(generated, f)
    print(f"Saved synthetic dataset (MIMIC-IV, fixed bias + anneal, no bias at sampling): {out_path} (n={len(generated)})")

    if args.do_eval:
        eval_dir = os.path.join(THIS_DIR, "evaluate_anneal_mimiciv", f"seed{args.seed}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Running downstream eval (MIMIC-IV data2) via: {EVAL_PY}")
        subprocess.run(
            [
                "python", EVAL_PY,
                "--base_data_dir", args.data_dir,
                "--mymodel2_path", out_path,
                "--save_dir", eval_dir,
                "--sources", "MyModel2",
                "--skip_real",
            ],
            check=True,
        )
        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="MyModel2")
        print(f"[MIMIC-IV fixed bias + Anneal seed={args.seed}] mean_acc={mean_acc:.6f}, mean_f1={mean_f1:.6f}")

        out_csv = os.path.join(OUTPUT_ROOT, "pcla_fixed_bias_anneal_mimiciv_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "mean_acc", "std_acc", "mean_f1", "std_f1", "note"])
            w.writerow([
                f"PCLA_fixed_bias_anneal_MIMIC-IV_seed{args.seed}",
                f"{mean_acc:.6f}", "0.000000", f"{mean_f1:.6f}", "0.000000",
                "MIMIC-IV data2; fixed bias + annealing; sampling with no bias (alpha=0)",
            ])
        out_json = os.path.join(OUTPUT_ROOT, "pcla_fixed_bias_anneal_mimiciv_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "fixed_bias_anneal_mimiciv": {"seed": args.seed, "mean_acc": mean_acc, "mean_f1": mean_f1},
                "data_dir": args.data_dir,
                "init_ckpt": args.init_ckpt_path,
                "eval_csv": eval_csv,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_csv} and {out_json}")


def parse_cli() -> Args:
    default_cfg = Model2Config()
    ap = argparse.ArgumentParser(description="MIMIC-IV data2 + Fixed bias + annealing (no bias at sampling)")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="MIMIC-IV data2 dir (train/val/codeToIndex/idToLabel).")
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    ap.add_argument("--sample_batch_size", type=int, default=default_cfg.sample_batch_size)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=default_cfg.lr)
    ap.add_argument("--pos_loss_weight", type=float, default=1.5)
    ap.add_argument("--total_samples", type=int, default=50000, help="Synthetic samples (align with model8).")
    ap.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device, e.g. cpu, cuda:1. Default: use cuda:1 if available, else cuda:0 or cpu.",
    )
    ap.add_argument("--eval", action="store_true", help="Run downstream 25-label eval and write summary")
    ap.add_argument(
        "--init_ckpt_path",
        default=DEFAULT_INIT_CKPT,
        help="HALO checkpoint trained on MIMIC-IV data2 (model8).",
    )
    # micro-probe logging options (for model6 experiments)
    ap.add_argument(
        "--log_micro_probe",
        action="store_true",
        help="Enable micro-probing: log probabilities for selected (context, disease) pairs during annealing.",
    )
    ap.add_argument(
        "--micro_probe_config",
        type=str,
        default=None,
        help="Path to micro-probing config (e.g. CSV with columns: context_id,disease_id,type).",
    )
    ap.add_argument(
        "--micro_probe_ckpt_per_epoch",
        type=int,
        default=0,
        help="How many micro-probe checkpoints to log per epoch (e.g. 10 => every 0.1 epoch).",
    )
    ap.add_argument(
        "--micro_probe_out_dir",
        type=str,
        default=None,
        help="Output dir for micro-probe logs (CSV files). If not set, a default under model6/micro_probe_logs is used.",
    )
    ap.add_argument(
        "--save_epoch_ckpts",
        type=int,
        default=0,
        help="If >0, save an extra checkpoint every N epochs (e.g. 1 => every epoch) into save_dir/epoch_ckpts/.",
    )
    ap.add_argument(
        "--save_micro_probe_ckpts",
        action="store_true",
        help="Save model state at each micro-probe step (every 0.1 epoch) => 100 checkpoints for eNTK trajectory.",
    )
    ap.add_argument(
        "--anneal_mode",
        type=str,
        default="base",
        choices=["base", "fast", "slow", "cosine"],
        help=(
            "Annealing schedule: "
            "base (30/40/30), fast (10/40/50), slow (50/40/10), "
            "cosine (CosineAnnealingLR-style from 1 to 0)."
        ),
    )
    ap.add_argument(
        "--config_variant",
        type=str,
        default="base",
        choices=["base", "kernel_strong", "kernel_weak"],
        help=(
            "Config variant for kernel strength: "
            "base (default), kernel_strong (scale loaded weights by 1.2), "
            "kernel_weak (scale loaded weights by 0.8)."
        ),
    )

    ns = ap.parse_args()
    # Resolve device default: avoid GPU 0 when multiple GPUs are present.
    if ns.device:
        device_str = ns.device
    else:
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            if n_gpu >= 2:
                device_str = "cuda:1"
            else:
                device_str = "cuda:0"
        else:
            device_str = "cpu"
    return Args(
        data_dir=ns.data_dir,
        save_dir=ns.save_dir,
        seed=int(ns.seed),
        epochs=int(ns.epochs),
        batch_size=int(ns.batch_size),
        sample_batch_size=int(ns.sample_batch_size),
        num_workers=int(ns.num_workers),
        lr=float(ns.lr),
        pos_loss_weight=float(ns.pos_loss_weight) if ns.pos_loss_weight is not None else None,
        total_samples=int(ns.total_samples),
        device=device_str,
        do_eval=bool(ns.eval),
        init_ckpt_path=ns.init_ckpt_path,
        log_micro_probe=bool(ns.log_micro_probe),
        micro_probe_config=ns.micro_probe_config,
        micro_probe_ckpt_per_epoch=int(ns.micro_probe_ckpt_per_epoch),
        micro_probe_out_dir=ns.micro_probe_out_dir,
        save_epoch_ckpts=int(ns.save_epoch_ckpts),
        save_micro_probe_ckpts=bool(ns.save_micro_probe_ckpts),
        anneal_mode=str(ns.anneal_mode),
        config_variant=str(ns.config_variant),
    )


if __name__ == "__main__":
    run_single_seed(parse_cli())

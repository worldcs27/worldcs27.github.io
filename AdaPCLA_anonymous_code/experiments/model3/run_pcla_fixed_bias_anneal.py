#!/usr/bin/env python3
"""
固定 bias + 退火 (Fixed Bias + Annealing)。

- 训练：out_logits = logits + α(epoch) * logit_adjust，α 从 1.0 分段线性降到 0。
- 采样/生成：不加 bias（α=0），即 logit_adjust=None。

所有结果与脚本放在 mywork/model3。
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

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_SAVE_DIR = os.path.join(THIS_DIR, "save_anneal")
DEFAULT_SEED = 1
OUTPUT_ROOT = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


import sys
if MODEL7_DIR not in sys.path:
    sys.path.insert(0, MODEL7_DIR)
from config import Model2Config  # type: ignore
from model import HALOModel  # type: ignore


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_logit_adjust(train_data, *, config: Model2Config) -> Tuple[np.ndarray, dict]:
    """与 model7/train.py 中 _compute_logit_adjust 一致，得到固定 bias 向量。"""
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


def alpha_anneal(epoch: int, total_epochs: int) -> float:
    """
    退火系数 α(epoch)：分段线性。
    - 前 30% epoch：α = 1.0
    - 30%–70%：线性 1.0 -> 0.0
    - 后 30%：α = 0.0
    """
    if total_epochs <= 0:
        return 1.0
    t = epoch / max(total_epochs - 1, 1)
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
    do_eval: bool
    init_ckpt_path: str | None


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


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


def run_single_seed(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    print(f"Vocab size (codes) = {cfg.code_vocab_size}, labels = {cfg.label_vocab_size}")

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
            # 若 checkpoint 来自 DDP，key 带 "module." 前缀，需去掉再加载到单卡 HALOModel
            if next(iter(state.keys()), "").startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            print(f"Loaded init from: {args.init_ckpt_path}")

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

    best_val = float("inf")
    ckpt_path = os.path.join(args.save_dir, "model_anneal.pt")

    for epoch in range(cfg.epoch):
        alpha = alpha_anneal(epoch, cfg.epoch)
        adj_epoch = (fixed_adj * alpha).to(device=device, dtype=torch.float32) if alpha > 0 else None

        model.train()
        train_loss_accum = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} alpha={alpha:.3f}")
        for batch_ehr, batch_mask in pbar:
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
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to: {ckpt_path}")

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
    print(f"Saved synthetic dataset (fixed bias + anneal, no bias at sampling): {out_path} (n={len(generated)})")

    if args.do_eval:
        eval_dir = os.path.join(THIS_DIR, "evaluate_anneal", f"seed{args.seed}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Running downstream eval via: {EVAL_PY}")
        subprocess.run(
            [
                "python", EVAL_PY,
                "--base_data_dir", args.data_dir,
                "--mymodel2_path", out_path,
                "--save_dir", eval_dir,
                "--sources", "MyModel2",
            ],
            check=True,
        )
        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="MyModel2")
        print(f"[Fixed bias + Anneal seed={args.seed}] mean_acc={mean_acc:.6f}, mean_f1={mean_f1:.6f}")

        fixed_csv = os.path.join(THIS_DIR, "..", "model1", "output", "pcla_best_seed3_summary.csv")
        fixed_mean_acc = fixed_std_acc = fixed_mean_f1 = fixed_std_f1 = float("nan")
        if os.path.exists(fixed_csv):
            with open(fixed_csv, newline="") as f:
                r = csv.reader(f)
                next(r, None)
                for row in r:
                    if not row:
                        continue
                    if row[0].startswith("mean±std"):
                        try:
                            m_a, s_a = row[1].split("±")
                            m_f, s_f = row[2].split("±")
                            fixed_mean_acc, fixed_std_acc = float(m_a), float(s_a)
                            fixed_mean_f1, fixed_std_f1 = float(m_f), float(s_f)
                        except Exception:
                            pass
                        break

        out_csv = os.path.join(OUTPUT_ROOT, "pcla_fixed_bias_anneal_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "mean_acc", "std_acc", "mean_f1", "std_f1", "note"])
            w.writerow([
                "PCLA_fixed_3seeds",
                f"{fixed_mean_acc:.6f}" if not np.isnan(fixed_mean_acc) else "",
                f"{fixed_std_acc:.6f}" if not np.isnan(fixed_std_acc) else "",
                f"{fixed_mean_f1:.6f}" if not np.isnan(fixed_mean_f1) else "",
                f"{fixed_std_f1:.6f}" if not np.isnan(fixed_std_f1) else "",
                "Original 3-seed fixed-bias PCLA (sampling with bias)",
            ])
            w.writerow([
                f"PCLA_fixed_bias_anneal_seed{args.seed}",
                f"{mean_acc:.6f}", "0.000000", f"{mean_f1:.6f}", "0.000000",
                "Fixed bias + annealing; sampling with no bias (alpha=0)",
            ])
        out_json = os.path.join(OUTPUT_ROOT, "pcla_fixed_bias_anneal_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "fixed_bias_anneal": {"seed": args.seed, "mean_acc": mean_acc, "mean_f1": mean_f1},
                "fixed_bias_3seeds": {
                    "mean_acc": fixed_mean_acc, "std_acc": fixed_std_acc,
                    "mean_f1": fixed_mean_f1, "std_f1": fixed_std_f1,
                    "source_csv": fixed_csv,
                },
                "eval_csv": eval_csv,
            }, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_csv} and {out_json}")


def parse_cli() -> Args:
    default_cfg = Model2Config()
    ap = argparse.ArgumentParser(description="Fixed bias + annealing (no bias at sampling)")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    ap.add_argument("--sample_batch_size", type=int, default=default_cfg.sample_batch_size)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=default_cfg.lr)
    ap.add_argument("--pos_loss_weight", type=float, default=1.5)
    ap.add_argument("--total_samples", type=int, default=33494)
    ap.add_argument("--eval", action="store_true", help="Run downstream 25-label eval and write summary")
    ap.add_argument(
        "--init_ckpt_path",
        default=HALO_MIMICIII_CKPT,
        help="HALO init checkpoint (same as PCLA)",
    )
    ns = ap.parse_args()
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
        do_eval=bool(ns.eval),
        init_ckpt_path=ns.init_ckpt_path,
    )


if __name__ == "__main__":
    run_single_seed(parse_cli())

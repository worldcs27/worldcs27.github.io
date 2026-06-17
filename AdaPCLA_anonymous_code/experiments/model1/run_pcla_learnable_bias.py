#!/usr/bin/env python3
"""
Run a single-seed PCLA (Model7) training + generation experiment with a *learnable* output bias.

约束与目标：
- 不修改发布包之外的外部数据与代码目录；
- 不再调用原来的多 seed / 对比 Real/HALO 的评估脚本，只做：
  1) 在 MIMIC-III 任务数据上训练一个 PCLA 生成器（单个 seed）；
  2) 用该生成器在同一任务空间中采样生成 `haloDataset.pkl`；
- 将原来的「外部传入 logit_adjust 向量」改为「模型内部的可学习 bias 参数」，并用统计先验 `b_stat` 进行初始化：

    self.output_bias = nn.Parameter(torch.tensor(calculated_b, dtype=torch.float32))

运行方式（示例）：

    cd EXPERIMENTS_ROOT/model1
    python run_pcla_learnable_bias.py \\
        --data_dir DATA_MIMICIII \\
        --save_dir ./save_learnable_bias/seed1

这会在 `save_learnable_bias/seed1/` 下生成：
- `model_lb.pt`：带可学习 bias 的 PCLA 生成器 checkpoint；
- `datasets/haloDataset.pkl`：用该模型生成的合成 EHR 数据（与输入任务空间兼容）。
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --------- 路径与默认参数（按原 model7 配置） ---------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = DATA_MIMICIII  # MIMIC-III 任务数据
DEFAULT_SAVE_DIR = os.path.join(THIS_DIR, "save_learnable_bias")

DEFAULT_SEED = 1
OUTPUT_ROOT = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


# --------- 引用原始的 Model2Config / HALOModel 定义（不改动原文件） ---------

import sys

if MODEL7_DIR not in sys.path:
    sys.path.insert(0, MODEL7_DIR)

from config import Model2Config  # type: ignore
from model import HALOModel as _HALOModelBase  # type: ignore

# 原始下游评估脚本路径（保持不变）


# --------- 工具与数据集定义（与原 train/test 逻辑兼容） ---------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MIMICDataset(Dataset):
    """与原 model7/train.py / test.py 中的 MIMICDataset 保持一致。"""

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


def compute_logit_adjust(train_data, *, config: Model2Config) -> Tuple[np.ndarray, dict]:
    """
    与原 train.py 中的 _compute_logit_adjust 一致：
    - 基于训练集统计每个 code 在多少个 visit 中出现过；
    - 得到先验 log-odds 向量 b_stat，截断到 [-clip, clip]；
    - 再扩展到 total_vocab_size 长度。
    """
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


# --------- Learnable-Bias 版本 HALOModel ---------


class HALOModelWithLearnableBias(_HALOModelBase):
    """
    在原 HALOModel 基础上，将 logit_adjust 向量内化为一个可学习的 bias 向量：

        self.output_bias: shape = [total_vocab_size]

    - 初始化时，用基于训练集统计得到的 logit_adjust 先验 b_stat 初始化；
    - 训练时，该 bias 会随梯度一起更新；
    - 前向时不再从外部传入 logit_adjust，而是始终使用内部的 self.output_bias。
    """

    def __init__(self, config: Model2Config, init_bias: torch.Tensor):
        super().__init__(config)
        if init_bias.ndim != 1 or int(init_bias.numel()) != int(config.total_vocab_size):
            raise ValueError(
                f"init_bias shape mismatch: got {tuple(init_bias.shape)} "
                f"expected ({config.total_vocab_size},)"
            )
        self.output_bias = nn.Parameter(init_bias.detach().clone().float())

    def forward(
        self,
        input_visits,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        past=None,
        pos_loss_weight=None,
    ):
        # 与原 HALOModel.forward 基本一致，只是：
        # - 不再接收外部 logit_adjust；
        # - 统一使用 self.output_bias 作为 bias，并参与反向传播。
        hidden_states = self.transformer(input_visits, position_ids, past)
        logits = self.ehr_head(hidden_states, input_visits)

        # 广播到 [B, T, V]
        bias = self.output_bias.view(1, 1, -1).to(device=logits.device, dtype=logits.dtype)

        out_logits = logits + bias
        probs = torch.sigmoid(out_logits)

        if ehr_labels is None:
            return probs

        shift_labels = ehr_labels[..., 1:, :].contiguous()

        pos_weight = None
        if pos_loss_weight is not None:
            pos_weight = torch.full(
                (logits.size(-1),),
                float(pos_loss_weight),
                device=logits.device,
                dtype=logits.dtype,
            )

        loss_logits = out_logits  # 注意：已经加过 bias 了，这里不再重复相加

        loss_elem = F.binary_cross_entropy_with_logits(
            loss_logits,
            shift_labels.to(dtype=loss_logits.dtype),
            pos_weight=pos_weight,
            reduction="none",
        )
        if ehr_masks is not None:
            mask = ehr_masks.to(dtype=loss_elem.dtype, device=loss_elem.device)
            loss_elem = loss_elem * mask
            denom = mask.sum().clamp(min=1.0) * float(loss_elem.size(-1))
            loss = loss_elem.sum() / denom
            return loss, probs * mask, shift_labels * mask

        loss = loss_elem.mean()
        return loss, probs, shift_labels


# --------- 采样（单机版） ---------


def sample_sequence(
    model: HALOModelWithLearnableBias,
    length: int,
    start_token: np.ndarray,
    batch_size: int,
    config: Model2Config,
    device: torch.device,
) -> np.ndarray:
    """
    简化版的采样逻辑：参考 model7/test.py 中的 sample_sequence，
    但不使用 DDP，只在单卡上生成。
    """
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
            prev = model.sample(torch.cat((prev, empty), dim=1), random=True)
            # 如果所有样本都已经生成了 end_record_token，则提前终止
            end_mask = (prev[:, :, config.end_record_token].sum(dim=1) > 0)
            if bool(end_mask.all()):
                break
    return prev.cpu().detach().numpy()


def convert_ehr(ehrs: np.ndarray, config: Model2Config):
    """与 model7/test.py 中的 convert_ehr 一致：从 one-hot 序列还原 visits + labels。"""
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


# --------- 训练 + 生成（单 seed，单机） ---------


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
            a = row.get("Accuracy")
            f1 = row.get("F1 Score")
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

    # 1) 加载数据与词表
    def load_pkl(name: str):
        path = os.path.join(args.data_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")
    val_data = load_pkl("valDataset.pkl")

    # 2) 构造配置对象
    cfg = Model2Config()
    cfg.lr = float(args.lr)
    cfg.epoch = int(args.epochs)
    cfg.batch_size = int(args.batch_size)
    cfg.sample_batch_size = int(args.sample_batch_size)
    cfg.pos_loss_weight = args.pos_loss_weight

    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    cfg.total_vocab_size = (
        cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size
    )

    print(f"Vocab size (codes) = {cfg.code_vocab_size}, labels = {cfg.label_vocab_size}")

    # 3) 计算先验 logit_adjust，并作为 learnable bias 的初始值
    adj_np, stats = compute_logit_adjust(train_data, config=cfg)
    print(f"Logit adjust stats: {stats}")
    adj_init = torch.from_numpy(adj_np)

    # 4) 构造带可学习 bias 的 HALOModel
    model = HALOModelWithLearnableBias(cfg, adj_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # 5) DataLoader
    train_ds = MIMICDataset(train_data, cfg)
    val_ds = MIMICDataset(val_data, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_val = float("inf")
    ckpt_path = os.path.join(args.save_dir, "model_lb.pt")

    # 6) 训练循环（单机版本）
    for epoch in range(cfg.epoch):
        model.train()
        train_loss_accum = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
            )
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.item())
            train_steps += 1
            pbar.set_postfix({"loss": float(loss.item())})

        # 验证
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
                )
                if vloss.dim() > 0:
                    vloss = vloss.mean()
                val_loss_accum += float(vloss.item())
                val_steps += 1

        avg_train = train_loss_accum / max(train_steps, 1)
        avg_val = val_loss_accum / max(val_steps, 1)
        print(f"Epoch {epoch} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint to: {ckpt_path}")

    # 7) 采样生成合成 EHR（haloDataset.pkl）
    print("Start synthetic generation with learnable-bias PCLA ...")
    model.eval()
    total_samples = int(args.total_samples)
    generated: List[dict] = []

    stoken = np.zeros(cfg.total_vocab_size, dtype=np.float32)
    stoken[cfg.start_record_token] = 1.0

    for i in tqdm(range(0, total_samples, cfg.sample_batch_size), desc="Generating"):
        bs = min(total_samples - i, cfg.sample_batch_size)
        batch_seq = sample_sequence(
            model,
            length=cfg.n_ctx,
            start_token=stoken,
            batch_size=bs,
            config=cfg,
            device=device,
        )
        generated += convert_ehr(batch_seq, cfg)

    ds_dir = os.path.join(args.save_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)
    out_path = os.path.join(ds_dir, "haloDataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(generated, f)
    print(f"Saved synthetic dataset (learnable bias): {out_path} (n={len(generated)})")

    # 8) （可选）下游 25-label 评估 + 与原 PCLA 固定 bias 对比
    if args.do_eval:
        eval_dir = os.path.join(THIS_DIR, "evaluate_learnable_bias", f"seed{args.seed}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"Running downstream eval via: {EVAL_PY}")
        # 只评估 MyModel2（即本次 learnable-bias PCLA），不强制评 HALO
        cmd = [
            "python",
            EVAL_PY,
            "--base_data_dir",
            args.data_dir,
            "--mymodel2_path",
            out_path,
            "--save_dir",
            eval_dir,
            "--sources",
            "MyModel2",
        ]
        subprocess.run(cmd, check=True)

        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        mean_acc_lb, mean_f1_lb = _parse_mean_metrics(eval_csv, source="MyModel2")
        print(
            f"[LearnableBias PCLA seed={args.seed}] mean_acc={mean_acc_lb:.6f}, "
            f"mean_f1={mean_f1_lb:.6f}"
        )

        # 读取原 PCLA 固定 bias 的 3-seed summary（pcla_best_seed3_summary.csv）以便对比
        fixed_csv = os.path.join(OUTPUT_ROOT, "pcla_best_seed3_summary.csv")
        fixed_mean_acc = fixed_std_acc = fixed_mean_f1 = fixed_std_f1 = float("nan")
        if os.path.exists(fixed_csv):
            with open(fixed_csv, newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if not row:
                        continue
                    if row[0].startswith("mean±std"):
                        # row 1: "mean±std", "0.901853±0.002840", "0.902251±0.002746", ...
                        acc_str = row[1]
                        f1_str = row[2]
                        try:
                            m_a, s_a = acc_str.split("±")
                            m_f, s_f = f1_str.split("±")
                            fixed_mean_acc = float(m_a)
                            fixed_std_acc = float(s_a)
                            fixed_mean_f1 = float(m_f)
                            fixed_std_f1 = float(s_f)
                        except Exception:
                            pass
                        break
        else:
            print(f"Warning: fixed-bias PCLA summary not found at {fixed_csv}")

        # 写对比 CSV / JSON
        out_csv = os.path.join(OUTPUT_ROOT, "pcla_fixed_vs_learnable_bias_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "variant",
                    "mean_acc",
                    "std_acc",
                    "mean_f1",
                    "std_f1",
                    "note",
                ]
            )
            w.writerow(
                [
                    "PCLA_fixed_3seeds",
                    f"{fixed_mean_acc:.6f}" if not np.isnan(fixed_mean_acc) else "",
                    f"{fixed_std_acc:.6f}" if not np.isnan(fixed_std_acc) else "",
                    f"{fixed_mean_f1:.6f}" if not np.isnan(fixed_mean_f1) else "",
                    f"{fixed_std_f1:.6f}" if not np.isnan(fixed_std_f1) else "",
                    "Original 3-seed fixed-bias PCLA summary (from pcla_best_seed3_summary.csv)",
                ]
            )
            w.writerow(
                [
                    f"PCLA_learnable_bias_seed{args.seed}",
                    f"{mean_acc_lb:.6f}",
                    "0.000000",  # 单次运行，std 记为 0
                    f"{mean_f1_lb:.6f}",
                    "0.000000",
                    "This run: single-seed PCLA with learnable output bias",
                ]
            )

        out_json = os.path.join(OUTPUT_ROOT, "pcla_fixed_vs_learnable_bias_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "learnable_bias": {
                        "seed": args.seed,
                        "mean_acc": mean_acc_lb,
                        "mean_f1": mean_f1_lb,
                    },
                    "fixed_bias_3seeds": {
                        "mean_acc": fixed_mean_acc,
                        "std_acc": fixed_std_acc,
                        "mean_f1": fixed_mean_f1,
                        "std_f1": fixed_std_f1,
                        "source_csv": fixed_csv,
                    },
                    "eval_csv_learnable": eval_csv,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Wrote comparison CSV: {out_csv}")
        print(f"Wrote comparison JSON: {out_json}")


def parse_cli() -> Args:
    default_cfg = Model2Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    ap.add_argument("--sample_batch_size", type=int, default=default_cfg.sample_batch_size)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=default_cfg.lr)
    ap.add_argument(
        "--pos_loss_weight",
        type=float,
        default=1.5,
        help="Positive class reweighting for BCEWithLogits (same含义 as原PCLA).",
    )
    ap.add_argument(
        "--total_samples",
        type=int,
        default=33494,
        help="Number of synthetic patients to generate (默认与 MIMIC-III 训练集等大)。",
    )
    ap.add_argument(
        "--eval",
        action="store_true",
        help="在训练+生成之后，自动调用原 evaluate_synthetic_training.py 对本次 PCLA (MyModel2) 做下游 25-label 评估，并与固定 bias 的 PCLA 3-seed summary 对比。",
    )
    args_ns = ap.parse_args()
    return Args(
        data_dir=args_ns.data_dir,
        save_dir=args_ns.save_dir,
        seed=int(args_ns.seed),
        epochs=int(args_ns.epochs),
        batch_size=int(args_ns.batch_size),
        sample_batch_size=int(args_ns.sample_batch_size),
        num_workers=int(args_ns.num_workers),
        lr=float(args_ns.lr),
        pos_loss_weight=float(args_ns.pos_loss_weight)
        if args_ns.pos_loss_weight is not None
        else None,
        total_samples=int(args_ns.total_samples),
        do_eval=bool(args_ns.eval),
    )


if __name__ == "__main__":
    cli_args = parse_cli()
    run_single_seed(cli_args)




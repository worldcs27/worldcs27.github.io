#!/usr/bin/env python3
"""
model4: Prior-only synthetic data experiment.

不使用任何 HALO/PCLA 模型，仅依赖训练集统计得到的先验：
- code 级先验：compute_logit_adjust(train_data, config) 得到的 logit_adjust / b；
- label 级先验：训练集中 25 维 CCS 标签的边际频率；
- 结构先验：训练集中每个病人的 visit 数分布、每个 visit 的 code 数分布。

根据这些先验直接生成 `haloDataset.pkl`，再复用原有
`evaluate_synthetic_training.py` 做下游 25-label 评估。
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


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_SAVE_DIR = os.path.join(THIS_DIR, "save_prior_only")
DEFAULT_SEED = 1
OUTPUT_ROOT = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


import sys

if MODEL7_DIR not in sys.path:
    sys.path.insert(0, MODEL7_DIR)
from config import Model2Config  # type: ignore


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_logit_adjust(train_data, *, config: Model2Config) -> Tuple[np.ndarray, dict]:
    """
    与 model7/train.py 中的 _compute_logit_adjust 一致：
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


def _collect_structure_stats(train_data, *, code_vocab_size: int) -> Tuple[List[int], List[int]]:
    """
    从训练集中收集结构统计：
    - num_visits_list: 每个病人的 visit 数分布
    - visit_len_list: 每个 visit 中 code 数分布
    """
    num_visits_list: List[int] = []
    visit_len_list: List[int] = []
    for p in train_data:
        visits = p.get("visits", [])
        num_visits_list.append(len(visits))
        for v in visits:
            # 保证索引在 code 词表范围内
            valid_codes = [int(c) for c in v if 0 <= int(c) < code_vocab_size]
            visit_len_list.append(len(valid_codes))
    # 避免后续 random.choice 出错
    if not visit_len_list:
        visit_len_list = [1]
    if not num_visits_list:
        num_visits_list = [1]
    return num_visits_list, visit_len_list


def _compute_label_prior(train_data, *, label_vocab_size: int) -> np.ndarray:
    """
    统计训练集中 25 维 CCS 标签的边际频率，作为独立 Bernoulli 先验。
    """
    label_counts = np.zeros((label_vocab_size,), dtype=np.int64)
    for p in train_data:
        y = np.asarray(p["labels"], dtype=np.int64)
        if y.shape[0] != label_vocab_size:
            raise ValueError(f"Label length mismatch: got {y.shape[0]} expected {label_vocab_size}")
        label_counts += y
    num_patients = len(train_data)
    if num_patients <= 0:
        return np.zeros((label_vocab_size,), dtype=np.float64)
    label_pi = label_counts.astype(np.float64) / float(num_patients)
    return label_pi


def sample_patient_prior_only(
    *,
    code_probs: np.ndarray,  # shape = [V_code]
    label_probs: np.ndarray,  # shape = [V_label]
    num_visits_list: List[int],
    visit_len_list: List[int],
    max_visits: int | None = None,
) -> dict:
    """
    根据先验（无模型）生成一个病人的 visits + labels。
    """
    V_code = int(code_probs.shape[0])

    # 1) 病人 visit 数量：从经验分布中随机取一个
    num_visits = random.choice(num_visits_list)
    if max_visits is not None:
        num_visits = min(num_visits, max_visits)

    visits_out: List[List[int]] = []

    for _ in range(num_visits):
        # 2) 该 visit 中 code 数量（粗略控制稀疏度）
        visit_len = max(1, random.choice(visit_len_list))

        # 3) 根据 code 先验生成该 visit 的 code 集合
        #    简单做法：按 Bernoulli 独立采样，然后截断到指定长度
        flags = np.random.rand(V_code) < code_probs  # shape=[V_code], 布尔向量
        candidate_idxs = np.nonzero(flags)[0].tolist()
        random.shuffle(candidate_idxs)
        visit_codes = candidate_idxs[:visit_len]
        visit_codes = sorted(set(int(c) for c in visit_codes))

        if visit_codes:
            visits_out.append(visit_codes)

    # 4) 病人的 25 维 labels（独立 Bernoulli，来自边际先验）
    labels = (np.random.rand(label_probs.shape[0]) < label_probs).astype(np.int32)

    return {"visits": visits_out, "labels": labels}


@dataclass
class Args:
    data_dir: str
    save_dir: str
    seed: int
    total_samples: int
    do_eval: bool


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
            a = row.get("Accuracy")
            f1 = row.get("F1 Score")
            if a not in (None, ""):
                accs.append(float(a))
            if f1 not in (None, ""):
                f1s.append(float(f1))
    if not accs or not f1s:
        raise ValueError(f"No rows for source={source} in {csv_path}")
    return _mean(accs), _mean(f1s)


def run_prior_only(args: Args) -> None:
    seed_everything(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    ds_dir = os.path.join(args.save_dir, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    # 1) 加载数据与词表
    def load_pkl(name: str):
        path = os.path.join(args.data_dir, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")

    # 2) 构造配置对象（仅用于 logit_adjust 参数与 vocab 大小）
    cfg = Model2Config()
    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size

    print(f"[Prior-only] code_vocab_size={cfg.code_vocab_size}, label_vocab_size={cfg.label_vocab_size}")

    # 3) code 先验：logit_adjust -> code_probs
    adj_np, stats = compute_logit_adjust(train_data, config=cfg)
    print(f"[Prior-only] Logit adjust stats: {stats}")
    V_code = int(cfg.code_vocab_size)
    b = adj_np[:V_code]
    code_probs = 1.0 / (1.0 + np.exp(-b))  # sigmoid(b)

    # 4) label 先验：边际频率 -> label_probs
    label_probs = _compute_label_prior(train_data, label_vocab_size=cfg.label_vocab_size)

    # 5) 结构先验：num_visits_list, visit_len_list
    num_visits_list, visit_len_list = _collect_structure_stats(train_data, code_vocab_size=V_code)

    # 6) 生成 prior-only 合成数据
    total_samples = int(args.total_samples)
    synthetic: List[dict] = []
    for i in range(total_samples):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"[Prior-only] Generating sample {i+1}/{total_samples} ...")
        synthetic.append(
            sample_patient_prior_only(
                code_probs=code_probs,
                label_probs=label_probs,
                num_visits_list=num_visits_list,
                visit_len_list=visit_len_list,
                max_visits=cfg.n_ctx,
            )
        )

    out_path = os.path.join(ds_dir, "haloDataset.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(synthetic, f)
    print(f"[Prior-only] Saved synthetic dataset to: {out_path} (n={len(synthetic)})")

    # 7) （可选）下游 25-label 评估
    if args.do_eval:
        eval_dir = os.path.join(THIS_DIR, "evaluate_prior_only", f"seed{args.seed}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"[Prior-only] Running downstream eval via: {EVAL_PY}")
        subprocess.run(
            [
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
            ],
            check=True,
        )

        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="MyModel2")
        print(f"[Prior-only seed={args.seed}] mean_acc={mean_acc:.6f}, mean_f1={mean_f1:.6f}")

        # 读取固定 PCLA 3-seed summary 以便对比
        fixed_csv = os.path.join(THIS_DIR, "..", "model1", "output", "pcla_best_seed3_summary.csv")
        fixed_mean_acc = fixed_std_acc = fixed_mean_f1 = fixed_std_f1 = float("nan")
        if os.path.exists(fixed_csv):
            with open(fixed_csv, newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    if not row:
                        continue
                    if row[0].startswith("mean±std"):
                        try:
                            m_a, s_a = row[1].split("±")
                            m_f, s_f = row[2].split("±")
                            fixed_mean_acc = float(m_a)
                            fixed_std_acc = float(s_a)
                            fixed_mean_f1 = float(m_f)
                            fixed_std_f1 = float(s_f)
                        except Exception:
                            pass
                        break

        out_csv = os.path.join(OUTPUT_ROOT, "prior_only_vs_pcla_summary.csv")
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
                    f"Prior_only_seed{args.seed}",
                    f"{mean_acc:.6f}",
                    "0.000000",
                    f"{mean_f1:.6f}",
                    "0.000000",
                    "This run: prior-only synthetic data (no generator model, only priors)",
                ]
            )

        out_json = os.path.join(OUTPUT_ROOT, "prior_only_vs_pcla_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prior_only": {
                        "seed": args.seed,
                        "mean_acc": mean_acc,
                        "mean_f1": mean_f1,
                    },
                    "fixed_bias_3seeds": {
                        "mean_acc": fixed_mean_acc,
                        "std_acc": fixed_std_acc,
                        "mean_f1": fixed_mean_f1,
                        "std_f1": fixed_std_f1,
                        "source_csv": fixed_csv,
                    },
                    "eval_csv_prior_only": eval_csv,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[Prior-only] Wrote summary CSV: {out_csv}")
        print(f"[Prior-only] Wrote summary JSON: {out_json}")


def parse_cli() -> Args:
    ap = argparse.ArgumentParser(description="Prior-only synthetic data (no generator model)")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--total_samples",
        type=int,
        default=33494,
        help="Number of synthetic patients to generate (默认与 MIMIC-III 训练集等大)。",
    )
    ap.add_argument(
        "--eval",
        action="store_true",
        help="在生成 prior-only 合成数据后，自动跑下游 25-label 评估，并与固定 bias PCLA 对比。",
    )
    ns = ap.parse_args()
    return Args(
        data_dir=ns.data_dir,
        save_dir=ns.save_dir,
        seed=int(ns.seed),
        total_samples=int(ns.total_samples),
        do_eval=bool(ns.eval),
    )


if __name__ == "__main__":
    run_prior_only(parse_cli())


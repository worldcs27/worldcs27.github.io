#!/usr/bin/env python3
"""
生成 model6 微观探测配置 CSV：context_id, disease_id, type。

- 从 MIMIC-IV data2 读取 train 与 code 词表，得到 context 数量与 code_vocab_size。
- 按训练集内 code 出现频次区分「长尾」与常见病，为每个 context 构造三类候选疾病
  related_rare / unrelated_rare / wrong（启发式，可后续手工微调）。
- 输出 CSV 供 run_micro_probing_fixed_bias_anneal_mimiciv.sh 使用。

用法:
  cd EXPERIMENTS_ROOT/model6
  python gen_micro_probe_config.py --data_dir /path/to/data2 --out mimiciv_long_tail_triplets_seed1.csv
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
import pickle
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch

DATA2_DEFAULT = DATA_MIMICIV
CONFIG_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_probe_configs")

PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL5_DIR = os.path.join(PROJECT_ROOT, "model5")


def build_model_and_dataset(data_dir: str):
    import sys

    if MODEL8_DIR not in sys.path:
        sys.path.insert(0, MODEL8_DIR)
    if MODEL5_DIR not in sys.path:
        sys.path.insert(0, MODEL5_DIR)

    from config import Model2Config  # type: ignore
    from model import HALOModel  # type: ignore
    from run_pcla_fixed_bias_anneal_mimiciv import MIMICDataset, DEFAULT_INIT_CKPT  # type: ignore

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

    # Prefer GPU 1 if available to avoid GPU 0.
    if torch.cuda.is_available():
        if torch.cuda.device_count() >= 2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = HALOModel(cfg).to(device)

    if DEFAULT_INIT_CKPT and os.path.exists(DEFAULT_INIT_CKPT):
        ckpt = torch.load(DEFAULT_INIT_CKPT, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
            if next(iter(state.keys()), "").startswith("module."):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        else:
            model.load_state_dict(ckpt, strict=True)

    train_ds = MIMICDataset(train_data, cfg)
    return model, cfg, train_ds


def compute_entk_sim(
    model: torch.nn.Module,
    cfg,
    ehr_np: np.ndarray,
    disease_id: int,
    device: torch.device,
) -> float:
    """
    Compute cosine similarity between:
      - grad wrt disease_id
      - average grad wrt all context codes
    as a simple eNTK similarity scalar.
    """
    model.eval()
    ehr_tensor = torch.from_numpy(ehr_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    def flat_grad() -> torch.Tensor:
        grads: List[torch.Tensor] = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads) if grads else torch.zeros(1, device=device)

    # disease gradient
    model.zero_grad(set_to_none=True)
    probs = model(
        ehr_tensor,
        position_ids=None,
        ehr_labels=None,
        ehr_masks=None,
        pos_loss_weight=None,
        logit_adjust=None,
    )
    # probs: (1, n_ctx-1, V)
    if probs.dim() != 3 or disease_id >= probs.size(-1):
        return float("nan")
    disease_probs = probs[:, :, disease_id]
    scalar = disease_probs.mean()
    scalar.backward(retain_graph=True)
    g_d = flat_grad()

    # context bundle gradient: average over codes that actually appear
    context_codes: List[int] = []
    for j in range(ehr_np.shape[0]):
        row = ehr_np[j]
        codes = np.nonzero(row[: int(cfg.code_vocab_size)])[0]
        for c in codes:
            cid = int(c)
            if cid not in context_codes:
                context_codes.append(cid)

    if not context_codes:
        return float("nan")

    g_ctx = None
    cnt = 0
    for cid in context_codes:
        if cid >= cfg.code_vocab_size:
            continue
        model.zero_grad(set_to_none=True)
        c_probs = probs[:, :, cid]
        c_scalar = c_probs.mean()
        c_scalar.backward(retain_graph=True)
        g_c = flat_grad()
        g_ctx = g_c if g_ctx is None else (g_ctx + g_c)
        cnt += 1

    if g_ctx is None or cnt == 0:
        return float("nan")
    g_ctx = g_ctx / float(cnt)

    n_d = torch.norm(g_d)
    n_c = torch.norm(g_ctx)
    if n_d.item() == 0.0 or n_c.item() == 0.0:
        return float("nan")
    cos = torch.dot(g_d, g_ctx) / (n_d * n_c)
    return float(cos.item())


def main():
    ap = argparse.ArgumentParser(description="Generate micro-probe config CSV for model6")
    ap.add_argument("--data_dir", default=DATA2_DEFAULT, help="MIMIC-IV data2 dir (trainDataset.pkl, codeToIndex.pkl)")
    ap.add_argument("--out", default="mimiciv_long_tail_triplets_seed1.csv", help="Output CSV filename (under micro_probe_configs/)")
    ap.add_argument("--n_ctx", type=int, default=30, help="Number of probe contexts (default 30)")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for sampling contexts/codes")
    ap.add_argument("--with_entk", action="store_true", help="If set, compute initial_entk_sim for each row.")
    ap.add_argument(
        "--base_config",
        type=str,
        default="",
        help=(
            "Optional existing config CSV whose context_id values must be included "
            "in the new config (e.g., 30ctx base when generating a 100ctx superset)."
        ),
    )
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    with open(os.path.join(data_dir, "codeToIndex.pkl"), "rb") as f:
        code_to_index = pickle.load(f)
    with open(os.path.join(data_dir, "trainDataset.pkl"), "rb") as f:
        train_data = pickle.load(f)

    code_vocab_size = len(code_to_index)
    n_train = len(train_data)
    print(f"[data2] train size = {n_train}, code_vocab_size = {code_vocab_size}")

    # 统计每个 code 在训练集中出现次数（按 visit 内出现即计 1）
    code_counts: Counter[int] = Counter()
    for p in train_data:
        for visit in p.get("visits", []):
            for c in visit:
                if 0 <= c < code_vocab_size:
                    code_counts[c] += 1

    # 长尾：出现次数少的 code（按次数排序取后半）
    sorted_codes = sorted(code_counts.keys(), key=lambda c: code_counts[c])
    n_rare = max(1, len(sorted_codes) // 3)
    rare_codes = sorted_codes[: n_rare]
    # 常见：出现次数多的 code
    common_codes = sorted_codes[-n_rare:] if n_rare else sorted_codes

    rng = random.Random(args.seed)

    # Optional base_config: ensure those context_ids are always included (e.g., 30ctx subset).
    base_context_ids: list[int] = []
    if args.base_config:
        base_path = args.base_config
        if not os.path.isabs(base_path):
            base_path = os.path.join(CONFIG_DIR_DEFAULT, base_path)
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"base_config not found: {base_path}")
        with open(base_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row["context_id"])
                except Exception:
                    continue
                base_context_ids.append(cid)
        base_context_ids = sorted(set(base_context_ids))
        print(f"[micro-probe] Loaded {len(base_context_ids)} base contexts from {base_path}")

    # 选 n_ctx 个 context（训练集下标）；若给定 base_context_ids，则在其基础上扩展为超集
    if n_train <= args.n_ctx:
        context_indices = list(range(n_train))
    else:
        if not base_context_ids:
            context_indices = rng.sample(range(n_train), args.n_ctx)
        else:
            if len(base_context_ids) > args.n_ctx:
                raise ValueError(
                    f"base_config has {len(base_context_ids)} unique contexts, "
                    f"which exceeds requested n_ctx={args.n_ctx}"
                )
            remaining = [i for i in range(n_train) if i not in base_context_ids]
            need = args.n_ctx - len(base_context_ids)
            extra = rng.sample(remaining, need) if need > 0 else []
            context_indices = sorted(set(base_context_ids + extra))

    context_indices.sort()

    # Optional: build model + dataset for initial eNTK similarity
    model = None
    cfg = None
    train_ds = None
    device = torch.device("cpu")
    if args.with_entk:
        model, cfg, train_ds = build_model_and_dataset(data_dir)
        device = next(model.parameters()).device

    os.makedirs(CONFIG_DIR_DEFAULT, exist_ok=True)
    out_path = os.path.join(CONFIG_DIR_DEFAULT, args.out)

    rows: list[dict[str, str | int | float]] = []
    for ctx_id in context_indices:
        # 每个 context 三条：related_rare, unrelated_rare, wrong
        # 启发式：related_rare / unrelated_rare 用不同长尾 code；wrong 用常见或随机
        rel_rare = rng.choice(rare_codes) if rare_codes else 0
        unrel_rare = rng.choice([c for c in rare_codes if c != rel_rare] or rare_codes) if rare_codes else 1
        wrong_code = rng.choice(common_codes) if common_codes else (rel_rare + 1) % code_vocab_size

        for disease_id, tag in [
            (rel_rare, "related_rare"),
            (unrel_rare, "unrelated_rare"),
            (wrong_code, "wrong"),
        ]:
            row: dict[str, str | int | float] = {
                "context_id": ctx_id,
                "disease_id": disease_id,
                "type": tag,
            }
            if args.with_entk and model is not None and train_ds is not None:
                ehr_np, _ = train_ds[ctx_id]
                sim = compute_entk_sim(model, cfg, ehr_np, int(disease_id), device)
                row["initial_entk_sim"] = sim
            rows.append(row)

    fieldnames = ["context_id", "disease_id", "type"]
    if args.with_entk:
        fieldnames.append("initial_entk_sim")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[micro-probe] Wrote {len(rows)} rows -> {out_path}")
    print(f"  contexts = {len(context_indices)}, code_vocab_size = {code_vocab_size}")


if __name__ == "__main__":
    main()

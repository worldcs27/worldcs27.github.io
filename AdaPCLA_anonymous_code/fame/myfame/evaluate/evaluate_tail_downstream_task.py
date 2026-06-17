#!/usr/bin/env python3
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import csv
import json
import os
import pickle
import random
from dataclasses import dataclass

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
        import sklearn  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing dependencies (torch/sklearn). Run with the conda env that has them, e.g.\n"
            "  sys.executable fame/myfame/evaluate/evaluate_tail_downstream_task.py ...\n"
            f"Original import error: {type(e).__name__}: {e}"
        )


_require_torch()
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


DEFAULT_DATA_DIR = "FAME_ROOT"
DEFAULT_REAL_TRAIN_PATH = os.path.join(DEFAULT_DATA_DIR, "trainDataset.pkl")
DEFAULT_REAL_TEST_PATH = os.path.join(DEFAULT_DATA_DIR, "testDataset.pkl")
DEFAULT_SYN_PATH = "FAME_ROOT"
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_COUNTS_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_counts.csv"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/tail_downstream"


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _infer_vocab_size(data_dir: str) -> int:
    p = os.path.join(data_dir, "codeToIndex.pkl")
    return int(len(_load_pkl(p)))


def _load_bucket_csv(bucket_csv: str):
    buckets = {"tail": [], "mid": [], "head": []}
    with open(bucket_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            b = (row.get("bucket") or "").strip()
            if b not in buckets:
                continue
            buckets[b].append(int(row["code_id"]))
    tail_ids = sorted(set(int(x) for x in buckets["tail"]))
    tail_id_to_out = {cid: i for i, cid in enumerate(tail_ids)}
    return buckets, tail_ids, tail_id_to_out


def _load_counts_csv(counts_csv: str) -> dict[int, int]:
    out = {}
    with open(counts_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row["code_id"])
            except Exception:
                continue
            try:
                c = int(float(row.get("train_visit_count", 0) or 0))
            except Exception:
                c = 0
            out[cid] = c
    return out


def _select_tail_codes_by_count(train_visit_count: dict[int, int], *, min_c: int, max_c: int) -> list[int]:
    min_c = int(min_c)
    max_c = int(max_c)
    if min_c < 1:
        min_c = 1
    if max_c < min_c:
        max_c = min_c
    return sorted([cid for cid, c in train_visit_count.items() if min_c <= int(c) <= max_c])


def _iter_visits(dataset):
    for p in dataset:
        for v in (p.get("visits") or []):
            yield v


def _visit_codes(v, *, code_vocab_size: int) -> set[int]:
    s = set()
    for c in v or []:
        try:
            cid = int(c)
        except Exception:
            continue
        if 0 <= cid < int(code_vocab_size):
            s.add(cid)
    return s


@dataclass
class Samples:
    x_codes: list[list[int]]
    y_tail: list[list[int]]  # tail output indices (0..tail_dim-1)
    n_visits_total: int
    n_visits_used: int
    n_pos_visits: int
    n_neg_visits: int

    def stats(self):
        x_lens = [len(x) for x in self.x_codes]
        y_lens = [len(y) for y in self.y_tail]
        def desc(v):
            if not v:
                return {"n": 0}
            a = np.asarray(v, dtype=np.float64)
            return {
                "n": int(a.size),
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)),
                "p99": float(np.percentile(a, 99)),
                "max": float(a.max()),
            }
        return {
            "n_visits_total": int(self.n_visits_total),
            "n_samples": int(self.n_visits_used),
            "n_pos_visits": int(self.n_pos_visits),
            "n_neg_visits": int(self.n_neg_visits),
            "x_len": desc(x_lens),
            "y_len": desc(y_lens),
        }


def _build_samples(
    dataset,
    *,
    code_vocab_size: int,
    tail_set: set[int],
    tail_id_to_out: dict[int, int],
    neg_ratio: float,
    max_pos: int | None,
    max_total: int | None,
    seed: int,
):
    rng = random.Random(int(seed))
    pos = []
    neg = []
    n_visits_total = 0
    for v in _iter_visits(dataset):
        n_visits_total += 1
        codes = _visit_codes(v, code_vocab_size=code_vocab_size)
        if not codes:
            continue
        y_ids = sorted(codes & tail_set)
        x_ids = sorted(codes - tail_set)
        if not x_ids:
            continue
        if y_ids:
            y = [tail_id_to_out[c] for c in y_ids if c in tail_id_to_out]
            if y:
                pos.append((x_ids, y))
        else:
            neg.append((x_ids, []))

    if max_pos is not None:
        rng.shuffle(pos)
        pos = pos[: int(max_pos)]

    # neg_ratio:
    # - <0: keep all negative visits (useful for evaluation)
    # - =0: keep no negatives
    # - >0: sample negatives at ratio * #pos
    if neg_ratio < 0:
        neg_keep = neg
    elif neg_ratio == 0:
        neg_keep = []
    else:
        k = int(round(float(neg_ratio) * len(pos)))
        rng.shuffle(neg)
        neg_keep = neg[:k]

    all_samples = pos + neg_keep
    rng.shuffle(all_samples)
    if max_total is not None:
        all_samples = all_samples[: int(max_total)]

    x_codes = [x for x, _ in all_samples]
    y_tail = [y for _, y in all_samples]

    return Samples(
        x_codes=x_codes,
        y_tail=y_tail,
        n_visits_total=int(n_visits_total),
        n_visits_used=int(len(all_samples)),
        n_pos_visits=int(len(pos)),
        n_neg_visits=int(len(neg_keep)),
    )


class TailDataset(torch.utils.data.Dataset):
    def __init__(self, x_codes: list[list[int]], y_tail: list[list[int]], tail_dim: int):
        self.x_codes = x_codes
        self.y_tail = y_tail
        self.tail_dim = int(tail_dim)

    def __len__(self):
        return len(self.x_codes)

    def __getitem__(self, i):
        return self.x_codes[i], self.y_tail[i]


def _collate(batch, *, tail_dim: int, device: torch.device):
    # EmbeddingBag: indices + offsets.
    flat = []
    offsets = [0]
    ys = torch.zeros((len(batch), int(tail_dim)), dtype=torch.float32)
    for i, (x_codes, y_tail) in enumerate(batch):
        flat.extend(x_codes)
        offsets.append(len(flat))
        for t in y_tail:
            if 0 <= int(t) < int(tail_dim):
                ys[i, int(t)] = 1.0
    indices = torch.tensor(flat, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1], dtype=torch.int64)
    return indices.to(device), offsets.to(device), ys.to(device)


class TailPredictor(nn.Module):
    def __init__(self, *, vocab_size: int, tail_dim: int, embed_dim: int, dropout: float):
        super().__init__()
        self.emb = nn.EmbeddingBag(int(vocab_size), int(embed_dim), mode="sum")
        self.drop = nn.Dropout(float(dropout))
        self.fc = nn.Linear(int(embed_dim), int(tail_dim))

    def forward(self, indices, offsets):
        x = self.emb(indices, offsets)
        x = self.drop(x)
        return self.fc(x)


def _compute_pos_weight(y_tail: list[list[int]], tail_dim: int, cap: float = 50.0) -> torch.Tensor:
    n = int(len(y_tail))
    pos = np.zeros((int(tail_dim),), dtype=np.int64)
    for ys in y_tail:
        for t in ys:
            if 0 <= int(t) < int(tail_dim):
                pos[int(t)] += 1
    neg = n - pos
    w = (neg.astype(np.float64) / np.maximum(1.0, pos.astype(np.float64))).clip(1.0, float(cap))
    return torch.tensor(w, dtype=torch.float32)


def _predict_proba(model: nn.Module, loader, *, device: torch.device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for indices, offsets, _ in loader:
            logits = model(indices, offsets)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            out.append(probs)
    return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)


def _build_y_true(y_tail: list[list[int]], tail_dim: int) -> np.ndarray:
    y = np.zeros((len(y_tail), int(tail_dim)), dtype=np.uint8)
    for i, ys in enumerate(y_tail):
        for t in ys:
            if 0 <= int(t) < int(tail_dim):
                y[i, int(t)] = 1
    return y


def _macro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    # Average over labels that have positives in y_true.
    if y_true.size == 0:
        return None
    ap_list = []
    for j in range(y_true.shape[1]):
        if int(y_true[:, j].sum()) == 0:
            continue
        ap_list.append(float(average_precision_score(y_true[:, j], y_score[:, j])))
    return float(np.mean(ap_list)) if ap_list else None


def _micro_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0:
        return None
    return float(average_precision_score(y_true.ravel(), y_score.ravel()))


def _recall_precision_at_k(y_true: np.ndarray, y_score: np.ndarray, ks=(5, 10, 20)):
    # Evaluate only on samples with at least one true tail label.
    pos_rows = np.where(y_true.sum(axis=1) > 0)[0]
    out = {}
    if pos_rows.size == 0:
        for k in ks:
            out[str(int(k))] = {"n": 0, "recall": None, "precision": None}
        return out
    y_true_pos = y_true[pos_rows]
    y_score_pos = y_score[pos_rows]
    for k in ks:
        k = int(k)
        topk = np.argpartition(-y_score_pos, kth=min(k - 1, y_score_pos.shape[1] - 1), axis=1)[:, :k]
        rec = []
        prec = []
        for i in range(y_true_pos.shape[0]):
            true_set = set(np.where(y_true_pos[i] > 0)[0].tolist())
            pred_set = set(topk[i].tolist())
            hit = len(true_set & pred_set)
            rec.append(hit / max(1, len(true_set)))
            prec.append(hit / max(1, k))
        out[str(k)] = {"n": int(y_true_pos.shape[0]), "recall": float(np.mean(rec)), "precision": float(np.mean(prec))}
    return out


def _train_and_eval(
    *,
    train_samples: Samples,
    test_samples: Samples,
    vocab_size: int,
    tail_dim: int,
    embed_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))

    model = TailPredictor(vocab_size=vocab_size, tail_dim=tail_dim, embed_dim=embed_dim, dropout=dropout).to(device)
    pos_weight = _compute_pos_weight(train_samples.y_tail, tail_dim).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    train_ds = TailDataset(train_samples.x_codes, train_samples.y_tail, tail_dim)
    test_ds = TailDataset(test_samples.x_codes, test_samples.y_tail, tail_dim)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate(b, tail_dim=tail_dim, device=device),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=lambda b: _collate(b, tail_dim=tail_dim, device=device),
    )

    for _ in range(int(epochs)):
        model.train()
        for indices, offsets, y in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(indices, offsets)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

    y_true = _build_y_true(test_samples.y_tail, tail_dim)
    y_score = _predict_proba(model, test_loader, device=device)
    metrics = {
        "micro_auprc_all": _micro_auprc(y_true, y_score),
        "macro_auprc_all": _macro_auprc(y_true, y_score),
        "recall_precision_at_k_pos_visits": _recall_precision_at_k(y_true, y_score, ks=(5, 10, 20)),
        "test_pos_visits": int((y_true.sum(axis=1) > 0).sum()),
        "test_samples": int(y_true.shape[0]),
        "tail_dim": int(tail_dim),
    }
    return metrics


def _freq_baseline_metrics(*, train_samples: Samples, test_samples: Samples, tail_dim: int):
    # Constant per-label score = training frequency.
    y_train = _build_y_true(train_samples.y_tail, tail_dim)
    freq = y_train.mean(axis=0).astype(np.float64)
    y_true = _build_y_true(test_samples.y_tail, tail_dim)
    y_score = np.tile(freq[None, :], (y_true.shape[0], 1))
    return {
        "micro_auprc_all": _micro_auprc(y_true, y_score),
        "macro_auprc_all": _macro_auprc(y_true, y_score),
        "recall_precision_at_k_pos_visits": _recall_precision_at_k(y_true, y_score, ks=(5, 10, 20)),
        "test_pos_visits": int((y_true.sum(axis=1) > 0).sum()),
        "test_samples": int(y_true.shape[0]),
        "tail_dim": int(tail_dim),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--counts_csv", default=DEFAULT_COUNTS_CSV, help="mimiciii_code_counts.csv (train_visit_count per code_id)")
    ap.add_argument("--tail_def", default="count", choices=["count", "bucket"])
    ap.add_argument("--tail_min_count", type=int, default=2, help="Used when --tail_def=count")
    ap.add_argument("--tail_max_count", type=int, default=10, help="Used when --tail_def=count")
    ap.add_argument("--real_train_path", default=DEFAULT_REAL_TRAIN_PATH)
    ap.add_argument("--real_test_path", default=DEFAULT_REAL_TEST_PATH)
    ap.add_argument("--synthetic_train_path", default=DEFAULT_SYN_PATH)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--neg_ratio", type=float, default=5.0, help="Sample neg visits at ratio * #pos visits.")
    ap.add_argument("--max_pos", type=int, default=None)
    ap.add_argument("--max_total", type=int, default=None)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    device = torch.device("cuda" if (args.device in ["auto", "cuda"] and torch.cuda.is_available()) else "cpu")

    vocab_size = _infer_vocab_size(args.data_dir)
    train_visit_count = _load_counts_csv(args.counts_csv) if args.tail_def == "count" else None
    if args.tail_def == "count":
        tail_ids = _select_tail_codes_by_count(train_visit_count, min_c=args.tail_min_count, max_c=args.tail_max_count)
        buckets = {"tail": tail_ids, "mid": [], "head": []}
    else:
        buckets, tail_ids, _ = _load_bucket_csv(args.bucket_csv)
    tail_id_to_out = {cid: i for i, cid in enumerate(tail_ids)}
    tail_set = set(tail_ids)
    tail_dim = int(len(tail_ids))

    real_train = _load_pkl(args.real_train_path)
    real_test = _load_pkl(args.real_test_path)
    syn_train = _load_pkl(args.synthetic_train_path)

    train_real_samples = _build_samples(
        real_train,
        code_vocab_size=vocab_size,
        tail_set=tail_set,
        tail_id_to_out=tail_id_to_out,
        neg_ratio=args.neg_ratio,
        max_pos=args.max_pos,
        max_total=args.max_total,
        seed=args.seed,
    )
    train_syn_samples = _build_samples(
        syn_train,
        code_vocab_size=vocab_size,
        tail_set=tail_set,
        tail_id_to_out=tail_id_to_out,
        neg_ratio=args.neg_ratio,
        max_pos=args.max_pos,
        max_total=args.max_total,
        seed=args.seed,
    )
    test_samples = _build_samples(
        real_test,
        code_vocab_size=vocab_size,
        tail_set=tail_set,
        tail_id_to_out=tail_id_to_out,
        neg_ratio=-1.0,  # evaluation on all usable visits (include all negatives)
        max_pos=None,
        max_total=None,
        seed=args.seed,
    )

    summary = {
        "task": "Tail code prediction (given non-tail codes in same visit)",
        "paths": {
            "bucket_csv": args.bucket_csv,
            "counts_csv": args.counts_csv,
            "real_train_path": args.real_train_path,
            "real_test_path": args.real_test_path,
            "synthetic_train_path": args.synthetic_train_path,
        },
        "vocab_size": int(vocab_size),
        "tail_dim": int(tail_dim),
        "tail_codes": int(len(tail_set)),
        "tail_def": {
            "mode": args.tail_def,
            "min_count": int(args.tail_min_count) if args.tail_def == "count" else None,
            "max_count": int(args.tail_max_count) if args.tail_def == "count" else None,
        },
        "buckets": {k: int(len(v)) for k, v in buckets.items()},
        "device": str(device),
        "params": {
            "seed": int(args.seed),
            "neg_ratio": float(args.neg_ratio),
            "max_pos": args.max_pos,
            "max_total": args.max_total,
            "embed_dim": int(args.embed_dim),
            "dropout": float(args.dropout),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "datasets": {
            "train_real": train_real_samples.stats(),
            "train_synthetic": train_syn_samples.stats(),
            "test_real": test_samples.stats(),
        },
    }

    summary["baseline_freq_train_real"] = _freq_baseline_metrics(
        train_samples=train_real_samples, test_samples=test_samples, tail_dim=tail_dim
    )
    summary["model_train_real"] = _train_and_eval(
        train_samples=train_real_samples,
        test_samples=test_samples,
        vocab_size=vocab_size,
        tail_dim=tail_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
    )
    summary["model_train_synthetic"] = _train_and_eval(
        train_samples=train_syn_samples,
        test_samples=test_samples,
        vocab_size=vocab_size,
        tail_dim=tail_dim,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
    )

    out_json = os.path.join(args.out_dir, "tail_downstream_summary.json")
    _write_json(summary, out_json)
    print(f"Wrote: {out_json}")
    print("Key metrics (higher is better):")
    for k in ["baseline_freq_train_real", "model_train_real", "model_train_synthetic"]:
        m = summary[k]
        print(
            f"- {k}: microAUPRC={m['micro_auprc_all']:.6f} macroAUPRC={m['macro_auprc_all'] if m['macro_auprc_all'] is not None else None} "
            f"R@10={m['recall_precision_at_k_pos_visits']['10']['recall']}"
        )


if __name__ == "__main__":
    main()

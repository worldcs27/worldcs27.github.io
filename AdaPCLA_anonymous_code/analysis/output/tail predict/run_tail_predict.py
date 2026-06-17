#!/usr/bin/env python3
"""
Tail-code prediction (task ii): given non-tail codes in a visit, predict tail code occurrence.
Train downstream classifier on synthetic data (one model per run), evaluate on real test set:
AUPRC (macro over tail codes), Top-K recall.
Runs on GPU when available (PyTorch). Results in output/.
Usage: python run_tail_predict.py [--dataset mimic3|mimic4] [--model MODEL] [--max_tail 500]
  Default: run all 6 models x 2 datasets (12 jobs).
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import DATASETS, MODELS, OUT_DIR, PCLA_ROOT, TOP_K

try:
    from sklearn.metrics import average_precision_score
except ImportError:
    average_precision_score = None


def load_buckets(bucket_csv: Path) -> tuple[set[int], list[int], int]:
    """Return (non_tail_ids, tail_id_list, vocab_size). tail_id_list preserves order for Y indices."""
    head_mid = set()
    tail_list = []
    max_code = -1
    with open(bucket_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            code_id = int(row["code_id"])
            b = row["bucket"].strip().lower()
            max_code = max(max_code, code_id)
            if b == "tail":
                tail_list.append(code_id)
            else:
                head_mid.add(code_id)
    non_tail = head_mid
    vocab_size = max_code + 1
    return non_tail, tail_list, vocab_size


def load_pkl(path: Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def visits_with_tail(
    data: list[dict],
    non_tail_set: set[int],
    tail_list: list[int],
    tail_set: set[int],
    vocab_size: int,
    n_tail: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten patients to visits; keep only visits with at least one tail code.
    X: (N, vocab_size) multi-hot of non-tail codes in visit (tail positions 0).
    Y: (N, n_tail) multi-hot of tail codes in visit.
    """
    tail_to_idx = {c: i for i, c in enumerate(tail_list)}
    X_list = []
    Y_list = []
    for p in data:
        for v in p.get("visits", []):
            if not v:
                continue
            v_set = set(v)
            tail_in_v = v_set & tail_set
            if not tail_in_v:
                continue
            x = np.zeros(vocab_size, dtype=np.float32)
            for c in v:
                if c in non_tail_set:
                    x[c] = 1.0
            y = np.zeros(n_tail, dtype=np.float32)
            for c in tail_in_v:
                y[tail_to_idx[c]] = 1.0
            X_list.append(x)
            Y_list.append(y)
    if not X_list:
        return np.zeros((0, vocab_size), dtype=np.float32), np.zeros((0, n_tail), dtype=np.float32)
    return np.stack(X_list), np.stack(Y_list)


def get_vocab_size_from_data(data: list[dict]) -> int:
    max_code = -1
    for p in data:
        for v in p.get("visits", []):
            for c in v:
                max_code = max(max_code, c)
    return max_code + 1


class TailMLP(nn.Module):
    """Single MLP outputting n_tail logits (multi-label). Input (N, vocab_size) -> (N, n_tail)."""

    def __init__(self, vocab_size: int, n_tail: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vocab_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_tail),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_and_eval(
    dataset_key: str,
    model_key: str,
    synth_train_path: Path,
    real_test_path: Path,
    bucket_csv: Path,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 256,
    hidden: int = 256,
    max_tail: int | None = None,
) -> dict:
    """Train on synthetic data (GPU), evaluate on real test. Return dict with auprc_macro, topk_recall."""
    non_tail_set, tail_list, _ = load_buckets(bucket_csv)
    if max_tail is not None:
        tail_list = tail_list[:max_tail]
    tail_set = set(tail_list)
    n_tail = len(tail_list)

    if not synth_train_path.exists():
        return {"error": f"synthetic path not found: {synth_train_path}"}
    train_data = load_pkl(synth_train_path)
    vocab_size = get_vocab_size_from_data(train_data)
    _, _, vocab_from_bucket = load_buckets(bucket_csv)
    vocab_size = max(vocab_size, vocab_from_bucket)

    X_train, Y_train = visits_with_tail(train_data, non_tail_set, tail_list, tail_set, vocab_size, n_tail)
    if X_train.shape[0] == 0:
        return {"error": "no training visits with tail codes"}

    test_data = load_pkl(real_test_path)
    X_test, Y_test = visits_with_tail(test_data, non_tail_set, tail_list, tail_set, vocab_size, n_tail)
    if X_test.shape[0] == 0:
        return {"error": "no test visits with tail codes"}

    # Mask: columns (tail codes) that have no positive in train -> don't contribute to loss
    col_has_pos = (Y_train.sum(axis=0) > 0).astype(np.float32)
    if col_has_pos.sum() == 0:
        return {"error": "no positive labels in training for any tail code"}

    X_tr = torch.from_numpy(X_train).to(device)
    Y_tr = torch.from_numpy(Y_train).to(device)
    mask_tr = torch.from_numpy(col_has_pos).to(device)
    dataset = TensorDataset(X_tr, Y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    net = TailMLP(vocab_size, n_tail, hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    n_active = max(1, int(col_has_pos.sum()))

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = net(xb)
            loss_per = loss_fn(logits, yb) * mask_tr.unsqueeze(0)
            loss = loss_per.sum() / (xb.size(0) * n_active)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        X_te = torch.from_numpy(X_test).to(device)
        logits_te = net(X_te)
        proba = torch.sigmoid(logits_te).cpu().numpy()

    # AUPRC macro (over tail codes that appear in test)
    ap_scores = []
    for j in range(n_tail):
        if Y_test[:, j].sum() == 0:
            continue
        ap_scores.append(average_precision_score(Y_test[:, j], proba[:, j]))
    auprc_macro = float(np.mean(ap_scores)) if ap_scores else 0.0

    recalls = []
    for i in range(len(Y_test)):
        true_set = set(np.where(Y_test[i] > 0.5)[0])
        if not true_set:
            continue
        top_k_idx = np.argsort(-proba[i])[:TOP_K]
        pred_set = set(top_k_idx)
        rec = len(true_set & pred_set) / len(true_set)
        recalls.append(rec)
    topk_recall = float(np.mean(recalls)) if recalls else 0.0

    return {
        "dataset": dataset_key,
        "model": model_key,
        "auprc_macro": auprc_macro,
        f"top{TOP_K}_recall": topk_recall,
    }


def main():
    parser = argparse.ArgumentParser(description="Tail-code prediction: train on synthetic (GPU), eval on real")
    parser.add_argument("--dataset", choices=["mimic3", "mimic4", "all"], default="all")
    parser.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max_tail", type=int, default=None, help="Cap tail codes (default: all)")
    parser.add_argument("--device", type=str, default="", help="cuda or cpu; default: auto (cuda if available)")
    parser.add_argument("--out_suffix", type=str, default="", help="Suffix for output files (e.g. quick -> summary_quick.csv)")
    args = parser.parse_args()

    if average_precision_score is None:
        print("Need sklearn. Install: pip install scikit-learn")
        sys.exit(1)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset != "all" else list(DATASETS.keys())
    models = [args.model] if args.model != "all" else list(MODELS.keys())

    results = []
    suf = f"_{args.out_suffix}" if args.out_suffix else ""
    out_csv = OUT_DIR / f"tail_predict_summary{suf}.csv"
    out_all = OUT_DIR / f"tail_predict_all{suf}.jsonl"
    if args.dataset == "all" and args.model == "all":
        out_csv.write_text("", encoding="utf-8")
        out_all.write_text("", encoding="utf-8")
    csv_header_written = out_csv.exists() and out_csv.stat().st_size > 0

    for dk in datasets:
        cfg = DATASETS[dk]
        real_test = cfg["real_test"]
        bucket_csv = cfg["bucket_csv"]
        if not real_test.exists() or not bucket_csv.exists():
            print(f"Skip {dk}: missing real_test or bucket_csv")
            continue
        for mk in models:
            synth_path = MODELS[mk].get(dk)
            if not synth_path:
                continue
            print(f"Running {dk} / {mk} ...")
            res = train_and_eval(
                dk, mk, synth_path, real_test, bucket_csv,
                device=device,
                epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
                max_tail=args.max_tail,
            )
            if "error" in res:
                print(f"  Error: {res['error']}")
            else:
                print(f"  AUPRC macro = {res['auprc_macro']:.4f}, Top{TOP_K} recall = {res[f'top{TOP_K}_recall']:.4f}")
            results.append(res)

            with open(out_all, "a", encoding="utf-8") as f:
                f.write(str(res) + "\n")
            if "error" not in res:
                with open(out_csv, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["dataset", "model", "auprc_macro", f"top{TOP_K}_recall"])
                    if not csv_header_written:
                        w.writeheader()
                        csv_header_written = True
                    w.writerow(res)
                print(f"  Appended to {out_csv}")

    if results:
        print(f"Done. Total: {len(results)} jobs, {len([r for r in results if 'error' not in r])} ok.")


if __name__ == "__main__":
    main()

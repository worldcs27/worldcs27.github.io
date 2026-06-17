#!/usr/bin/env python3
"""
Evaluate distribution controllability: how well a synthetic dataset matches a target prior.

We treat pi(c) as the probability that code c appears in a visit (visit-level, de-duplicated).
Given a synthetic dataset (list of {"visits": [[code_ids], ...]} dicts) and a pi_target vector,
we compute pi_syn and report alignment metrics.
"""

from __future__ import annotations
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
from dataclasses import dataclass

import numpy as np


DEFAULT_SYN_PATH = "FAME_ROOT"
DEFAULT_TARGET_PRIOR_PATH = ""
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/distribution_control"


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return (x + eps) / (s + eps * x.size)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log((p + 1e-12) / (m + 1e-12))))
    kl_qm = float(np.sum(q * np.log((q + 1e-12) / (m + 1e-12))))
    return 0.5 * (kl_pm + kl_qm)


def _rankdata_with_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return None
    rx = _rankdata_with_ties(x)
    ry = _rankdata_with_ties(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return None
    return float(np.sum(rx * ry) / denom)


@dataclass(frozen=True)
class Buckets:
    tail: set[int]
    mid: set[int]
    head: set[int]


def _load_buckets(bucket_csv: str) -> Buckets:
    tail: set[int] = set()
    mid: set[int] = set()
    head: set[int] = set()
    with open(bucket_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                cid = int(row.get("code_id") or row.get("code") or row.get("id"))
            except Exception:
                continue
            b = (row.get("bucket") or "").strip().lower()
            if b == "tail":
                tail.add(cid)
            elif b == "mid":
                mid.add(cid)
            elif b == "head":
                head.add(cid)
    return Buckets(tail=tail, mid=mid, head=head)


def _collect_visit_counts(dataset, *, code_vocab_size: int) -> tuple[np.ndarray, int]:
    total_visits = 0
    visit_counts = np.zeros((code_vocab_size,), dtype=np.int64)
    for p in dataset:
        visits = p.get("visits", [])
        total_visits += int(len(visits))
        for v in visits:
            if not v:
                continue
            for c in set(v):
                ci = int(c)
                if 0 <= ci < code_vocab_size:
                    visit_counts[ci] += 1
    return visit_counts, int(total_visits)


def _bucket_summary(pi: np.ndarray, ids: set[int]) -> dict:
    xs = [float(pi[i]) for i in ids if 0 <= i < pi.size]
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_path", default=DEFAULT_SYN_PATH)
    ap.add_argument("--target_prior_path", required=True, help="Path to pi_target .npy (len=code_vocab_size, visit-level marginal)")
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--mask_mode", choices=["all", "nonzero_union", "target_nonzero"], default="nonzero_union")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    syn = _load_pkl(args.synthetic_path)
    pi_t = np.load(args.target_prior_path).astype(np.float64).reshape(-1)
    code_vocab_size = int(pi_t.size)

    syn_visit_counts, total_visits = _collect_visit_counts(syn, code_vocab_size=code_vocab_size)
    pi_syn = syn_visit_counts.astype(np.float64) / max(1.0, float(total_visits))

    if args.mask_mode == "all":
        mask = np.ones_like(pi_t, dtype=bool)
    elif args.mask_mode == "target_nonzero":
        mask = pi_t > 0
    else:
        mask = (pi_t > 0) | (pi_syn > 0)

    diff = pi_syn - pi_t
    mae_all = float(np.mean(np.abs(diff)))
    rmse_all = float(np.sqrt(np.mean(diff * diff)))
    mae_masked = float(np.mean(np.abs(diff[mask]))) if np.any(mask) else None
    rmse_masked = float(np.sqrt(np.mean(diff[mask] * diff[mask]))) if np.any(mask) else None
    sp = _spearman_corr(np.log1p(pi_t[mask]), np.log1p(pi_syn[mask])) if np.any(mask) else None

    # Treat prevalence profile as a distribution over codes by normalizing counts.
    target_counts_like = np.clip(pi_t, 0.0, 1.0) * float(total_visits)
    js_profile = float(_js_divergence(target_counts_like, syn_visit_counts.astype(np.float64)))

    buckets = _load_buckets(args.bucket_csv)
    out = {
        "paths": {"synthetic_path": args.synthetic_path, "target_prior_path": args.target_prior_path, "bucket_csv": args.bucket_csv},
        "code_vocab_size": int(code_vocab_size),
        "synthetic_total_visits": int(total_visits),
        "mask_mode": args.mask_mode,
        "metrics": {
            "mae_all": mae_all,
            "rmse_all": rmse_all,
            "mae_masked": mae_masked,
            "rmse_masked": rmse_masked,
            "spearman_log1p_masked": sp,
            "js_prevalence_profile": js_profile,
        },
        "bucket_stats": {
            "target": {
                "tail": _bucket_summary(pi_t, buckets.tail),
                "mid": _bucket_summary(pi_t, buckets.mid),
                "head": _bucket_summary(pi_t, buckets.head),
            },
            "synthetic": {
                "tail": _bucket_summary(pi_syn, buckets.tail),
                "mid": _bucket_summary(pi_syn, buckets.mid),
                "head": _bucket_summary(pi_syn, buckets.head),
            },
        },
    }

    out_path = os.path.join(args.out_dir, "distribution_control_summary.json")
    _write_json(out, out_path)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


DEFAULT_SYN_PATH = "FAME_ROOT"
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_DATA_DIR = "FAME_ROOT"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/generation_distribution"


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _desc(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return (x + eps) / (s + eps * x.size)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # Jensen-Shannon divergence (natural log).
    p = _normalize(p)
    q = _normalize(q)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log((p + 1e-12) / (m + 1e-12))))
    kl_qm = float(np.sum(q * np.log((q + 1e-12) / (m + 1e-12))))
    return 0.5 * (kl_pm + kl_qm)


def _rankdata_with_ties(x: np.ndarray) -> np.ndarray:
    # Average rank for ties (1..n).
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


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
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


def _write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_rank_frequency_csv(counts: np.ndarray, out_csv: str):
    freq = [int(c) for c in counts.tolist() if int(c) > 0]
    freq.sort(reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "count"])
        w.writeheader()
        for r, c in enumerate(freq, start=1):
            w.writerow({"rank": int(r), "count": int(c)})


def _plot_rank_frequency_overlay(train_counts: np.ndarray, syn_counts: np.ndarray, out_png: str, title: str):
    # Optional: only used if matplotlib exists.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def prep(x):
        v = [int(c) for c in x.tolist() if int(c) > 0]
        v.sort(reverse=True)
        return v

    t = prep(train_counts)
    s = prep(syn_counts)
    if not t or not s:
        return {"ok": False, "reason": "empty counts"}

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(t) + 1), t, label="Real(train)", linewidth=1.2)
    plt.plot(range(1, len(s) + 1), s, label="Synthetic", linewidth=1.2)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("rank (log)")
    plt.ylabel("count (log)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return {"ok": True, "png": out_png}


def _load_bucket_csv(bucket_csv: str):
    buckets = {"tail": [], "mid": [], "head": []}
    train_visit = {}
    train_patient = {}
    code_name = {}
    with open(bucket_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            b = (row.get("bucket") or "").strip()
            if b not in buckets:
                continue
            cid = int(row["code_id"])
            buckets[b].append(cid)
            code_name[cid] = row.get("code", "")
            train_visit[cid] = int(float(row.get("train_visit_count", 0) or 0))
            train_patient[cid] = int(float(row.get("train_patient_count", 0) or 0))
    return buckets, train_visit, train_patient, code_name


def _infer_code_vocab_size(data_dir: str, bucket_code_ids: list[int]) -> int:
    code_pkl = os.path.join(data_dir, "codeToIndex.pkl")
    if os.path.exists(code_pkl):
        try:
            return int(len(_load_pkl(code_pkl)))
        except Exception:
            pass
    return int(max(bucket_code_ids) + 1 if bucket_code_ids else 0)


def _collect_dataset_stats(dataset, *, code_vocab_size: int, code_to_bucket: dict[int, str] | None = None):
    visit_count = Counter()
    patient_count = Counter()
    total_patients = 0
    total_visits = 0
    total_code_mentions = 0

    visit_len_list = []
    codes_per_visit_list = []
    unique_codes_per_patient_list = []

    bucket_codes_per_visit = defaultdict(list)
    bucket_unique_codes_per_patient = defaultdict(list)

    for p in dataset:
        total_patients += 1
        visits = p.get("visits", [])
        total_visits += int(len(visits))
        visit_len_list.append(int(len(visits)))

        seen_patient = set()
        seen_patient_bucket = defaultdict(set)
        for v in visits:
            if not v:
                codes_per_visit_list.append(0)
                if code_to_bucket is not None:
                    for b in ["tail", "mid", "head"]:
                        bucket_codes_per_visit[b].append(0)
                continue
            uniq = set()
            for c in v:
                try:
                    cid = int(c)
                except Exception:
                    continue
                if 0 <= cid < int(code_vocab_size):
                    uniq.add(cid)
            codes_per_visit_list.append(int(len(uniq)))
            total_code_mentions += int(len(uniq))
            if code_to_bucket is not None:
                seen_visit_bucket = defaultdict(set)
                for cid in uniq:
                    b = code_to_bucket.get(cid)
                    if b:
                        seen_visit_bucket[b].add(cid)
                        seen_patient_bucket[b].add(cid)
                for b in ["tail", "mid", "head"]:
                    bucket_codes_per_visit[b].append(int(len(seen_visit_bucket[b])))
            for cid in uniq:
                visit_count[cid] += 1
                seen_patient.add(cid)
        unique_codes_per_patient_list.append(int(len(seen_patient)))
        if code_to_bucket is not None:
            for b in ["tail", "mid", "head"]:
                bucket_unique_codes_per_patient[b].append(int(len(seen_patient_bucket[b])))
        for cid in seen_patient:
            patient_count[cid] += 1

    out = {
        "visit_count": visit_count,
        "patient_count": patient_count,
        "total_patients": int(total_patients),
        "total_visits": int(total_visits),
        "total_code_mentions": int(total_code_mentions),
        "visit_len_list": visit_len_list,
        "codes_per_visit_list": codes_per_visit_list,
        "unique_codes_per_patient_list": unique_codes_per_patient_list,
    }
    if code_to_bucket is not None:
        out["bucket_codes_per_visit"] = {k: v for k, v in bucket_codes_per_visit.items()}
        out["bucket_unique_codes_per_patient"] = {k: v for k, v in bucket_unique_codes_per_patient.items()}
    return out


def _dense(counter: Counter, size: int) -> np.ndarray:
    x = np.zeros((size,), dtype=np.int64)
    for k, v in counter.items():
        if 0 <= int(k) < int(size):
            x[int(k)] = int(v)
    return x


@dataclass
class BucketMetrics:
    codes: int
    covered_codes: int
    coverage: float
    mass_frac: float
    patient_mass_frac: float
    median_count: float
    mean_count: float


def _bucket_metrics(bucket_ids: list[int], syn_visit_arr: np.ndarray, syn_patient_arr: np.ndarray):
    ids = np.asarray(bucket_ids, dtype=np.int64)
    if ids.size == 0:
        return BucketMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    v = syn_visit_arr[ids]
    p = syn_patient_arr[ids]
    covered = int(np.sum(v > 0))
    total_mass = float(np.sum(syn_visit_arr))
    total_patient_mass = float(np.sum(syn_patient_arr))
    mass_frac = float(np.sum(v) / total_mass) if total_mass > 0 else 0.0
    patient_mass_frac = float(np.sum(p) / total_patient_mass) if total_patient_mass > 0 else 0.0
    return BucketMetrics(
        codes=int(ids.size),
        covered_codes=covered,
        coverage=float(covered / ids.size),
        mass_frac=mass_frac,
        patient_mass_frac=patient_mass_frac,
        median_count=float(np.median(v)),
        mean_count=float(np.mean(v)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_path", default=DEFAULT_SYN_PATH)
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="For inferring code vocab size (codeToIndex.pkl)")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--compare_real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute distribution/length stats on a real dataset split for comparison.",
    )
    ap.add_argument(
        "--real_path",
        default=None,
        help="Path to real dataset pkl (default: <data_dir>/trainDataset.pkl when --compare_real).",
    )
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    buckets, train_visit_map, train_patient_map, code_name = _load_bucket_csv(args.bucket_csv)
    all_bucket_ids = buckets["tail"] + buckets["mid"] + buckets["head"]
    code_vocab_size = _infer_code_vocab_size(args.data_dir, all_bucket_ids)

    code_to_bucket = {}
    for b in ["tail", "mid", "head"]:
        for cid in buckets[b]:
            code_to_bucket[int(cid)] = b

    syn = _load_pkl(args.synthetic_path)
    syn_stats = _collect_dataset_stats(syn, code_vocab_size=code_vocab_size, code_to_bucket=code_to_bucket)

    syn_visit_arr = _dense(syn_stats["visit_count"], code_vocab_size)
    syn_patient_arr = _dense(syn_stats["patient_count"], code_vocab_size)

    train_visit_arr = np.zeros((code_vocab_size,), dtype=np.int64)
    train_patient_arr = np.zeros((code_vocab_size,), dtype=np.int64)
    for cid, c in train_visit_map.items():
        if 0 <= int(cid) < code_vocab_size:
            train_visit_arr[int(cid)] = int(c)
    for cid, c in train_patient_map.items():
        if 0 <= int(cid) < code_vocab_size:
            train_patient_arr[int(cid)] = int(c)

    # Global divergence and correlation (on codes present in buckets).
    ids_present = np.asarray(sorted(set(all_bucket_ids)), dtype=np.int64)
    tv = train_visit_arr[ids_present]
    sv = syn_visit_arr[ids_present]
    tp = train_patient_arr[ids_present]
    sp = syn_patient_arr[ids_present]

    train_present_mask = train_visit_arr > 0
    syn_present_mask = syn_visit_arr > 0
    syn_oov_mask = syn_present_mask & (~train_present_mask)
    syn_total_visit_mass = float(np.sum(syn_visit_arr))
    syn_total_patient_mass = float(np.sum(syn_patient_arr))

    summary = {
        "paths": {"synthetic_path": args.synthetic_path, "bucket_csv": args.bucket_csv},
        "code_vocab_size": int(code_vocab_size),
        "synthetic_total_patients": int(syn_stats["total_patients"]),
        "synthetic_total_visits": int(syn_stats["total_visits"]),
        "synthetic_total_code_mentions": int(syn_stats["total_code_mentions"]),
        "synthetic_present_codes": int(np.sum(syn_visit_arr > 0)),
        "train_present_codes_in_bucket_file": int(len(ids_present)),
        "synthetic_oov_codes_in_vocab": int(np.sum(syn_oov_mask)),
        "synthetic_oov_mass_frac_visit": float(np.sum(syn_visit_arr[syn_oov_mask]) / syn_total_visit_mass)
        if syn_total_visit_mass > 0
        else 0.0,
        "synthetic_oov_mass_frac_patient": float(np.sum(syn_patient_arr[syn_oov_mask]) / syn_total_patient_mass)
        if syn_total_patient_mass > 0
        else 0.0,
        "js_divergence_visit_present_codes": float(_js_divergence(tv, sv)),
        "js_divergence_patient_present_codes": float(_js_divergence(tp, sp)),
        "spearman_visit_present_codes": _spearman_corr(np.log1p(tv), np.log1p(sv)),
        "spearman_patient_present_codes": _spearman_corr(np.log1p(tp), np.log1p(sp)),
    }

    # Bucket metrics.
    bucket_out = {}
    for b in ["tail", "mid", "head"]:
        m = _bucket_metrics(buckets[b], syn_visit_arr, syn_patient_arr)
        mt = _bucket_metrics(buckets[b], train_visit_arr, train_patient_arr)
        bucket_out[b] = {
            "codes": m.codes,
            "covered_codes": m.covered_codes,
            "coverage": m.coverage,
            "mass_frac_visit": m.mass_frac,
            "mass_frac_patient": m.patient_mass_frac,
            "train_mass_frac_visit": mt.mass_frac,
            "train_mass_frac_patient": mt.patient_mass_frac,
            "mass_frac_visit_lift": (m.mass_frac / mt.mass_frac) if mt.mass_frac > 0 else None,
            "mass_frac_patient_lift": (m.patient_mass_frac / mt.patient_mass_frac) if mt.patient_mass_frac > 0 else None,
            "median_visit_count": m.median_count,
            "mean_visit_count": m.mean_count,
        }
        # Bucket-specific JS divergence against train (only within bucket).
        b_ids = np.asarray(buckets[b], dtype=np.int64)
        if b_ids.size:
            bucket_out[b]["js_divergence_visit_within_bucket"] = float(
                _js_divergence(train_visit_arr[b_ids], syn_visit_arr[b_ids])
            )
            bucket_out[b]["spearman_visit_within_bucket"] = _spearman_corr(
                np.log1p(train_visit_arr[b_ids]), np.log1p(syn_visit_arr[b_ids])
            )
            nz = syn_visit_arr[b_ids] > 0
            bucket_out[b]["covered_codes_syn_nonzero"] = int(np.sum(nz))
            bucket_out[b]["spearman_visit_within_bucket_syn_nonzero"] = (
                _spearman_corr(np.log1p(train_visit_arr[b_ids][nz]), np.log1p(syn_visit_arr[b_ids][nz]))
                if int(np.sum(nz)) >= 2
                else None
            )
            abs_log_ratio = np.abs(np.log((syn_visit_arr[b_ids].astype(np.float64) + 1.0) / (train_visit_arr[b_ids] + 1.0)))
            bucket_out[b]["median_abs_log_ratio_visit"] = float(np.median(abs_log_ratio))
            bucket_out[b]["mean_abs_log_ratio_visit"] = float(np.mean(abs_log_ratio))

    summary["buckets"] = bucket_out

    # Rare-code recall by train frequency threshold (visit-count definition).
    thresholds = [1, 2, 3, 5, 10]
    rare = {}
    for t in thresholds:
        m = (train_visit_arr > 0) & (train_visit_arr <= int(t))
        denom = int(np.sum(m))
        num = int(np.sum(syn_visit_arr[m] > 0)) if denom > 0 else 0
        rare[str(t)] = {
            "train_codes": denom,
            "covered_codes": num,
            "recall": float(num / denom) if denom > 0 else None,
        }
    summary["rare_code_recall_by_train_visit_count"] = rare

    # Simple distribution summaries to detect pathological generation.
    summary["synthetic_visit_len_stats"] = _desc(syn_stats["visit_len_list"])
    summary["synthetic_codes_per_visit_stats"] = _desc(syn_stats["codes_per_visit_list"])
    summary["synthetic_unique_codes_per_patient_stats"] = _desc(syn_stats["unique_codes_per_patient_list"])
    if "bucket_codes_per_visit" in syn_stats:
        summary["synthetic_bucket_codes_per_visit_stats"] = {
            b: _desc(syn_stats["bucket_codes_per_visit"].get(b, [])) for b in ["tail", "mid", "head"]
        }
    if "bucket_unique_codes_per_patient" in syn_stats:
        summary["synthetic_bucket_unique_codes_per_patient_stats"] = {
            b: _desc(syn_stats["bucket_unique_codes_per_patient"].get(b, [])) for b in ["tail", "mid", "head"]
        }

    if args.compare_real:
        real_path = args.real_path or os.path.join(args.data_dir, "trainDataset.pkl")
        if os.path.exists(real_path):
            real = _load_pkl(real_path)
            real_stats = _collect_dataset_stats(real, code_vocab_size=code_vocab_size, code_to_bucket=code_to_bucket)
            summary["real_path"] = real_path
            summary["real_total_patients"] = int(real_stats["total_patients"])
            summary["real_total_visits"] = int(real_stats["total_visits"])
            summary["real_total_code_mentions"] = int(real_stats["total_code_mentions"])
            summary["real_visit_len_stats"] = _desc(real_stats["visit_len_list"])
            summary["real_codes_per_visit_stats"] = _desc(real_stats["codes_per_visit_list"])
            summary["real_unique_codes_per_patient_stats"] = _desc(real_stats["unique_codes_per_patient_list"])
            if "bucket_codes_per_visit" in real_stats:
                summary["real_bucket_codes_per_visit_stats"] = {
                    b: _desc(real_stats["bucket_codes_per_visit"].get(b, [])) for b in ["tail", "mid", "head"]
                }
            if "bucket_unique_codes_per_patient" in real_stats:
                summary["real_bucket_unique_codes_per_patient_stats"] = {
                    b: _desc(real_stats["bucket_unique_codes_per_patient"].get(b, [])) for b in ["tail", "mid", "head"]
                }
        else:
            summary["real_path"] = real_path
            summary["real_stats_error"] = "real_path_not_found"

    _write_json(summary, os.path.join(args.out_dir, "generation_distribution_summary.json"))

    # Per-code CSV (present bucket codes only).
    per_code_csv = os.path.join(args.out_dir, "generation_distribution_per_code.csv")
    with open(per_code_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "code_id",
                "bucket",
                "code",
                "train_visit_count",
                "syn_visit_count",
                "train_patient_count",
                "syn_patient_count",
            ],
        )
        w.writeheader()
        for cid in ids_present.tolist():
            cid = int(cid)
            w.writerow(
                {
                    "code_id": cid,
                    "bucket": code_to_bucket.get(cid, ""),
                    "code": code_name.get(cid, ""),
                    "train_visit_count": int(train_visit_arr[cid]),
                    "syn_visit_count": int(syn_visit_arr[cid]),
                    "train_patient_count": int(train_patient_arr[cid]),
                    "syn_patient_count": int(syn_patient_arr[cid]),
                }
            )

    # Rank-frequency export + optional plots.
    _write_rank_frequency_csv(syn_visit_arr, os.path.join(args.out_dir, "synthetic_rank_frequency_visit.csv"))
    _write_rank_frequency_csv(syn_patient_arr, os.path.join(args.out_dir, "synthetic_rank_frequency_patient.csv"))
    _write_rank_frequency_csv(train_visit_arr, os.path.join(args.out_dir, "train_rank_frequency_visit.csv"))
    _write_rank_frequency_csv(train_patient_arr, os.path.join(args.out_dir, "train_rank_frequency_patient.csv"))

    plot_status = {}
    if args.plot:
        try:
            plot_status["overlay_rank_frequency_visit"] = _plot_rank_frequency_overlay(
                train_visit_arr,
                syn_visit_arr,
                os.path.join(args.out_dir, "overlay_rank_frequency_visit.png"),
                title="Rank-frequency (train vs synthetic) | per-visit presence",
            )
            plot_status["overlay_rank_frequency_patient"] = _plot_rank_frequency_overlay(
                train_patient_arr,
                syn_patient_arr,
                os.path.join(args.out_dir, "overlay_rank_frequency_patient.png"),
                title="Rank-frequency (train vs synthetic) | per-patient presence",
            )
        except Exception as e:
            plot_status["error"] = f"{type(e).__name__}: {e}"
    _write_json(plot_status, os.path.join(args.out_dir, "plots.json"))

    print(f"Wrote results to: {args.out_dir}")
    print("- generation_distribution_summary.json")
    print("- generation_distribution_per_code.csv")
    print("- synthetic_rank_frequency_*.csv, train_rank_frequency_*.csv")
    print("- overlay_rank_frequency_*.png (if --plot and matplotlib available)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate distribution fit + tail semantic plausibility for multiple generators on MIMIC-III,
aggregating results across 3 synthetic generations (seeds 1/2/3).

Outputs:
  1) per-seed CSV (numeric): one row per (method, seed)
  2) mean±std CSV: one row per method with "mean±std" formatted values

This is a multi-seed counterpart of evaluate_baseline_suite.py, but it uses the
already-generated synthetic datasets under each baseline's save/gen_seed* folders,
and model7 "best_*_seed{1,2,3}" folders.
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
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_PYTHON = sys.executable
DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/table1_mimiciii_3seeds"
DEFAULT_PER_SEED_CSV = "FAME_ROOT/output/table1_mimiciii_distribution_tail_per_seed.csv"
DEFAULT_MEAN_STD_CSV = "FAME_ROOT/output/table1_mimiciii_distribution_tail_mean_std.csv"


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> None:
    _safe_mkdir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
        f.write(f"(exit {rc})\n\n")
        if rc != 0:
            raise RuntimeError(f"command failed (exit={rc}): {' '.join(shlex.quote(x) for x in cmd)}")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get(d: dict, path: list[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _glob_one(pattern: str) -> Path:
    """Pick the newest match for a glob pattern (by mtime)."""
    matches = [Path(p) for p in sorted(Path().glob(pattern))]
    if not matches:
        raise FileNotFoundError(f"no match for glob: {pattern}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _find_synthetic_paths(method: str) -> dict[int, str]:
    """
    Return mapping: seed -> synthetic_path (Dataset.pkl)
    """
    # Baselines follow: fame/myfame/baseline/<name>/save/gen_seed{seed}_*/datasets/<dataset>.pkl
    # PCLA (MIMIC-III best) uses model7/save/best_*_seed{seed}_*/datasets/haloDataset.pkl
    base = "fame/myfame/baseline"

    if method.upper() == "PCLA":
        out = {}
        for seed in (1, 2, 3):
            d = _glob_one(f"{base}/model7/save/best_*_seed{seed}_*/datasets/haloDataset.pkl")
            out[seed] = str(d.resolve())
        return out

    if method.upper() == "HALO":
        # Use HALO2 generation folders for the 3 seeds (HALO and HALO2 are the same architecture in this repo).
        out = {}
        for seed in (1, 2, 3):
            d = _glob_one(f"{base}/HALO2/save/gen_seed{seed}_*/datasets/haloDataset.pkl")
            out[seed] = str(d.resolve())
        return out

    mapping = {
        "GPT": ("gpt", "gptDataset.pkl"),
        "LSTM": ("lstm", "lstmDataset.pkl"),
        "EVA": ("eva", "evaDataset.pkl"),
        "SYNTEG": ("synteg", "syntegDataset.pkl"),
    }
    key = method.upper()
    if key not in mapping:
        raise ValueError(f"unknown method: {method}")
    folder, filename = mapping[key]

    out = {}
    for seed in (1, 2, 3):
        d = _glob_one(f"{base}/{folder}/save/gen_seed{seed}_*/datasets/{filename}")
        out[seed] = str(d.resolve())
    return out


def _fmt_mean_std(vals: list[float]) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return ""
    # Report sample std (ddof=1) for repeated runs, consistent with common paper practice.
    ddof = 1 if arr.size > 1 else 0
    return f"{arr.mean():.6f}±{arr.std(ddof=ddof):.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--csv_per_seed_out", default=DEFAULT_PER_SEED_CSV)
    ap.add_argument("--csv_mean_std_out", default=DEFAULT_MEAN_STD_CSV)
    ap.add_argument(
        "--methods",
        default="HALO,PCLA,EVA,GPT,LSTM,SynTEG",
        help="Comma-separated methods (HALO,PCLA,EVA,GPT,LSTM,SynTEG). PCLA maps to model7 best (MIMIC-III).",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir) / _now_tag()
    _safe_mkdir(out_root)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    per_seed_rows: list[dict[str, object]] = []

    for method in methods:
        syn_paths = _find_synthetic_paths(method)
        for seed, syn_path in syn_paths.items():
            row: dict[str, object] = {
                "method": method,
                "seed": seed,
                "synthetic_path": syn_path,
                "error": "",
            }
            if not os.path.exists(syn_path):
                row["error"] = "missing_synthetic_path"
                per_seed_rows.append(row)
                continue

            method_dir = out_root / method.replace("/", "_").replace(" ", "_") / f"seed{seed}"
            _safe_mkdir(method_dir)

            try:
                # (1) Distribution fit
                dist_dir = method_dir / "generation_distribution"
                _safe_mkdir(dist_dir)
                _run(
                    [
                        str(args.python),
                        str(EVAL_DIR / "evaluate_generation_distribution.py"),
                        "--synthetic_path",
                        syn_path,
                        "--bucket_csv",
                        str(args.bucket_csv),
                        "--data_dir",
                        str(args.data_dir),
                        "--out_dir",
                        str(dist_dir),
                        "--no-plot",
                    ],
                    cwd=REPO_ROOT,
                    log_path=method_dir / "run.log",
                )
                dist_sum = _read_json(dist_dir / "generation_distribution_summary.json")

                # (2) Tail plausibility
                tail_dir = method_dir / "tail_semantic_plausibility"
                _safe_mkdir(tail_dir)
                _run(
                    [
                        str(args.python),
                        str(EVAL_DIR / "evaluate_tail_semantic_plausibility.py"),
                        "--data_dir",
                        str(args.data_dir),
                        "--bucket_csv",
                        str(args.bucket_csv),
                        "--ref_path",
                        str(Path(args.data_dir) / "trainDataset.pkl"),
                        "--real_path",
                        str(Path(args.data_dir) / "testDataset.pkl"),
                        "--synthetic_path",
                        syn_path,
                        "--out_dir",
                        str(tail_dir),
                    ],
                    cwd=REPO_ROOT,
                    log_path=method_dir / "run.log",
                )
                tail_sum = _read_json(tail_dir / "tail_semantic_plausibility_summary.json")

                # Flat metrics (same schema as baseline_suite_distribution_tail.csv).
                row.update(
                    {
                        "synthetic_total_patients": _get(dist_sum, ["synthetic_total_patients"]),
                        "synthetic_total_visits": _get(dist_sum, ["synthetic_total_visits"]),
                        "js_visit": _get(dist_sum, ["js_divergence_visit_present_codes"]),
                        "js_patient": _get(dist_sum, ["js_divergence_patient_present_codes"]),
                        "spearman_visit": _get(dist_sum, ["spearman_visit_present_codes"]),
                        "spearman_patient": _get(dist_sum, ["spearman_patient_present_codes"]),
                        "rare_recall_t10": _get(dist_sum, ["rare_code_recall_by_train_visit_count", "10", "recall"]),
                        "tail_codes_per_visit_mean": _get(dist_sum, ["synthetic_bucket_codes_per_visit_stats", "tail", "mean"]),
                        "pair_seen_rate": _get(tail_sum, ["synthetic", "pair_seen_rate"]),
                        "tail_involved_pair_seen_rate": _get(tail_sum, ["synthetic", "tail_involved_pair_seen_rate"]),
                        "tail_codes_present_frac": _get(tail_sum, ["synthetic", "tail_codes_present_frac"]),
                        "tail_context_js": _get(tail_sum, ["synthetic", "tail_context_js_weighted_by_ref_defined"]),
                        "tail_context_topk_jaccard": _get(tail_sum, ["synthetic", "tail_context_topk_jaccard_weighted_by_ref_defined"]),
                    }
                )
            except Exception as e:
                row["error"] = f"eval_failed: {type(e).__name__}"

            per_seed_rows.append(row)

    # Write per-seed CSV (numeric).
    csv_per_seed_out = Path(args.csv_per_seed_out)
    _safe_mkdir(csv_per_seed_out.parent)

    fieldnames = [
        "method",
        "seed",
        "synthetic_total_patients",
        "synthetic_total_visits",
        "js_visit",
        "js_patient",
        "spearman_visit",
        "spearman_patient",
        "rare_recall_t10",
        "tail_codes_per_visit_mean",
        "pair_seen_rate",
        "tail_involved_pair_seen_rate",
        "tail_codes_present_frac",
        "tail_context_js",
        "tail_context_topk_jaccard",
        "synthetic_path",
        "error",
    ]
    with open(csv_per_seed_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_seed_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Aggregate mean±std per method (skip rows with error).
    metrics = [
        "js_visit",
        "js_patient",
        "spearman_visit",
        "spearman_patient",
        "rare_recall_t10",
        "tail_codes_per_visit_mean",
        "pair_seen_rate",
        "tail_involved_pair_seen_rate",
        "tail_codes_present_frac",
        "tail_context_js",
        "tail_context_topk_jaccard",
    ]

    mean_std_rows: list[dict[str, object]] = []
    for method in methods:
        rows_ok = [r for r in per_seed_rows if r.get("method") == method and not r.get("error")]
        agg: dict[str, object] = {"method": method, "seed": "mean±std", "synthetic_path": "", "error": ""}
        for k in metrics:
            vals = []
            for r in rows_ok:
                v = r.get(k, None)
                try:
                    if v is None or v == "":
                        continue
                    vals.append(float(v))
                except Exception:
                    continue
            agg[k] = _fmt_mean_std(vals)
        mean_std_rows.append(agg)

    csv_mean_std_out = Path(args.csv_mean_std_out)
    _safe_mkdir(csv_mean_std_out.parent)
    fieldnames_mean = ["method", "seed"] + metrics + ["synthetic_path", "error"]
    with open(csv_mean_std_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_mean)
        w.writeheader()
        for r in mean_std_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_mean})

    print(f"wrote per-seed: {csv_per_seed_out}")
    print(f"wrote mean±std: {csv_mean_std_out}")
    print(f"details: {out_root}")


if __name__ == "__main__":
    main()

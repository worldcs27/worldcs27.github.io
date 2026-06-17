#!/usr/bin/env python3
"""
Evaluate multiple baseline synthesizers and export a single CSV summary.

This script runs:
  - evaluate_generation_distribution.py (distribution fit)
  - evaluate_tail_semantic_plausibility.py (tail plausibility / context consistency)

and aggregates key metrics into a CSV for paper tables.
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


REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_PYTHON = sys.executable
DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/baseline_suite"
DEFAULT_CSV_OUT = "FAME_ROOT/output/baseline_suite_distribution_tail.csv"
DEFAULT_COMPARE_JSON = "FAME_ROOT/evaluate/save/halo_vs_model2_tau1_clip15_1225_000518/compare_longtail_summary.json"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=DEFAULT_PYTHON)
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--csv_out", default=DEFAULT_CSV_OUT)
    ap.add_argument(
        "--methods",
        default="HALO,Model2,EVA,GPT,LSTM,SynTEG",
        help="Comma-separated methods to include (subset of HALO,Model2,EVA,GPT,LSTM,SynTEG,HALO2,HALO-Coarse,Model).",
    )
    ap.add_argument(
        "--compare_json",
        default=DEFAULT_COMPARE_JSON,
        help="Optional compare_longtail_summary.json to reuse metrics for HALO/Model2 (skip re-evaluation).",
    )
    args = ap.parse_args()

    out_root = Path(args.out_dir) / _now_tag()
    _safe_mkdir(out_root)

    baselines_all = {
        "HALO": "DATA_MIMICIII/datasets/haloDataset.pkl",
        "HALO2": "DATA_MIMICIII/datasets/haloDataset.pkl",
        "Model": "FAME_ROOT/baseline/model/save/datasets/mymodelDataset.pkl",
        "Model2": "FAME_ROOT/baseline/model2/save/datasets/haloDataset.pkl",
        "GPT": "FAME_ROOT/baseline/gpt/save/datasets/gptDataset.pkl",
        "LSTM": "FAME_ROOT/baseline/lstm/save/datasets/lstmDataset.pkl",
        "EVA": "FAME_ROOT/baseline/eva/save/datasets/evaDataset.pkl",
        "HALO-Coarse": "FAME_ROOT/baseline/haloCoarse/save/datasets/haloCoarseDataset.pkl",
        "SynTEG": "FAME_ROOT/baseline/synteg/save/datasets/syntegDataset.pkl",
    }

    requested = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    baselines = {k: baselines_all[k] for k in requested if k in baselines_all}

    compare = None
    if args.compare_json and os.path.exists(args.compare_json):
        compare = _read_json(Path(args.compare_json))

    rows: list[dict[str, object]] = []
    for name, syn_path in baselines.items():
        if compare is not None and name in ["HALO", "Model2"]:
            # Reuse existing summary for HALO/Model2 to avoid re-running heavy evaluation.
            gen = compare.get("generation_distribution", {}).get(name, {})
            tail = compare.get("tail_semantic_plausibility", {}).get(name, {})
            rows.append(
                {
                    "method": name,
                    "synthetic_path": syn_path,
                    "synthetic_total_patients": "",
                    "synthetic_total_visits": "",
                    "js_visit": gen.get("js_visit", ""),
                    "js_patient": gen.get("js_patient", ""),
                    "spearman_visit": gen.get("spearman_visit", ""),
                    "spearman_patient": gen.get("spearman_patient", ""),
                    "rare_recall_t10": gen.get("rare_recall_t10", ""),
                    "tail_codes_per_visit_mean": gen.get("tail_codes_per_visit_mean", ""),
                    "pair_seen_rate": tail.get("pair_seen_rate", ""),
                    "tail_involved_pair_seen_rate": tail.get("tail_involved_pair_seen_rate", ""),
                    "tail_codes_present_frac": tail.get("tail_codes_present_frac", ""),
                    "tail_context_js": tail.get("tail_context_js", ""),
                    "tail_context_topk_jaccard": tail.get("tail_context_topk_jaccard", ""),
                    "error": "",
                }
            )
            continue

        if not os.path.exists(syn_path):
            rows.append({"method": name, "synthetic_path": syn_path, "error": "missing_synthetic_path"})
            continue

        method_dir = out_root / name.replace("/", "_").replace(" ", "_")
        _safe_mkdir(method_dir)

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

        # Aggregate into a flat row (paper-friendly names).
        row = {
            "method": name,
            "synthetic_path": syn_path,
            "synthetic_total_patients": _get(dist_sum, ["synthetic_total_patients"]),
            "synthetic_total_visits": _get(dist_sum, ["synthetic_total_visits"]),
            # Distribution fit
            "js_visit": _get(dist_sum, ["js_divergence_visit_present_codes"]),
            "js_patient": _get(dist_sum, ["js_divergence_patient_present_codes"]),
            "spearman_visit": _get(dist_sum, ["spearman_visit_present_codes"]),
            "spearman_patient": _get(dist_sum, ["spearman_patient_present_codes"]),
            "rare_recall_t10": _get(dist_sum, ["rare_code_recall_by_train_visit_count", "10", "recall"]),
            "tail_codes_per_visit_mean": _get(dist_sum, ["synthetic_bucket_codes_per_visit_stats", "tail", "mean"]),
            # Tail plausibility
            "pair_seen_rate": _get(tail_sum, ["synthetic", "pair_seen_rate"]),
            "tail_involved_pair_seen_rate": _get(tail_sum, ["synthetic", "tail_involved_pair_seen_rate"]),
            "tail_codes_present_frac": _get(tail_sum, ["synthetic", "tail_codes_present_frac"]),
            "tail_context_js": _get(tail_sum, ["synthetic", "tail_context_js_weighted_by_ref_defined"]),
            "tail_context_topk_jaccard": _get(tail_sum, ["synthetic", "tail_context_topk_jaccard_weighted_by_ref_defined"]),
            "error": "",
        }
        rows.append(row)

    csv_out = Path(args.csv_out)
    _safe_mkdir(csv_out.parent)

    fieldnames = [
        "method",
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
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"wrote: {csv_out}")
    print(f"details: {out_root}")


if __name__ == "__main__":
    main()

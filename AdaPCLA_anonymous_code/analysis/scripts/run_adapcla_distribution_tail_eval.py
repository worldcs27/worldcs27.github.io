#!/usr/bin/env python3
"""
Run FAME's distribution + tail plausibility evaluation for AdaPCLA synthetic data.

Evaluates AdaPCLA haloDataset.pkl from model3 (MIMIC-III) and model5 (MIMIC-IV),
calling evaluate_generation_distribution.py and evaluate_tail_semantic_plausibility.py.
Outputs per-seed and mean±std CSV files to mywork/output/.
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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Paths: PCLA = ADAPCLA_ROOT, experiments = ADAPCLA_ROOT/experiments
PCLA_ROOT = Path(__file__).resolve().parents[2]  # scripts -> mywork -> PCLA
MYWORK_ROOT = PCLA_ROOT / "mywork"
FAME_ROOT = PCLA_ROOT / "fame" / "myfame"
EVAL_DIR = FAME_ROOT / "evaluate"
OUTPUT_DIR = MYWORK_ROOT / "output"

# Configs for each dataset
CONFIGS = {
    "mimiciii": {
        "method": "AdaPCLA",
        "data_dir": str(FAME_ROOT / "baseline" / "HALO" / "save"),
        "bucket_csv": str(FAME_ROOT / "output" / "长尾分布问题分析" / "mimiciii_code_buckets.csv"),
        "synthetic_tpl": str(MYWORK_ROOT / "model3" / "save_anneal" / "seed{seed}" / "datasets" / "haloDataset.pkl"),
        "per_seed_csv": "table1_adapcla_mimiciii_distribution_tail_per_seed.csv",
        "mean_std_csv": "table1_adapcla_mimiciii_distribution_tail_mean_std.csv",
    },
    "mimiciv": {
        "method": "AdaPCLA",
        "data_dir": str(FAME_ROOT / "data2"),
        "bucket_csv": str(FAME_ROOT / "output" / "长尾分布问题分析" / "mimiciv_code_buckets.csv"),
        "synthetic_tpl": str(MYWORK_ROOT / "model5" / "save_anneal_mimiciv" / "seed{seed}" / "datasets" / "haloDataset.pkl"),
        "per_seed_csv": "table1_adapcla_mimiciv_distribution_tail_per_seed.csv",
        "mean_std_csv": "table1_adapcla_mimiciv_distribution_tail_mean_std.csv",
    },
}

FIELDNAMES = [
    "method", "seed", "synthetic_total_patients", "synthetic_total_visits",
    "js_visit", "js_patient", "spearman_visit", "spearman_patient",
    "rare_recall_t10", "tail_codes_per_visit_mean",
    "pair_seen_rate", "tail_involved_pair_seen_rate", "tail_codes_present_frac",
    "tail_context_js", "tail_context_topk_jaccard",
    "synthetic_path", "error",
]


def _get(d: dict, path: list, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt_mean_std(vals: list[float]) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return ""
    ddof = 1 if arr.size > 1 else 0
    return f"{arr.mean():.6f}±{arr.std(ddof=ddof):.6f}"


def _run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.call(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"(exit {rc})\n\n")
    if rc != 0:
        raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def eval_one_seed(
    cfg: dict,
    seed: int,
    python_exe: str,
    work_dir: Path,
) -> dict:
    """Run distribution + tail eval for one (dataset, seed)."""
    syn_path = cfg["synthetic_tpl"].format(seed=seed)
    if not os.path.exists(syn_path):
        return {"method": cfg["method"], "seed": seed, "synthetic_path": syn_path, "error": "missing_synthetic_path"}

    ref_path = str(Path(cfg["data_dir"]) / "trainDataset.pkl")
    real_path = str(Path(cfg["data_dir"]) / "testDataset.pkl")

    method_dir = work_dir / f"AdaPCLA_seed{seed}"
    method_dir.mkdir(parents=True, exist_ok=True)

    try:
        # (1) Distribution
        dist_dir = method_dir / "generation_distribution"
        dist_dir.mkdir(exist_ok=True)
        _run(
            [
                python_exe,
                str(EVAL_DIR / "evaluate_generation_distribution.py"),
                "--synthetic_path", syn_path,
                "--bucket_csv", cfg["bucket_csv"],
                "--data_dir", cfg["data_dir"],
                "--real_path", ref_path,
                "--out_dir", str(dist_dir),
                "--no-plot",
            ],
            cwd=PCLA_ROOT,
            log_path=method_dir / "run.log",
        )
        with open(dist_dir / "generation_distribution_summary.json") as f:
            dist_sum = json.load(f)

        # (2) Tail plausibility
        tail_dir = method_dir / "tail_semantic_plausibility"
        tail_dir.mkdir(exist_ok=True)
        _run(
            [
                python_exe,
                str(EVAL_DIR / "evaluate_tail_semantic_plausibility.py"),
                "--data_dir", cfg["data_dir"],
                "--bucket_csv", cfg["bucket_csv"],
                "--ref_path", ref_path,
                "--real_path", real_path,
                "--synthetic_path", syn_path,
                "--out_dir", str(tail_dir),
            ],
            cwd=PCLA_ROOT,
            log_path=method_dir / "run.log",
        )
        with open(tail_dir / "tail_semantic_plausibility_summary.json") as f:
            tail_sum = json.load(f)

        return {
            "method": cfg["method"],
            "seed": seed,
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
            "synthetic_path": syn_path,
            "error": "",
        }
    except Exception as e:
        return {"method": cfg["method"], "seed": seed, "synthetic_path": syn_path, "error": f"eval_failed: {type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mimiciii", "mimiciv", "all"], default="all")
    ap.add_argument("--python", default=shutil.which("python3") or sys.executable)
    ap.add_argument("--work_dir", default=None, help="Working dir for intermediate outputs (default: output/adapcla_eval_<timestamp>)")
    ap.add_argument("--keep_work", action="store_true", help="Keep intermediate work dir after run")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_base = Path(args.work_dir) if args.work_dir else OUTPUT_DIR / f"adapcla_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    work_base.mkdir(parents=True, exist_ok=True)

    datasets = ["mimiciii", "mimiciv"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        cfg = CONFIGS[ds]
        print(f"[{ds}] Evaluating AdaPCLA seeds 1,2,3...")
        work_dir = work_base / ds
        work_dir.mkdir(parents=True, exist_ok=True)

        per_seed_rows = []
        for seed in (1, 2, 3):
            row = eval_one_seed(cfg, seed, args.python, work_dir)
            per_seed_rows.append(row)
            if row.get("error"):
                print(f"  seed{seed}: ERROR {row['error']}")
            else:
                ps = row.get("pair_seen_rate") or 0
                tps = row.get("tail_involved_pair_seen_rate") or 0
                print(f"  seed{seed}: pair_seen={ps:.4f}, tail_pair_seen={tps:.4f}")

        # Write per-seed CSV
        per_seed_out = OUTPUT_DIR / cfg["per_seed_csv"]
        with open(per_seed_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for r in per_seed_rows:
                w.writerow({k: r.get(k, "") for k in FIELDNAMES})
        print(f"  Wrote {per_seed_out}")

        # Aggregate mean±std (skip rows with error)
        ok_rows = [r for r in per_seed_rows if not r.get("error")]
        mean_std_row = {"method": cfg["method"], "seed": "mean±std", "synthetic_path": "", "error": ""}
        for key in FIELDNAMES:
            if key in ("method", "seed", "synthetic_path", "error"):
                continue
            vals = [r.get(key) for r in ok_rows if r.get(key) is not None]
            try:
                vals = [float(v) for v in vals]
            except (TypeError, ValueError):
                continue
            if vals:
                mean_std_row[key] = _fmt_mean_std(vals)

        mean_std_out = OUTPUT_DIR / cfg["mean_std_csv"]
        with open(mean_std_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerow({k: mean_std_row.get(k, "") for k in FIELDNAMES})
        print(f"  Wrote {mean_std_out}")

    if not args.keep_work and not args.work_dir:
        try:
            shutil.rmtree(work_base)
            print(f"Removed work dir {work_base}")
        except OSError:
            print(f"Note: could not remove work dir {work_base}")

    print("Done.")


if __name__ == "__main__":
    main()


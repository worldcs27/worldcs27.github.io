#!/usr/bin/env python3
"""
Zero-shot evaluation for baselines: GPT, LSTM, EVA, SynTEG, HALO.
For each baseline and direction (III→IV, IV→III):
  1. Generate synthetic data with model trained on source domain (no target prior).
  2. Map generated codes from source index space to target domain.
  3. Run downstream 25-label evaluation on target real test set.
  4. Collect Acc, F1, AUPRC and write zeroshot_baselines_table.csv.
"""
from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import csv
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

ZERO_SHOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ZERO_SHOT_DIR))
from paths_baselines import (
    BASELINES,
    DATA_III,
    DATA_IV,
    EVAL_PY,
    OUT_DIR,
    TOTAL_SAMPLES,
)

NUM_GPUS = int(__import__("os").environ.get("NUM_GPUS", "1"))
TORCHRUN = "torchrun"  # must be in PATH (same env as python)
# Per-baseline master_port to avoid EADDRINUSE when running multiple baselines in sequence
MASTER_PORT_BASE = int(__import__("os").environ.get("MASTER_PORT_BASE", "29510"))


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_index_to_code(code_to_index: dict) -> dict:
    return {int(v): k for k, v in code_to_index.items()}


def map_visits_to_target_domain(
    records: list,
    src_index_to_code: dict,
    tgt_code_to_index: dict,
) -> list:
    out = []
    for rec in records:
        visits = rec.get("visits", [])
        labels = rec.get("labels")
        if labels is None:
            labels = __import__("numpy").zeros(25, dtype="float32")
        mapped_visits = []
        for v in visits:
            mapped_v = []
            for idx in v:
                code_str = src_index_to_code.get(int(idx))
                if code_str is not None and code_str in tgt_code_to_index:
                    mapped_v.append(tgt_code_to_index[code_str])
            if mapped_v:
                mapped_visits.append(mapped_v)
        if mapped_visits:
            out.append({"visits": mapped_visits, "labels": labels})
    return out


def run_generation(
    baseline: str,
    source_data: Path,
    model_dir: Path,
    save_dir: Path,
    *,
    ckpt_path: Path | None = None,
) -> Path | None:
    """Run baseline's test/generate script; return path to merged dataset pkl or None on failure."""
    info = BASELINES[baseline]
    script_dir = info["script_dir"]
    test_script = info["test_script"]
    dataset_name = info["dataset_name"]
    if not script_dir.exists() or not (script_dir / test_script).exists():
        print(f"Skip {baseline}: script dir or test script not found.")
        return None
    port = MASTER_PORT_BASE + list(BASELINES.keys()).index(baseline)
    cmd = [
        TORCHRUN,
        f"--nproc_per_node={NUM_GPUS}",
        f"--master_port={port}",
        test_script,
        "--data_dir", str(source_data),
        "--save_dir", str(save_dir),
        "--total_samples", str(TOTAL_SAMPLES),
    ]
    if baseline == "halo" and ckpt_path is not None and ckpt_path.exists():
        cmd += ["--ckpt_path", str(ckpt_path)]
    elif baseline == "synteg":
        cmd += ["--model_dir", str(model_dir)]
    else:
        ckpt = model_dir / (info["ckpt_name_iii"] or info["ckpt_name_iv"])
        if ckpt.exists():
            cmd += ["--ckpt_path", str(ckpt)]
        else:
            cmd += ["--model_dir", str(model_dir)]
    print(f"Running: {' '.join(cmd)} (cwd={script_dir})")
    ret = subprocess.run(cmd, cwd=str(script_dir))
    if ret.returncode != 0:
        print(f"Generation failed for {baseline} (exit {ret.returncode})")
        return None
    pkl_path = save_dir / "datasets" / dataset_name
    if not pkl_path.exists():
        print(f"Expected {pkl_path} after generation.")
        return None
    return pkl_path


def run_eval(base_data_dir: Path, syn_path: Path, save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            EVAL_PY,
            "--base_data_dir", str(base_data_dir),
            "--mymodel2_path", str(syn_path),
            "--save_dir", str(save_dir),
            "--sources", "MyModel2",
        ],
        check=True,
        cwd=str(ZERO_SHOT_DIR),
    )
    csv_path = save_dir / "compare_real_halo_mymodel2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} after eval")
    return csv_path


def parse_mean_acc_f1_auprc(csv_path: Path, source: str = "MyModel2") -> tuple[float, float, float]:
    import numpy as np
    accs, f1s, auprcs = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") != source:
                continue
            a, f1, au = row.get("Accuracy"), row.get("F1 Score"), row.get("AUPRC")
            if a not in (None, ""):
                accs.append(float(a))
            if f1 not in (None, ""):
                f1s.append(float(f1))
            if au not in (None, ""):
                auprcs.append(float(au))
    return (
        float(np.mean(accs)) if accs else 0.0,
        float(np.mean(f1s)) if f1s else 0.0,
        float(np.mean(auprcs)) if auprcs else 0.0,
    )


def run_one_baseline_one_direction(
    baseline: str,
    direction: str,
) -> tuple[float, float, float] | None:
    """direction: 'iii_to_iv' or 'iv_to_iii'. Returns (acc, f1, auprc) or None."""
    info = BASELINES[baseline]
    if direction == "iii_to_iv":
        source_data = DATA_III
        target_data = DATA_IV
        model_dir = info["script_dir"] / info["save_subdir_iii"]
        out_mapped = OUT_DIR / f"{baseline}_iii_to_iv_mapped.pkl"
        eval_dir = OUT_DIR / f"eval_{baseline}_to_iv"
        method_name = f"{baseline.upper()} (III→IV zero-shot)"
        target_name = "MIMIC-IV"
    else:
        source_data = DATA_IV
        target_data = DATA_III
        model_dir = info["script_dir"] / info["save_subdir_iv"]
        out_mapped = OUT_DIR / f"{baseline}_iv_to_iii_mapped.pkl"
        eval_dir = OUT_DIR / f"eval_{baseline}_to_iii"
        method_name = f"{baseline.upper()} (IV→III zero-shot)"
        target_name = "MIMIC-III"
    if not model_dir.exists():
        print(f"Skip {baseline} {direction}: model_dir not found {model_dir}")
        return None
    ckpt_path = None
    if baseline == "halo":
        ckpt_path = model_dir / (info["ckpt_name_iii"] if direction == "iii_to_iv" else info["ckpt_name_iv"])
    with tempfile.TemporaryDirectory(prefix="zeroshot_") as tmp:
        save_dir = Path(tmp)
        pkl_path = run_generation(baseline, source_data, model_dir, save_dir, ckpt_path=ckpt_path)
        if pkl_path is None:
            return None
        raw = load_pkl(pkl_path)
        code_iii = load_pkl(DATA_III / "codeToIndex.pkl")
        code_iv = load_pkl(DATA_IV / "codeToIndex.pkl")
        if direction == "iii_to_iv":
            idx2code = build_index_to_code(code_iii)
            mapped = map_visits_to_target_domain(raw, idx2code, code_iv)
        else:
            idx2code = build_index_to_code(code_iv)
            mapped = map_visits_to_target_domain(raw, idx2code, code_iii)
        if not mapped:
            print(f"After mapping {baseline} {direction}: 0 records, skip eval.")
            return None
        with open(out_mapped, "wb") as f:
            pickle.dump(mapped, f)
        print(f"Mapped {len(mapped)} records -> {out_mapped}")
        run_eval(target_data, out_mapped, eval_dir)
        acc, f1, auprc = parse_mean_acc_f1_auprc(eval_dir / "compare_real_halo_mymodel2.csv")
        print(f"{method_name}: Acc={acc:.4f}, F1={f1:.4f}, AUPRC={auprc:.4f}")
        return (acc, f1, auprc)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default="gpt,lstm,eva,synteg,halo", help="Comma-separated baseline names.")
    ap.add_argument("--directions", default="iii_to_iv,iv_to_iii", help="Comma-separated: iii_to_iv,iv_to_iii.")
    args = ap.parse_args()
    baselines = [s.strip() for s in args.baselines.split(",") if s.strip()]
    directions = [s.strip() for s in args.directions.split(",") if s.strip()]
    for b in baselines:
        if b not in BASELINES:
            print(f"Unknown baseline: {b}. Valid: {list(BASELINES.keys())}")
            return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for baseline in baselines:
        for direction in directions:
            res = run_one_baseline_one_direction(baseline, direction)
            if res is None:
                continue
            acc, f1, auprc = res
            target_name = "MIMIC-IV" if direction == "iii_to_iv" else "MIMIC-III"
            method_name = f"{baseline.upper()} ({'III→IV' if direction == 'iii_to_iv' else 'IV→III'} zero-shot)"
            rows.append({"target": target_name, "method": method_name, "Acc": acc, "F1": f1, "AUPRC": auprc})
    if rows:
        out_csv = OUT_DIR / "zeroshot_baselines_table.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["target", "method", "Acc", "F1", "AUPRC"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
    print("Done.")


if __name__ == "__main__":
    main()

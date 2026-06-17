#!/usr/bin/env python3
"""
Hyperparameter tuning for Model2 prior correction (logit adjustment).

Optimizes the downstream classifier metrics reported by:
  EVAL_PY

Tuned params:
  - logit_adjust_tau
  - logit_adjust_clip

This script will:
  1) Train Model2 with (tau, clip)
  2) Generate a synthetic dataset via model2/test.py
  3) Run evaluate_synthetic_training.py with that dataset as MyModel2
  4) Parse compare_real_halo_mymodel2.csv and compute mean Acc/F1 for MyModel2

Notes:
  - Training/evaluation are expensive; tune with small grids first.
  - This script contains no extra dependencies (no optuna).
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]  # FAME_ROOT/baseline/model2/../../..
MODEL2_DIR = Path(__file__).resolve().parent


DEFAULT_PYTHON = sys.executable
DEFAULT_EVAL_SCRIPT = str(REPO_ROOT / "fame" / "myfame" / "evaluate" / "evaluate_synthetic_training.py")

# Defaults match the rest of the repo.
DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_HALO_PATH = "FAME_ROOT"


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty float list")
    return out


def _optional_float(s: str | None) -> float | None:
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in {"none", "null", "nil", ""}:
        return None
    return float(t)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool) -> None:
    _safe_mkdir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"$ {_quote_cmd(cmd)}\n")
        f.flush()
        if dry_run:
            f.write("(dry-run)\n\n")
            return
        p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
        f.write(f"\n(exit {rc})\n\n")
        if rc != 0:
            raise RuntimeError(f"command failed (exit={rc}): {_quote_cmd(cmd)}")


@dataclass(frozen=True)
class TrialResult:
    tau: float
    clip: float | None
    mean_acc: float
    mean_f1: float
    score: float
    eval_csv: str
    trial_dir: str


def _mean_acc_f1_from_eval_csv(csv_path: Path, *, source: str = "MyModel2") -> tuple[float, float]:
    acc: list[float] = []
    f1: list[float] = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("source") or "").strip() != source:
                continue
            acc.append(float(row["Accuracy"]))
            f1.append(float(row["F1 Score"]))
    if not acc or not f1 or len(acc) != len(f1):
        raise ValueError(f"no rows found for source={source} in {csv_path}")
    return (sum(acc) / len(acc), sum(f1) / len(f1))


def _score(mean_acc: float, mean_f1: float, objective: str) -> float:
    objective = (objective or "avg").strip().lower()
    if objective == "f1":
        return float(mean_f1)
    if objective == "acc":
        return float(mean_acc)
    if objective == "avg":
        return 0.5 * float(mean_acc + mean_f1)
    raise ValueError(f"unknown objective: {objective}")


def _iter_grid(taus: Iterable[float], clips: Iterable[float | None]):
    for tau in taus:
        for clip in clips:
            yield float(tau), clip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(MODEL2_DIR / "save" / f"tune_logit_adjust_{_now_tag()}"))
    ap.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable used to run eval script.")
    ap.add_argument("--eval_script", default=DEFAULT_EVAL_SCRIPT)
    ap.add_argument("--objective", default="avg", choices=["avg", "acc", "f1"])
    ap.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)

    # Data + paths
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Dataset directory with train/val/testDataset.pkl.")
    ap.add_argument("--halo_path", default=DEFAULT_HALO_PATH, help="Reference HALO synthetic dataset path.")

    # Training config
    ap.add_argument("--num_gpus", type=int, default=int(os.environ.get("NUM_GPUS", "1")))
    ap.add_argument("--cuda_visible_devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    ap.add_argument("--master_port_base", type=int, default=int(os.environ.get("MASTER_PORT_BASE", "29560")))
    ap.add_argument("--train_epochs", type=int, default=None, help="Override Model2Config.epoch for tuning (optional).")
    ap.add_argument("--train_lr", type=float, default=None, help="Override Model2Config.lr for tuning (optional).")
    ap.add_argument("--no_resume", action="store_true", help="Disable checkpoint resume for every trial.")
    ap.add_argument("--total_samples", type=int, default=33494, help="Total synthetic records to generate per trial.")

    # Tuned hyperparameters
    ap.add_argument("--taus", required=True, help="Comma-separated taus, e.g. 0.2,0.5,1.0")
    ap.add_argument("--clips", required=True, help="Comma-separated clips, e.g. 5,10,15 (or 'none')")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    _safe_mkdir(out_dir)

    taus = _parse_float_list(args.taus)
    clips: list[float | None] = [_optional_float(x.strip()) for x in (args.clips or "").split(",") if x.strip()]
    if not clips:
        raise ValueError("empty clips list")

    # Environment for torchrun.
    env = dict(os.environ)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    # Persist config.
    _write_json(
        out_dir / "tune_config.json",
        {
            "out_dir": str(out_dir),
            "python": args.python,
            "eval_script": args.eval_script,
            "objective": args.objective,
            "dry_run": bool(args.dry_run),
            "data_dir": args.data_dir,
            "halo_path": args.halo_path,
            "num_gpus": int(args.num_gpus),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "master_port_base": int(args.master_port_base),
            "train_epochs": args.train_epochs,
            "train_lr": args.train_lr,
            "no_resume": bool(args.no_resume),
            "total_samples": int(args.total_samples),
            "taus": taus,
            "clips": clips,
        },
    )

    results_csv = out_dir / "results.csv"
    if not results_csv.exists():
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "tau",
                    "clip",
                    "mean_acc",
                    "mean_f1",
                    "score",
                    "eval_csv",
                    "trial_dir",
                ],
            )
            w.writeheader()

    trial_idx = 0
    for tau, clip in _iter_grid(taus, clips):
        trial_idx += 1
        trial_name = f"trial_{trial_idx:04d}_tau={tau:g}_clip={'none' if clip is None else f'{clip:g}'}"
        trial_dir = out_dir / trial_name
        model_save_dir = trial_dir / "model2"
        eval_save_dir = trial_dir / "eval"
        trial_log = trial_dir / "run.log"

        _safe_mkdir(trial_dir)
        _write_json(
            trial_dir / "trial.json",
            {"trial": int(trial_idx), "tau": float(tau), "clip": clip, "model_save_dir": str(model_save_dir), "eval_save_dir": str(eval_save_dir)},
        )

        eval_csv = eval_save_dir / "compare_real_halo_mymodel2.csv"
        if eval_csv.exists():
            mean_acc, mean_f1 = _mean_acc_f1_from_eval_csv(eval_csv)
            score = _score(mean_acc, mean_f1, args.objective)
            tr = TrialResult(
                tau=float(tau),
                clip=clip,
                mean_acc=float(mean_acc),
                mean_f1=float(mean_f1),
                score=float(score),
                eval_csv=str(eval_csv),
                trial_dir=str(trial_dir),
            )
        else:
            _safe_mkdir(model_save_dir)
            _safe_mkdir(eval_save_dir)

            port = int(args.master_port_base) + int(trial_idx)

            train_cmd = [
                "torchrun",
                f"--nproc_per_node={int(args.num_gpus)}",
                f"--master_port={port}",
                "train.py",
                "--data_dir",
                str(args.data_dir),
                "--save_dir",
                str(model_save_dir),
                "--logit_adjust_tau",
                str(float(tau)),
            ]
            if clip is None:
                # CLI requires float; disable clipping by setting it to a very large number.
                train_cmd += ["--logit_adjust_clip", "1e9"]
            else:
                train_cmd += ["--logit_adjust_clip", str(float(clip))]
            if args.train_epochs is not None:
                train_cmd += ["--epoch", str(int(args.train_epochs))]
            if args.train_lr is not None:
                train_cmd += ["--lr", str(float(args.train_lr))]
            if args.no_resume:
                train_cmd += ["--no-resume"]

            test_cmd = [
                "torchrun",
                f"--nproc_per_node={int(args.num_gpus)}",
                f"--master_port={port + 200}",
                "test.py",
                "--data_dir",
                str(args.data_dir),
                "--save_dir",
                str(model_save_dir),
                "--total_samples",
                str(int(args.total_samples)),
            ]

            synth_path = model_save_dir / "datasets" / "haloDataset.pkl"
            eval_cmd = [
                str(args.python),
                str(args.eval_script),
                "--base_data_dir",
                str(args.data_dir),
                "--halo_path",
                str(args.halo_path),
                "--mymodel2_path",
                str(synth_path),
                "--save_dir",
                str(eval_save_dir),
            ]

            # Run the pipeline.
            _run(train_cmd, cwd=MODEL2_DIR, env=env, log_path=trial_log, dry_run=bool(args.dry_run))
            _run(test_cmd, cwd=MODEL2_DIR, env=env, log_path=trial_log, dry_run=bool(args.dry_run))
            _run(eval_cmd, cwd=REPO_ROOT, env=env, log_path=trial_log, dry_run=bool(args.dry_run))

            if args.dry_run:
                # Placeholder; real metrics will be available after running.
                mean_acc = float("nan")
                mean_f1 = float("nan")
                score = float("nan")
            else:
                mean_acc, mean_f1 = _mean_acc_f1_from_eval_csv(eval_csv)
                score = _score(mean_acc, mean_f1, args.objective)

            tr = TrialResult(
                tau=float(tau),
                clip=clip,
                mean_acc=float(mean_acc),
                mean_f1=float(mean_f1),
                score=float(score),
                eval_csv=str(eval_csv),
                trial_dir=str(trial_dir),
            )

        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "tau",
                    "clip",
                    "mean_acc",
                    "mean_f1",
                    "score",
                    "eval_csv",
                    "trial_dir",
                ],
            )
            w.writerow(
                {
                    "trial": int(trial_idx),
                    "tau": tr.tau,
                    "clip": "" if tr.clip is None else tr.clip,
                    "mean_acc": tr.mean_acc,
                    "mean_f1": tr.mean_f1,
                    "score": tr.score,
                    "eval_csv": tr.eval_csv,
                    "trial_dir": tr.trial_dir,
                }
            )

        print(f"[trial {trial_idx}] tau={tau:g} clip={'none' if clip is None else clip:g} -> score={tr.score}", flush=True)

    print(f"Wrote results to: {results_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


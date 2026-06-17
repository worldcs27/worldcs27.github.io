from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass


DEFAULT_DEVICES = "0,1,2,3,4,6,7"
DEFAULT_BASE_PORT = 29680
DEFAULT_HALO_SYN_PATH = "FAME_ROOT"


@dataclass(frozen=True)
class RunResult:
    seed: int
    mean_acc: float
    mean_f1: float
    eval_dir: str


def _which_torchrun() -> str:
    if os.environ.get("TORCHRUN"):
        return os.environ["TORCHRUN"]
    p = shutil.which("torchrun")
    if p:
        return p
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        cand = os.path.join(conda, "bin", "torchrun")
        if os.path.exists(cand):
            return cand
        raise FileNotFoundError("torchrun not found (set TORCHRUN=/path/to/torchrun or activate conda env).")


def _parse_mean_metrics(csv_path: str, *, source: str) -> tuple[float, float]:
    accs: list[float] = []
    f1s: list[float] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") != source:
                continue
            if row.get("Accuracy") not in (None, ""):
                accs.append(float(row["Accuracy"]))
            if row.get("F1 Score") not in (None, ""):
                f1s.append(float(row["F1 Score"]))
    if not accs or not f1s:
        raise ValueError(f"No rows for source={source} in {csv_path}")
    return sum(accs) / len(accs), sum(f1s) / len(f1s)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Empty values")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, var**0.5


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--devices", default=DEFAULT_DEVICES)
    p.add_argument("--base_port", type=int, default=DEFAULT_BASE_PORT)
    p.add_argument("--halo_path", default=DEFAULT_HALO_SYN_PATH)
    p.add_argument("--output_csv", default="FAME_ROOT/output/halo_seed3_summary.csv")
    p.add_argument("--output_json", default="FAME_ROOT/output/halo_seed3_summary.json")
    p.add_argument("--force", action="store_true", help="Re-run even if eval CSV exists.")
    args = p.parse_args()

    devices = str(args.devices)
    num_gpus = len([x for x in devices.split(",") if x.strip() != ""])

    torchrun = _which_torchrun()
    eval_py = EVAL_PY
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    results: list[RunResult] = []
    for idx, seed in enumerate(args.seeds):
        eval_dir = f"FAME_ROOT/evaluate/save/halo_eval_seed{seed}_{ts}"
        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        if (not args.force) and os.path.exists(eval_csv):
            mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="HALO")
            results.append(RunResult(seed=seed, mean_acc=mean_acc, mean_f1=mean_f1, eval_dir=eval_dir))
            continue

        os.makedirs(eval_dir, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["MASTER_PORT"] = str(int(args.base_port) + idx * 10)

        subprocess.run(
            [
                torchrun,
                f"--nproc_per_node={num_gpus}",
                f"--master_port={env['MASTER_PORT']}",
                eval_py,
                "--halo_path",
                str(args.halo_path),
                "--save_dir",
                eval_dir,
                "--seed",
                str(int(seed)),
                "--sources",
                "HALO",
            ],
            check=True,
            env=env,
        )

        mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="HALO")
        results.append(RunResult(seed=seed, mean_acc=mean_acc, mean_f1=mean_f1, eval_dir=eval_dir))

    accs = [r.mean_acc for r in results]
    f1s = [r.mean_f1 for r in results]
    mean_acc, std_acc = _mean_std(accs)
    mean_f1, std_f1 = _mean_std(f1s)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mean_acc", "mean_f1", "delta_acc", "delta_f1", "eval_dir"])
        for r in results:
            w.writerow(
                [
                    r.seed,
                    f"{r.mean_acc:.6f}",
                    f"{r.mean_f1:.6f}",
                    f"{(r.mean_acc - mean_acc):+.6f}",
                    f"{(r.mean_f1 - mean_f1):+.6f}",
                    r.eval_dir,
                ]
            )
        w.writerow([])
        w.writerow(["mean±std", f"{mean_acc:.6f}±{std_acc:.6f}", f"{mean_f1:.6f}±{std_f1:.6f}", "", "", ""])

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "seeds": [r.seed for r in results],
                "halo_path": str(args.halo_path),
                "metric": "mean over 25 labels from compare_real_halo_mymodel2.csv (source=HALO)",
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "runs": [
                    {
                        "seed": r.seed,
                        "mean_acc": r.mean_acc,
                        "mean_f1": r.mean_f1,
                        "delta_acc": r.mean_acc - mean_acc,
                        "delta_f1": r.mean_f1 - mean_f1,
                        "eval_dir": r.eval_dir,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



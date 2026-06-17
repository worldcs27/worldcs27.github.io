#!/usr/bin/env python3
"""
汇总当前 mywork/model1 下 PCLA 复现实验的 Real / HALO / PCLA（三路）下游 25-label 性能。

数据来源：
- 每个 seed 的评估结果位于：
  mywork/model1/evaluate/<run_tag>/compare_real_halo_mymodel2.csv
  其中包含 25 个 label × 多个 source 行（Real / HALO / MyModel2）。

本脚本会：
1. 遍历 mywork/model1/evaluate/*/compare_real_halo_mymodel2.csv；
2. 对每个文件、每个 source（Real / HALO / MyModel2）：
   - 取 25 个标签的 Accuracy / F1 Score 的均值；
3. 在 seed 维度上再次对 Real / HALO / PCLA（MyModel2）做 mean±std；
4. 写出一个简洁的总表：
   - output/real_halo_pcla_seed3_summary.csv
   - output/real_halo_pcla_seed3_summary.json

用法：
    cd EXPERIMENTS_ROOT/model1
    python summary_real_halo_pcla.py
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_ROOT = os.path.join(THIS_DIR, "evaluate")
OUTPUT_ROOT = os.path.join(THIS_DIR, "output")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


SOURCES = ("Real", "HALO", "MyModel2")  # 对应 compare_real_halo_mymodel2.csv 里的 source 字段


@dataclass
class RunMetrics:
    eval_dir: str
    per_source_mean_acc: Dict[str, float]
    per_source_mean_f1: Dict[str, float]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def parse_eval_csv(path: str) -> RunMetrics:
    """
    从单个 compare_real_halo_mymodel2.csv 中解析：
    - 对每个 source（Real/HALO/MyModel2），在所有 label 上的 mean Accuracy / mean F1。
    """
    accs: Dict[str, List[float]] = {s: [] for s in SOURCES}
    f1s: Dict[str, List[float]] = {s: [] for s in SOURCES}

    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row.get("source")
            if src not in SOURCES:
                continue
            a = row.get("Accuracy")
            f1 = row.get("F1 Score")
            if a not in (None, ""):
                accs[src].append(float(a))
            if f1 not in (None, ""):
                f1s[src].append(float(f1))

    per_source_mean_acc = {s: _mean(accs[s]) for s in SOURCES}
    per_source_mean_f1 = {s: _mean(f1s[s]) for s in SOURCES}
    return RunMetrics(
        eval_dir=os.path.dirname(path),
        per_source_mean_acc=per_source_mean_acc,
        per_source_mean_f1=per_source_mean_f1,
    )


def main() -> int:
    # 1. 收集所有 compare_real_halo_mymodel2.csv
    eval_files: List[str] = []
    if os.path.isdir(EVAL_ROOT):
        for name in os.listdir(EVAL_ROOT):
            sub = os.path.join(EVAL_ROOT, name)
            if not os.path.isdir(sub):
                continue
            csv_path = os.path.join(sub, "compare_real_halo_mymodel2.csv")
            if os.path.exists(csv_path):
                eval_files.append(csv_path)

    if not eval_files:
        print(f"No compare_real_halo_mymodel2.csv found under {EVAL_ROOT}")
        return 1

    print("Found eval CSVs:")
    for p in eval_files:
        print("  -", p)

    # 2. 对每个 eval 文件，得到一次 run 的 Real/HALO/PCLA 均值
    runs: List[RunMetrics] = [parse_eval_csv(p) for p in sorted(eval_files)]

    # 3. 在 seed 维度上对每个 source 做 mean±std
    per_source_accs: Dict[str, List[float]] = {s: [] for s in SOURCES}
    per_source_f1s: Dict[str, List[float]] = {s: [] for s in SOURCES}
    for run in runs:
        for s in SOURCES:
            per_source_accs[s].append(run.per_source_mean_acc[s])
            per_source_f1s[s].append(run.per_source_mean_f1[s])

    summary_rows = []
    for s in SOURCES:
        mean_acc = _mean(per_source_accs[s])
        std_acc = _std(per_source_accs[s])
        mean_f1 = _mean(per_source_f1s[s])
        std_f1 = _std(per_source_f1s[s])
        summary_rows.append(
            {
                "source": s,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "n_runs": len(per_source_accs[s]),
            }
        )

    # 4. 写 CSV
    out_csv = os.path.join(OUTPUT_ROOT, "real_halo_pcla_seed3_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "mean_acc",
                "std_acc",
                "mean_f1",
                "std_f1",
                "n_runs",
            ]
        )
        for row in summary_rows:
            w.writerow(
                [
                    row["source"],
                    f"{row['mean_acc']:.6f}",
                    f"{row['std_acc']:.6f}",
                    f"{row['mean_f1']:.6f}",
                    f"{row['std_f1']:.6f}",
                    row["n_runs"],
                ]
            )

    # 5. 写 JSON
    out_json = os.path.join(OUTPUT_ROOT, "real_halo_pcla_seed3_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"runs": [r.__dict__ for r in runs], "summary": summary_rows}, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


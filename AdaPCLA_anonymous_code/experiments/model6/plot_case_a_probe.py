#!/usr/bin/env python3
"""
子图 C：Case A 单 context 探针 — Anticoagulant vs Traffic accident + Oracle 水平线。

从 micro_probe_ckpt_*.csv 读该 context 的三条轨迹（related_rare / unrelated_rare / wrong），
从 micro_probe_oracle.csv 读 related_rare 的 Oracle log_prob，画水平虚线。
针对 context_id=552（Case A 心血管）时，图例使用 Anticoagulant (V5861) / Traffic accident (E8192)。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_trajectories(
    logs_dir: str,
    context_id: int,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Load mean_log_prob vs global_ckpt_idx for the three types."""
    pattern = os.path.join(logs_dir, "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    per_ckpt: Dict[int, Dict[str, float]] = {}
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ctx = int(row["context_id"])
                    if ctx != context_id:
                        continue
                    gidx = int(row["global_ckpt_idx"])
                    tag = row.get("type", "")
                    mlp = float(row["mean_log_prob"])
                except Exception:
                    continue
                if gidx not in per_ckpt:
                    per_ckpt[gidx] = {}
                per_ckpt[gidx][tag] = mlp

    if not per_ckpt:
        raise RuntimeError(f"No rows for context_id={context_id} in {len(paths)} files.")

    ts_sorted = sorted(per_ckpt.keys())
    curves: Dict[str, List[float]] = defaultdict(list)
    for t in ts_sorted:
        slot = per_ckpt[t]
        for tag in ["related_rare", "unrelated_rare", "wrong"]:
            curves[tag].append(slot.get(tag, float("nan")))
    return [float(t) for t in ts_sorted], dict(curves)


def load_oracle_for_related_rare(oracle_csv: str, context_id: int) -> float:
    """Read Oracle mean_log_prob for related_rare at this context_id (global_ckpt_idx=-1)."""
    if not os.path.isfile(oracle_csv):
        return float("nan")
    with open(oracle_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row["context_id"]) != context_id:
                    continue
                if row.get("type") != "related_rare":
                    continue
                return float(row["mean_log_prob"])
            except Exception:
                continue
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Case A probe: Anticoagulant vs Traffic accident + Oracle.")
    ap.add_argument("--logs_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_probe_logs", "seed1"))
    ap.add_argument("--oracle_csv", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_probe_logs", "seed1", "micro_probe_oracle.csv"))
    ap.add_argument("--context_id", type=int, default=552, help="Case A cardiovascular context.")
    ap.add_argument("--ckpt_per_epoch", type=int, default=10)
    ap.add_argument("--out", type=str, default="fig_case_a_probe.png")
    args = ap.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    oracle_csv = os.path.abspath(args.oracle_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    ts, curves = load_trajectories(logs_dir, args.context_id)
    oracle_val = load_oracle_for_related_rare(oracle_csv, args.context_id)
    t_epoch = [t / float(args.ckpt_per_epoch) for t in ts]

    # Case A 图例
    labels = {
        "related_rare": "Anticoagulant (V5861)",
        "unrelated_rare": "Unrelated rare",
        "wrong": "Traffic accident (E8192)",
    }
    colors = {"related_rare": "C0", "unrelated_rare": "C1", "wrong": "C2"}

    plt.figure(figsize=(5, 4))
    for tag in ["related_rare", "unrelated_rare", "wrong"]:
        ys = curves.get(tag)
        if not ys:
            continue
        plt.plot(t_epoch, ys, label=labels[tag], color=colors[tag])

    if not (oracle_val != oracle_val):  # not nan
        plt.axhline(y=oracle_val, color="C0", linestyle="--", alpha=0.8, label="Oracle (θ₀+bias)")

    plt.xlabel("Annealing progress (epoch)")
    plt.ylabel("log p (last-step)")
    plt.title(f"Case A probe (context_id={args.context_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

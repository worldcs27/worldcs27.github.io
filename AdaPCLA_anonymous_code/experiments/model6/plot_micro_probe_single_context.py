#!/usr/bin/env python3
"""
Plot micro-probe probability trajectories for a single context:
related_rare / unrelated_rare / wrong.

Data source: micro_probe_ckpt_XXXX.csv files produced by
model5/run_pcla_fixed_bias_anneal_mimiciv.py with --log_micro_probe.

Each CSV row:
  epoch, step_in_epoch, alpha, global_ckpt_idx,
  context_id, disease_id, type, mean_prob, mean_log_prob

This script:
  - scans all micro_probe_ckpt_*.csv under a logs directory
  - filters rows for a given context_id
  - groups by type (related_rare / unrelated_rare / wrong)
  - plots mean_log_prob vs. (epoch + fraction) or global_ckpt_idx
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
    """
    Load trajectories for a single context across all checkpoints.

    Returns:
      ts: list of time indices (global_ckpt_idx)
      curves: dict[type] -> list of mean_log_prob aligned with ts
    """
    pattern = os.path.join(logs_dir, "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    # Map global_ckpt_idx -> {type -> mean_log_prob}
    per_ckpt: Dict[int, Dict[str, float]] = {}

    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ctx = int(row["context_id"])
                except Exception:
                    continue
                if ctx != context_id:
                    continue
                try:
                    gidx = int(row["global_ckpt_idx"])
                    tag = row.get("type", "")
                    mlp = float(row["mean_log_prob"])
                except Exception:
                    continue
                if gidx not in per_ckpt:
                    per_ckpt[gidx] = {}
                per_ckpt[gidx][tag] = mlp

    if not per_ckpt:
        raise RuntimeError(f"No rows found for context_id={context_id} in {len(paths)} files.")

    ts_sorted = sorted(per_ckpt.keys())
    curves: Dict[str, List[float]] = defaultdict(list)
    for t in ts_sorted:
        slot = per_ckpt[t]
        for tag in ["related_rare", "unrelated_rare", "wrong"]:
            curves[tag].append(slot.get(tag, float("nan")))
    return [float(t) for t in ts_sorted], dict(curves)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot related/unrelated/wrong trajectories for a single context."
    )
    ap.add_argument(
        "--logs_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "micro_probe_logs",
            "seed1",
        ),
        help="Directory containing micro_probe_ckpt_*.csv files.",
    )
    ap.add_argument(
        "--context_id",
        type=int,
        required=True,
        help="Context id to visualize (as in micro_probe_config CSV).",
    )
    ap.add_argument(
        "--ckpt_per_epoch",
        type=int,
        default=10,
        help="Number of micro-probe checkpoints per epoch (for x-axis scaling).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="micro_probe_single_context.png",
        help="Output figure path (PNG).",
    )
    args = ap.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ts, curves = load_trajectories(logs_dir, args.context_id)

    # Map global_ckpt_idx -> epoch_fraction for nicer x-axis
    # t_epoch = global_ckpt_idx / ckpt_per_epoch
    t_epoch = [t / float(args.ckpt_per_epoch) for t in ts]

    plt.figure(figsize=(6, 4))
    for tag, color in [
        ("related_rare", "C0"),
        ("unrelated_rare", "C1"),
        ("wrong", "C2"),
    ]:
        ys = curves.get(tag)
        if not ys:
            continue
        plt.plot(t_epoch, ys, label=tag, color=color)

    plt.xlabel("Epoch (approx, global_ckpt_idx / ckpt_per_epoch)")
    plt.ylabel("mean_log_prob")
    plt.title(f"Micro-probe trajectories (context_id={args.context_id})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[micro-probe-plot] Saved figure to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()


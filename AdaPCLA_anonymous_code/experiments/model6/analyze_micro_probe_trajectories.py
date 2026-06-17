#!/usr/bin/env python3
"""
Summarize micro-probe Δlog p trajectories per context and type
(related_rare / unrelated_rare / wrong), and compute AUC and final Δlog p.

Usage examples:

  # 30-context setup
  python analyze_micro_probe_trajectories.py \\
    --logs_dir micro_probe_logs/seed1 \\
    --ckpt_per_epoch 10 \\
    --out_per_context micro_probe_summary_30ctx.csv \\
    --out_summary micro_probe_summary_30ctx_agg.csv

  # 100-context setup
  python analyze_micro_probe_trajectories.py \\
    --logs_dir micro_probe_logs_100ctx/seed1 \\
    --ckpt_per_epoch 10 \\
    --out_per_context micro_probe_summary_100ctx.csv \\
    --out_summary micro_probe_summary_100ctx_agg.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


ProbeKey = Tuple[int, int, str]  # (context_id, disease_id, type)


def load_all_trajectories(logs_dir: str):
    pattern = os.path.join(logs_dir, "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    trajectories: Dict[ProbeKey, Dict[int, float]] = defaultdict(dict)
    all_gidx: set[int] = set()
    for p in paths:
        with open(p, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gidx = int(row["global_ckpt_idx"])
                    ctx = int(row["context_id"])
                    did = int(row["disease_id"])
                    tag = row.get("type", "")
                    mlp = float(row["mean_log_prob"])
                except Exception:
                    continue
                key = (ctx, did, tag)
                trajectories[key][gidx] = mlp
                all_gidx.add(gidx)
    ts_sorted = sorted(all_gidx)
    return ts_sorted, dict(trajectories)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize micro-probe Δlog p trajectories per context and type.")
    ap.add_argument("--logs_dir", type=str, required=True, help="Directory containing micro_probe_ckpt_*.csv")
    ap.add_argument("--ckpt_per_epoch", type=int, default=10, help="Checkpoints per epoch (default: 10)")
    ap.add_argument("--out_per_context", type=str, required=True, help="Output CSV for per-context summary.")
    ap.add_argument("--out_summary", type=str, required=True, help="Output CSV for aggregated summary.")
    args = ap.parse_args()

    ts, trajectories = load_all_trajectories(args.logs_dir)
    if not ts:
        raise RuntimeError("No checkpoints found")
    t_epoch = np.array(ts, dtype=float) / float(args.ckpt_per_epoch)

    # Group by (context_id, type)
    ctx_type_to_keys: Dict[Tuple[int, str], List[ProbeKey]] = defaultdict(list)
    for (ctx, did, tag), _traj in trajectories.items():
        ctx_type_to_keys[(ctx, tag)].append((ctx, did, tag))

    per_rows = []
    for (ctx, tag), keys in ctx_type_to_keys.items():
        # Merge probes of same (ctx, tag): Δlog p(t) = mean over keys of (log_prob(t) - log_prob(0))
        deltas = []
        for t in ts:
            vals = []
            for key in keys:
                traj = trajectories[key]
                p0 = traj.get(min(traj.keys()), float("nan"))
                pt = traj.get(t, float("nan"))
                if p0 == p0 and pt == pt:
                    vals.append(pt - p0)
            deltas.append(np.mean(vals) if vals else np.nan)
        deltas = np.array(deltas, dtype=float)
        mask = ~np.isnan(deltas)
        if mask.any():
            auc = float(np.trapz(deltas[mask], t_epoch[mask]))
            final_delta = float(deltas[mask][-1])
        else:
            auc = float("nan")
            final_delta = float("nan")
        per_rows.append(
            {
                "context_id": ctx,
                "type": tag,
                "auc": auc,
                "final_delta": final_delta,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_per_context)) or ".", exist_ok=True)
    with open(args.out_per_context, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["context_id", "type", "auc", "final_delta"])
        writer.writeheader()
        writer.writerows(per_rows)

    # Aggregate by type
    summary_rows = []
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for row in per_rows:
        by_type[row["type"]].append(row)
    for tag, rows in by_type.items():
        aucs = np.array([r["auc"] for r in rows if r["auc"] == r["auc"]], dtype=float)
        finals = np.array([r["final_delta"] for r in rows if r["final_delta"] == r["final_delta"]], dtype=float)
        summary_rows.append(
            {
                "type": tag,
                "n_contexts": len(rows),
                "auc_mean": float(aucs.mean()) if len(aucs) else float("nan"),
                "auc_std": float(aucs.std()) if len(aucs) else float("nan"),
                "final_delta_mean": float(finals.mean()) if len(finals) else float("nan"),
                "final_delta_std": float(finals.std()) if len(finals) else float("nan"),
            }
        )

    with open(args.out_summary, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "n_contexts",
                "auc_mean",
                "auc_std",
                "final_delta_mean",
                "final_delta_std",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()


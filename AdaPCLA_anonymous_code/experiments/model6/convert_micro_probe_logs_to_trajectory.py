#!/usr/bin/env python3
"""
Convert micro_probe_ckpt_*.csv logs to a single trajectory CSV matching
anneal_tail_trajectory_*.csv format.

Usage:
  python model6/convert_micro_probe_logs_to_trajectory.py \\
    --logs_dir model6/micro_probe_logs_100ctx/seed1 \\
    --out_csv model6/anneal_tail_trajectory_100ctx.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert micro_probe logs to trajectory CSV.")
    ap.add_argument("--logs_dir", type=str, required=True, help="Directory with micro_probe_ckpt_*.csv")
    ap.add_argument("--out_csv", type=str, required=True, help="Output trajectory CSV path.")
    args = ap.parse_args()

    pattern = os.path.join(os.path.abspath(args.logs_dir), "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    rows: list[dict] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gidx = int(row["global_ckpt_idx"])
                    ctx = int(row["context_id"])
                    did = int(row["disease_id"])
                    mean_prob = float(row["mean_prob"])
                    mean_log_prob = float(row["mean_log_prob"])
                except (KeyError, ValueError):
                    continue
                rows.append({
                    "checkpoint_idx": gidx,
                    "context_id": ctx,
                    "disease_id": did,
                    "mean_prob": mean_prob,
                    "mean_log_prob": mean_log_prob,
                })

    out_path = os.path.abspath(args.out_csv)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["checkpoint_idx", "context_id", "disease_id", "mean_prob", "mean_log_prob"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[convert] Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()

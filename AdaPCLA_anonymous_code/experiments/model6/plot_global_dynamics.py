#!/usr/bin/env python3
"""
子图 A：全局概率演化 — 30 个 probing contexts 上三类（Related Rare / Unrelated Rare / Wrong）
的平均 Δlog p 随退火步数的变化。

从 micro_probe_ckpt_*.csv 读所有 (context_id, disease_id, type) 的 mean_log_prob 轨迹，
对每类计算平均 Δlog p(t) = mean over probes of type of [ log_prob(t) - log_prob(0) ]，
可选画 related_rare 的 Oracle 目标水平线（mean over related_rare probes of [ Oracle - log_prob(0) ]）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_all_trajectories(logs_dir: str) -> Tuple[List[int], Dict[Tuple[int, int, str], Dict[int, float]]]:
    """
    Load per (context_id, disease_id, type) trajectory: gidx -> mean_log_prob.
    Returns: sorted checkpoint indices, and probe_key -> {gidx: mlp}
    """
    pattern = os.path.join(logs_dir, "micro_probe_ckpt_*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")

    # (ctx, did, type) -> {gidx -> mean_log_prob}
    trajectories: Dict[Tuple[int, int, str], Dict[int, float]] = defaultdict(dict)
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


def load_oracle_delta_for_related(
    logs_dir: str,
    oracle_csv: str,
    trajectories: Dict[Tuple[int, int, str], Dict[int, float]],
) -> float:
    """Mean over related_rare probes of (Oracle_log_prob - log_prob(0))."""
    if not os.path.isfile(oracle_csv):
        return float("nan")
    oracle_by_key: Dict[Tuple[int, int, str], float] = {}
    with open(oracle_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ctx = int(row["context_id"])
                did = int(row["disease_id"])
                tag = row.get("type", "")
                mlp = float(row["mean_log_prob"])
                oracle_by_key[(ctx, did, tag)] = mlp
            except Exception:
                continue

    deltas = []
    for key, traj in trajectories.items():
        if key[2] != "related_rare":
            continue
        log_p0 = traj.get(min(traj.keys()), float("nan"))
        oracle = oracle_by_key.get(key, float("nan"))
        if log_p0 == log_p0 and oracle == oracle:
            deltas.append(oracle - log_p0)
    return float(sum(deltas) / len(deltas)) if deltas else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Global dynamics: average Δlog p over 30 contexts.")
    ap.add_argument(
        "--logs_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_probe_logs", "seed1"),
    )
    ap.add_argument(
        "--oracle_csv",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_probe_logs", "seed1", "micro_probe_oracle.csv"),
    )
    ap.add_argument("--ckpt_per_epoch", type=int, default=10)
    ap.add_argument("--out", type=str, default="fig_global_dynamics.png")
    args = ap.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    oracle_csv = os.path.abspath(args.oracle_csv)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    ts, trajectories = load_all_trajectories(logs_dir)
    if not ts:
        raise RuntimeError("No checkpoint data found.")

    # Per type: at each t, average over probes of (log_prob(t) - log_prob(0))
    type_to_probes: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    for key in trajectories:
        type_to_probes[key[2]].append(key)

    curves: Dict[str, List[float]] = {}
    for tag in ["related_rare", "unrelated_rare", "wrong"]:
        probes = type_to_probes.get(tag, [])
        if not probes:
            curves[tag] = [float("nan")] * len(ts)
            continue
        curve = []
        for t in ts:
            vals = []
            for key in probes:
                traj = trajectories[key]
                p0 = traj.get(min(traj.keys()), float("nan"))
                pt = traj.get(t, float("nan"))
                if p0 == p0 and pt == pt:
                    vals.append(pt - p0)
            curve.append(sum(vals) / len(vals)) if vals else float("nan")
        curves[tag] = curve

    oracle_delta = load_oracle_delta_for_related(logs_dir, oracle_csv, trajectories)
    t_epoch = [t / float(args.ckpt_per_epoch) for t in ts]

    plt.figure(figsize=(5, 4))
    colors = {"related_rare": "C0", "unrelated_rare": "C1", "wrong": "C2"}
    labels = {"related_rare": "Related rare", "unrelated_rare": "Unrelated rare", "wrong": "Wrong"}
    for tag in ["related_rare", "unrelated_rare", "wrong"]:
        ys = curves.get(tag)
        if ys:
            plt.plot(t_epoch, ys, label=labels[tag], color=colors[tag])
    if oracle_delta == oracle_delta:
        plt.axhline(y=oracle_delta, color="C0", linestyle="--", alpha=0.8, label="Oracle (θ₀+bias)")

    plt.xlabel("Annealing progress (epoch)")
    plt.ylabel("Average Δ log p")
    plt.title("Global dynamics (30 probing contexts)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

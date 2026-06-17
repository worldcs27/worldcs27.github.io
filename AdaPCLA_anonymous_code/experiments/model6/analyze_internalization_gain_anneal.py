#!/usr/bin/env python3
"""
Analyze annealing-speed internalization via related-rare gain / AUC.

Motivation
----------
Previous global norm-style internalization metrics (||z_T - z_0 - b_stat||, etc.)
were dominated by the large static bias b_stat and the full code-vocab, so
differences between annealing schedules (fast / base / slow / cosine) were
essentially drowned out.

Here we instead:
  - Focus ONLY on micro-probe pairs (context_id, disease_id) with type == 'related_rare',
    i.e., codes that theory says should receive internalized signal from the curriculum;
  - Work directly with the *trajectory* of last-step mean_log_prob over 100 micro-probe
    checkpoints for each schedule;
  - Define per-pair metrics that capture how much and how consistently the model
    internalizes tail-relevant signal along the annealing path:

    a) Final gain: Δlog p = log p_T - log p_0
    b) AUC:        sum_t (max(0, log p_t - log p_0)) * Δt, measuring cumulative
                   positive log-prob improvement over the trajectory
    c) Normalized AUC: AUC divided by (T-1) * max_t max(0, log p_t - log p_0),
       to get a [0,1]-like measure of how steadily the gain is accumulated.

These measures are much closer to the theoretical notion of "internalization of
tail-relevant signal": for related-rare codes we expect slower annealing (and
well-shaped cosine schedules) to allow the model to gradually internalize the
curriculum bias into its own logits, reflected as larger and smoother gains in
log-probability along the trajectory.

Usage
-----
1) First generate the four trajectory CSVs (see main paper experiments):

   - model6/anneal_tail_trajectory_base.csv
   - model6/anneal_tail_trajectory_fast.csv
   - model6/anneal_tail_trajectory_slow.csv
   - model6/anneal_tail_trajectory_cosine.csv

   each produced by `tail_code_trajectory.py` with the SAME micro_probe_config
   (e.g. mimiciv_long_tail_triplets_seed1.csv) but different ckpt_dir.

2) Then run this script from PCLA/mywork:

   cd EXPERIMENTS_ROOT
   python model6/analyze_internalization_gain_anneal.py

It will:
  - Filter to type == 'related_rare' pairs using the micro_probe_config;
  - Compute per-pair Δlog p, AUC, normalized AUC for each schedule;
  - Aggregate mean / std / n_pairs per schedule;
  - Write a summary CSV and generate barplots / boxplots for gain and AUC.
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = EXPERIMENTS_ROOT
MODEL6_DIR = os.path.join(PROJECT_ROOT, "model6")

# Trajectory CSVs produced by tail_code_trajectory.py
TRAJ_FILES = {
    "base": os.path.join(MODEL6_DIR, "anneal_tail_trajectory_base.csv"),
    "fast": os.path.join(MODEL6_DIR, "anneal_tail_trajectory_fast.csv"),
    "slow": os.path.join(MODEL6_DIR, "anneal_tail_trajectory_slow.csv"),
    "cosine": os.path.join(MODEL6_DIR, "anneal_tail_trajectory_cosine.csv"),
}

# Micro-probe config with type column (related_rare / unrelated_rare / wrong)
MICRO_PROBE_CONFIG = os.path.join(
    MODEL6_DIR, "micro_probe_configs", "mimiciv_long_tail_triplets_seed1.csv"
)


@dataclass(frozen=True)
class Pair:
    context_id: int
    disease_id: int


def load_related_rare_pairs(config_path: str) -> List[Pair]:
    pairs: List[Pair] = []
    with open(config_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("type") or "").strip().lower()
            if "related_rare" not in t:
                continue
            try:
                ctx = int(row["context_id"])
                did = int(row["disease_id"])
            except (KeyError, ValueError):
                continue
            pairs.append(Pair(ctx, did))
    # 去重，保持顺序
    seen = set()
    uniq: List[Pair] = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def load_trajectory(path: str, focus_pairs: List[Pair]) -> pd.DataFrame:
    """
    Load one trajectory CSV and keep only rows for focus_pairs.
    Returns a DataFrame with at least columns:
      - checkpoint_idx
      - context_id
      - disease_id
      - mean_log_prob
    """
    df = pd.read_csv(path)
    # 标准列名检查
    required_cols = {"checkpoint_idx", "context_id", "disease_id", "mean_log_prob"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    focus_set = {(p.context_id, p.disease_id) for p in focus_pairs}
    mask = df.apply(
        lambda r: (int(r["context_id"]), int(r["disease_id"])) in focus_set,
        axis=1,
    )
    df = df.loc[mask].copy()
    return df


def compute_gain_and_auc_one_pair(sub: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Given all rows for a single (context_id, disease_id) and schedule, compute:
      - gain: Δlog p = log p_T - log p_0
      - auc:  sum_t (max(0, log p_t - log p_0)) over t (simple Riemann sum, Δt=1)
      - auc_norm: auc / ((T-1) * max_t max(0, log p_t - log p_0)), in [0,1] if gain>0
    """
    if sub.empty:
        return float("nan"), float("nan"), float("nan")

    sub = sub.sort_values("checkpoint_idx")
    logs = sub["mean_log_prob"].to_numpy().astype("float64")
    T = logs.shape[0]
    if T < 2:
        return float("nan"), float("nan"), float("nan")

    log0 = logs[0]
    logT = logs[-1]
    gain = float(logT - log0)

    y = logs - log0
    y_pos = np.maximum(y, 0.0)

    # AUC: simple left Riemann sum with Δt=1
    auc = float(np.sum(y_pos[1:]))  # skip t=0, since log p_0 - log p_0 = 0

    max_y = float(np.max(y_pos))
    if T > 1 and max_y > 1e-8:
        auc_norm = float(auc / ((T - 1) * max_y))
    else:
        auc_norm = float("nan")

    return gain, auc, auc_norm


def compute_metrics_for_schedule(
    schedule: str,
    traj_path: str,
    related_pairs: List[Pair],
) -> pd.DataFrame:
    """
    For a given schedule and its trajectory file:
      - Filter to related_rare pairs
      - For each pair, compute gain / AUC / AUC_norm
    Returns a DataFrame with columns:
      schedule, context_id, disease_id, gain, auc, auc_norm
    """
    df = load_trajectory(traj_path, related_pairs)
    results: List[Dict[str, float]] = []

    # group by (context_id, disease_id)
    grouped = df.groupby(["context_id", "disease_id"])
    for (ctx, did), sub in grouped:
        gain, auc, auc_norm = compute_gain_and_auc_one_pair(sub)
        results.append(
            {
                "schedule": schedule,
                "context_id": int(ctx),
                "disease_id": int(did),
                "gain": gain,
                "auc": auc,
                "auc_norm": auc_norm,
            }
        )

    if not results:
        return pd.DataFrame(
            columns=["schedule", "context_id", "disease_id", "gain", "auc", "auc_norm"]
        )
    return pd.DataFrame(results)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate gain / auc / auc_norm per schedule: mean, std, n_pairs.
    """
    if df.empty:
        raise ValueError("No metrics to aggregate (empty DataFrame).")

    agg_map = {
        "gain": ["mean", "std"],
        "auc": ["mean", "std"],
        "auc_norm": ["mean", "std"],
    }
    grouped = df.groupby("schedule").agg(agg_map)
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.values]
    grouped["n_pairs"] = df.groupby("schedule")["gain"].count()
    summary = grouped.reset_index()
    return summary


def plot_boxplot(df: pd.DataFrame, metric: str, out_path: str):
    plt.figure(figsize=(6, 4))
    desired_order = ["slow", "base", "cosine", "fast"]
    available = [s for s in desired_order if s in df["schedule"].unique()]
    if not available:
        raise ValueError(f"No schedules available for {metric} boxplot.")

    df["schedule"] = pd.Categorical(df["schedule"], categories=available, ordered=True)
    df_sorted = df.sort_values("schedule")

    data = [df_sorted[df_sorted["schedule"] == s][metric].dropna().values for s in available]
    plt.boxplot(
        data,
        tick_labels=available,
        showfliers=True,
    )
    plt.xlabel("Annealing schedule")
    plt.ylabel(metric)
    plt.title(f"{metric} by annealing schedule (related_rare only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved boxplot to {out_path}")
    plt.close()


def plot_bar(summary: pd.DataFrame, metric_mean: str, metric_std: str, out_path: str):
    plt.figure(figsize=(6, 4))
    desired_order = ["slow", "base", "cosine", "fast"]
    available = [s for s in desired_order if s in summary["schedule"].values]
    if not available:
        raise ValueError(f"No schedules available for {metric_mean} barplot.")

    s_ord = summary.set_index("schedule").loc[available].reset_index()
    x = range(len(available))
    means = s_ord[metric_mean].values
    stds = s_ord[metric_std].values

    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, available)
    plt.xlabel("Annealing schedule")
    plt.ylabel(metric_mean)
    plt.title(f"{metric_mean} (± {metric_std}) by schedule (related_rare only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved barplot to {out_path}")
    plt.close()


def main() -> None:
    print(f"[INFO] Using micro-probe config: {MICRO_PROBE_CONFIG}")
    related_pairs = load_related_rare_pairs(MICRO_PROBE_CONFIG)
    print(f"[INFO] Loaded {len(related_pairs)} related_rare pairs from config.")

    all_metrics: List[pd.DataFrame] = []
    for sched, path in TRAJ_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] Trajectory file for {sched} not found: {path}")
            continue
        print(f"[INFO] Processing schedule={sched}, trajectory={path}")
        df_sched = compute_metrics_for_schedule(sched, path, related_pairs)
        all_metrics.append(df_sched)

    if not all_metrics:
        raise SystemExit("No trajectory metrics computed (no CSVs found).")

    df_all = pd.concat(all_metrics, ignore_index=True)
    base_dir = os.path.dirname(__file__)

    # Save per-pair metrics
    per_pair_csv = os.path.join(base_dir, "internalization_gain_per_pair_anneal.csv")
    df_all.to_csv(per_pair_csv, index=False)
    print(f"[INFO] Saved per-pair metrics to {per_pair_csv}")

    # Aggregate per schedule
    summary = aggregate_metrics(df_all)
    print("\n=== Related-rare gain / AUC summary by schedule ===")
    cols_to_print = [
        "schedule",
        "gain_mean",
        "gain_std",
        "auc_mean",
        "auc_std",
        "auc_norm_mean",
        "auc_norm_std",
        "n_pairs",
    ]
    print(summary[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    summary_csv = os.path.join(base_dir, "internalization_gain_summary_anneal.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[INFO] Saved summary CSV to {summary_csv}")

    # Plots: gain
    plot_boxplot(
        df_all,
        metric="gain",
        out_path=os.path.join(base_dir, "internalization_gain_boxplot_anneal.png"),
    )
    plot_bar(
        summary,
        metric_mean="gain_mean",
        metric_std="gain_std",
        out_path=os.path.join(base_dir, "internalization_gain_bar_anneal.png"),
    )

    # Plots: AUC (raw)
    plot_boxplot(
        df_all,
        metric="auc",
        out_path=os.path.join(base_dir, "internalization_auc_boxplot_anneal.png"),
    )
    plot_bar(
        summary,
        metric_mean="auc_mean",
        metric_std="auc_std",
        out_path=os.path.join(base_dir, "internalization_auc_bar_anneal.png"),
    )


if __name__ == "__main__":
    main()
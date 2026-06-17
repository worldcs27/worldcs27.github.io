#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt


FILES = {
    "base": "internalization_E_base.csv",
    "fast": "internalization_E_fast.csv",
    "slow": "internalization_E_slow.csv",
    "cosine": "internalization_E_cosine.csv",
}


def load_and_concat(base_dir: str) -> pd.DataFrame:
    dfs = []
    for schedule, fname in FILES.items():
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue
        df = pd.read_csv(path)
        if "schedule" not in df.columns:
            df["schedule"] = schedule
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No CSV files loaded.")
    return pd.concat(dfs, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # 主指标：E_l2, E_tail_l2, E_mae
    agg_map = {
        "E_l2": ["mean", "std"],
        "E_tail_l2": ["mean", "std"],
        "E_mae": ["mean", "std"],
        "norm_dz_l2": ["mean"],
        "norm_b_l2": ["mean"],
        "norm_dz_minus_b_l2": ["mean"],
    }
    grouped = df.groupby("schedule").agg(agg_map)
    # 展平多级列名
    grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.values]
    grouped["n_contexts"] = df.groupby("schedule")["context_id"].count()
    summary = grouped.reset_index()

    print("\n=== Summary by schedule ===")
    cols_to_print = [
        "schedule",
        "E_l2_mean",
        "E_l2_std",
        "E_tail_l2_mean",
        "E_tail_l2_std",
        "E_mae_mean",
        "E_mae_std",
        "norm_dz_l2_mean",
        "norm_b_l2_mean",
        "norm_dz_minus_b_l2_mean",
        "n_contexts",
    ]
    print(summary[cols_to_print].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return summary


def _available_schedules(series, desired_order):
    return [s for s in desired_order if s in series.values]


def plot_metric_boxplot(df: pd.DataFrame, metric: str, out_path: str):
    plt.figure(figsize=(6, 4))
    desired_order = ["slow", "base", "cosine", "fast"]
    available = _available_schedules(df["schedule"], desired_order)
    if not available:
        raise ValueError(f"No valid schedules found for {metric} boxplot.")

    df["schedule"] = pd.Categorical(df["schedule"], categories=available, ordered=True)
    df_sorted = df.sort_values("schedule")
    data = [df_sorted[df_sorted["schedule"] == s][metric].values for s in available]

    plt.boxplot(
        data,
        tick_labels=available,
        showfliers=True,
    )
    plt.ylabel(metric)
    plt.xlabel("Annealing schedule")
    plt.title(f"{metric} by annealing schedule")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved boxplot to {out_path}")
    plt.close()


def plot_metric_bar(summary: pd.DataFrame, metric_mean: str, metric_std: str, out_path: str):
    plt.figure(figsize=(6, 4))
    desired_order = ["slow", "base", "cosine", "fast"]
    available = _available_schedules(summary["schedule"], desired_order)
    if not available:
        raise ValueError(f"No valid schedules found for {metric_mean} barplot.")

    s_ordered = summary.set_index("schedule").loc[available].reset_index()
    x = range(len(available))
    means = s_ordered[metric_mean].values
    stds = s_ordered[metric_std].values

    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, available)
    plt.ylabel(metric_mean)
    plt.xlabel("Annealing schedule")
    plt.title(f"{metric_mean} (± {metric_std}) by schedule")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved barplot to {out_path}")
    plt.close()


def main():
    base_dir = os.path.dirname(__file__)
    df = load_and_concat(base_dir)
    summary = summarize(df)

    # 保存汇总 CSV
    out_summary = os.path.join(base_dir, "internalization_E_summary_anneal.csv")
    summary.to_csv(out_summary, index=False)
    print(f"[INFO] Saved summary CSV to {out_summary}")

    # 绘图：E_tail_l2
    plot_metric_boxplot(df, metric="E_tail_l2", out_path=os.path.join(base_dir, "internalization_E_tail_l2_boxplot.png"))
    plot_metric_bar(
        summary,
        metric_mean="E_tail_l2_mean",
        metric_std="E_tail_l2_std",
        out_path=os.path.join(base_dir, "internalization_E_tail_l2_bar.png"),
    )

    # 绘图：E_mae
    plot_metric_boxplot(df, metric="E_mae", out_path=os.path.join(base_dir, "internalization_E_mae_boxplot.png"))
    plot_metric_bar(
        summary,
        metric_mean="E_mae_mean",
        metric_std="E_mae_std",
        out_path=os.path.join(base_dir, "internalization_E_mae_bar.png"),
    )


if __name__ == "__main__":
    main()
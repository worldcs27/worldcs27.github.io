#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os


def main() -> None:
    df = pd.read_csv("entk_vs_dlogp.csv")

    plt.figure(figsize=(5, 4))
    for t, c in [("related_rare", "C0"), ("unrelated_rare", "C1"), ("wrong", "C2")]:
        sub = df[df["type"] == t]
        if sub.empty:
            continue
        plt.scatter(
            sub["entk_sim_ctx_bundle"],
            sub["delta_log_prob"],
            s=20,
            alpha=0.7,
            label=t,
            c=c,
        )

    plt.xlabel("eNTK sim (A vs context bundle)")
    plt.ylabel("Δ log p")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.abspath("fig_entk_vs_dlogp.png")
    plt.savefig(out_path, dpi=200)
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()


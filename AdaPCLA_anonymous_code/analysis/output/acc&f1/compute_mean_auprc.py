#!/usr/bin/env python3
"""Compute mean AUPRC from compare_real_halo_mymodel2.csv for each model/seed.
Output: augmented summary CSVs with mean_auprc column, and LaTeX table values.
"""
import csv
import os
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path("ADAPCLA_ROOT")
SUMMARY_MIMIC3 = Path(__file__).parent / "seed3_summary_with_synteg_and_adapcla_mimic3.csv"
SUMMARY_MIMIC4 = Path(__file__).parent / "seed3_summary_with_synteg_and_adapcla_mimic4.csv"


def mean_auprc_from_compare(csv_path: str, model: str = "") -> float | None:
    """Compute mean AUPRC over labels for the synthetic model.
    If model given and CSV has multiple sources (HALO, GPT, etc.), filter by source==model.
    Else use first non-Real source (MyModel2).
    """
    p = Path(csv_path)
    if not p.exists():
        p = WORKSPACE / csv_path.replace("ADAPCLA_ROOT", "").lstrip("/")
    if not p.exists():
        p = Path(csv_path)
    if not p.exists():
        return None
    auprcs = []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row.get("source", "").strip()
            if src.lower() == "real":
                continue
            if model:
                # MIMIC-III: per-model files use source "MyModel2"
                # MIMIC-IV suite: source is model name (HALO, GPT, etc.)
                # AdaPCLA MIMIC-IV: own file with source "MyModel2"
                if src == model or src == "MyModel2":
                    pass
                else:
                    continue
            try:
                auprcs.append(float(row["AUPRC"]))
            except (KeyError, ValueError):
                pass
    return sum(auprcs) / len(auprcs) if auprcs else None


def process_summary(summary_path: Path, dataset: str) -> tuple[list[dict], dict]:
    """Process one summary CSV, add mean_auprc, return rows and model means."""
    rows = []
    model_seeds = defaultdict(list)  # model -> [(seed, mean_auprc), ...]
    with open(summary_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames + ["mean_auprc"] if "mean_auprc" not in (r.fieldnames or []) else r.fieldnames
        for row in r:
            model = row.get("model", "")
            seed = row.get("seed", "")
            if seed == "mean±std" or model == "Real":
                row["mean_auprc"] = ""
                rows.append(row)
                continue
            eval_csv = row.get("eval_csv", "")
            auprc = mean_auprc_from_compare(eval_csv, model=model) if eval_csv else None
            row["mean_auprc"] = f"{auprc:.4f}" if auprc is not None else ""
            rows.append(row)
            if auprc is not None and model != "Real":
                model_seeds[model].append((seed, auprc))
    # Compute mean±std per model
    model_stats = {}
    for model, vals in model_seeds.items():
        if vals:
            auprcs = [a for _, a in vals]
            m = sum(auprcs) / len(auprcs)
            v = sum((x - m) ** 2 for x in auprcs) / len(auprcs) if len(auprcs) > 1 else 0
            std = v ** 0.5
            model_stats[model] = (m, std)
    return rows, model_stats


def main():
    for summary_path, dataset in [(SUMMARY_MIMIC3, "MIMIC-III"), (SUMMARY_MIMIC4, "MIMIC-IV")]:
        if not summary_path.exists():
            continue
        rows, stats = process_summary(summary_path, dataset)
        out_path = summary_path.parent / f"{summary_path.stem}_with_auprc.csv"
        fieldnames = list(rows[0].keys()) if rows else []
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n{dataset} -> {out_path}")
        print("Model mean_auprc mean±std:")
        for m in ["GPT", "LSTM", "EVA", "SynTEG", "HALO", "PCLA", "AdaPCLA"]:
            if m in stats:
                mn, sd = stats[m]
                print(f"  {m}: {mn:.4f}±{sd:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
从 output/ 下已有的 eval_*/compare_real_halo_mymodel2.csv 汇总成 zeroshot_baselines_table.csv。
用于：主流程中途中断时，用已有结果生成表格（不必重跑全部）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ZERO_SHOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ZERO_SHOT_DIR / "output"
# AdaPCLA 结果表（由 ../output/zero-shot/run_zeroshot.sh 产出）
ADAPCLA_CSV = ZERO_SHOT_DIR.parent / "output" / "zero-shot" / "output" / "zeroshot_table3.csv"


def parse_mean_acc_f1_auprc(csv_path: Path, source: str = "MyModel2") -> tuple[float, float, float]:
    accs, f1s, auprcs = [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") != source:
                continue
            a, f1, au = row.get("Accuracy"), row.get("F1 Score"), row.get("AUPRC")
            if a not in (None, ""):
                accs.append(float(a))
            if f1 not in (None, ""):
                f1s.append(float(f1))
            if au not in (None, ""):
                auprcs.append(float(au))
    return (
        float(np.mean(accs)) if accs else 0.0,
        float(np.mean(f1s)) if f1s else 0.0,
        float(np.mean(auprcs)) if auprcs else 0.0,
    )


def main():
    # eval_gpt_to_iv -> GPT, III→IV, MIMIC-IV
    # eval_gpt_to_iii -> GPT, IV→III, MIMIC-III
    name_map = [
        ("eval_gpt_to_iv", "GPT", "MIMIC-IV", "GPT (III→IV zero-shot)"),
        ("eval_gpt_to_iii", "GPT", "MIMIC-III", "GPT (IV→III zero-shot)"),
        ("eval_lstm_to_iv", "LSTM", "MIMIC-IV", "LSTM (III→IV zero-shot)"),
        ("eval_lstm_to_iii", "LSTM", "MIMIC-III", "LSTM (IV→III zero-shot)"),
        ("eval_eva_to_iv", "EVA", "MIMIC-IV", "EVA (III→IV zero-shot)"),
        ("eval_eva_to_iii", "EVA", "MIMIC-III", "EVA (IV→III zero-shot)"),
        ("eval_synteg_to_iv", "SynTEG", "MIMIC-IV", "SynTEG (III→IV zero-shot)"),
        ("eval_synteg_to_iii", "SynTEG", "MIMIC-III", "SynTEG (IV→III zero-shot)"),
        ("eval_halo_to_iv", "HALO", "MIMIC-IV", "HALO (III→IV zero-shot)"),
        ("eval_halo_to_iii", "HALO", "MIMIC-III", "HALO (IV→III zero-shot)"),
    ]
    rows = []
    for dir_name, _bl, target_name, method_name in name_map:
        csv_path = OUT_DIR / dir_name / "compare_real_halo_mymodel2.csv"
        if not csv_path.exists():
            continue
        acc, f1, auprc = parse_mean_acc_f1_auprc(csv_path)
        rows.append({"target": target_name, "method": method_name, "Acc": acc, "F1": f1, "AUPRC": auprc})
    # 并入 AdaPCLA 结果（若存在）
    if ADAPCLA_CSV.exists():
        with open(ADAPCLA_CSV, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append({
                    "target": row.get("target", ""),
                    "method": row.get("method", ""),
                    "Acc": float(row.get("Acc", 0) or 0),
                    "F1": float(row.get("F1", 0) or 0),
                    "AUPRC": float(row.get("AUPRC", 0) or 0),
                })
    if not rows:
        print("No eval CSV found under output/; nothing to write.")
        return
    out_csv = OUT_DIR / "zeroshot_baselines_table.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["target", "method", "Acc", "F1", "AUPRC"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows (baselines + AdaPCLA if present).")


if __name__ == "__main__":
    main()

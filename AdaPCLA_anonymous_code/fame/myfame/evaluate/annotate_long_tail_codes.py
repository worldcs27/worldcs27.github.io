#!/usr/bin/env python3
"""
Annotate long-tail code lists with ICD-9 diagnosis/procedure titles.

Inputs:
  - top1pct CSV and remaining90pct CSV produced by plot_mimiciii_long_tail.py
  - D_ICD_DIAGNOSES.csv (required)
  - D_ICD_PROCEDURES.csv (optional)

Outputs:
  - annotated CSVs with SHORT_TITLE/LONG_TITLE when available
  - summary JSON with top chapters and example codes
"""

from __future__ import annotations

import argparse
import csv
import json
import os


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_icd_map(path: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            code = (row.get("ICD9_CODE") or "").strip().strip('"')
            if not code:
                continue
            out[code] = {
                "short": (row.get("SHORT_TITLE") or "").strip().strip('"'),
                "long": (row.get("LONG_TITLE") or "").strip().strip('"'),
            }
    return out


def _write_csv(rows: list[dict], path: str, fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory containing mimiciii_train_top1pct_codes.csv etc.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--diag_csv", required=True, help="Path to D_ICD_DIAGNOSES.csv")
    ap.add_argument("--proc_csv", default=None, help="Optional path to D_ICD_PROCEDURES.csv")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    diag = _load_icd_map(args.diag_csv)
    proc = _load_icd_map(args.proc_csv) if args.proc_csv else {}

    def lookup(code: str) -> tuple[str, str, str]:
        c = str(code).strip()
        if c in diag:
            return "diagnosis", diag[c]["short"], diag[c]["long"]
        if c in proc:
            return "procedure", proc[c]["short"], proc[c]["long"]
        return "", "", ""

    summaries = {}
    for name in ["mimiciii_train_top1pct_codes.csv", "mimiciii_train_remaining90pct_codes.csv"]:
        in_path = os.path.join(args.in_dir, name)
        rows = _read_csv(in_path)
        out_rows: list[dict] = []
        for row in rows:
            code = row.get("code", "")
            typ, short, long = lookup(code)
            out = dict(row)
            out["icd_type"] = typ
            out["short_title"] = short
            out["long_title"] = long
            out_rows.append(out)

        out_path = os.path.join(args.out_dir, name.replace(".csv", "_annotated.csv"))
        fieldnames = list(rows[0].keys()) + ["icd_type", "short_title", "long_title"] if rows else ["icd_type", "short_title", "long_title"]
        _write_csv(out_rows, out_path, fieldnames=fieldnames)

        # Quick summary: top-10 by visit_count.
        def vc(r):
            try:
                return int(r.get("visit_count") or 0)
            except Exception:
                return 0

        top10 = sorted(out_rows, key=vc, reverse=True)[:10]
        summaries[name] = [
            {
                "code": r.get("code", ""),
                "visit_count": vc(r),
                "chapter": r.get("chapter", ""),
                "icd_type": r.get("icd_type", ""),
                "short_title": r.get("short_title", ""),
            }
            for r in top10
        ]

    summary_path = os.path.join(args.out_dir, "icd_annotation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()


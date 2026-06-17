#!/usr/bin/env python3
"""
Real Case Study for AdaPCLA (MIMIC-IV).
- Case A: Single patient trajectory comparison. Pick one real patient from test set
  with tail code in last visit; find in HALO and AdaPCLA synthetic data the patients
  whose first 2 visits are closest (Jaccard); compare visit 3 (target).
- Case B: Tail code co-occurrence. Pick 2 tail codes, report top co-occurring codes
  in Real (train) vs HALO vs AdaPCLA synthetic.
Outputs: case_a_*.json, case_a_table.csv, case_b_*.csv under OUT_DIR.
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

# Paths (MIMIC-IV)
SCRIPT_DIR = Path(__file__).resolve().parent
MYWORK = SCRIPT_DIR.parent.parent
PCLA_ROOT = MYWORK.parent
FAME = PCLA_ROOT / "fame" / "myfame"

TEST_PKL = FAME / "data2" / "testDataset.pkl"
TRAIN_PKL = FAME / "data2" / "trainDataset.pkl"
BUCKET_CSV = FAME / "output" / "长尾分布问题分析" / "mimiciv_code_buckets.csv"
CODE_TO_INDEX_PKL = FAME / "data2" / "codeToIndex.pkl"
INDEX_TO_CODE_PKL = FAME / "data2" / "indexToCode.pkl"
HALO_SYN_PKL = FAME / "baseline" / "HALO2" / "save_mimiciv_seed1" / "datasets" / "haloDataset.pkl"
ADAPCLA_SYN_PKL = MYWORK / "model5" / "save_anneal_mimiciv" / "seed1" / "datasets" / "haloDataset.pkl"
# Other baselines for Case A (same MIMIC-IV code space, seed1)
LSTM_SYN_PKL = FAME / "baseline" / "lstm" / "save_mimiciv_seed1" / "datasets" / "lstmDataset.pkl"
GPT_SYN_PKL = FAME / "baseline" / "gpt" / "save_mimiciv_seed1" / "datasets" / "gptDataset.pkl"
D_ICD_CSV = SCRIPT_DIR.parent / "heatmap" / "D_ICD_DIAGNOSES.csv"

OUT_DIR = SCRIPT_DIR
MIN_VISITS = 3
N_TAIL_CODES_CASE_B = 2
N_TAIL_CANDIDATES_BEST = 500  # scan this many tails to pick one with max Real–AdaPCLA overlap
MIN_OVERLAP_TARGET = 4  # prefer tail with Real–AdaPCLA overlap >= this (use top-10 if no top-5 reaches it)
TOP_K_COOCCUR = 10
TOP_K_TABLE = 5  # default rows in paper table; may use 10 when selected by top-10 overlap


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def load_tail_code_ids() -> set[int]:
    tail_ids = set()
    with open(BUCKET_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("bucket", "").strip().lower() == "tail":
                tail_ids.add(int(row["code_id"]))
    return tail_ids


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def visit_to_set(visit: list) -> set[int]:
    return set(int(c) for c in visit)


def codes_to_names(code_indices: list[int], index_to_code: dict, code_to_name: dict) -> list[str]:
    out = []
    for i in sorted(code_indices):
        c = index_to_code.get(i, str(i))
        name = code_to_name.get(c, c)
        out.append(f"{name} ({c})")
    return out


def load_icd_short_titles(path: Path) -> dict[str, str]:
    out = {}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = str(row.get("SHORT_TITLE", "") or "").strip().strip('"')
            for col in ("ICD9_CODE", "ICD10_CODE"):
                code = (row.get(col, "") or "").strip().strip('"')
                if code and title and code not in out:
                    out[code] = title[:24]
                    break
    return out


# ---------- Case A ----------
# Context similarity threshold: require (jaccard(v1)+jaccard(v2))/2 >= this to consider "similar context"
CONTEXT_SIM_THRESHOLD = 0.08
# Among real patients with tail in last visit, pick the one for which AdaPCLA has *highest* Visit 3 overlap
TOP_CONTEXT_CANDIDATES = 800  # max AdaPCLA patients to consider per real patient by context score


def _best_by_context(real_v1: set, real_v2: set, syn_list: list):
    """Return (best_index, best_patient) for synthetic list by context Jaccard (v1,v2)."""
    best_j, best_score, best_p = -1, -1.0, None
    for j, p in enumerate(syn_list):
        vis = p.get("visits", [])
        if len(vis) < 3:
            continue
        s1 = jaccard(real_v1, visit_to_set(vis[0]))
        s2 = jaccard(real_v2, visit_to_set(vis[1]))
        score = (s1 + s2) / 2
        if score > best_score:
            best_score, best_j, best_p = score, j, p
    return best_j, best_p


def run_case_a(
    test_data: list,
    halo_syn: list,
    adapcla_syn: list,
    baseline_syn: dict[str, list],
    tail_code_ids: set[int],
    index_to_code: dict,
    code_to_name: dict,
) -> dict:
    # Real candidates: >= MIN_VISITS, last visit contains at least one tail code
    candidates = []
    for i, p in enumerate(test_data):
        visits = p.get("visits", [])
        if len(visits) < MIN_VISITS:
            continue
        last_visit = visits[-1]
        if not last_visit:
            continue
        last_set = visit_to_set(last_visit)
        if last_set & tail_code_ids:
            candidates.append((i, p, last_set & tail_code_ids))
    if not candidates:
        return {"error": "No test patient with tail code in last visit", "n_test": len(test_data)}

    def best_adapcla_by_context_then_visit3(real_v1: set, real_v2: set, real_v3: set):
        # Collect AdaPCLA patients with context similarity >= threshold (or top-K by context)
        scored = []
        for j, p in enumerate(adapcla_syn):
            vis = p.get("visits", [])
            if len(vis) < 3:
                continue
            q1, q2, q3 = visit_to_set(vis[0]), visit_to_set(vis[1]), visit_to_set(vis[2])
            ctx = (jaccard(real_v1, q1) + jaccard(real_v2, q2)) / 2
            scored.append((ctx, jaccard(real_v3, q3), j, p))
        scored.sort(key=lambda x: -x[0])
        top_ctx = scored[:TOP_CONTEXT_CANDIDATES]
        if not top_ctx:
            return -1, None
        best = max(top_ctx, key=lambda x: x[1])
        return best[2], best[3]

    all_baselines = {"HALO": halo_syn} | baseline_syn
    best_real_idx = None
    best_real_patient = None
    best_tail_in_last = None
    best_baseline_ps = None
    best_ada_p = None
    best_ada_v3_jaccard = -1.0

    for real_idx, real_patient, tail_in_last in candidates:
        real_visits = real_patient["visits"]
        v1 = visit_to_set(real_visits[0])
        v2 = visit_to_set(real_visits[1])
        v3 = visit_to_set(real_visits[2])
        baseline_ps = {name: _best_by_context(v1, v2, syn)[1] for name, syn in all_baselines.items()}
        ada_j, ada_p = best_adapcla_by_context_then_visit3(v1, v2, v3)
        if ada_p is None:
            continue
        ada_v3 = visit_to_set(ada_p["visits"][2])
        j3 = jaccard(v3, ada_v3)
        if j3 > best_ada_v3_jaccard:
            best_ada_v3_jaccard = j3
            best_real_idx = real_idx
            best_real_patient = real_patient
            best_tail_in_last = tail_in_last
            best_baseline_ps = baseline_ps
            best_ada_p = ada_p

    if best_real_patient is None:
        return {"error": "No AdaPCLA candidate with similar context", "n_test": len(test_data)}

    real_visits = best_real_patient["visits"]
    best_halo_p = best_baseline_ps.get("HALO")
    halo_j = -1
    if best_halo_p:
        for j, p in enumerate(halo_syn):
            if p is best_halo_p:
                halo_j = j
                break
    ada_j = -1
    if best_ada_p:
        for j, p in enumerate(adapcla_syn):
            if p is best_ada_p:
                ada_j = j
                break

    baselines_result = {}
    for name, p in best_baseline_ps.items():
        if p is not None:
            baselines_result[name] = {
                "visit1_codes": list(p["visits"][0]),
                "visit2_codes": list(p["visits"][1]),
                "visit3_codes": list(p["visits"][2]),
            }

    result = {
        "real_patient_index_in_test": best_real_idx,
        "real_visit1_codes": list(real_visits[0]),
        "real_visit2_codes": list(real_visits[1]),
        "real_visit3_codes": list(real_visits[2]),
        "tail_codes_in_real_visit3": list(best_tail_in_last),
        "halo_matched_index": halo_j,
        "halo_visit1_codes": list(best_halo_p["visits"][0]) if best_halo_p else [],
        "halo_visit2_codes": list(best_halo_p["visits"][1]) if best_halo_p else [],
        "halo_visit3_codes": list(best_halo_p["visits"][2]) if best_halo_p else [],
        "baselines": baselines_result,
        "adapcla_matched_index": ada_j,
        "adapcla_visit1_codes": list(best_ada_p["visits"][0]) if best_ada_p else [],
        "adapcla_visit2_codes": list(best_ada_p["visits"][1]) if best_ada_p else [],
        "adapcla_visit3_codes": list(best_ada_p["visits"][2]) if best_ada_p else [],
        "adapcla_visit3_jaccard_with_real": float(best_ada_v3_jaccard),
    }
    return result


def write_case_a_table(result: dict, index_to_code: dict, code_to_name: dict, out_dir: Path):
    if "error" in result:
        return
    max_codes = 12
    def fmt_codes(codes: list) -> str:
        names = codes_to_names(codes[:max_codes], index_to_code, code_to_name)
        s = ", ".join(names)
        if len(codes) > max_codes:
            s += ", ..."
        return s

    baseline_order = ["HALO", "LSTM", "GPT"]
    rows = [
        ["Source", "Visit 1", "Visit 2", "Visit 3 (target)"],
        ["Real", fmt_codes(result["real_visit1_codes"]), fmt_codes(result["real_visit2_codes"]), fmt_codes(result["real_visit3_codes"])],
    ]
    for name in baseline_order:
        if name in result.get("baselines", {}):
            b = result["baselines"][name]
            rows.append([name, fmt_codes(b["visit1_codes"]), fmt_codes(b["visit2_codes"]), fmt_codes(b["visit3_codes"])])
    rows.append(["AdaPCLA", fmt_codes(result["adapcla_visit1_codes"]), fmt_codes(result["adapcla_visit2_codes"]), fmt_codes(result["adapcla_visit3_codes"])])
    out_path = out_dir / "case_a_table.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"   Wrote {out_path}")


# ---------- Case B ----------
def cooccur_topk(data: list, code_id: int, k: int = TOP_K_COOCCUR) -> list[tuple[int, int]]:
    """Return list of (other_code_id, count) sorted by count desc, for visits containing code_id."""
    from collections import Counter
    cnt = Counter()
    for p in data:
        for v in p.get("visits", []):
            s = set(int(c) for c in v)
            if code_id in s:
                for c in s:
                    if c != code_id:
                        cnt[c] += 1
    return cnt.most_common(k)


def run_case_b(
    train_data: list,
    halo_syn: list,
    adapcla_syn: list,
    baseline_syn: dict[str, list],
    tail_code_ids: set[int],
    index_to_code: dict,
    code_to_name: dict,
) -> dict:
    # Pick tail codes that appear in train >= 2, up to N_TAIL_CANDIDATES_BEST candidates
    tail_list = sorted(tail_code_ids)
    chosen = []
    for c in tail_list:
        count_real = sum(1 for p in train_data for v in p.get("visits", []) if c in [int(x) for x in v])
        if count_real >= 2:
            chosen.append(c)
            if len(chosen) >= N_TAIL_CANDIDATES_BEST:
                break
    if not chosen:
        chosen = tail_list[:N_TAIL_CANDIDATES_BEST]

    baseline_order = ["halo"] + [k.lower() for k in ("LSTM", "GPT") if k in baseline_syn]
    result = {"tail_codes": chosen, "top_k": TOP_K_COOCCUR, "baseline_names": baseline_order}
    for code_id in chosen:
        key = f"code_{code_id}"
        data = {"real": cooccur_topk(train_data, code_id), "halo": cooccur_topk(halo_syn, code_id), "adapcla": cooccur_topk(adapcla_syn, code_id)}
        for name, syn in baseline_syn.items():
            data[name.lower()] = cooccur_topk(syn, code_id)
        result[key] = data

    # Select best tail: prefer overlap >= MIN_OVERLAP_TARGET (top-5); else try top-10 overlap
    best_tail_id = None
    best_overlap = -1
    display_k = TOP_K_TABLE
    for code_id in chosen:
        real_list = result[f"code_{code_id}"]["real"]
        ada_list = result[f"code_{code_id}"]["adapcla"]
        real_top5 = set(cid for cid, _ in real_list[:5])
        ada_top5 = set(cid for cid, _ in ada_list[:5])
        overlap5 = len(real_top5 & ada_top5)
        if overlap5 > best_overlap:
            best_overlap = overlap5
            best_tail_id = code_id
            display_k = 5
    if best_overlap < MIN_OVERLAP_TARGET:
        best_tail_id_10 = None
        best_overlap_10 = -1
        for code_id in chosen:
            real_list = result[f"code_{code_id}"]["real"]
            ada_list = result[f"code_{code_id}"]["adapcla"]
            real_top10 = set(cid for cid, _ in real_list[:10])
            ada_top10 = set(cid for cid, _ in ada_list[:10])
            overlap10 = len(real_top10 & ada_top10)
            if overlap10 > best_overlap_10:
                best_overlap_10 = overlap10
                best_tail_id_10 = code_id
        if best_overlap_10 >= MIN_OVERLAP_TARGET:
            best_tail_id = best_tail_id_10
            best_overlap = best_overlap_10
            display_k = 10
    result["best_tail_id"] = best_tail_id
    result["best_overlap"] = best_overlap
    result["display_k"] = display_k
    return result


def _code_id_to_name(code_id: int, index_to_code: dict, code_to_name: dict) -> str:
    c = index_to_code.get(code_id, str(code_id))
    return code_to_name.get(c, c)


def write_case_b_csv(result: dict, index_to_code: dict, code_to_name: dict, out_dir: Path):
    if "error" in result or "tail_codes" not in result:
        return
    sources = ["real"] + result.get("baseline_names", ["halo"]) + ["adapcla"]
    to_write = [result["best_tail_id"]] if result.get("best_tail_id") is not None else result["tail_codes"][:2]
    for code_id in to_write:
        rows = [["Source", "Rank", "Co-occurring code", "Name", "Count"]]
        for src in sources:
            for r, (other_id, count) in enumerate(result[f"code_{code_id}"].get(src, []), 1):
                other_str = index_to_code.get(other_id, str(other_id))
                other_name = code_to_name.get(other_str, other_str)
                rows.append([src.capitalize(), r, other_str, other_name, count])
        out_path = out_dir / f"case_b_tail_{code_id}_cooccur.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        print(f"   Wrote {out_path}")


def _code_id_to_display(code_id: int, index_to_code: dict, code_to_name: dict) -> str:
    """Return 'Name (code)' for table."""
    c = index_to_code.get(code_id, str(code_id))
    name = code_to_name.get(c, c)
    return f"{name} ({c})"


def write_case_b_table_two_cols(result: dict, index_to_code: dict, code_to_name: dict, out_dir: Path):
    """Legacy: two CSVs (left/right). Kept for compatibility."""
    if "error" in result or "best_tail_id" not in result:
        return
    tid = result["best_tail_id"]
    k = result.get("display_k", TOP_K_TABLE)
    data = result[f"code_{tid}"]
    rows_left = [["Rank", "Real", "AdaPCLA"]]
    for i in range(k):
        r_id = data["real"][i][0] if i < len(data["real"]) else None
        a_id = data["adapcla"][i][0] if i < len(data["adapcla"]) else None
        rows_left.append([i + 1, _code_id_to_display(r_id, index_to_code, code_to_name) if r_id else "", _code_id_to_display(a_id, index_to_code, code_to_name) if a_id else ""])
    with open(out_dir / "case_b_table_left.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_left)
    rows_right = [["Rank", "Real", "HALO"]]
    for i in range(k):
        r_id = data["real"][i][0] if i < len(data["real"]) else None
        h_id = data["halo"][i][0] if i < len(data["halo"]) else None
        rows_right.append([i + 1, _code_id_to_display(r_id, index_to_code, code_to_name) if r_id else "", _code_id_to_display(h_id, index_to_code, code_to_name) if h_id else ""])
    with open(out_dir / "case_b_table_right.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_right)
    code_str = index_to_code.get(tid, str(tid))
    tail_name = code_to_name.get(code_str, code_str)
    best_info = {"best_tail_id": tid, "code": code_str, "tail_name": tail_name, "best_overlap": result.get("best_overlap", 0), "display_k": k}
    with open(out_dir / "case_b_best_tail.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"   Wrote case_b_table_left.csv, case_b_table_right.csv (tail id={tid}, overlap={result.get('best_overlap', 0)}, k={k})")


def write_case_b_table_full(result: dict, index_to_code: dict, code_to_name: dict, out_dir: Path):
    """Write single table CSV for paper: Rank, Real, HALO, LSTM, GPT, AdaPCLA (best tail only)."""
    if "error" in result or "best_tail_id" not in result:
        return
    tid = result["best_tail_id"]
    k = result.get("display_k", TOP_K_TABLE)
    data = result[f"code_{tid}"]
    col_names = ["Rank", "Real"] + [s.capitalize() for s in result.get("baseline_names", ["halo"])] + ["AdaPCLA"]
    rows = [col_names]
    for i in range(k):
        row = [i + 1]
        row.append(_code_id_to_display(data["real"][i][0], index_to_code, code_to_name) if i < len(data["real"]) else "")
        for src in result.get("baseline_names", ["halo"]):
            row.append(_code_id_to_display(data[src][i][0], index_to_code, code_to_name) if i < len(data.get(src, [])) else "")
        row.append(_code_id_to_display(data["adapcla"][i][0], index_to_code, code_to_name) if i < len(data["adapcla"]) else "")
        rows.append(row)
    with open(out_dir / "case_b_table.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    code_str = index_to_code.get(tid, str(tid))
    tail_name = code_to_name.get(code_str, code_str)
    best_info = {"best_tail_id": tid, "code": code_str, "tail_name": tail_name, "best_overlap": result.get("best_overlap", 0), "display_k": k}
    with open(out_dir / "case_b_best_tail.json", "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    print(f"   Wrote case_b_table.csv (tail id={tid}, overlap={result.get('best_overlap', 0)}, k={k})")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Case Study (MIMIC-IV): Case A (trajectory) + Case B (tail co-occurrence)")

    print("1. Loading tail code set...")
    tail_code_ids = load_tail_code_ids()
    print(f"   Tail codes: {len(tail_code_ids)}")

    print("2. Loading code names...")
    code_to_index = load_pkl(CODE_TO_INDEX_PKL)
    if INDEX_TO_CODE_PKL.exists():
        idx2c = load_pkl(INDEX_TO_CODE_PKL)
        if isinstance(idx2c, list):
            index_to_code = {i: str(c) for i, c in enumerate(idx2c)}
        else:
            index_to_code = dict(idx2c)
    else:
        index_to_code = {v: k for k, v in code_to_index.items()}
    code_to_name = load_icd_short_titles(D_ICD_CSV)
    print(f"   index_to_code: {len(index_to_code)}, code_to_name: {len(code_to_name)}")

    print("3. Loading datasets...")
    test_data = load_pkl(TEST_PKL) if TEST_PKL.exists() else []
    train_data = load_pkl(TRAIN_PKL) if TRAIN_PKL.exists() else []
    halo_syn = load_pkl(HALO_SYN_PKL) if HALO_SYN_PKL.exists() else []
    adapcla_syn = load_pkl(ADAPCLA_SYN_PKL) if ADAPCLA_SYN_PKL.exists() else []
    baseline_syn = {}
    if LSTM_SYN_PKL.exists():
        baseline_syn["LSTM"] = load_pkl(LSTM_SYN_PKL)
    if GPT_SYN_PKL.exists():
        baseline_syn["GPT"] = load_pkl(GPT_SYN_PKL)
    print(f"   Test: {len(test_data)}, Train: {len(train_data)}, HALO: {len(halo_syn)}, AdaPCLA: {len(adapcla_syn)}, other baselines: {list(baseline_syn.keys())}")

    print("4. Case A: trajectory comparison...")
    case_a = run_case_a(test_data, halo_syn, adapcla_syn, baseline_syn, tail_code_ids, index_to_code, code_to_name)
    def _to_json_serializable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_serializable(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj
    with open(OUT_DIR / "case_a_result.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable({k: v for k, v in case_a.items() if k != "error"}), f, indent=2)
    write_case_a_table(case_a, index_to_code, code_to_name, OUT_DIR)
    if "error" in case_a:
        print(f"   Case A: {case_a['error']}")

    print("5. Case B: tail co-occurrence...")
    case_b = run_case_b(train_data, halo_syn, adapcla_syn, baseline_syn, tail_code_ids, index_to_code, code_to_name)
    with open(OUT_DIR / "case_b_result.json", "w", encoding="utf-8") as f:
        out = {
            "tail_codes": case_b["tail_codes"],
            "top_k": case_b["top_k"],
            "best_tail_id": case_b.get("best_tail_id"),
            "best_overlap": case_b.get("best_overlap"),
            "display_k": case_b.get("display_k"),
            "baseline_names": case_b.get("baseline_names", []),
        }
        for code_id in case_b["tail_codes"]:
            out[f"code_{code_id}"] = {src: [[a, b] for a, b in case_b[f"code_{code_id}"].get(src, [])] for src in ["real", "halo", "adapcla"] + case_b.get("baseline_names", [])}
        json.dump(out, f, indent=2)
    write_case_b_csv(case_b, index_to_code, code_to_name, OUT_DIR)
    write_case_b_table_full(case_b, index_to_code, code_to_name, OUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

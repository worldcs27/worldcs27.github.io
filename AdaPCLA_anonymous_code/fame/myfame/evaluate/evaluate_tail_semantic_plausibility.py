#!/usr/bin/env python3
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import csv
import json
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np


DEFAULT_DATA_DIR = "FAME_ROOT"
DEFAULT_REF_PATH = os.path.join(DEFAULT_DATA_DIR, "trainDataset.pkl")
DEFAULT_REAL_EVAL_PATH = os.path.join(DEFAULT_DATA_DIR, "testDataset.pkl")
DEFAULT_SYN_PATH = "FAME_ROOT"
DEFAULT_BUCKET_CSV = "FAME_ROOT/output/长尾分布问题分析/mimiciii_code_buckets.csv"
DEFAULT_OUT_DIR = "FAME_ROOT/evaluate/save/tail_semantic_plausibility"


def _load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _desc(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _load_bucket_csv(bucket_csv: str):
    buckets = {"tail": [], "mid": [], "head": []}
    with open(bucket_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            b = (row.get("bucket") or "").strip()
            if b not in buckets:
                continue
            buckets[b].append(int(row["code_id"]))
    code_to_bucket = {}
    for b, ids in buckets.items():
        for cid in ids:
            code_to_bucket[int(cid)] = b
    return buckets, code_to_bucket


def _infer_code_vocab_size(data_dir: str) -> int:
    code_pkl = os.path.join(data_dir, "codeToIndex.pkl")
    if os.path.exists(code_pkl):
        try:
            return int(len(_load_pkl(code_pkl)))
        except Exception:
            pass
    raise FileNotFoundError(f"Cannot infer vocab size; missing or unreadable: {code_pkl}")


def _iter_visits(dataset):
    for p in dataset:
        for v in (p.get("visits") or []):
            yield v


def _visit_code_set(v, *, code_vocab_size: int) -> list[int]:
    # Unique, sorted, int-casted, and in-vocab.
    s = set()
    for c in v or []:
        try:
            cid = int(c)
        except Exception:
            continue
        if 0 <= cid < int(code_vocab_size):
            s.add(cid)
    if not s:
        return []
    return sorted(s)


def _pair_key(i: int, j: int, *, vocab: int) -> int:
    # Precondition: 0<=i<j<vocab
    return i * vocab + j


def _js_divergence_sparse(p: Counter, q: Counter, eps: float = 1e-12) -> float | None:
    if not p and not q:
        return 0.0
    p_total = float(sum(p.values()))
    q_total = float(sum(q.values()))
    if p_total <= 0 or q_total <= 0:
        return None
    keys = set(p.keys()) | set(q.keys())
    js = 0.0
    for k in keys:
        pv = float(p.get(k, 0.0)) / p_total
        qv = float(q.get(k, 0.0)) / q_total
        mv = 0.5 * (pv + qv)
        if pv > 0:
            js += 0.5 * pv * math.log((pv + eps) / (mv + eps))
        if qv > 0:
            js += 0.5 * qv * math.log((qv + eps) / (mv + eps))
    return float(js)


def _topk_keys(counter: Counter, k: int) -> set[int]:
    if k <= 0 or not counter:
        return set()
    return {int(i) for i, _ in counter.most_common(int(k))}


def _build_reference_stats(
    ref_dataset,
    *,
    code_vocab_size: int,
    tail_set: set[int],
    code_to_bucket: dict[int, str],
):
    pair_set = set()
    tail_context = defaultdict(Counter)  # tail_code -> Counter(other_code -> count)
    tail_visit_count = Counter()
    any_tail_other = Counter()  # pooled: other codes in visits where any tail appears
    any_tail_other_bucket_mix = Counter()
    visits_with_tail = 0
    visits = 0
    total_pairs = 0
    total_pairs_by_bucket_pair = Counter()

    for v in _iter_visits(ref_dataset):
        codes = _visit_code_set(v, code_vocab_size=code_vocab_size)
        if len(codes) < 2:
            continue
        visits += 1

        # Pair set (unordered).
        for a_idx in range(len(codes)):
            a = codes[a_idx]
            ba = code_to_bucket.get(a, "other")
            for b_idx in range(a_idx + 1, len(codes)):
                b = codes[b_idx]
                bb = code_to_bucket.get(b, "other")
                pair_set.add(_pair_key(a, b, vocab=code_vocab_size))
                total_pairs += 1
                key = "/".join(sorted((ba, bb)))
                total_pairs_by_bucket_pair[key] += 1

        # Tail-conditioned context.
        tails_here = [c for c in codes if c in tail_set]
        if tails_here:
            visits_with_tail += 1
            codes_set = set(codes)
            tails_set = set(tails_here)
            others_set = codes_set - tails_set
            for other in others_set:
                any_tail_other[other] += 1
                any_tail_other_bucket_mix[code_to_bucket.get(other, "other")] += 1
            for t in tails_here:
                tail_visit_count[t] += 1
                for other in codes_set:
                    if other != t:
                        tail_context[t][other] += 1

    return {
        "pair_set": pair_set,
        "tail_context": tail_context,
        "tail_visit_count": tail_visit_count,
        "any_tail_other": any_tail_other,
        "any_tail_other_bucket_mix": any_tail_other_bucket_mix,
        "ref_visits_with_tail": int(visits_with_tail),
        "ref_visits_with_2plus_codes": int(visits),
        "ref_total_pairs": int(total_pairs),
        "ref_total_pairs_by_bucket_pair": dict(total_pairs_by_bucket_pair),
    }


def _evaluate_dataset(
    dataset,
    *,
    name: str,
    code_vocab_size: int,
    tail_set: set[int],
    code_to_bucket: dict[int, str],
    ref_pair_set: set[int],
    ref_tail_context: dict[int, Counter],
    ref_tail_visit_count: Counter,
    ref_any_tail_other: Counter,
    ref_any_tail_other_bucket_mix: Counter,
    ref_visits_with_tail: int,
    topk: int,
    min_ref_tail_visits: int,
    min_eval_tail_visits: int,
):
    total_visits = 0
    visits_with_2plus_codes = 0
    visits_with_tail = 0
    tail_code_present = Counter()

    total_pairs = 0
    seen_pairs = 0
    total_pairs_tail_involved = 0
    seen_pairs_tail_involved = 0
    total_pairs_by_bucket_pair = Counter()
    seen_pairs_by_bucket_pair = Counter()

    tail_context = defaultdict(Counter)
    js_by_tail = {}
    topk_jaccard_by_tail = {}
    any_tail_other = Counter()
    any_tail_other_bucket_mix = Counter()

    pair_seen_rate_per_visit = []
    tail_pair_seen_rate_per_visit = []

    for v in _iter_visits(dataset):
        total_visits += 1
        codes = _visit_code_set(v, code_vocab_size=code_vocab_size)
        if not codes:
            continue

        tails_here = [c for c in codes if c in tail_set]
        if tails_here:
            visits_with_tail += 1
            for t in tails_here:
                tail_code_present[t] += 1

        if len(codes) < 2:
            continue

        visits_with_2plus_codes += 1

        # Pair plausibility vs reference.
        local_total = 0
        local_seen = 0
        local_tail_total = 0
        local_tail_seen = 0
        for a_idx in range(len(codes)):
            a = codes[a_idx]
            ba = code_to_bucket.get(a, "other")
            for b_idx in range(a_idx + 1, len(codes)):
                b = codes[b_idx]
                bb = code_to_bucket.get(b, "other")
                local_total += 1
                total_pairs += 1
                key = "/".join(sorted((ba, bb)))
                total_pairs_by_bucket_pair[key] += 1

                in_ref = _pair_key(a, b, vocab=code_vocab_size) in ref_pair_set
                if in_ref:
                    local_seen += 1
                    seen_pairs += 1
                    seen_pairs_by_bucket_pair[key] += 1

                if a in tail_set or b in tail_set:
                    local_tail_total += 1
                    total_pairs_tail_involved += 1
                    if in_ref:
                        local_tail_seen += 1
                        seen_pairs_tail_involved += 1

        pair_seen_rate_per_visit.append((local_seen / local_total) if local_total else 0.0)
        tail_pair_seen_rate_per_visit.append((local_tail_seen / local_tail_total) if local_tail_total else 0.0)

        # Tail context counters (conditioned on tail code presence).
        if tails_here:
            codes_set = set(codes)
            tails_set = set(tails_here)
            others_set = codes_set - tails_set
            for other in others_set:
                any_tail_other[other] += 1
                any_tail_other_bucket_mix[code_to_bucket.get(other, "other")] += 1
            for t in tails_here:
                for other in codes_set:
                    if other != t:
                        tail_context[t][other] += 1

    # Tail-context divergence vs reference.
    total_ref_tail_visits = int(sum(ref_tail_visit_count.values()))
    weighted_js_sum_all_defined = 0.0
    weighted_n_all_defined = 0
    macro_js_list_all_defined = []

    weighted_js_sum_filtered = 0.0
    weighted_n_filtered = 0
    macro_js_list_filtered = []

    weighted_jacc_sum_all_defined = 0.0
    weighted_n_jacc_all_defined = 0
    macro_jacc_list_all_defined = []

    weighted_jacc_sum_filtered = 0.0
    weighted_n_jacc_filtered = 0
    macro_jacc_list_filtered = []

    for t, ref_n in ref_tail_visit_count.items():
        ref_n = int(ref_n)
        if ref_n <= 0:
            continue
        t_int = int(t)
        ref_ctx = ref_tail_context.get(t, Counter())
        cur_ctx = tail_context.get(t, Counter())
        ref_ctx_total = int(sum(ref_ctx.values()))
        cur_ctx_total = int(sum(cur_ctx.values()))
        eval_n = int(tail_code_present.get(t_int, 0))
        eval_has_t = eval_n > 0

        js = _js_divergence_sparse(ref_ctx, cur_ctx)
        if js is not None:
            js_by_tail[t_int] = float(js)
            macro_js_list_all_defined.append(float(js))
            weighted_js_sum_all_defined += float(js) * ref_n
            weighted_n_all_defined += ref_n
            if (
                ref_n >= int(min_ref_tail_visits)
                and eval_n >= int(min_eval_tail_visits)
                and ref_ctx_total > 0
                and cur_ctx_total > 0
            ):
                macro_js_list_filtered.append(float(js))
                weighted_js_sum_filtered += float(js) * ref_n
                weighted_n_filtered += ref_n

        ref_top = _topk_keys(ref_ctx, topk)
        cur_top = _topk_keys(cur_ctx, topk)
        denom = len(ref_top | cur_top)
        j = (len(ref_top & cur_top) / denom) if denom else 1.0
        topk_jaccard_by_tail[t_int] = float(j)
        if ref_ctx_total > 0:
            macro_jacc_list_all_defined.append(float(j))
            weighted_jacc_sum_all_defined += float(j) * ref_n
            weighted_n_jacc_all_defined += ref_n
            if (
                ref_n >= int(min_ref_tail_visits)
                and eval_n >= int(min_eval_tail_visits)
                and cur_ctx_total > 0
            ):
                macro_jacc_list_filtered.append(float(j))
                weighted_jacc_sum_filtered += float(j) * ref_n
                weighted_n_jacc_filtered += ref_n

    # Pooled tail context: P(other_code | any tail present in visit).
    pooled_other_js = _js_divergence_sparse(ref_any_tail_other, any_tail_other)
    ref_pooled_top = _topk_keys(ref_any_tail_other, topk)
    eval_pooled_top = _topk_keys(any_tail_other, topk)
    denom = len(ref_pooled_top | eval_pooled_top)
    pooled_other_topk_jaccard = (len(ref_pooled_top & eval_pooled_top) / denom) if denom else 1.0

    pooled_bucket_js = _js_divergence_sparse(ref_any_tail_other_bucket_mix, any_tail_other_bucket_mix)
    ref_bucket_total = float(sum(ref_any_tail_other_bucket_mix.values()))
    eval_bucket_total = float(sum(any_tail_other_bucket_mix.values()))
    ref_bucket_frac = (
        {k: float(v / ref_bucket_total) for k, v in ref_any_tail_other_bucket_mix.items()} if ref_bucket_total > 0 else {}
    )
    eval_bucket_frac = {k: float(v / eval_bucket_total) for k, v in any_tail_other_bucket_mix.items()} if eval_bucket_total > 0 else {}

    out = {
        "name": name,
        "total_visits": int(total_visits),
        "visits_with_2plus_codes": int(visits_with_2plus_codes),
        "visits_with_tail": int(visits_with_tail),
        "tail_codes_present": int(len(tail_code_present)),
        "tail_codes_present_frac": float(len(tail_code_present) / len(tail_set)) if tail_set else None,
        "pair_seen_rate": float(seen_pairs / total_pairs) if total_pairs > 0 else None,
        "tail_involved_pair_seen_rate": float(seen_pairs_tail_involved / total_pairs_tail_involved)
        if total_pairs_tail_involved > 0
        else None,
        "pair_seen_rate_by_bucket_pair": {
            k: (float(seen_pairs_by_bucket_pair[k] / v) if v > 0 else None) for k, v in total_pairs_by_bucket_pair.items()
        },
        "pair_seen_rate_per_visit_stats": _desc(pair_seen_rate_per_visit),
        "tail_pair_seen_rate_per_visit_stats": _desc(tail_pair_seen_rate_per_visit),
        "tail_context_js_weighted_by_ref_defined": (
            float(weighted_js_sum_all_defined / weighted_n_all_defined) if weighted_n_all_defined > 0 else None
        ),
        "tail_context_js_macro_defined": (float(np.mean(macro_js_list_all_defined)) if macro_js_list_all_defined else None),
        "tail_context_js_weighted_by_ref_filtered": (
            float(weighted_js_sum_filtered / weighted_n_filtered) if weighted_n_filtered > 0 else None
        ),
        "tail_context_js_macro_filtered": (float(np.mean(macro_js_list_filtered)) if macro_js_list_filtered else None),
        "tail_context_topk_jaccard_weighted_by_ref_defined": (
            float(weighted_jacc_sum_all_defined / weighted_n_jacc_all_defined) if weighted_n_jacc_all_defined > 0 else None
        ),
        "tail_context_topk_jaccard_macro_defined": (
            float(np.mean(macro_jacc_list_all_defined)) if macro_jacc_list_all_defined else None
        ),
        "tail_context_topk_jaccard_weighted_by_ref_filtered": (
            float(weighted_jacc_sum_filtered / weighted_n_jacc_filtered) if weighted_n_jacc_filtered > 0 else None
        ),
        "tail_context_topk_jaccard_macro_filtered": (
            float(np.mean(macro_jacc_list_filtered)) if macro_jacc_list_filtered else None
        ),
        "tail_context_filter": {
            "min_ref_tail_visits": int(min_ref_tail_visits),
            "min_eval_tail_visits": int(min_eval_tail_visits),
        },
        "tail_context_pooled": {
            "ref_visits_with_tail": int(ref_visits_with_tail),
            "eval_visits_with_tail": int(visits_with_tail),
            "ref_any_tail_other_total": int(sum(ref_any_tail_other.values())),
            "eval_any_tail_other_total": int(sum(any_tail_other.values())),
            "js_other_codes": pooled_other_js,
            "topk_other_codes_jaccard": float(pooled_other_topk_jaccard),
            "js_other_bucket_mix": pooled_bucket_js,
            "ref_other_bucket_mix_frac": ref_bucket_frac,
            "eval_other_bucket_mix_frac": eval_bucket_frac,
        },
        "topk": int(topk),
        "ref_total_tail_visits": int(total_ref_tail_visits),
    }

    per_tail_rows = []
    for t, ref_n in ref_tail_visit_count.items():
        t = int(t)
        per_tail_rows.append(
            {
                "tail_code_id": t,
                "ref_tail_visit_count": int(ref_n),
                "eval_tail_visit_count": int(tail_code_present.get(t, 0)),
                "tail_context_js": js_by_tail.get(t, ""),
                "tail_context_topk_jaccard": topk_jaccard_by_tail.get(t, ""),
            }
        )

    return out, per_tail_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--bucket_csv", default=DEFAULT_BUCKET_CSV)
    ap.add_argument("--ref_path", default=DEFAULT_REF_PATH, help="Reference real dataset for semantic statistics (default: trainDataset.pkl)")
    ap.add_argument("--real_path", default=DEFAULT_REAL_EVAL_PATH, help="Real dataset to evaluate (default: testDataset.pkl)")
    ap.add_argument("--synthetic_path", default=DEFAULT_SYN_PATH)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--topk", type=int, default=20, help="Top-K co-occurring codes for overlap metric")
    ap.add_argument(
        "--min_ref_tail_visits",
        type=int,
        default=1,
        help="Only aggregate per-tail-code context metrics over tail codes with at least this many ref visits.",
    )
    ap.add_argument(
        "--min_eval_tail_visits",
        type=int,
        default=1,
        help="Only aggregate tail-context metrics over tail codes appearing at least this many times in eval dataset.",
    )
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)
    buckets, code_to_bucket = _load_bucket_csv(args.bucket_csv)
    tail_set = set(int(x) for x in buckets.get("tail", []))
    code_vocab_size = _infer_code_vocab_size(args.data_dir)

    ref = _load_pkl(args.ref_path)
    ref_stats = _build_reference_stats(
        ref,
        code_vocab_size=code_vocab_size,
        tail_set=tail_set,
        code_to_bucket=code_to_bucket,
    )

    real = _load_pkl(args.real_path)
    syn = _load_pkl(args.synthetic_path)

    real_summary, real_rows = _evaluate_dataset(
        real,
        name="real",
        code_vocab_size=code_vocab_size,
        tail_set=tail_set,
        code_to_bucket=code_to_bucket,
        ref_pair_set=ref_stats["pair_set"],
        ref_tail_context=ref_stats["tail_context"],
        ref_tail_visit_count=ref_stats["tail_visit_count"],
        ref_any_tail_other=ref_stats["any_tail_other"],
        ref_any_tail_other_bucket_mix=ref_stats["any_tail_other_bucket_mix"],
        ref_visits_with_tail=ref_stats["ref_visits_with_tail"],
        topk=args.topk,
        min_ref_tail_visits=args.min_ref_tail_visits,
        min_eval_tail_visits=args.min_eval_tail_visits,
    )
    syn_summary, syn_rows = _evaluate_dataset(
        syn,
        name="synthetic",
        code_vocab_size=code_vocab_size,
        tail_set=tail_set,
        code_to_bucket=code_to_bucket,
        ref_pair_set=ref_stats["pair_set"],
        ref_tail_context=ref_stats["tail_context"],
        ref_tail_visit_count=ref_stats["tail_visit_count"],
        ref_any_tail_other=ref_stats["any_tail_other"],
        ref_any_tail_other_bucket_mix=ref_stats["any_tail_other_bucket_mix"],
        ref_visits_with_tail=ref_stats["ref_visits_with_tail"],
        topk=args.topk,
        min_ref_tail_visits=args.min_ref_tail_visits,
        min_eval_tail_visits=args.min_eval_tail_visits,
    )

    out = {
        "paths": {
            "bucket_csv": args.bucket_csv,
            "ref_path": args.ref_path,
            "real_path": args.real_path,
            "synthetic_path": args.synthetic_path,
        },
        "code_vocab_size": int(code_vocab_size),
        "tail_codes": int(len(tail_set)),
        "reference": {
            "ref_visits_with_2plus_codes": ref_stats["ref_visits_with_2plus_codes"],
            "ref_visits_with_tail": ref_stats["ref_visits_with_tail"],
            "ref_total_tail_occurrences": int(sum(ref_stats["tail_visit_count"].values())),
            "ref_any_tail_other_total": int(sum(ref_stats["any_tail_other"].values())),
            "ref_total_pairs": ref_stats["ref_total_pairs"],
            "ref_total_pairs_by_bucket_pair": ref_stats["ref_total_pairs_by_bucket_pair"],
            "ref_tail_codes_with_context": int(len(ref_stats["tail_visit_count"])),
        },
        "real": real_summary,
        "synthetic": syn_summary,
    }

    _write_json(out, os.path.join(args.out_dir, "tail_semantic_plausibility_summary.json"))

    # Per-tail-code CSV (real + synthetic side-by-side, aligned to reference tail codes).
    by_tail = {}
    for row in real_rows:
        by_tail[int(row["tail_code_id"])] = {"tail_code_id": int(row["tail_code_id"]), **row}
    for row in syn_rows:
        t = int(row["tail_code_id"])
        base = by_tail.get(t, {"tail_code_id": t, "ref_tail_visit_count": row["ref_tail_visit_count"]})
        # Rename fields to keep both.
        base["real_eval_tail_visit_count"] = base.pop("eval_tail_visit_count", "")
        base["real_tail_context_js"] = base.pop("tail_context_js", "")
        base["real_tail_context_topk_jaccard"] = base.pop("tail_context_topk_jaccard", "")
        base["syn_eval_tail_visit_count"] = row["eval_tail_visit_count"]
        base["syn_tail_context_js"] = row["tail_context_js"]
        base["syn_tail_context_topk_jaccard"] = row["tail_context_topk_jaccard"]
        by_tail[t] = base

    per_tail_csv = os.path.join(args.out_dir, "tail_semantic_plausibility_per_tail_code.csv")
    fieldnames = [
        "tail_code_id",
        "ref_tail_visit_count",
        "real_eval_tail_visit_count",
        "syn_eval_tail_visit_count",
        "real_tail_context_js",
        "syn_tail_context_js",
        "real_tail_context_topk_jaccard",
        "syn_tail_context_topk_jaccard",
    ]
    with open(per_tail_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in sorted(by_tail.keys()):
            w.writerow({k: by_tail[t].get(k, "") for k in fieldnames})

    print(f"Wrote results to: {args.out_dir}")
    print("- tail_semantic_plausibility_summary.json")
    print("- tail_semantic_plausibility_per_tail_code.csv")


if __name__ == "__main__":
    main()

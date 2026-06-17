#!/usr/bin/env python3
"""
Zero-Shot Controllability (Option A: code mapping).
- model3 (III) + π_target(IV) → generate (III indices) → map III→IV code space → eval on real IV.
- model5 (IV) + π_target(III) → generate (IV indices) → map IV→III code space → eval on real III.
Results in output/ for Table 3. Baseline from main table (理解 B).
"""
from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import csv
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ZERO_SHOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ZERO_SHOT_DIR))
from paths import (
    DATA_III,
    DATA_IV,
    EVAL_PY,
    LOGIT_ADJUST_CLIP,
    LOGIT_ADJUST_EPS,
    LOGIT_ADJUST_TAU,
    MODEL3_CKPT,
    MODEL5_CKPT,
    MODEL7_DIR,
    MODEL8_DIR,
    N_CTX,
    OUT_DIR,
    SAMPLE_BATCH_SIZE,
    TOTAL_SAMPLES,
)

sys.path.insert(0, MODEL7_DIR)
from config import Model2Config
from model import HALOModel


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_logit_adjust_from_train(train_data, code_vocab: int, total_vocab_size: int):
    """Compute b_target from target train data; return array of length total_vocab_size."""
    visit_counts = np.zeros((code_vocab,), dtype=np.int64)
    total_visits = 0
    for p in train_data:
        for v in p.get("visits", []):
            if not v:
                continue
            total_visits += 1
            for c in set(v):
                ci = int(c)
                if 0 <= ci < code_vocab:
                    visit_counts[ci] += 1
    eps = LOGIT_ADJUST_EPS
    if total_visits <= 0:
        return np.zeros((total_vocab_size,), dtype=np.float32)
    pi = visit_counts.astype(np.float64) / float(total_visits)
    b = np.log((1.0 - pi + eps) / (pi + eps)) * float(LOGIT_ADJUST_TAU)
    if LOGIT_ADJUST_CLIP is not None:
        b = np.clip(b, -float(LOGIT_ADJUST_CLIP), float(LOGIT_ADJUST_CLIP))
    b = np.where(visit_counts > 0, b, 0.0)
    adj = np.zeros((total_vocab_size,), dtype=np.float32)
    adj[:code_vocab] = b.astype(np.float32)
    return adj


def build_config(data_dir: Path) -> Model2Config:
    code_to_index = load_pkl(data_dir / "codeToIndex.pkl")
    id_to_label = load_pkl(data_dir / "idToLabel.pkl")
    cfg = Model2Config()
    cfg.code_vocab_size = len(code_to_index)
    cfg.label_vocab_size = len(id_to_label)
    cfg.total_vocab_size = cfg.code_vocab_size + cfg.label_vocab_size + cfg.special_vocab_size
    return cfg


def sample_sequence(model, length: int, start_token: np.ndarray, batch_size: int, config: Model2Config, device, logit_adjust=None):
    empty = torch.zeros((1, 1, config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
    context = torch.tensor(start_token, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    prev = context.unsqueeze(1)
    model.eval()
    with torch.no_grad():
        for _ in range(length - 1):
            prev = model.sample(torch.cat((prev, empty), dim=1), random=True, logit_adjust=logit_adjust)
            end_mask = prev[:, :, config.end_record_token].sum(dim=1) > 0
            if bool(end_mask.all()):
                break
    return prev.cpu().detach().numpy()


def convert_ehr(ehrs: np.ndarray, config: Model2Config):
    out = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        labels_output = ehr[1][config.code_vocab_size : config.code_vocab_size + config.label_vocab_size]
        visit_output = []
        for j in range(2, len(ehr)):
            visit = ehr[j]
            indices = np.nonzero(visit)[0]
            codes = []
            end = False
            for idx in indices:
                if idx < config.code_vocab_size:
                    codes.append(int(idx))
                elif idx == config.end_record_token:
                    end = True
            if codes:
                visit_output.append(codes)
            if end:
                break
        out.append({"visits": visit_output, "labels": labels_output})
    return out


def generate_and_save(
    ckpt_path: Path,
    model_config: Model2Config,
    target_train_data: list,
    out_pkl: Path,
    device: torch.device,
):
    """Load model, compute b_target from target_train_data, generate, save pkl."""
    code_vocab = model_config.code_vocab_size
    total_vocab = model_config.total_vocab_size
    # b_target from target train: use model's code_vocab size (target may use same ontology)
    b_target = compute_logit_adjust_from_train(target_train_data, code_vocab, total_vocab)
    b_tensor = torch.from_numpy(b_target).to(device=device, dtype=torch.float32)

    model = HALOModel(model_config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    if next(iter(state.keys()), "").startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    stoken = np.zeros(total_vocab, dtype=np.float32)
    stoken[model_config.start_record_token] = 1.0
    generated = []
    n_samples = int(TOTAL_SAMPLES)
    for i in range(0, n_samples, SAMPLE_BATCH_SIZE):
        bs = min(n_samples - i, SAMPLE_BATCH_SIZE)
        batch_seq = sample_sequence(
            model, length=N_CTX, start_token=stoken, batch_size=bs, config=model_config, device=device, logit_adjust=b_tensor
        )
        generated += convert_ehr(batch_seq, model_config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(generated, f)
    print(f"Saved {out_pkl} (n={len(generated)})")
    return out_pkl


def build_index_to_code(code_to_index: dict) -> dict:
    """code_to_index: code_string -> index. Return index -> code_string."""
    return {int(v): k for k, v in code_to_index.items()}


def map_visits_to_target_domain(
    records: list[dict],
    src_index_to_code: dict[int, str],
    tgt_code_to_index: dict[str, int],
) -> list[dict]:
    """
    Map visit code indices from source domain to target domain.
    Only codes that exist in target are kept; III/IV-only codes are dropped for that visit.
    Labels are unchanged.
    """
    out = []
    for rec in records:
        visits = rec.get("visits", [])
        labels = rec.get("labels")
        if labels is None:
            labels = np.zeros(25, dtype=np.float32)
        mapped_visits = []
        for v in visits:
            mapped_v = []
            for idx in v:
                code_str = src_index_to_code.get(int(idx))
                if code_str is not None and code_str in tgt_code_to_index:
                    mapped_v.append(tgt_code_to_index[code_str])
            if mapped_v:
                mapped_visits.append(mapped_v)
        if mapped_visits:
            out.append({"visits": mapped_visits, "labels": labels})
    return out


def run_eval(base_data_dir: Path, syn_path: Path, save_dir: Path, source_name: str = "ZeroShot"):
    """Run evaluate_synthetic_training.py; outputs compare_real_halo_mymodel2.csv in save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            EVAL_PY,
            "--base_data_dir",
            str(base_data_dir),
            "--mymodel2_path",
            str(syn_path),
            "--save_dir",
            str(save_dir),
            "--sources",
            "MyModel2",
        ],
        check=True,
        cwd=str(ZERO_SHOT_DIR),
    )
    csv_path = save_dir / "compare_real_halo_mymodel2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} after eval")
    return csv_path


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- model3 (III) → target IV (with code mapping III index → IV index) ---
    print("=== model3 (trained on III) + π_target(IV) → generate → map to IV code space → eval on real IV ===")
    cfg_iii = build_config(DATA_III)
    train_iv = load_pkl(DATA_IV / "trainDataset.pkl")
    syn_3_to_iv = OUT_DIR / "model3_to_iv_syn.pkl"
    syn_3_to_iv_mapped = OUT_DIR / "model3_to_iv_syn_mapped.pkl"
    if not Path(MODEL3_CKPT).exists():
        print(f"Skip model3: checkpoint not found {MODEL3_CKPT}")
    else:
        generate_and_save(Path(MODEL3_CKPT), cfg_iii, train_iv, syn_3_to_iv, device)
        code_iii = load_pkl(DATA_III / "codeToIndex.pkl")
        code_iv = load_pkl(DATA_IV / "codeToIndex.pkl")
        idx2code_iii = build_index_to_code(code_iii)
        raw_3 = load_pkl(syn_3_to_iv)
        mapped_3 = map_visits_to_target_domain(raw_3, idx2code_iii, code_iv)
        with open(syn_3_to_iv_mapped, "wb") as f:
            pickle.dump(mapped_3, f)
        print(f"Mapped III→IV code space: saved {syn_3_to_iv_mapped} (n={len(mapped_3)})")
        eval_dir_3 = OUT_DIR / "eval_model3_to_iv"
        run_eval(DATA_IV, syn_3_to_iv_mapped, eval_dir_3)
        acc_3, f1_3, auprc_3 = parse_mean_acc_f1_auprc(eval_dir_3 / "compare_real_halo_mymodel2.csv")
        print(f"model3→IV: Acc={acc_3:.4f}, F1={f1_3:.4f}, AUPRC={auprc_3:.4f}")

    # --- model5 (IV) → target III (with code mapping IV index → III index) ---
    print("=== model5 (trained on IV) + π_target(III) → generate → map to III code space → eval on real III ===")
    cfg_iv = build_config(DATA_IV)
    train_iii = load_pkl(DATA_III / "trainDataset.pkl")
    syn_5_to_iii = OUT_DIR / "model5_to_iii_syn.pkl"
    syn_5_to_iii_mapped = OUT_DIR / "model5_to_iii_syn_mapped.pkl"
    if not Path(MODEL5_CKPT).exists():
        print(f"Skip model5: checkpoint not found {MODEL5_CKPT}")
    else:
        generate_and_save(Path(MODEL5_CKPT), cfg_iv, train_iii, syn_5_to_iii, device)
        code_iii = load_pkl(DATA_III / "codeToIndex.pkl")
        code_iv = load_pkl(DATA_IV / "codeToIndex.pkl")
        idx2code_iv = build_index_to_code(code_iv)
        raw_5 = load_pkl(syn_5_to_iii)
        mapped_5 = map_visits_to_target_domain(raw_5, idx2code_iv, code_iii)
        with open(syn_5_to_iii_mapped, "wb") as f:
            pickle.dump(mapped_5, f)
        print(f"Mapped IV→III code space: saved {syn_5_to_iii_mapped} (n={len(mapped_5)})")
        eval_dir_5 = OUT_DIR / "eval_model5_to_iii"
        run_eval(DATA_III, syn_5_to_iii_mapped, eval_dir_5)
        acc_5, f1_5, auprc_5 = parse_mean_acc_f1_auprc(eval_dir_5 / "compare_real_halo_mymodel2.csv")
        print(f"model5→III: Acc={acc_5:.4f}, F1={f1_5:.4f}, AUPRC={auprc_5:.4f}")

    # --- Table 3 CSV (ours only; baseline from main table) ---
    rows = []
    if Path(MODEL3_CKPT).exists() and (OUT_DIR / "eval_model3_to_iv" / "compare_real_halo_mymodel2.csv").exists():
        acc_3, f1_3, auprc_3 = parse_mean_acc_f1_auprc(OUT_DIR / "eval_model3_to_iv" / "compare_real_halo_mymodel2.csv")
        rows.append({"target": "MIMIC-IV", "method": "AdaPCLA (III→IV zero-shot)", "Acc": acc_3, "F1": f1_3, "AUPRC": auprc_3})
    if Path(MODEL5_CKPT).exists() and (OUT_DIR / "eval_model5_to_iii" / "compare_real_halo_mymodel2.csv").exists():
        acc_5, f1_5, auprc_5 = parse_mean_acc_f1_auprc(OUT_DIR / "eval_model5_to_iii" / "compare_real_halo_mymodel2.csv")
        rows.append({"target": "MIMIC-III", "method": "AdaPCLA (IV→III zero-shot)", "Acc": acc_5, "F1": f1_5, "AUPRC": auprc_5})
    if rows:
        out_csv = OUT_DIR / "zeroshot_table3.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["target", "method", "Acc", "F1", "AUPRC"])
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
    print("Done. Baseline (GPT, LSTM, ...) use main table results (理解 B).")


if __name__ == "__main__":
    main()

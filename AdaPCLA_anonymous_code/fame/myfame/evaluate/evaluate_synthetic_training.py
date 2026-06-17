from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import os
import sys
import csv
import pickle
import random
import itertools
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn import metrics
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Keep architecture consistent with HALO_Inpatient/evaluate_synthetic_training.py
DEFAULT_SEED = 4
LR = 0.001
EPOCHS = 25
LABEL_IDX_LIST = list(range(25))
BATCH_SIZE = 512
LSTM_HIDDEN_DIM = 32
EMBEDDING_DIM = 64
NUM_TRAIN_EXAMPLES = 5000
NUM_TEST_EXAMPLES = 1000
NUM_VAL_EXAMPLES = 500


def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return local_rank, 0, 1, device
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())
    device = torch.device("cuda", local_rank)
    return local_rank, rank, world_size, device


local_rank, rank, world_size, device = setup_ddp()


# Paths (MyFAME evaluation uses HALO_Inpatient real data + MyFAME synthetic outputs)
DEFAULT_BASE_DATA_DIR = DATA_MIMICIII
DEFAULT_HALO_SYN_PATH = "FAME_ROOT"
DEFAULT_MYMODEL2_SYN_PATH = "FAME_ROOT/baseline/model2/save/datasets/haloDataset.pkl"
DEFAULT_SAVE_DIR = "FAME_ROOT/evaluate/save"

parser = argparse.ArgumentParser()
parser.add_argument("--base_data_dir", default=DEFAULT_BASE_DATA_DIR)
parser.add_argument("--halo_path", default=DEFAULT_HALO_SYN_PATH, help="HALO synthetic dataset path.")
parser.add_argument("--mymodel2_path", default=DEFAULT_MYMODEL2_SYN_PATH, help="MyModel2 synthetic dataset path.")
parser.add_argument(
    "--extra_source",
    action="append",
    default=[],
    help="Add an extra synthetic source as NAME=PATH. Can be repeated.",
)
parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
parser.add_argument(
    "--sources",
    default="HALO,MyModel2",
    help="Comma-separated synthetic sources to evaluate: HALO,MyModel2. Real is always included.",
)
parser.add_argument("--epochs", type=int, default=EPOCHS, help="Epochs for each downstream classifier.")
parser.add_argument("--n_train", type=int, default=NUM_TRAIN_EXAMPLES, help="Train examples per label (balanced pos/neg).")
parser.add_argument("--n_val", type=int, default=NUM_VAL_EXAMPLES, help="Val examples per label (balanced pos/neg).")
parser.add_argument("--n_test", type=int, default=NUM_TEST_EXAMPLES, help="Test examples per label (balanced pos/neg).")
parser.add_argument(
    "--skip_real",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Skip training/evaluating the classifier trained on real data (still uses real val/test for evaluation).",
)
parser.add_argument(
    "--label_indices",
    default="",
    help="Optional comma-separated label indices to evaluate (e.g., '0,1,5'). Empty => all labels.",
)
args, _unknown = parser.parse_known_args()

BASE_DATA_DIR = args.base_data_dir
HALO_SYN_PATH = args.halo_path
MYMODEL2_SYN_PATH = args.mymodel2_path
SAVE_DIR = args.save_dir
SEED = int(args.seed)
EPOCHS = int(args.epochs)
NUM_TRAIN_EXAMPLES = int(args.n_train)
NUM_VAL_EXAMPLES = int(args.n_val)
NUM_TEST_EXAMPLES = int(args.n_test)
_LABEL_INDICES_RAW = str(args.label_indices or "").strip()
SKIP_REAL = bool(args.skip_real)

# Seed after parsing so callers can override via --seed.
random.seed(SEED + rank)
np.random.seed(SEED + rank)
torch.manual_seed(SEED + rank)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED + rank)

if local_rank != -1:
    if rank == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
else:
    os.makedirs(SAVE_DIR, exist_ok=True)

# Per-rank log file to avoid multi-rank console spam.
LOG_DIR = os.path.join(SAVE_DIR, "logs")
if local_rank != -1:
    if rank == 0:
        os.makedirs(LOG_DIR, exist_ok=True)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
else:
    os.makedirs(LOG_DIR, exist_ok=True)

_rank_log_f = open(os.path.join(LOG_DIR, f"rank{rank}.log"), "a", buffering=1)


def log(msg: str, *, console: bool = False):
    _rank_log_f.write(str(msg).rstrip() + "\n")
    if console and rank == 0:
        print(msg, flush=True)


# Import HALOConfig from the baseline HALO code to match BASE_DATA_DIR preprocessing.
# Use path relative to this script so it works on any machine (not hardcoded FAME_ROOT
_evalscript_dir = os.path.dirname(os.path.abspath(__file__))
_halo_baseline_dir = os.path.abspath(os.path.join(_evalscript_dir, "..", "baseline", "HALO"))
sys.path.insert(0, _halo_baseline_dir)
from config import HALOConfig  # noqa: E402


index_to_code = pickle.load(open(os.path.join(BASE_DATA_DIR, "idToLabel.pkl"), "rb"))
code_to_index = pickle.load(open(os.path.join(BASE_DATA_DIR, "codeToIndex.pkl"), "rb"))

config = HALOConfig()
# Ensure vocab matches the actual processed dataset indices in BASE_DATA_DIR.
config.code_vocab_size = len(code_to_index)
train_ehr_dataset = pickle.load(open(os.path.join(BASE_DATA_DIR, "trainDataset.pkl"), "rb"))
val_ehr_dataset = pickle.load(open(os.path.join(BASE_DATA_DIR, "valDataset.pkl"), "rb"))
test_ehr_dataset = pickle.load(open(os.path.join(BASE_DATA_DIR, "testDataset.pkl"), "rb"))


def _parse_label_indices(raw: str, *, n_labels: int) -> list[int]:
    raw = str(raw or "").strip()
    if raw == "":
        return list(range(int(n_labels)))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= int(n_labels):
            raise ValueError(f"--label_indices contains out-of-range index: {idx} (n_labels={n_labels})")
        out.append(idx)
    # Deduplicate but keep order.
    seen = set()
    out2: list[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        out2.append(i)
    return out2


LABEL_IDX_LIST = _parse_label_indices(_LABEL_INDICES_RAW, n_labels=len(index_to_code))


def _load_synth(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing synthetic dataset for {name}: {path}")
    data = pickle.load(open(path, "rb"))
    if not isinstance(data, list):
        raise ValueError(f"Synthetic dataset {name} must be a list, got {type(data)} from {path}")
    if len(data) == 0:
        raise ValueError(f"Synthetic dataset {name} is empty: {path}")
    return data


_source_to_path = {"HALO": HALO_SYN_PATH, "MyModel2": MYMODEL2_SYN_PATH}
for item in getattr(args, "extra_source", []) or []:
    if not item:
        continue
    if "=" not in item:
        raise ValueError(f"--extra_source must be NAME=PATH, got: {item}")
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError(f"--extra_source has empty NAME: {item}")
    if not path:
        raise ValueError(f"--extra_source has empty PATH: {item}")
    if name in _source_to_path:
        raise ValueError(f"Duplicate synthetic source name: {name}")
    _source_to_path[name] = path
sources = [s.strip() for s in str(args.sources).split(",") if s.strip()]
unknown_sources = [s for s in sources if s not in _source_to_path]
if unknown_sources:
    raise ValueError(f"Unknown --sources: {unknown_sources}. Supported: {sorted(_source_to_path)}")
if not sources:
    raise ValueError("--sources is empty; at least one of HALO,MyModel2 is required.")

synth_datasets = {src: _load_synth(_source_to_path[src], src) for src in sources}

log(f"BASE_DATA_DIR={BASE_DATA_DIR}", console=(rank == 0))
for src in sources:
    log(f"{src}_SYN_PATH={_source_to_path[src]} (n={len(synth_datasets[src])})", console=(rank == 0))


class DiagnosisModel(nn.Module):
    def __init__(self, config):
        super(DiagnosisModel, self).__init__()
        self.embedding = nn.Linear(config.code_vocab_size, EMBEDDING_DIM, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=2,
            dropout=0.5,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * LSTM_HIDDEN_DIM, 1)

    def forward(self, input_visits, lengths):
        visit_emb = self.embedding(input_visits)
        visit_emb = self.dropout(visit_emb)
        packed_input = pack_padded_sequence(visit_emb, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        out_forward = output[range(len(output)), lengths - 1, :LSTM_HIDDEN_DIM]
        out_reverse = output[:, 0, LSTM_HIDDEN_DIM:]
        out_combined = torch.cat((out_forward, out_reverse), 1)

        patient_embedding = self.fc(out_combined)
        patient_embedding = torch.squeeze(patient_embedding, 1)
        prob = torch.sigmoid(patient_embedding)
        return prob


def get_batch(ehr_dataset, loc, batch_size, label_idx):
    ehr = ehr_dataset[loc : loc + batch_size]
    batch_ehr = np.zeros((len(ehr), config.n_ctx, config.code_vocab_size))
    batch_labels = np.array([p["labels"][label_idx] for p in ehr])
    batch_lens = np.zeros(len(ehr), dtype=np.int64)
    for i, p in enumerate(ehr):
        visits = p["visits"]
        # Some synthetic records may have 0 visits; pack_padded_sequence requires lengths>0.
        # Also, we truncate to config.n_ctx when building the dense (B, T, V) tensor, so
        # lengths must be capped to avoid invalid PackedSequence shapes.
        batch_lens[i] = max(1, min(int(len(visits)), int(config.n_ctx)))
        for j, v in enumerate(visits):
            if j >= config.n_ctx:
                break
            batch_ehr[i, j][v] = 1
    return batch_ehr, batch_labels, batch_lens


def train_model(model, train_dataset, val_dataset, save_path, label_idx):
    global_loss = 1e10
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    epoch_iter = tqdm(
        range(EPOCHS),
        desc=f"[rank{rank}] train label{label_idx}",
        leave=False,
        disable=(rank != 0),
    )
    for e in epoch_iter:
        np.random.shuffle(train_dataset)
        train_losses = []
        for i in range(0, len(train_dataset), BATCH_SIZE):
            model.train()
            batch_ehr, batch_labels, batch_lens = get_batch(train_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            prob = model(batch_ehr, batch_lens)
            loss = bce(prob, batch_labels)
            train_losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
        cur_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        model.eval()
        with torch.no_grad():
            val_losses = []
            for v_i in range(0, len(val_dataset), BATCH_SIZE):
                batch_ehr, batch_labels, batch_lens = get_batch(val_dataset, v_i, BATCH_SIZE, label_idx)
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
                prob = model(batch_ehr, batch_lens)
                val_loss = bce(prob, batch_labels)
                val_losses.append(val_loss.cpu().detach().numpy())
            cur_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            if rank == 0:
                epoch_iter.set_postfix(train=f"{cur_train_loss:.5f}", val=f"{cur_val_loss:.5f}", best=f"{global_loss:.5f}")
            if cur_val_loss < global_loss:
                global_loss = cur_val_loss
                state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(state, save_path)


def test_model(model, test_dataset, label_idx, *, dataset_name: str):
    loss_list = []
    pred_list = []
    true_list = []
    bce = nn.BCELoss()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch_ehr, batch_labels, batch_lens = get_batch(test_dataset, i, BATCH_SIZE, label_idx)
            batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
            prob = model(batch_ehr, batch_lens)
            val_loss = bce(prob, batch_labels)
            loss_list.append(val_loss.cpu().detach().numpy())
            pred_list += list(prob.cpu().detach().numpy())
            true_list += list(batch_labels.cpu().detach().numpy())

    round_list = np.around(pred_list)
    avg_loss = np.mean(loss_list) if loss_list else 0.0
    cmatrix = metrics.confusion_matrix(true_list, round_list)
    acc = metrics.accuracy_score(true_list, round_list)
    prc = metrics.precision_score(true_list, round_list, zero_division=0)
    rec = metrics.recall_score(true_list, round_list, zero_division=0)
    f1 = metrics.f1_score(true_list, round_list, zero_division=0)
    try:
        auroc = metrics.roc_auc_score(true_list, pred_list)
    except Exception:
        auroc = 0.0
    try:
        (precisions, recalls, _) = metrics.precision_recall_curve(true_list, pred_list)
        auprc = metrics.auc(recalls, precisions)
    except Exception:
        auprc = 0.0

    metrics_dict = {}
    metrics_dict["Test Loss"] = float(avg_loss)
    metrics_dict["Confusion Matrix"] = cmatrix
    metrics_dict["Accuracy"] = float(acc)
    metrics_dict["Precision"] = float(prc)
    metrics_dict["Recall"] = float(rec)
    metrics_dict["F1 Score"] = float(f1)
    metrics_dict["AUROC"] = float(auroc)
    metrics_dict["AUPRC"] = float(auprc)

    log(
        f"[{dataset_name}] label {label_idx} metrics: "
        f"Acc={acc:.4f} P={prc:.4f} R={rec:.4f} F1={f1:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f}",
        console=(rank == 0),
    )

    return metrics_dict


def _write_csv(results_dict: dict, csv_path: str, sources: list[str]):
    rows = []
    for label_idx, label_name in enumerate(index_to_code):
        if label_name not in results_dict:
            continue
        by_src = results_dict[label_name]
        for src in sources:
            if src not in by_src:
                continue
            m = by_src[src]
            if not m:
                continue
            rows.append(
                {
                    "label_idx": label_idx,
                    "label_name": label_name,
                    "source": src,
                    "Accuracy": m.get("Accuracy", ""),
                    "Precision": m.get("Precision", ""),
                    "Recall": m.get("Recall", ""),
                    "F1 Score": m.get("F1 Score", ""),
                    "AUROC": m.get("AUROC", ""),
                    "AUPRC": m.get("AUPRC", ""),
                }
            )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["label_idx", "label_name", "source", "Accuracy", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC"],
        )
        w.writeheader()
        w.writerows(rows)

def _sample_balanced(pos_set: list, neg_set: list, n_total: int) -> list:
    n_total = int(n_total)
    n_total = max(0, n_total)
    if n_total == 0:
        return []
    if len(pos_set) == 0 and len(neg_set) == 0:
        return []
    if len(pos_set) == 0:
        replace = len(neg_set) < n_total
        return list(np.random.choice(neg_set, n_total, replace=replace))
    if len(neg_set) == 0:
        replace = len(pos_set) < n_total
        return list(np.random.choice(pos_set, n_total, replace=replace))

    n_pos = n_total // 2
    n_neg = n_total - n_pos
    pos_replace = len(pos_set) < n_pos
    neg_replace = len(neg_set) < n_neg
    return list(np.random.choice(pos_set, n_pos, replace=pos_replace)) + list(np.random.choice(neg_set, n_neg, replace=neg_replace))


results_local = {}
for i in LABEL_IDX_LIST:
    if (i % world_size) != rank:
        continue

    log(f"[rank{rank}] Evaluating label {i} ({index_to_code[i]})...", console=(rank == 0))

    label_results = {}

    # Prepare datasets (Real + synthetic sources)
    synth_pos_neg = {}
    for src_name, ehr_dataset in synth_datasets.items():
        synth_pos_neg[src_name] = (
            [p for p in ehr_dataset if p["labels"][i] == 1],
            [p for p in ehr_dataset if p["labels"][i] == 0],
        )

    train_pos_label_dataset = [p for p in train_ehr_dataset if p["labels"][i] == 1]
    train_neg_label_dataset = [p for p in train_ehr_dataset if p["labels"][i] == 0]
    val_pos_label_dataset = [p for p in val_ehr_dataset if p["labels"][i] == 1]
    val_neg_label_dataset = [p for p in val_ehr_dataset if p["labels"][i] == 0]
    test_pos_label_dataset = [p for p in test_ehr_dataset if p["labels"][i] == 1]
    test_neg_label_dataset = [p for p in test_ehr_dataset if p["labels"][i] == 0]

    val_dataset = _sample_balanced(val_pos_label_dataset, val_neg_label_dataset, int(NUM_VAL_EXAMPLES))
    test_dataset = _sample_balanced(test_pos_label_dataset, test_neg_label_dataset, int(NUM_TEST_EXAMPLES))
    if rank == 0:
        if len(val_pos_label_dataset) == 0 or len(val_neg_label_dataset) == 0:
            log(
                f"[WARN] Real val split is single-class for label {i}: pos={len(val_pos_label_dataset)} neg={len(val_neg_label_dataset)}",
                console=True,
            )
        if len(test_pos_label_dataset) == 0 or len(test_neg_label_dataset) == 0:
            log(
                f"[WARN] Real test split is single-class for label {i}: pos={len(test_pos_label_dataset)} neg={len(test_neg_label_dataset)}",
                console=True,
            )

    # NOTE: Keep the original HALO_Inpatient script behavior (including its replacement condition).
    train_dataset_real = _sample_balanced(train_pos_label_dataset, train_neg_label_dataset, int(NUM_TRAIN_EXAMPLES))
    if rank == 0 and (len(train_pos_label_dataset) == 0 or len(train_neg_label_dataset) == 0):
        log(
            f"[WARN] Real train split is single-class for label {i}: pos={len(train_pos_label_dataset)} neg={len(train_neg_label_dataset)}",
            console=True,
        )

    train_datasets_synth = {}
    for src_name, (pos_set, neg_set) in synth_pos_neg.items():
        if rank == 0 and (len(pos_set) == 0 or len(neg_set) == 0):
            log(
                f"[WARN] Synthetic {src_name} is single-class for label {i}: pos={len(pos_set)} neg={len(neg_set)}",
                console=True,
            )
        train_datasets_synth[src_name] = _sample_balanced(pos_set, neg_set, int(NUM_TRAIN_EXAMPLES))

    # Perform the different experiments
    if not SKIP_REAL:
        model_real = DiagnosisModel(config).to(device)
        save_real = os.path.join(SAVE_DIR, f"syn_diag_Real_{i}.pt")
        log(f"[rank{rank}] Training classifier on Real for label {i}...", console=(rank == 0))
        if os.path.exists(save_real):
            log(f"[rank{rank}] Found cached classifier: {save_real}", console=(rank == 0))
        else:
            train_model(model_real, train_dataset_real, val_dataset, save_real, i)
        state = torch.load(save_real, map_location=device, weights_only=False)
        model_real.load_state_dict(state["model"])
        test_results_real = test_model(model_real, test_dataset, i, dataset_name="Real")
        label_results["Real"] = test_results_real

    for src_name, train_dataset in train_datasets_synth.items():
        model_syn = DiagnosisModel(config).to(device)
        save_syn = os.path.join(SAVE_DIR, f"syn_diag_{src_name}_{i}.pt")
        log(f"[rank{rank}] Training classifier on {src_name} for label {i}...", console=(rank == 0))
        if os.path.exists(save_syn):
            log(f"[rank{rank}] Found cached classifier: {save_syn}", console=False)
        else:
            train_model(model_syn, train_dataset, val_dataset, save_syn, i)
        state = torch.load(save_syn, map_location=device, weights_only=False)
        model_syn.load_state_dict(state["model"])
        test_results_syn = test_model(model_syn, test_dataset, i, dataset_name=src_name)
        label_results[src_name] = test_results_syn

    results_local[index_to_code[i]] = label_results


# Merge results across ranks and write outputs on rank 0.
# NOTE: Avoid all_gather_object with NCCL (uses GPU memory and can OOM); merge via per-rank files instead.
if local_rank != -1:
    part_path = os.path.join(SAVE_DIR, f"fully_synthetic_stats_rank{rank}.pkl")
    pickle.dump(results_local, open(part_path, "wb"))
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    if rank == 0:
        results = {}
        for r in range(world_size):
            pth = os.path.join(SAVE_DIR, f"fully_synthetic_stats_rank{r}.pkl")
            if not os.path.exists(pth):
                continue
            part = pickle.load(open(pth, "rb"))
            if part:
                results.update(part)
        pickle.dump(results, open(os.path.join(SAVE_DIR, "fully_synthetic_stats.pkl"), "wb"))
        csv_sources = (["Real"] if not SKIP_REAL else []) + sources
        _write_csv(results, os.path.join(SAVE_DIR, "compare_real_halo_mymodel2.csv"), csv_sources)
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
else:
    results = results_local
    pickle.dump(results, open(os.path.join(SAVE_DIR, "fully_synthetic_stats.pkl"), "wb"))
    csv_sources = (["Real"] if not SKIP_REAL else []) + sources
    _write_csv(results, os.path.join(SAVE_DIR, "compare_real_halo_mymodel2.csv"), csv_sources)

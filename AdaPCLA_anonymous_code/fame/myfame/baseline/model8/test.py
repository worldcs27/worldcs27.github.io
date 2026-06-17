import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import datetime
import json
import os
import pickle
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import Model2Config
from model import HALOModel


SEED = 4
DEFAULT_DATA_DIR = DATA_MIMICIII
DEFAULT_SAVE_DIR = "MODEL8_DIR/save"


def setup(local_rank: int):
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))


def cleanup():
    dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MIMICDataset(Dataset):
    def __init__(self, data, config: Model2Config):
        self.data = data
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        visits = p["visits"]

        sample_ehr = np.zeros((self.config.n_ctx, self.config.total_vocab_size), dtype=np.float32)
        sample_mask = np.zeros((self.config.n_ctx, 1), dtype=np.float32)

        for j, v in enumerate(visits):
            if j + 2 < self.config.n_ctx:
                sample_ehr[j + 2][v] = 1
                sample_mask[j + 2] = 1

        sample_ehr[1, self.config.code_vocab_size : self.config.code_vocab_size + self.config.label_vocab_size] = np.array(p["labels"])

        if len(visits) + 1 < self.config.n_ctx:
            sample_ehr[len(visits) + 1, self.config.end_record_token] = 1
        if len(visits) + 2 < self.config.n_ctx:
            sample_ehr[len(visits) + 2 :, self.config.pad_visit_token] = 1

        sample_mask[1] = 1
        sample_ehr[0, self.config.start_record_token] = 1
        sample_mask = sample_mask[1:, :]
        return sample_ehr, sample_mask


def conf_mat(x, y):
    totaltrue = np.sum(x)
    totalfalse = len(x) - totaltrue
    truepos, totalpos = np.sum(x & y), np.sum(y)
    falsepos = totalpos - truepos
    return np.array([[totalfalse - falsepos, falsepos], [totaltrue - truepos, truepos]])

def _load_logit_adjust_from_path(path: str, *, expected_size: int, device) -> torch.Tensor:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if int(arr.size) != int(expected_size):
        raise ValueError(f"logit_adjust size mismatch: got {int(arr.size)} expected {int(expected_size)} from {path}")
    return torch.tensor(arr, device=device, dtype=torch.float32)


def sample_sequence(model, length, context, batch_size, config: Model2Config, device, *, logit_adjust=None, sample=True):
    empty = torch.zeros((1, 1, config.total_vocab_size), device=device, dtype=torch.float32).repeat(batch_size, 1, 1)
    context = torch.tensor(context, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    prev = context.unsqueeze(1)

    model_ref = model.module if isinstance(model, DDP) else model
    with torch.no_grad():
        for _ in range(length - 1):
            prev = model_ref.sample(torch.cat((prev, empty), dim=1), sample, logit_adjust=logit_adjust)
            if (
                torch.sum(torch.sum(prev[:, :, config.end_record_token], dim=1).bool().int(), dim=0).item()
                == batch_size
            ):
                break
    return prev.cpu().detach().numpy()


def convert_ehr(ehrs, config: Model2Config):
    ehr_outputs = []
    for i in range(len(ehrs)):
        ehr = ehrs[i]
        ehr_output = []
        labels_output = ehr[1][config.code_vocab_size : config.code_vocab_size + config.label_vocab_size]
        for j in range(2, len(ehr)):
            visit = ehr[j]
            visit_output = []
            indices = np.nonzero(visit)[0]
            end = False
            for idx in indices:
                if idx < config.code_vocab_size:
                    visit_output.append(int(idx))
                elif idx == config.end_record_token:
                    end = True
            if visit_output:
                ehr_output.append(visit_output)
            if end:
                break
        ehr_outputs.append({"visits": ehr_output, "labels": labels_output})
    return ehr_outputs


def main(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup(local_rank)
    device = torch.device("cuda", local_rank)

    base_seed = int(getattr(args, "seed", SEED))
    seed_everything(base_seed + rank)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "datasets"), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, "testing_stats"), exist_ok=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    config = Model2Config()
    data_dir = args.data_dir
    config.batch_size = int(args.batch_size)
    config.sample_batch_size = int(args.sample_batch_size)

    def load_pkl(name):
        return pickle.load(open(os.path.join(data_dir, name), "rb"))

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    test_data = load_pkl("testDataset.pkl")

    config.code_vocab_size = len(code_to_index)
    config.label_vocab_size = len(id_to_label)
    config.total_vocab_size = config.code_vocab_size + config.label_vocab_size + config.special_vocab_size

    test_dataset = MIMICDataset(test_data, config)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)

    model = HALOModel(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    ckpt_path = args.ckpt_path or os.path.join(args.save_dir, "model8.pt")
    if not os.path.exists(ckpt_path):
        if rank == 0:
            print(f"Missing checkpoint: {ckpt_path}")
        cleanup()
        return

    map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    ckpt_logit_adjust = ckpt.get("logit_adjust", None)
    if isinstance(ckpt_logit_adjust, torch.Tensor):
        ckpt_logit_adjust = ckpt_logit_adjust.to(device=device, dtype=torch.float32)
    else:
        ckpt_logit_adjust = None

    override_logit_adjust = None
    if args.logit_adjust_path:
        override_logit_adjust = _load_logit_adjust_from_path(
            args.logit_adjust_path, expected_size=int(config.total_vocab_size), device=device
        )

    # If override is provided, use it either for both eval+sampling or for sampling only.
    logit_adjust_for_eval = ckpt_logit_adjust
    logit_adjust_for_sampling = ckpt_logit_adjust
    if override_logit_adjust is not None:
        if args.override_for_sampling_only:
            logit_adjust_for_sampling = override_logit_adjust
        else:
            logit_adjust_for_eval = override_logit_adjust
            logit_adjust_for_sampling = override_logit_adjust

    model.eval()
    if not bool(args.skip_eval):
        confusion_matrix = None
        loss_list = []
        n_visits = 0
        n_pos_codes = 0
        n_total_codes = 0

        with torch.no_grad():
            iterator = tqdm(test_loader, desc="Test") if rank == 0 else test_loader
            for batch_ehr, batch_mask in iterator:
                batch_ehr = batch_ehr.to(device)
                batch_mask = batch_mask.to(device)
                test_loss, predictions, labels = model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=config.pos_loss_weight,
                    logit_adjust=logit_adjust_for_eval,
                )
                if test_loss.dim() > 0:
                    test_loss = test_loss.mean()
                loss_list.append(float(test_loss.item()))

                batch_mask_array = batch_mask.squeeze().cpu().numpy()
                rounded_preds = np.around(predictions.squeeze().cpu().numpy()).transpose((2, 0, 1))
                rounded_preds = rounded_preds + batch_mask_array - 1
                rounded_preds = rounded_preds.flatten()
                true_values = labels.squeeze().cpu().numpy().transpose((2, 0, 1))
                true_values = true_values + batch_mask_array - 1
                true_values = true_values.flatten()

                n_visits += torch.sum(batch_mask).cpu().item()
                n_pos_codes += torch.sum(labels).cpu().item()
                n_total_codes += (torch.sum(batch_mask) * config.total_vocab_size).cpu().item()

                batch_cmatrix = conf_mat(true_values == 1, rounded_preds == 1)
                batch_cmatrix[0][0] = (
                    torch.sum(batch_mask).cpu().item() * config.total_vocab_size
                    - batch_cmatrix[0][1]
                    - batch_cmatrix[1][0]
                    - batch_cmatrix[1][1]
                )
                confusion_matrix = batch_cmatrix if confusion_matrix is None else confusion_matrix + batch_cmatrix

        # Aggregate scalar stats.
        stats_t = torch.tensor(
            [float(sum(loss_list)), float(len(loss_list)), float(n_visits), float(n_pos_codes), float(n_total_codes)],
            device=device,
            dtype=torch.float64,
        )
        cm_t = (
            torch.tensor(confusion_matrix.flatten(), device=device, dtype=torch.float64)
            if confusion_matrix is not None
            else torch.zeros(4, device=device, dtype=torch.float64)
        )
        dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cm_t, op=dist.ReduceOp.SUM)

        if rank == 0:
            total_sum_loss, total_batches, total_n_visits, total_n_pos_codes, total_n_total_codes = stats_t.tolist()
            avg_loss = total_sum_loss / max(1.0, total_batches)
            total_cm = cm_t.cpu().numpy().reshape(2, 2)
            tn, fp, fn, tp = total_cm.ravel()
            acc = (tn + tp) / (tn + fp + fn + tp + 1e-12)
            prc = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1 = (2 * prc * rec) / (prc + rec + 1e-12)
            metrics = {
                "Test Loss": float(avg_loss),
                "Confusion Matrix": total_cm.tolist(),
                "Accuracy": float(acc),
                "Precision": float(prc),
                "Recall": float(rec),
                "F1 Score": float(f1),
            }
            out_path = os.path.join(args.save_dir, "testing_stats", "model7_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"Wrote metrics: {out_path}")

    # --- Generation ---
    total_samples = int(args.total_samples)
    samples_per_gpu = total_samples // world_size
    remainder = total_samples % world_size
    if rank < remainder:
        samples_per_gpu += 1

    # Re-seed for stable generation.
    seed_everything(base_seed + 100_000 + rank)

    local_synthetic_dataset = []
    stoken = np.zeros(config.total_vocab_size, dtype=np.float32)
    stoken[config.start_record_token] = 1

    model_ref = model.module if isinstance(model, DDP) else model
    apply_adj = logit_adjust_for_sampling if bool(args.apply_logit_adjust_in_sampling) else None

    iterator = tqdm(range(0, samples_per_gpu, config.sample_batch_size), desc=f"Rank {rank} Generating") if rank == 0 else range(0, samples_per_gpu, config.sample_batch_size)
    for i in iterator:
        bs = min([samples_per_gpu - i, config.sample_batch_size])
        batch_synthetic = sample_sequence(
            model_ref,
            config.n_ctx,
            stoken,
            batch_size=bs,
            config=config,
            device=device,
            logit_adjust=apply_adj,
            sample=True,
        )
        local_synthetic_dataset += convert_ehr(batch_synthetic, config)

    part_path = os.path.join(args.save_dir, "datasets", f"haloDataset_rank_{rank}.pkl")
    pickle.dump(local_synthetic_dataset, open(part_path, "wb"))
    dist.barrier(device_ids=[torch.cuda.current_device()])

    if rank == 0:
        full = []
        for r in range(world_size):
            pth = os.path.join(args.save_dir, "datasets", f"haloDataset_rank_{r}.pkl")
            full += pickle.load(open(pth, "rb"))
        final_path = os.path.join(args.save_dir, "datasets", "haloDataset.pkl")
        pickle.dump(full, open(final_path, "wb"))
        print(f"Saved synthetic dataset: {final_path} (n={len(full)})")

    dist.barrier(device_ids=[torch.cuda.current_device()])
    cleanup()


if __name__ == "__main__":
    default_cfg = Model2Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ckpt_path", default=None, help="Optional checkpoint path (default: <save_dir>/model8.pt)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--total_samples", type=int, default=33494)
    p.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    p.add_argument("--sample_batch_size", type=int, default=default_cfg.sample_batch_size)
    p.add_argument("--skip_eval", action=argparse.BooleanOptionalAction, default=False, help="Skip test-set evaluation and only generate synthetic data.")
    p.add_argument("--logit_adjust_path", default=None, help="Optional path to a logit_adjust .npy vector (length=total_vocab_size)")
    p.add_argument(
        "--override_for_sampling_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, a provided --logit_adjust_path is used only for sampling, not for test loss evaluation.",
    )
    p.add_argument(
        "--apply_logit_adjust_in_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to apply logit adjustment during sampling generation.",
    )
    args = p.parse_args()
    main(args)

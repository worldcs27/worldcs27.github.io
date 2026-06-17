import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import datetime
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


def _compute_logit_adjust(train_data, *, config: Model2Config):
    code_vocab = int(config.code_vocab_size)
    total_visits = 0
    visit_counts = np.zeros((code_vocab,), dtype=np.int64)
    for p in train_data:
        visits = p.get("visits", [])
        total_visits += int(len(visits))
        for v in visits:
            if not v:
                continue
            for c in set(v):
                ci = int(c)
                if 0 <= ci < code_vocab:
                    visit_counts[ci] += 1

    eps = float(config.logit_adjust_eps)
    if total_visits <= 0:
        return np.zeros((config.total_vocab_size,), dtype=np.float32), {"total_visits": 0}

    pi = visit_counts.astype(np.float64) / float(total_visits)
    b = np.log((1.0 - pi + eps) / (pi + eps)) * float(config.logit_adjust_tau)
    if config.logit_adjust_clip is not None:
        b = np.clip(b, -float(config.logit_adjust_clip), float(config.logit_adjust_clip))
    b = np.where(visit_counts > 0, b, 0.0)

    adj = np.zeros((config.total_vocab_size,), dtype=np.float32)
    adj[:code_vocab] = b.astype(np.float32)

    stats = {
        "total_visits": int(total_visits),
        "codes_with_pos": int(np.sum(visit_counts > 0)),
        "tau": float(config.logit_adjust_tau),
        "clip": float(config.logit_adjust_clip) if config.logit_adjust_clip is not None else None,
        "adj_min": float(adj[:code_vocab].min()) if code_vocab else 0.0,
        "adj_max": float(adj[:code_vocab].max()) if code_vocab else 0.0,
    }
    return adj, stats


def train(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(local_rank)
    base_seed = int(getattr(args, "seed", SEED))
    seed_everything(base_seed + rank)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
    dist.barrier(device_ids=[torch.cuda.current_device()])

    config = Model2Config()
    config.lr = float(args.lr)
    config.epoch = int(args.epoch)
    config.batch_size = int(args.batch_size)
    config.sample_batch_size = int(args.sample_batch_size)
    if args.pos_loss_weight in (None, "", "None", "none"):
        config.pos_loss_weight = None
    else:
        config.pos_loss_weight = float(args.pos_loss_weight)
    config.logit_adjust_tau = float(args.logit_adjust_tau)
    config.logit_adjust_clip = float(args.logit_adjust_clip) if args.logit_adjust_clip is not None else None
    config.apply_logit_adjust_in_sampling = bool(args.apply_logit_adjust_in_sampling)

    data_dir = args.data_dir

    def load_pkl(name):
        return pickle.load(open(os.path.join(data_dir, name), "rb"))

    code_to_index = load_pkl("codeToIndex.pkl")
    id_to_label = load_pkl("idToLabel.pkl")
    train_data = load_pkl("trainDataset.pkl")
    val_data = load_pkl("valDataset.pkl")

    config.code_vocab_size = len(code_to_index)
    config.label_vocab_size = len(id_to_label)
    config.total_vocab_size = config.code_vocab_size + config.label_vocab_size + config.special_vocab_size

    # Build logit adjustment vector on rank 0 and broadcast.
    if rank == 0:
        adj_np, stats = _compute_logit_adjust(train_data, config=config)
        np.save(os.path.join(args.save_dir, "logit_adjust.npy"), adj_np)
        print(f"Seed={base_seed} | Logit adjust stats: {stats}", flush=True)
    else:
        adj_np = np.zeros((config.total_vocab_size,), dtype=np.float32)
    adj = torch.tensor(adj_np, device=torch.device("cuda", local_rank), dtype=torch.float32)
    dist.broadcast(adj, src=0)

    train_dataset = MIMICDataset(train_data, config)
    val_dataset = MIMICDataset(val_data, config)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    model = HALOModel(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    ckpt_path = os.path.join(args.save_dir, "model8.pt")
    start_epoch = 0
    best_val = 1e10

    if args.init_ckpt_path:
        init_path = str(args.init_ckpt_path)
        if os.path.exists(init_path):
            map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
            init_ckpt = torch.load(init_path, map_location=map_location, weights_only=False)
            if isinstance(init_ckpt, dict) and "model" in init_ckpt:
                model.load_state_dict(init_ckpt["model"], strict=True)
                if rank == 0:
                    print(f"Initialized model weights from: {init_path}", flush=True)
            else:
                if rank == 0:
                    print(f"Init checkpoint missing 'model' key: {init_path}", flush=True)
        else:
            if rank == 0:
                print(f"Init checkpoint not found: {init_path}", flush=True)

    if args.resume and os.path.exists(ckpt_path):
        map_location = {"cuda:%d" % 0: "cuda:%d" % local_rank}
        checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_val = float(checkpoint.get("best_val_loss", 1e10))
        if rank == 0:
            print(f"Resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, config.epoch):
        train_sampler.set_epoch(epoch)
        model.train()

        train_loss_accum = 0.0
        train_steps = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}") if rank == 0 else train_loader
        for batch_ehr, batch_mask in pbar:
            batch_ehr = batch_ehr.to(local_rank)
            batch_mask = batch_mask.to(local_rank)
            optimizer.zero_grad()
            loss, _, _ = model(
                batch_ehr,
                position_ids=None,
                ehr_labels=batch_ehr,
                ehr_masks=batch_mask,
                pos_loss_weight=config.pos_loss_weight,
                logit_adjust=adj,
            )
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss_accum += float(loss.item())
            train_steps += 1
            if rank == 0:
                pbar.set_postfix({"loss": float(loss.item())})

        model.eval()
        val_loss_accum = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_ehr, batch_mask in val_loader:
                batch_ehr = batch_ehr.to(local_rank)
                batch_mask = batch_mask.to(local_rank)
                vloss, _, _ = model(
                    batch_ehr,
                    position_ids=None,
                    ehr_labels=batch_ehr,
                    ehr_masks=batch_mask,
                    pos_loss_weight=config.pos_loss_weight,
                    logit_adjust=adj,
                )
                if vloss.dim() > 0:
                    vloss = vloss.mean()
                val_loss_accum += float(vloss.item())
                val_steps += 1

        val_loss_tensor = torch.tensor(val_loss_accum / val_steps if val_steps > 0 else 0.0, device=local_rank, dtype=torch.float32)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = float((val_loss_tensor / world_size).item())

        if rank == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss_accum/train_steps:.6f} | Val Loss: {avg_val_loss:.6f}", flush=True)
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val,
                    "logit_adjust": adj.detach().cpu(),
                    "config": vars(config),
                    "seed": int(base_seed),
                }
                torch.save(state, ckpt_path)
                print("Saved best checkpoint", flush=True)

    cleanup()


if __name__ == "__main__":
    default_cfg = Model2Config()
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init_ckpt_path", default=None, help="Optional init checkpoint (loads weights before training).")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--lr", type=float, default=default_cfg.lr)
    p.add_argument("--epoch", type=int, default=default_cfg.epoch)
    p.add_argument("--batch_size", type=int, default=default_cfg.batch_size)
    p.add_argument("--sample_batch_size", type=int, default=default_cfg.sample_batch_size)
    p.add_argument("--pos_loss_weight", default=None, help="Optional scalar pos_weight for BCEWithLogits (empty => None).")

    p.add_argument("--logit_adjust_tau", type=float, default=default_cfg.logit_adjust_tau)
    p.add_argument("--logit_adjust_clip", type=float, default=default_cfg.logit_adjust_clip)
    p.add_argument("--apply_logit_adjust_in_sampling", action=argparse.BooleanOptionalAction, default=default_cfg.apply_logit_adjust_in_sampling)

    train(p.parse_args())

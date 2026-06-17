import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT
import argparse
import csv
import os
import subprocess
import time

import numpy as np


TORCHRUN = torchrun
PYTHON = sys.executable


def _run(cmd: list[str], *, env: dict[str, str] | None = None, log_path: str | None = None):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-4000:]}")
    return p.stdout


def _write_scaled_logit_adjust(*, base_path: str, out_path: str, base_tau: float, target_tau: float, clip: float | None):
    base = np.load(base_path).astype(np.float32).reshape(-1)
    scale = float(target_tau) / float(base_tau) if float(base_tau) != 0.0 else 0.0
    out = base * scale
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    np.save(out_path, out.astype(np.float32))


def _mean_acc_f1_from_compare_csv(path: str, *, source: str = "PCLA") -> dict[str, float]:
    acc: list[float] = []
    f1: list[float] = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["source"] != source:
                continue
            acc.append(float(row["Accuracy"]))
            f1.append(float(row["F1 Score"]))
    if not acc:
        raise RuntimeError(f"No rows found for source={source} in {path}")
    return {"mean_acc": sum(acc) / len(acc), "mean_f1": sum(f1) / len(f1), "n_labels": float(len(acc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_dir", default=DATA_MIMICIV)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda_visible_devices", default="0,1,2,3,4,6,7")
    ap.add_argument("--num_gpus", type=int, default=7)
    ap.add_argument("--master_port_base", type=int, default=31400)

    ap.add_argument("--lrs", default="5e-5,1e-4,2e-4")
    ap.add_argument("--train_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--sample_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--train_logit_tau", type=float, default=0.2)
    ap.add_argument("--train_logit_clip", type=float, default=15.0)

    ap.add_argument("--sample_tau", type=float, default=0.1, help="Sampling-time tau (scales saved logit_adjust.npy).")
    ap.add_argument("--sample_clip", type=float, default=15.0)

    ap.add_argument("--quick_total_samples", type=int, default=20000)
    ap.add_argument("--eval_epochs", type=int, default=10)
    ap.add_argument("--eval_n_train", type=int, default=2000)
    ap.add_argument("--eval_n_val", type=int, default=200)
    ap.add_argument("--eval_n_test", type=int, default=500)
    ap.add_argument("--eval_label_indices", default="", help="Empty => all 25 labels.")
    ap.add_argument("--skip_real", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument(
        "--init_ckpt_path",
        default="MODEL8_DIR/save_mimiciv_seed1/model8.pt",
        help="Warm-start checkpoint for training.",
    )
    ap.add_argument("--model8_dir", default=MODEL8_DIR)
    ap.add_argument("--save_root", default="MODEL8_DIR/save_mimiciv_lr_sweep")
    ap.add_argument("--out_csv", default="FAME_ROOT/output/model8_mimiciv_lr_sweep_quick.csv")
    args = ap.parse_args()

    lrs = [float(x.strip()) for x in str(args.lrs).split(",") if x.strip() != ""]
    if not lrs:
        raise ValueError("--lrs is empty")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    header = [
        "lr",
        "train_epochs",
        "batch_size",
        "train_logit_tau",
        "train_logit_clip",
        "sample_tau",
        "sample_clip",
        "quick_total_samples",
        "eval_epochs",
        "eval_n_train",
        "eval_n_val",
        "eval_n_test",
        "eval_label_indices",
        "mean_acc",
        "mean_f1",
        "ckpt_path",
        "synth_path",
        "compare_csv",
        "train_dir",
        "eval_dir",
    ]
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    rows: list[dict[str, str]] = []

    for idx, lr in enumerate(lrs):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_tag = f"lr{lr:g}_e{int(args.train_epochs)}_seed{int(args.seed)}_{stamp}"
        train_dir = os.path.join(args.save_root, run_tag)
        eval_dir = os.path.join(args.save_root, run_tag, "eval_quick")
        os.makedirs(train_dir, exist_ok=True)

        # ---- Train ----
        _run(
            [
                TORCHRUN,
                "--standalone",
                f"--nproc-per-node={int(args.num_gpus)}",
                f"--master_port={int(args.master_port_base) + idx}",
                os.path.join(args.model8_dir, "train.py"),
                "--data_dir",
                args.base_data_dir,
                "--save_dir",
                train_dir,
                "--seed",
                str(int(args.seed)),
                "--no-resume",
                "--init_ckpt_path",
                str(args.init_ckpt_path),
                "--num_workers",
                str(int(args.num_workers)),
                "--lr",
                str(float(lr)),
                "--epoch",
                str(int(args.train_epochs)),
                "--batch_size",
                str(int(args.batch_size)),
                "--sample_batch_size",
                str(int(args.sample_batch_size)),
                "--logit_adjust_tau",
                str(float(args.train_logit_tau)),
                "--logit_adjust_clip",
                str(float(args.train_logit_clip)),
            ],
            env=env,
            log_path=os.path.join(train_dir, "train.log"),
        )

        ckpt_path = os.path.join(train_dir, "model8.pt")
        base_adj = os.path.join(train_dir, "logit_adjust.npy")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"Missing checkpoint after training: {ckpt_path}")
        if not os.path.exists(base_adj):
            raise RuntimeError(f"Missing logit_adjust after training: {base_adj}")

        # ---- Generate (quick) ----
        scaled_adj = os.path.join(train_dir, "logit_adjust_scaled_for_sampling.npy")
        _write_scaled_logit_adjust(
            base_path=base_adj,
            out_path=scaled_adj,
            base_tau=float(args.train_logit_tau),
            target_tau=float(args.sample_tau),
            clip=float(args.sample_clip) if args.sample_clip is not None else None,
        )

        _run(
            [
                TORCHRUN,
                "--standalone",
                f"--nproc-per-node={int(args.num_gpus)}",
                f"--master_port={int(args.master_port_base) + 100 + idx}",
                os.path.join(args.model8_dir, "test.py"),
                "--data_dir",
                args.base_data_dir,
                "--save_dir",
                train_dir,
                "--seed",
                str(int(args.seed)),
                "--ckpt_path",
                ckpt_path,
                "--skip_eval",
                "--total_samples",
                str(int(args.quick_total_samples)),
                "--logit_adjust_path",
                scaled_adj,
            ],
            env=env,
            log_path=os.path.join(train_dir, "generate_quick.log"),
        )

        synth_path = os.path.join(train_dir, "datasets", "haloDataset.pkl")
        if not os.path.exists(synth_path):
            raise RuntimeError(f"Missing synthetic dataset: {synth_path}")

        # ---- Downstream eval (quick) ----
        os.makedirs(eval_dir, exist_ok=True)
        eval_cmd = [
            TORCHRUN,
            "--standalone",
            f"--nproc-per-node={int(args.num_gpus)}",
            f"--master_port={int(args.master_port_base) + 200 + idx}",
            EVAL_PY,
            "--base_data_dir",
            args.base_data_dir,
            "--save_dir",
            eval_dir,
            "--seed",
            str(int(args.seed)),
            "--sources",
            "PCLA",
            "--extra_source",
            f"PCLA={synth_path}",
            "--epochs",
            str(int(args.eval_epochs)),
            "--n_train",
            str(int(args.eval_n_train)),
            "--n_val",
            str(int(args.eval_n_val)),
            "--n_test",
            str(int(args.eval_n_test)),
        ]
        if str(args.eval_label_indices).strip() != "":
            eval_cmd += ["--label_indices", str(args.eval_label_indices).strip()]
        if bool(args.skip_real):
            eval_cmd.append("--skip_real")
        _run(eval_cmd, env=env, log_path=os.path.join(eval_dir, "eval.log"))

        compare_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")
        means = _mean_acc_f1_from_compare_csv(compare_csv, source="PCLA")

        row = {
            "lr": str(float(lr)),
            "train_epochs": str(int(args.train_epochs)),
            "batch_size": str(int(args.batch_size)),
            "train_logit_tau": str(float(args.train_logit_tau)),
            "train_logit_clip": str(float(args.train_logit_clip)),
            "sample_tau": str(float(args.sample_tau)),
            "sample_clip": str(float(args.sample_clip)),
            "quick_total_samples": str(int(args.quick_total_samples)),
            "eval_epochs": str(int(args.eval_epochs)),
            "eval_n_train": str(int(args.eval_n_train)),
            "eval_n_val": str(int(args.eval_n_val)),
            "eval_n_test": str(int(args.eval_n_test)),
            "eval_label_indices": str(args.eval_label_indices),
            "mean_acc": f"{means['mean_acc']:.6f}",
            "mean_f1": f"{means['mean_f1']:.6f}",
            "ckpt_path": ckpt_path,
            "synth_path": synth_path,
            "compare_csv": compare_csv,
            "train_dir": train_dir,
            "eval_dir": eval_dir,
        }
        rows.append(row)
        with open(args.out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

        print(f"lr={lr:g} mean_acc={means['mean_acc']:.6f} mean_f1={means['mean_f1']:.6f}")

    best = max(rows, key=lambda r: (float(r["mean_f1"]), float(r["mean_acc"])))
    print("BEST", best["lr"], "mean_acc", best["mean_acc"], "mean_f1", best["mean_f1"])
    print("CSV", args.out_csv)


if __name__ == "__main__":
    main()


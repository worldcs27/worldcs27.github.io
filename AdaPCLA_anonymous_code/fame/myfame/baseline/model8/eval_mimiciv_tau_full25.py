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


def _mean_acc_f1_from_compare_csv(path: str, *, source: str) -> dict[str, float]:
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
    ap.add_argument("--taus", default="0.18,0.2,0.22")
    ap.add_argument("--clip", type=float, default=15.0)
    ap.add_argument("--base_tau", type=float, default=0.2)
    ap.add_argument("--model8_dir", default=MODEL8_DIR)
    ap.add_argument("--ckpt_path", default="MODEL8_DIR/save_mimiciv_seed1/model8.pt")
    ap.add_argument("--base_logit_adjust", default="MODEL8_DIR/save_mimiciv_seed1/logit_adjust.npy")
    ap.add_argument("--total_samples", type=int, default=50000)
    ap.add_argument("--cuda_visible_devices", default="0,3,4")
    ap.add_argument("--num_gpus", type=int, default=3)
    ap.add_argument("--master_port_base", type=int, default=31000)
    ap.add_argument("--skip_real", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out_csv", default="FAME_ROOT/output/model8_mimiciv_tau_full25.csv")
    ap.add_argument("--reuse_generation", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reuse_eval", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip() != ""]
    if not taus:
        raise ValueError("--taus is empty")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    header = [
        "tau",
        "clip",
        "seed",
        "total_samples",
        "mean_acc",
        "mean_f1",
        "synth_path",
        "compare_csv",
        "save_dir",
        "eval_dir",
    ]
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    for idx, tau in enumerate(taus):
        tau_tag = f"{tau:g}"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.model8_dir, f"save_mimiciv_seed{int(args.seed)}_tau{tau_tag}_50k")
        eval_dir = os.path.join("FAME_ROOT/evaluate/save", f"mimiciv_data2_seed{int(args.seed)}_model8_tau{tau_tag}_50k_{stamp}")
        os.makedirs(save_dir, exist_ok=True)

        adj_path = os.path.join(save_dir, "logit_adjust_scaled.npy")
        _write_scaled_logit_adjust(
            base_path=args.base_logit_adjust,
            out_path=adj_path,
            base_tau=float(args.base_tau),
            target_tau=float(tau),
            clip=float(args.clip) if args.clip is not None else None,
        )

        synth_path = os.path.join(save_dir, "datasets", "haloDataset.pkl")
        compare_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")

        need_gen = not (args.reuse_generation and os.path.exists(synth_path))
        if need_gen:
            _run(
                [
                    torchrun,
                    "--standalone",
                    f"--nproc-per-node={int(args.num_gpus)}",
                    f"--master_port={int(args.master_port_base) + idx}",
                    os.path.join(args.model8_dir, "test.py"),
                    "--data_dir",
                    args.base_data_dir,
                    "--save_dir",
                    save_dir,
                    "--seed",
                    str(int(args.seed)),
                    "--ckpt_path",
                    args.ckpt_path,
                    "--skip_eval",
                    "--total_samples",
                    str(int(args.total_samples)),
                    "--logit_adjust_path",
                    adj_path,
                ],
                env=env,
                log_path=os.path.join(save_dir, f"generate_seed{int(args.seed)}_{int(args.total_samples)}.log"),
            )

        need_eval = not (args.reuse_eval and os.path.exists(compare_csv))
        if need_eval:
            os.makedirs(eval_dir, exist_ok=True)
            cmd = [
                torchrun,
                "--standalone",
                f"--nproc-per-node={int(args.num_gpus)}",
                f"--master_port={int(args.master_port_base) + 100 + idx}",
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
            ]
            if bool(args.skip_real):
                cmd.append("--skip_real")
            _run(cmd, env=env, log_path=os.path.join(eval_dir, "run.log"))

        means = _mean_acc_f1_from_compare_csv(compare_csv, source="PCLA")
        row = {
            "tau": str(tau),
            "clip": str(args.clip),
            "seed": str(int(args.seed)),
            "total_samples": str(int(args.total_samples)),
            "mean_acc": f"{means['mean_acc']:.6f}",
            "mean_f1": f"{means['mean_f1']:.6f}",
            "synth_path": synth_path,
            "compare_csv": compare_csv,
            "save_dir": save_dir,
            "eval_dir": eval_dir,
        }
        with open(args.out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

        print(f"tau={tau:g} mean_acc={means['mean_acc']:.6f} mean_f1={means['mean_f1']:.6f}")


if __name__ == "__main__":
    main()


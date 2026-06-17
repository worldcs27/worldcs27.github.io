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
    return {
        "mean_acc": sum(acc) / len(acc),
        "mean_f1": sum(f1) / len(f1),
        "n_labels": float(len(acc)),
    }


def _write_scaled_logit_adjust(
    *,
    base_path: str,
    out_path: str,
    base_tau: float,
    target_tau: float,
    clip: float | None,
):
    base = np.load(base_path).astype(np.float32).reshape(-1)
    scale = float(target_tau) / float(base_tau) if float(base_tau) != 0.0 else 0.0
    out = base * scale
    if clip is not None:
        out = np.clip(out, -float(clip), float(clip))
    np.save(out_path, out.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_dir", default=DATA_MIMICIV)
    ap.add_argument("--model8_dir", default=MODEL8_DIR)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cuda_visible_devices", default="1,2,4,6,7")
    ap.add_argument("--num_gpus", type=int, default=5)
    ap.add_argument("--master_port_base", type=int, default=29700)
    ap.add_argument(
        "--skip_final",
        action="store_true",
        help="Only run the quick sweep; skip the final 50k generation + full 25-label evaluation.",
    )

    ap.add_argument("--ckpt_path", default="MODEL8_DIR/save_mimiciv_seed1/model8.pt")
    ap.add_argument("--base_logit_adjust", default="MODEL8_DIR/save_mimiciv_seed1/logit_adjust.npy")
    ap.add_argument("--base_tau", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=15.0)

    ap.add_argument("--quick_total_samples", type=int, default=20000)
    ap.add_argument("--quick_epochs", type=int, default=10)
    ap.add_argument("--quick_n_train", type=int, default=2000)
    ap.add_argument("--quick_n_val", type=int, default=200)
    ap.add_argument("--quick_n_test", type=int, default=500)
    ap.add_argument("--quick_label_indices", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument(
        "--skip_real_in_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, skip the Real baseline in evaluate_synthetic_training.py for faster tuning.",
    )

    ap.add_argument("--final_total_samples", type=int, default=50000)
    ap.add_argument(
        "--final_save_dir",
        default="",
        help="Optional final model8 save_dir for 50k generation (default: <model8_dir>/save_mimiciv_seed1).",
    )
    ap.add_argument("--final_eval_dir", default="FAME_ROOT/evaluate/save/mimiciv_data2_seed1_25label_model8_tuned")
    ap.add_argument("--final_summary_csv", default="FAME_ROOT/output/mimiciv_data2_seed1_model8_tuned_summary.csv")
    ap.add_argument("--final_cuda_visible_devices", default="", help="Optional override for the final stage CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--final_num_gpus", type=int, default=0, help="Optional override for the final stage nproc_per_node.")

    ap.add_argument(
        "--taus",
        default="0.0,0.05,0.1,0.15,0.2",
        help="Comma-separated tau values to try (sampling-time scaling of logit_adjust).",
    )
    args = ap.parse_args()

    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip() != ""]
    if not taus:
        raise ValueError("--taus is empty")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    save_root = os.path.join(args.model8_dir, "save_mimiciv_seed1", "tune_tau")
    os.makedirs(save_root, exist_ok=True)
    out_csv = os.path.join(save_root, "tune_results.csv")

    header = [
        "tau",
        "clip",
        "quick_total_samples",
        "quick_epochs",
        "quick_label_indices",
        "save_dir",
        "eval_dir",
        "mean_acc",
        "mean_f1",
        "compare_csv",
        "synth_path",
    ]
    if not os.path.exists(out_csv):
        with open(out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    rows: list[dict[str, str]] = []

    for idx, tau in enumerate(taus):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"tau{tau:g}_clip{args.clip:g}_{stamp}"
        save_dir = os.path.join(save_root, run_name)
        eval_dir = os.path.join(save_dir, "eval_quick")
        os.makedirs(save_dir, exist_ok=True)

        adj_path = os.path.join(save_dir, "logit_adjust_scaled.npy")
        _write_scaled_logit_adjust(
            base_path=args.base_logit_adjust,
            out_path=adj_path,
            base_tau=float(args.base_tau),
            target_tau=float(tau),
            clip=float(args.clip) if args.clip is not None else None,
        )

        # 1) Generate synthetic dataset (skip test-set eval inside test.py).
        gen_log = os.path.join(save_dir, "generate_quick.log")
        _run(
            [
                torchrun,
                "--standalone",
                f"--nproc_per_node={int(args.num_gpus)}",
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
                str(int(args.quick_total_samples)),
                "--logit_adjust_path",
                adj_path,
            ],
            env=env,
            log_path=gen_log,
        )

        synth_path = os.path.join(save_dir, "datasets", "haloDataset.pkl")
        compare_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")

        # 2) Quick downstream evaluation on a subset of labels.
        eval_log = os.path.join(save_dir, "eval_quick.log")
        eval_cmd = [
            torchrun,
            "--standalone",
            f"--nproc_per_node={int(args.num_gpus)}",
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
            "--epochs",
            str(int(args.quick_epochs)),
            "--n_train",
            str(int(args.quick_n_train)),
            "--n_val",
            str(int(args.quick_n_val)),
            "--n_test",
            str(int(args.quick_n_test)),
            "--label_indices",
            str(args.quick_label_indices),
        ]
        if bool(args.skip_real_in_eval):
            eval_cmd.append("--skip_real")
        _run(
            eval_cmd,
            env=env,
            log_path=eval_log,
        )

        means = _mean_acc_f1_from_compare_csv(compare_csv, source="PCLA")
        row = {
            "tau": str(tau),
            "clip": str(args.clip),
            "quick_total_samples": str(args.quick_total_samples),
            "quick_epochs": str(args.quick_epochs),
            "quick_label_indices": str(args.quick_label_indices),
            "save_dir": save_dir,
            "eval_dir": eval_dir,
            "mean_acc": f"{means['mean_acc']:.6f}",
            "mean_f1": f"{means['mean_f1']:.6f}",
            "compare_csv": compare_csv,
            "synth_path": synth_path,
        }
        rows.append(row)
        with open(out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)

    # Pick best (mean_f1 first, then mean_acc).
    best = max(rows, key=lambda r: (float(r["mean_f1"]), float(r["mean_acc"])))
    best_tau = float(best["tau"])

    if args.skip_final:
        print(f"[DONE] best_tau={best_tau} (quick sweep only)")
        print(f"Quick sweep CSV: {out_csv}")
        return

    # Final run: regenerate 50k with best tau and run full 25-label eval.
    final_cuda = str(args.final_cuda_visible_devices or "").strip() or str(args.cuda_visible_devices)
    final_num_gpus = int(args.final_num_gpus) if int(args.final_num_gpus) > 0 else int(args.num_gpus)
    env_final = env.copy()
    env_final["CUDA_VISIBLE_DEVICES"] = final_cuda

    os.makedirs(args.final_eval_dir, exist_ok=True)
    final_adj_path = os.path.join(save_root, "best_logit_adjust_scaled.npy")
    _write_scaled_logit_adjust(
        base_path=args.base_logit_adjust,
        out_path=final_adj_path,
        base_tau=float(args.base_tau),
        target_tau=float(best_tau),
        clip=float(args.clip) if args.clip is not None else None,
    )

    final_save_dir = str(args.final_save_dir or "").strip() or os.path.join(args.model8_dir, "save_mimiciv_seed1")
    os.makedirs(final_save_dir, exist_ok=True)

    _run(
        [
            torchrun,
            "--standalone",
            f"--nproc_per_node={final_num_gpus}",
            f"--master_port={int(args.master_port_base) + 999}",
            os.path.join(args.model8_dir, "test.py"),
            "--data_dir",
            args.base_data_dir,
            "--save_dir",
            final_save_dir,
            "--seed",
            str(int(args.seed)),
            "--ckpt_path",
            args.ckpt_path,
            "--skip_eval",
            "--total_samples",
            str(int(args.final_total_samples)),
            "--logit_adjust_path",
            final_adj_path,
        ],
        env=env_final,
        log_path=os.path.join(save_root, "generate_final_50k.log"),
    )

    final_synth_path = os.path.join(final_save_dir, "datasets", "haloDataset.pkl")
    final_eval_cmd = [
        torchrun,
        "--standalone",
        f"--nproc_per_node={final_num_gpus}",
        f"--master_port={int(args.master_port_base) + 1999}",
        EVAL_PY,
        "--base_data_dir",
        args.base_data_dir,
        "--save_dir",
        args.final_eval_dir,
        "--seed",
        str(int(args.seed)),
        "--sources",
        "PCLA",
        "--extra_source",
        f"PCLA={final_synth_path}",
    ]
    if bool(args.skip_real_in_eval):
        final_eval_cmd.append("--skip_real")
    _run(final_eval_cmd, env=env_final, log_path=os.path.join(save_root, "eval_final_25label.log"))

    final_compare = os.path.join(args.final_eval_dir, "compare_real_halo_mymodel2.csv")
    final_means = _mean_acc_f1_from_compare_csv(final_compare, source="PCLA")

    os.makedirs(os.path.dirname(args.final_summary_csv), exist_ok=True)
    with open(args.final_summary_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "tau",
                "clip",
                "total_samples",
                "mean_acc",
                "mean_f1",
                "compare_csv",
                "synth_path",
                "eval_dir",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "model": "model8",
                "seed": str(int(args.seed)),
                "tau": str(best_tau),
                "clip": str(args.clip),
                "total_samples": str(int(args.final_total_samples)),
                "mean_acc": f"{final_means['mean_acc']:.6f}",
                "mean_f1": f"{final_means['mean_f1']:.6f}",
                "compare_csv": final_compare,
                "synth_path": final_synth_path,
                "eval_dir": args.final_eval_dir,
            }
        )

    print(f"[DONE] best_tau={best_tau} final mean_acc={final_means['mean_acc']:.6f} mean_f1={final_means['mean_f1']:.6f}")
    print(f"Quick sweep CSV: {out_csv}")
    print(f"Final summary CSV: {args.final_summary_csv}")


if __name__ == "__main__":
    main()

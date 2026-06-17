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


def _run(cmd: list[str], env: dict[str, str] | None = None):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def _mean_acc_f1_from_compare_csv(path: str):
    acc = {"Real": [], "HALO": [], "MyModel2": []}
    f1 = {"Real": [], "HALO": [], "MyModel2": []}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row["source"]
            if src not in acc:
                continue
            acc[src].append(float(row["Accuracy"]))
            f1[src].append(float(row["F1 Score"]))
    out = {}
    for src in acc:
        out[src] = {
            "mean_acc": sum(acc[src]) / max(1, len(acc[src])),
            "mean_f1": sum(f1[src]) / max(1, len(f1[src])),
            "n_labels": len(acc[src]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda_visible_devices", default="0,1,2,3,4,6,7")
    ap.add_argument("--num_gpus", type=int, default=7)
    ap.add_argument("--master_port_base", type=int, default=29600)
    ap.add_argument("--total_samples", type=int, default=33494)
    ap.add_argument("--data_dir", default=DATA_MIMICIII)
    ap.add_argument("--init_ckpt_path", default=HALO_MIMICIII_CKPT)
    ap.add_argument("--model7_dir", default=MODEL7_DIR)
    ap.add_argument("--save_root", default="MODEL7_DIR/save")
    ap.add_argument("--eval_save_root", default="FAME_ROOT/evaluate/save")
    ap.add_argument("--out_csv", default="FAME_ROOT/output/model7_hparam_sweep.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Small, practical grid (keep runtime reasonable).
    sweep = [
        {"name": "tau0.0_clip15_lr1e-4", "tau": 0.0, "clip": 15, "lr": 1e-4, "pos_loss_weight": "", "apply_adj_sampling": 1},
        {"name": "tau0.1_clip15_lr1e-4", "tau": 0.1, "clip": 15, "lr": 1e-4, "pos_loss_weight": "", "apply_adj_sampling": 1},
        {"name": "tau0.2_clip15_lr1e-4", "tau": 0.2, "clip": 15, "lr": 1e-4, "pos_loss_weight": "", "apply_adj_sampling": 1},
        {"name": "tau0.2_clip15_lr5e-5", "tau": 0.2, "clip": 15, "lr": 5e-5, "pos_loss_weight": "", "apply_adj_sampling": 1},
        {"name": "tau0.2_clip15_lr2e-4", "tau": 0.2, "clip": 15, "lr": 2e-4, "pos_loss_weight": "", "apply_adj_sampling": 1},
        {"name": "tau0.2_clip15_lr1e-4_pos2", "tau": 0.2, "clip": 15, "lr": 1e-4, "pos_loss_weight": "2.0", "apply_adj_sampling": 1},
    ]

    header = [
        "exp_name",
        "save_dir",
        "eval_dir",
        "tau",
        "clip",
        "lr",
        "epochs",
        "pos_loss_weight",
        "apply_adj_sampling",
        "mean_acc_MyModel2",
        "mean_f1_MyModel2",
        "mean_acc_HALO",
        "mean_f1_HALO",
        "compare_csv",
        "synth_path",
    ]
    if not os.path.exists(args.out_csv):
        with open(args.out_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writeheader()

    for idx, cfg in enumerate(sweep):
        exp_name = cfg["name"]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_root, f"{exp_name}_{stamp}")
        eval_dir = os.path.join(args.eval_save_root, f"model7_{exp_name}_{stamp}")
        synth_path = os.path.join(save_dir, "datasets", "haloDataset.pkl")
        compare_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["NUM_GPUS"] = str(args.num_gpus)
        env["MASTER_PORT"] = str(args.master_port_base + idx)
        env["DATA_DIR"] = args.data_dir
        env["SAVE_DIR"] = save_dir
        env["EPOCHS"] = str(args.epochs)
        env["LR"] = str(cfg["lr"])
        env["LOGIT_ADJUST_TAU"] = str(cfg["tau"])
        env["LOGIT_ADJUST_CLIP"] = str(cfg["clip"])
        env["POS_LOSS_WEIGHT"] = str(cfg["pos_loss_weight"])
        env["APPLY_LOGIT_ADJUST_IN_SAMPLING"] = "1" if int(cfg["apply_adj_sampling"]) else "0"
        env["RESUME"] = "0"
        env["INIT_CKPT_PATH"] = args.init_ckpt_path
        env["TOTAL_SAMPLES"] = str(args.total_samples)

        run_sh = os.path.join(args.model7_dir, "run.sh")
        if args.dry_run:
            print("DRY RUN:", exp_name, save_dir)
            continue

        _run(["bash", run_sh, "train"], env=env)
        _run(["bash", run_sh, "test", "--skip_eval"], env=env)

        # Downstream 25-label evaluation (DDP).
        _run(
            [
                torchrun,
                f"--nproc_per_node={args.num_gpus}",
                f"--master_port={args.master_port_base + 100 + idx}",
                EVAL_PY,
                "--mymodel2_path",
                synth_path,
                "--save_dir",
                eval_dir,
            ],
            env=env,
        )

        means = _mean_acc_f1_from_compare_csv(compare_csv)
        row = {
            "exp_name": exp_name,
            "save_dir": save_dir,
            "eval_dir": eval_dir,
            "tau": cfg["tau"],
            "clip": cfg["clip"],
            "lr": cfg["lr"],
            "epochs": args.epochs,
            "pos_loss_weight": cfg["pos_loss_weight"],
            "apply_adj_sampling": cfg["apply_adj_sampling"],
            "mean_acc_MyModel2": means["MyModel2"]["mean_acc"],
            "mean_f1_MyModel2": means["MyModel2"]["mean_f1"],
            "mean_acc_HALO": means["HALO"]["mean_acc"],
            "mean_f1_HALO": means["HALO"]["mean_f1"],
            "compare_csv": compare_csv,
            "synth_path": synth_path,
        }
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writerow(row)


if __name__ == "__main__":
    main()


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


def _run(cmd: list[str], env: dict[str, str]):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout


def _mean_acc_f1(path: str):
    acc = {"HALO": [], "MyModel2": []}
    f1 = {"HALO": [], "MyModel2": []}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row["source"]
            if src not in acc:
                continue
            acc[src].append(float(row["Accuracy"]))
            f1[src].append(float(row["F1 Score"]))
    return {
        "HALO": (sum(acc["HALO"]) / len(acc["HALO"]), sum(f1["HALO"]) / len(f1["HALO"])),
        "MyModel2": (sum(acc["MyModel2"]) / len(acc["MyModel2"]), sum(f1["MyModel2"]) / len(f1["MyModel2"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda_visible_devices", default="0,1,2,3,4,6,7")
    ap.add_argument("--num_gpus", type=int, default=7)
    ap.add_argument("--train_port_base", type=int, default=29560)
    ap.add_argument("--eval_port_base", type=int, default=29660)
    ap.add_argument("--total_samples", type=int, default=33494)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--clip", type=float, default=15.0)
    ap.add_argument("--init_ckpt_path", default=HALO_MIMICIII_CKPT)
    ap.add_argument("--data_dir", default=DATA_MIMICIII)
    ap.add_argument("--model7_dir", default=MODEL7_DIR)
    ap.add_argument("--save_root", default="MODEL7_DIR/save")
    ap.add_argument("--eval_save_root", default="FAME_ROOT/evaluate/save")
    ap.add_argument("--out_csv", default="FAME_ROOT/output/model7_hparam_sweep.csv")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # Focused scan around current best:
    # - vary pos_loss_weight around 2.0
    # - test whether applying logit_adjust during sampling matters
    ring = [
        {"exp_name": "ring_pos1.5", "pos_loss_weight": "1.5", "apply_adj_sampling": 1},
        {"exp_name": "ring_pos2.5", "pos_loss_weight": "2.5", "apply_adj_sampling": 1},
        {"exp_name": "ring_pos3.0", "pos_loss_weight": "3.0", "apply_adj_sampling": 1},
        {"exp_name": "ring_pos2.0_no_adj_sampling", "pos_loss_weight": "2.0", "apply_adj_sampling": 0},
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

    run_sh = os.path.join(args.model7_dir, "run.sh")
    for idx, cfg in enumerate(ring):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"{cfg['exp_name']}_tau{args.tau}_clip{int(args.clip)}_lr{args.lr:g}_e{args.epochs}"
        save_dir = os.path.join(args.save_root, f"{exp_name}_{stamp}")
        eval_dir = os.path.join(args.eval_save_root, f"model7_{exp_name}_{stamp}")
        synth_path = os.path.join(save_dir, "datasets", "haloDataset.pkl")
        compare_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env["NUM_GPUS"] = str(args.num_gpus)
        env["MASTER_PORT"] = str(args.train_port_base + idx)
        env["DATA_DIR"] = args.data_dir
        env["SAVE_DIR"] = save_dir
        env["EPOCHS"] = str(args.epochs)
        env["LR"] = str(args.lr)
        env["LOGIT_ADJUST_TAU"] = str(args.tau)
        env["LOGIT_ADJUST_CLIP"] = str(args.clip)
        env["POS_LOSS_WEIGHT"] = str(cfg["pos_loss_weight"])
        env["APPLY_LOGIT_ADJUST_IN_SAMPLING"] = "1" if int(cfg["apply_adj_sampling"]) else "0"
        env["RESUME"] = "0"
        env["INIT_CKPT_PATH"] = args.init_ckpt_path
        env["TOTAL_SAMPLES"] = str(args.total_samples)

        if args.dry_run:
            print("DRY RUN:", save_dir, eval_dir)
            continue

        _run(["bash", run_sh, "train"], env=env)
        _run(["bash", run_sh, "test", "--skip_eval"], env=env)
        _run(
            [
                torchrun,
                f"--nproc_per_node={args.num_gpus}",
                f"--master_port={args.eval_port_base + idx}",
                EVAL_PY,
                "--mymodel2_path",
                synth_path,
                "--save_dir",
                eval_dir,
            ],
            env=env,
        )

        means = _mean_acc_f1(compare_csv)
        row = {
            "exp_name": exp_name,
            "save_dir": save_dir,
            "eval_dir": eval_dir,
            "tau": args.tau,
            "clip": args.clip,
            "lr": args.lr,
            "epochs": args.epochs,
            "pos_loss_weight": cfg["pos_loss_weight"],
            "apply_adj_sampling": cfg["apply_adj_sampling"],
            "mean_acc_MyModel2": means["MyModel2"][0],
            "mean_f1_MyModel2": means["MyModel2"][1],
            "mean_acc_HALO": means["HALO"][0],
            "mean_f1_HALO": means["HALO"][1],
            "compare_csv": compare_csv,
            "synth_path": synth_path,
        }
        with open(args.out_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)


if __name__ == "__main__":
    main()


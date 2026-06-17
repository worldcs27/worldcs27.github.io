#!/usr/bin/env python3
"""
Minimal reproduction of PCLA (Model7) best-3-seeds experiment.

关键点：
- 不改动 `dataset/` 与 `fame/` 目录下任何代码或数据；
- 直接复用 `fame/myfame/baseline/model7` 中已有的训练 / 采样逻辑（bash.sh, train.py, test.py）；
- 只在 `mywork/model1` 下生成新的 `save/`、`evaluate/`、`output/` 结果。

用法（在含 torchrun 的环境中）：
    cd EXPERIMENTS_ROOT/model1
    python run_pcla_best_3seeds.py  --seeds 1 2 3

这会在当前目录下生成：
- save/            ：每个 seed 对应一套 PCLA 生成器（model7）及其 `datasets/haloDataset.pkl`
- evaluate/        ：对应的下游 25-label 评估结果（compare_real_halo_mymodel2.csv）
- output/pcla_best_seed3_summary.{csv,json} ：和原 `model7_best_seed3_summary` 结构等价的汇总结果
"""

from __future__ import annotations
import os, sys
_ROOT = os.environ.get('ADAPCLA_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 3))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from paths_config import FAME_ROOT, MODEL7_DIR, MODEL8_DIR, EVAL_PY, DATA_MIMICIII, DATA_MIMICIV, EXPERIMENTS_ROOT, HALO_MIMICIII_CKPT, HALO_MIMICIV_CKPT

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
from dataclasses import dataclass


DEFAULT_DEVICES = "0,1,2,3,4,6,7"
DEFAULT_BASE_PORT = 29540

# 固定引用原仓库中的脚本（不修改原代码）


@dataclass(frozen=True)
class RunResult:
    seed: int
    mean_acc: float
    mean_f1: float
    save_dir: str
    eval_dir: str


def _which_torchrun() -> str:
    """尽量复用 repeat_best_3seeds.py 里的查找逻辑，只是本地简化版。"""
    if os.environ.get("TORCHRUN"):
        return os.environ["TORCHRUN"]
    p = shutil.which("torchrun")
    if p:
        return p
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        cand = os.path.join(conda, "bin", "torchrun")
        if os.path.exists(cand):
            return cand
    # 退回到原工程里用过的默认路径（如果存在）
        raise FileNotFoundError(
        "torchrun not found. Activate your conda env or set TORCHRUN=/path/to/torchrun."
    )


def _parse_mean_metrics(csv_path: str, *, source: str = "MyModel2") -> tuple[float, float]:
    """从 compare_real_halo_mymodel2.csv 中提取 source=MyModel2 的 mean acc / f1。"""
    accs: list[float] = []
    f1s: list[float] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") != source:
                continue
            a = row.get("Accuracy")
            f1 = row.get("F1 Score")
            if a not in (None, ""):
                accs.append(float(a))
            if f1 not in (None, ""):
                f1s.append(float(f1))
    if not accs or not f1s:
        raise ValueError(f"No rows for source={source} in {csv_path}")
    return sum(accs) / len(accs), sum(f1s) / len(f1s)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Empty values")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, var**0.5


def main() -> int:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    save_root = os.path.join(this_dir, "save")
    eval_root = os.path.join(this_dir, "evaluate")
    output_root = os.path.join(this_dir, "output")
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(eval_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3,4,6,7",
    )
    p.add_argument("--base_port", type=int, default=DEFAULT_BASE_PORT)
    p.add_argument("--total_samples", type=int, default=33494)

    # 与 repeat_best_3seeds.py 保持一致的 best 配置（不改内在逻辑）
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--clip", type=float, default=15.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--pos_loss_weight", type=float, default=1.5)
    p.add_argument("--apply_logit_adjust_in_sampling", type=int, choices=[0, 1], default=1)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if eval CSV exists.",
    )

    p.add_argument(
        "--output_csv",
        default=os.path.join(output_root, "pcla_best_seed3_summary.csv"),
    )
    p.add_argument(
        "--output_json",
        default=os.path.join(output_root, "pcla_best_seed3_summary.json"),
    )
    args = p.parse_args()

    devices = str(args.devices)
    num_gpus = len([x for x in devices.split(",") if x.strip() != ""])

    torchrun = _which_torchrun()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[RunResult] = []

    for idx, seed in enumerate(args.seeds):
        run_tag = (
            f"best_pos{args.pos_loss_weight}_tau{args.tau}_clip{args.clip}"
            f"_lr{args.lr}_e{args.epochs}_seed{seed}_{ts}"
        )
        # 与原逻辑不同：我们把 SAVE_DIR / eval_dir 都放在 mywork/model1 下
        save_dir = os.path.join(save_root, run_tag)
        eval_dir = os.path.join(eval_root, run_tag)
        eval_csv = os.path.join(eval_dir, "compare_real_halo_mymodel2.csv")

        if (not args.force) and os.path.exists(eval_csv):
            mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="MyModel2")
            results.append(
                RunResult(
                    seed=seed,
                    mean_acc=mean_acc,
                    mean_f1=mean_f1,
                    save_dir=save_dir,
                    eval_dir=eval_dir,
                )
            )
            continue

        # 环境变量基本仿照原 repeat_best_3seeds.py，只是 SAVE_DIR 改到当前目录下
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices
        env["NUM_GPUS"] = str(num_gpus)
        env["MASTER_PORT"] = str(int(args.base_port) + idx * 10)
        env["SAVE_DIR"] = save_dir
        env["TOTAL_SAMPLES"] = str(int(args.total_samples))
        # 显式指定数据目录与初始化 ckpt，避免 bash.sh 中默认的 FAME_ROOT 路径在当前服务器上无权限/不存在
        env["DATA_DIR"] = DATA_MIMICIII
        env["INIT_CKPT_PATH"] = (
            HALO_MIMICIII_CKPT
        )

        env["SEED"] = str(int(seed))
        env["LR"] = str(float(args.lr))
        env["EPOCHS"] = str(int(args.epochs))
        env["POS_LOSS_WEIGHT"] = str(float(args.pos_loss_weight))
        env["LOGIT_ADJUST_TAU"] = str(float(args.tau))
        env["LOGIT_ADJUST_CLIP"] = str(float(args.clip))
        env["APPLY_LOGIT_ADJUST_IN_SAMPLING"] = str(int(args.apply_logit_adjust_in_sampling))
        env["RESUME"] = "0"

        # 1) 训练 + 生成：直接调用原仓库的 bash.sh（不改其内部逻辑）
        subprocess.run(
            ["bash", os.path.join(MODEL7_DIR, "bash.sh"), "all"],
            check=True,
            env=env,
        )

        # 2) 下游 25-label 评估：调用原 evaluate_synthetic_training.py
        eval_env = env.copy()
        eval_env["MASTER_PORT"] = str(int(args.base_port) + idx * 10 + 1)
        os.makedirs(eval_dir, exist_ok=True)
        subprocess.run(
            [
                torchrun,
                f"--nproc_per_node={num_gpus}",
                f"--master_port={eval_env['MASTER_PORT']}",
                EVAL_PY,
                "--mymodel2_path",
                os.path.join(save_dir, "datasets", "haloDataset.pkl"),
                "--save_dir",
                eval_dir,
                "--seed",
                str(int(seed)),
            ],
            check=True,
            env=eval_env,
        )

        mean_acc, mean_f1 = _parse_mean_metrics(eval_csv, source="MyModel2")
        results.append(
            RunResult(
                seed=seed,
                mean_acc=mean_acc,
                mean_f1=mean_f1,
                save_dir=save_dir,
                eval_dir=eval_dir,
            )
        )

    accs = [r.mean_acc for r in results]
    f1s = [r.mean_f1 for r in results]
    mean_acc, std_acc = _mean_std(accs)
    mean_f1, std_f1 = _mean_std(f1s)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mean_acc", "mean_f1", "delta_acc", "delta_f1", "save_dir", "eval_dir"])
        for r in results:
            w.writerow(
                [
                    r.seed,
                    f"{r.mean_acc:.6f}",
                    f"{r.mean_f1:.6f}",
                    f"{(r.mean_acc - mean_acc):+.6f}",
                    f"{(r.mean_f1 - mean_f1):+.6f}",
                    r.save_dir,
                    r.eval_dir,
                ]
            )
        w.writerow([])
        w.writerow(
            [
                "mean±std",
                f"{mean_acc:.6f}±{std_acc:.6f}",
                f"{mean_f1:.6f}±{std_f1:.6f}",
                "",
                "",
                "",
                "",
            ]
        )

    with open(args.output_json, "w") as f:
        json.dump(
            {
                "seeds": [r.seed for r in results],
                "metric": "mean over 25 labels from compare_real_halo_mymodel2.csv (source=MyModel2)",
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "runs": [
                    {
                        "seed": r.seed,
                        "mean_acc": r.mean_acc,
                        "mean_f1": r.mean_f1,
                        "delta_acc": r.mean_acc - mean_acc,
                        "delta_f1": r.mean_f1 - mean_f1,
                        "save_dir": r.save_dir,
                        "eval_dir": r.eval_dir,
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



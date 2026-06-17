# PCLA (Model7) 三种子实验复现说明（mywork/model1）

本目录用于在 **不改动原工程代码与数据** 的前提下，复现论文中展示的 **PCLA（model7）best-3-seeds** 结果。

核心思路：

- 所有「真实数据 + 训练逻辑 + 采样逻辑 + 下游评估逻辑」仍然使用原仓库中的代码（`fame/myfame/baseline/model7` 与 `fame/myfame/evaluate`）；
- 只是在 `mywork/model1` 下提供一个额外的驱动脚本，把 **保存路径全部重定向到当前目录**，而不写回 `fame/myfame`；
- 这样可以方便你多次复现实验、对比结果，同时保证 `dataset/` 与 `fame/` 目录完全不被修改。

---

## 1. 代码入口：`run_pcla_best_3seeds.py`

路径：

- `ADAPCLA_ROOT/experiments/model1/run_pcla_best_3seeds.py`

功能概览：

- 逻辑上等价于原工程的 `fame/myfame/baseline/model7/repeat_best_3seeds.py`；
- 但是：
  - **训练 / 采样** 仍然调用原始的 `baseline/model7/bash.sh`、`train.py`、`test.py`；
  - **下游 25-label 评估** 仍然调用原始的 `evaluate/evaluate_synthetic_training.py`；
  - 唯一改变的是：`SAVE_DIR`、评估结果与 summary CSV/JSON 的输出目录改为 `mywork/model1` 下的 `save/`、`evaluate/`、`output/`。

---

## 2. 运行方式

在包含 `torchrun` 的 conda 环境中（例如你之前用于跑 baseline 的环境）：

```bash
cd EXPERIMENTS_ROOT/model1

# 默认复现 seeds = [1, 2, 3]
python run_pcla_best_3seeds.py

# 如需指定 seeds，可以显式传参，例如：
# python run_pcla_best_3seeds.py --seeds 1 2 3
```

脚本会自动：

1. 为每个 seed 设定与原论文中相同的「best 配置」：
   - `pos_loss_weight = 1.5`
   - `logit_adjust_tau = 0.2`
   - `logit_adjust_clip = 15.0`
   - `lr = 1e-4`
   - `epochs = 10`
   - `TOTAL_SAMPLES = 33494`
2. 设置对应的环境变量（`SEED`, `LR`, `EPOCHS`, `POS_LOSS_WEIGHT`, `LOGIT_ADJUST_TAU`, `LOGIT_ADJUST_CLIP`, `APPLY_LOGIT_ADJUST_IN_SAMPLING`, `RESUME=0`, `TOTAL_SAMPLES` 等）。
3. 调用原仓库中的：
   - `fame/myfame/baseline/model7/bash.sh all`  
     - 内部会先运行 `train.py`（DDP 训练 PCLA），再运行 `test.py`（用训练好的模型生成 `haloDataset.pkl`）。
   - 然后用 `fame/myfame/evaluate/evaluate_synthetic_training.py` 做下游 25-label 任务的训练 + 评估。

注意：通过在 `run_pcla_best_3seeds.py` 里设置环境变量：

```python
env["DATA_DIR"] = DATA_MIMICIII
env["INIT_CKPT_PATH"] = HALO_MIMICIII_CKPT
```

我们已经将 `bash.sh` 中的默认路径从 `FAME_ROOT 显式重定向到你当前仓库下的实际路径。  
因此，真实数据来源是 **MIMIC‑III 任务数据**（`fame/myfame/baseline/HALO/save`），模型初始化使用的是你本地的 `model2_halo_logit.pt`，与原工程的逻辑完全一致，但不会访问 `FAME_ROOT

---

## 3. 生成的结果放在哪？

运行完成后，在 `mywork/model1` 下会看到三个子目录：

1. **`save/`**：每个 seed 对应一套 PCLA 生成器 + 其生成的合成数据

   - 结构类似：
     - `save/best_pos1.5_tau0.2_clip15.0_lr0.0001_e10_seed1_YYYYMMDD_HHMMSS/`
       - `model7.pt`（或若干 checkpoint）
       - `datasets/haloDataset.pkl`（该 seed 下 PCLA 生成的合成 EHR 数据）
     - `save/best_pos1.5_tau0.2_clip15.0_lr0.0001_e10_seed2_.../`
     - `save/best_pos1.5_tau0.2_clip15.0_lr0.0001_e10_seed3_.../`

2. **`evaluate/`**：下游 25-label 评估结果

   - 每个 seed 对应一个子目录：
     - `evaluate/best_pos1.5_tau0.2_clip15.0_lr0.0001_e10_seed1_.../`
       - `compare_real_halo_mymodel2.csv`（与原工程相同格式）
       - 日志等文件

3. **`output/`**：总体 summary（仅在 `model1` 下）

   - `output/pcla_best_seed3_summary.csv`
   - `output/pcla_best_seed3_summary.json`

CSV 中会列出每个 seed 的：

- `mean_acc`：在 25 个标签上的平均 Accuracy（`compare_real_halo_mymodel2.csv` 按行平均）；
- `mean_f1`：在 25 个标签上的平均 F1；
- `delta_acc` / `delta_f1`：相对 overall mean 的偏差；
- `save_dir` / `eval_dir`：对应的模型与评估目录路径。

JSON 中则保存了相同的信息，方便后续脚本读取。

---

## 4. 与原工程对比（逻辑等价，但路径不同）

原工程的 PCLA 3-seed summary 由以下脚本产生：

- `fame/myfame/baseline/model7/repeat_best_3seeds.py`
  - 使用同样的 best 超参训练 model7（PCLA）3 次（不同 seeds）；
  - 在 `fame/myfame/baseline/model7/save/` 下保存模型与 `haloDataset.pkl`；
  - 在 `fame/myfame/evaluate/save/model7_.../` 下保存下游评估结果；
  - 在 `fame/myfame/output/model7_best_seed3_summary.{csv,json}` 写出 summary。

本目录下的 `run_pcla_best_3seeds.py`：

- **完全重用** 原脚本的训练 / 采样 / 评估逻辑（调用同一份 `bash.sh`、`train.py`、`test.py`、`evaluate_synthetic_training.py`）；
- **唯一的区别**：通过环境变量把 `SAVE_DIR` 等路径改到 `mywork/model1/save/`、`mywork/model1/evaluate/` 和 `mywork/model1/output/`；
- 因此，它在数学上 / 逻辑上等价于原工程的实验，只是把所有新生成的内容“沙箱化”到了 `mywork/model1` 中，方便你复现和对比，而不会污染原始结果。

---

## 5. 建议后续操作

- 如果你只需要 **重新生成 PCLA 的合成数据 + 下游结果**，可以直接跑本目录的脚本即可；
- 如果你之后希望：
  - 对不同超参组合做对比；
  - 只用已有的 ckpt 做「eval-only」实验（不重训 PCLA，只是换评估设置）；  
  可以在本目录再新建独立的 Python 脚本，调用：
  - `evaluate_synthetic_training.py` + `--mymodel2_path <某个 save_dir/datasets/haloDataset.pkl>`，
  - 把新的结果同样写回 `mywork/model1` 下的 `evaluate/` 与 `output/` 中。

此外，本目录还提供了 `run_pcla_learnable_bias.py` 与配套的 `README_PCLA_learnable_bias.md`，用于在 **MIMIC‑III 任务数据上训练带可学习输出偏置（learnable bias）的 PCLA 生成器**，并可选地通过 `--eval` 与原始固定 bias 的 PCLA 结果做一对一的下游 25-label 性能对比。




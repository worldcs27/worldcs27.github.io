# model3：固定 Bias + 退火实验报告

## 1. 实验目的

在 PCLA 生成器上验证 **Bias Annealing（偏置退火）** 方案：

- **训练阶段**：使用与原始 PCLA 相同的**固定** logit_adjust（基于训练集统计、只算一次），但在训练过程中按 **α(epoch)** 从 1.0 逐步退火到 0，使模型前期在 bias 引导下收敛，后期在无 bias 条件下微调。
- **采样/生成阶段**：**完全不加 bias**（`logit_adjust=None`），即最终用于生成合成数据的模型在推理时不再依赖任何先验向量。
- **目标**：在「生成时彻底去掉 bias」的前提下，看下游 25-label 任务上的表现能否接近甚至超过「采样时仍加 bias」的固定 PCLA。

---

## 2. 实验设计

### 2.1 模型与数据

- **模型**：原始 `HALOModel`（来自 `fame/myfame/baseline/model7`），无 learnable bias，与 PCLA 基线结构一致。
- **数据**：MIMIC-III 任务数据（`fame/myfame/baseline/HALO/save`），与 model1/model2 复现一致。
  - 训练集：`trainDataset.pkl`（33,494 条）
  - 验证集：`valDataset.pkl`
  - 词表：`codeToIndex.pkl`（6,984 codes）、`idToLabel.pkl`（25 CCS 标签）

### 2.2 固定 Bias（logit_adjust）计算

与 model7 完全一致，基于训练集 visit 级统计：

- 统计每个 code 在多少个 visit 中出现过，得到 visit 比例 \( \pi_i \)；
- \( b_i = \tau \cdot \log\frac{1-\pi_i+\epsilon}{\pi_i+\epsilon} \)，再截断到 \( [-\text{clip}, \text{clip}] \)；
- 仅对在训练集中出现过的 code 赋非零值，其余为 0；label 与特殊 token 维度为 0。

**超参数**（与 PCLA 一致）：

| 参数 | 值 | 说明 |
|------|-----|------|
| `logit_adjust_tau` | 0.2 | 缩放系数 |
| `logit_adjust_clip` | 15.0 | 截断范围 |
| `logit_adjust_eps` | 1e-8 | 数值稳定 |

### 2.3 退火系数 α 的设置

- **α(epoch)** 为分段线性，epoch 从 0 到 `total_epochs - 1`，代码中的计算方式为：

  ```text
  t = epoch / max(total_epochs - 1, 1)   # 归一化到 [0, 1]

  α(epoch) =
      1.0,                   若 t ≤ 0.3   （前 30% epoch）
      1.0 - (t - 0.3) / 0.4, 若 0.3 < t ≤ 0.7   （中间 40%，线性从 1.0 到 0.0）
      0.0,                   若 t > 0.7   （后 30% epoch）
  ```

- **分段含义**：
  - **前 30% epoch**：α = 1.0，与固定 PCLA 训练一致，充分利用 bias 稳定梯度与长尾。
  - **30%–70% epoch**：α 从 1.0 线性降到 0.0，逐步撤掉 bias。
  - **后 30% epoch**：α = 0.0，无 bias 微调，强迫主干在不依赖先验的条件下收敛。

- **本次运行（total_epochs = 10）时各 epoch 的 α 取值**：

  | epoch | t = epoch/9 | α(epoch) |
  |-------|-------------|----------|
  | 0     | 0.000       | 1.0      |
  | 1     | 0.111       | 1.0      |
  | 2     | 0.222       | 1.0      |
  | 3     | 0.333       | 0.917    |
  | 4     | 0.444       | 0.639    |
  | 5     | 0.556       | 0.361    |
  | 6     | 0.667       | 0.083    |
  | 7     | 0.778       | 0.0      |
  | 8     | 0.889       | 0.0      |
  | 9     | 1.000       | 0.0      |

- 训练时：`out_logits = logits + α(epoch) * logit_adjust`，每个 epoch 内 α 固定。
- 采样时：**始终** `logit_adjust=None`，即生成 `haloDataset.pkl` 时不加任何 bias。

### 2.4 训练与生成流程

1. 加载数据与词表，构造 `Model2Config`，计算一次固定 `logit_adjust`。
2. 从 HALO 预训练 checkpoint 初始化 `HALOModel`（单卡，无 DDP）。
3. 每个 epoch：  
   - 计算当前 α(epoch)；  
   - 训练/验证时传入 `logit_adjust = α * fixed_adj`（α=0 时传 `None`）。
4. 保存验证 loss 最优的 checkpoint 到 `save_anneal/seed1/model_anneal.pt`。
5. 使用该 checkpoint 生成合成数据：调用 `model.sample(..., logit_adjust=None)`，得到 `haloDataset.pkl`。
6. （可选）调用下游评估脚本，在合成数据上训练 25-label 分类器，得到 Accuracy / F1 等指标。

---

## 3. 超参数汇总

### 3.1 脚本默认（本次运行）

| 类别 | 参数 | 值 | 备注 |
|------|------|-----|------|
| 数据 | `--data_dir` | `fame/myfame/baseline/HALO/save` | MIMIC-III 任务目录 |
| 数据 | `--save_dir` | `./save_anneal/seed1` | 本实验输出目录 |
| 随机种子 | `--seed` | 1 | 单 seed 实验 |
| 训练 | `--epochs` | 10 | 总 epoch 数 |
| 训练 | `--batch_size` | 48 | 与 Model2Config 默认一致 |
| 训练 | `--sample_batch_size` | 256 | 生成时每批样本数 |
| 训练 | `--lr` | 1e-4 | 学习率（Model2Config 默认） |
| 训练 | `--pos_loss_weight` | 1.5 | BCE 正类权重 |
| 训练 | `--num_workers` | 4 | DataLoader 线程数 |
| 生成 | `--total_samples` | 33,494 | 与训练集等大 |
| 初始化 | `--init_ckpt_path` | `fame/.../model2/save/model2_halo_logit.pt` | HALO 预训练（DDP 保存，加载时已去掉 `module.` 前缀） |

### 3.2 模型与 logit_adjust 相关（来自 Model2Config）

| 参数 | 值 |
|------|-----|
| `n_ctx` | 48 |
| `n_embd` | 768 |
| `n_layer` | 12 |
| `n_head` | 12 |
| `logit_adjust_tau` | 0.2 |
| `logit_adjust_clip` | 15.0 |
| `logit_adjust_eps` | 1e-8 |

### 3.3 本次运行得到的 logit_adjust 统计

（脚本打印的 `Logit adjust stats`）

- `total_visits`: 42,533  
- `codes_with_pos`: 6,444  
- `tau`: 0.2, `clip`: 15.0  
- `adj_min`: 0.0, `adj_max`: ≈ 2.13  

---

## 4. 实验结果

### 4.1 下游 25-label 评估（本次运行：seed=1，带 `--eval`）

评估方式：在本次生成的 `haloDataset.pkl` 上训练下游分类器，在真实测试集上计算各 label 的 Accuracy、F1、AUROC、AUPRC，再对 25 个 label 取平均。

| 指标 | 本实验（Fixed bias + Anneal, 采样无 bias） | 固定 PCLA 3-seed（采样带 bias） |
|------|--------------------------------------------|----------------------------------|
| **mean_acc** | **0.9058** | 0.9021 ± 0.0022 |
| **mean_f1**  | **0.9069** | 0.9028 ± 0.0024 |

- 本实验为单 seed（seed=1），std 记为 0。  
- 固定 PCLA 3-seed 来自 `model1/output/pcla_best_seed3_summary.csv`。

### 4.2 结论摘要

- 在 **采样阶段完全不加 bias** 的前提下，固定 bias + 退火（model3）的 **mean_acc 与 mean_f1 略高于** 采样时仍加 bias 的固定 PCLA 3-seed 均值。
- 说明：通过退火，模型在训练后期已在不依赖 bias 的条件下收敛，生成质量足以支撑下游任务，且满足「最终模型在推理/生成环节不再使用 bias」的要求。

---

## 5. 输出文件结构（model3 目录下）

```
model3/
├── run_pcla_fixed_bias_anneal.py    # 主脚本
├── README_fixed_bias_anneal.md      # 简要使用说明
├── EXPERIMENT_fixed_bias_anneal.md  # 本实验报告
├── save_anneal/
│   └── seed1/
│       ├── model_anneal.pt          # 最佳 checkpoint（含 logit_adjust、anneal_schedule 等）
│       └── datasets/
│           └── haloDataset.pkl      # 合成数据（采样时无 bias）
├── evaluate_anneal/
│   └── seed1/
│       └── compare_real_halo_mymodel2.csv   # 下游 25-label 明细（Real vs MyModel2）
└── output/
    ├── pcla_fixed_bias_anneal_summary.csv   # 与固定 PCLA 的对比表
    └── pcla_fixed_bias_anneal_summary.json  # 同上，JSON 格式
```

---

## 6. 复现命令

```bash
cd EXPERIMENTS_ROOT/model3

python run_pcla_fixed_bias_anneal.py \
  --data_dir DATA_MIMICIII \
  --save_dir ./save_anneal/seed1 \
  --eval
```

如需更换初始化、epoch 数或学习率，可使用 `--init_ckpt_path`、`--epochs`、`--lr` 等参数。

---

## 7. 与 model1 / model2 的对比（概念）

| 版本 | 训练时 bias | 采样时 bias | 说明 |
|------|--------------|-------------|------|
| model1 固定 PCLA | 固定 logit_adjust | 使用 logit_adjust | 原始 PCLA 复现 |
| model1 learnable bias | 可学习 output_bias | 使用学到的 bias | 下游表现弱于固定 PCLA |
| model2 learnable bias (sep LR) | 可学习 bias，lr=0.01×其他 | 使用学到的 bias | 下游仍约 79% |
| **model3 固定 bias + 退火** | **固定 logit_adjust × α(epoch)，α→0** | **无（None）** | 下游 ~90.6%，满足「生成时无 bias」 |

---

*文档生成自 model3 单次完整运行（seed=1，带 `--eval`），结果对应 `output/pcla_fixed_bias_anneal_summary.csv` 与终端打印的 mean_acc / mean_f1。*

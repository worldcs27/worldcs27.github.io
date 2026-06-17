# model4：Prior-only 合成数据实验报告

## 1. 实验目的

在 **完全不使用 HALO/PCLA 等生成模型** 的前提下，仅依赖训练集统计得到的 **先验（bias + 简单结构统计）** 生成合成数据，并用与 Real / HALO / PCLA 相同的下游 25-label 评估流程得到 mean_acc / mean_f1，作为 **“无模型、仅先验”** 的基线。

目标：观察 **只依赖 bias 与结构先验** 时下游表现如何，与 PCLA 等模型结果对比，说明生成模型带来的增益。

---

## 2. 实验配置

### 2.1 数据与词表

- **数据目录**：与 PCLA 一致，MIMIC-III 任务数据  
  `fame/myfame/baseline/HALO/save`
- **加载文件**：
  - `codeToIndex.pkl` → code 词表（`code_vocab_size`）
  - `idToLabel.pkl` → 25 维 CCS 标签（`label_vocab_size`）
  - `trainDataset.pkl` → 训练集，用于统计先验与结构分布

### 2.2 先验计算（与 PCLA 一致）

- **Code 先验（logit_adjust）**  
  使用与 model7 完全相同的 `compute_logit_adjust(train_data, config)`：
  - 统计每个 code 在多少个 visit 中出现 → `visit_counts`，总 visit 数 `total_visits`
  - \( \pi_i = \text{visit\_counts}[i] / \text{total\_visits} \)，\( b_i = \tau \cdot \log\frac{1-\pi_i+\epsilon}{\pi_i+\epsilon} \)，再截断到 \( [-\text{clip}, \text{clip}] \)
  - 仅对训练集中出现过的 code 赋非零 \( b_i \)，其余为 0  
  - **Code 出现概率**：\( p_i = \sigma(b_i) = 1/(1+e^{-b_i}) \)，用于生成时 Bernoulli 采样

- **Label 先验**  
  从 `train_data` 中统计 25 维 CCS 标签的边际频率 → `label_probs`，生成时对 25 维独立 Bernoulli 采样。

- **结构先验**  
  从 `train_data` 中抽取：
  - `num_visits_list`：每个病人的 visit 数分布
  - `visit_len_list`：每个 visit 中 code 数分布  
  生成时每个病人的 visit 数、每个 visit 的 code 数从上述经验分布中随机抽样。

### 2.3 生成协议（无模型）

- 对每个“病人”：
  1. 从 `num_visits_list` 中随机取一个数作为该病人的 visit 数；
  2. 对每个 visit：从 `visit_len_list` 中取一个长度；对 `V_code` 个 code 按 `code_probs` 独立 Bernoulli 采样，得到候选 code 集合，截断到该长度并去重排序；
  3. 对 25 维 label 按 `label_probs` 独立 Bernoulli 采样。
- 输出格式与原有 `haloDataset.pkl` 一致：`List[{"visits": List[List[int]], "labels": np.ndarray (25,)}, ...]`。

### 2.4 脚本默认参数（本次运行）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--data_dir` | `fame/myfame/baseline/HALO/save` | MIMIC-III 任务数据目录 |
| `--save_dir` | `./save_prior_only/seed1` | 本实验输出目录 |
| `--seed` | 1 | 随机种子 |
| `--total_samples` | 33,494 | 合成病人数（与训练集等大） |
| `--eval` | 已开启 | 生成后自动跑下游评估并写 summary |

---

## 3. 实验结果

### 3.1 下游 25-label 评估（本次运行：seed=1，带 `--eval`）

评估方式：在 prior-only 生成的 `haloDataset.pkl` 上训练下游分类器，在真实测试集上计算各 label 的 Accuracy、F1 等，再对 25 个 label 取平均。

| 指标 | Prior-only (无模型，仅先验) | PCLA 固定 bias 3-seed |
|------|-----------------------------|------------------------|
| **mean_acc** | **0.5087** | 0.9021 ± 0.0022 |
| **mean_f1**  | **0.5610** | 0.9028 ± 0.0024 |

- Prior-only 为单 seed（seed=1），std 记为 0。  
- PCLA 固定 3-seed 来自 `model1/output/pcla_best_seed3_summary.csv`。

### 3.2 结论摘要

- **仅依赖先验 bias + 简单结构统计** 的合成数据，下游 mean_acc ≈ 0.51、mean_f1 ≈ 0.56，远低于 **PCLA 固定 bias 3-seed**（mean_acc ≈ 0.90、mean_f1 ≈ 0.90）。
- 说明：**生成模型（PCLA/HALO）学到了先验之外的可迁移结构**，单纯“先验 + 结构分布”不足以支撑与真实数据或 PCLA 合成数据相当的下游表现；prior-only 可作为论文中 **“无模型基线”** 的一行，用于对比模型带来的增益。

---

## 4. 输出文件结构（model4 目录下）

```
model4/
├── README_prior_only_protocol.md   # 方法草案与伪代码
├── EXPERIMENT_prior_only.md        # 本实验报告（配置 + 结果）
├── run_prior_only_bias.py          # 主脚本
├── save_prior_only/
│   └── seed1/
│       └── datasets/
│           └── haloDataset.pkl     # prior-only 合成数据
├── evaluate_prior_only/
│   └── seed1/
│       └── compare_real_halo_mymodel2.csv   # 下游 25-label 明细
└── output/
    ├── prior_only_vs_pcla_summary.csv   # 与 PCLA 固定 3-seed 对比表
    └── prior_only_vs_pcla_summary.json  # 同上，JSON 格式
```

---

## 5. 复现命令

```bash
cd EXPERIMENTS_ROOT/model4

python run_prior_only_bias.py \
  --data_dir DATA_MIMICIII \
  --save_dir ./save_prior_only/seed1 \
  --eval
```

---

*本报告对应本次运行结果：Prior-only seed=1，mean_acc=0.508720，mean_f1=0.561025；对比数据来自 model1/output/pcla_best_seed3_summary.csv。*

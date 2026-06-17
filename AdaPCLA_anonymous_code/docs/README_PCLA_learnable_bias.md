# PCLA (Model7) 带可学习输出偏置的单种子实验说明

本文档说明 `run_pcla_learnable_bias.py` 的用途、依赖和运行方式，并解释它与原始 PCLA（固定 logit\_adjust）的关系，方便你在论文中描述实验设置与对比结果。

## 1. 背景与目标

原始 PCLA（`fame/myfame/baseline/model7`）基于 HALO 生成器，在 **MIMIC‑III 任务数据**上进行训练与采样：

- 输入：病人的就诊序列（多次 visit，每次 visit 是一组 ICD‑9-CM 诊断代码的 one-hot 向量）；
- 输出：下一个 visit 的多标签（基于代码空间的自回归多标签预测）；
- 同时使用外部预先计算好的 `logit_adjust` 向量（`logit_adjust.npy` / checkpoint 内的 `logit_adjust`）作为对输出 logits 的偏移，用于纠正长尾分布中的类别不平衡。


本次实验的目标是：

- 在 **不修改原始工程代码** 的前提下，在 `mywork/model1` 目录中定义一个新的驱动脚本 `run_pcla_learnable_bias.py`；
- 将 PCLA 生成器中的 `logit_adjust` 从「外部提供的固定 buffer」改为「模型内部的可学习向量参数」；
- 在 MIMIC‑III 任务数据上训练 **单个 seed** 的「PCLA + Learnable Bias」生成器，并可选地用 `--eval` 与原始固定 bias 的 PCLA 结果做下游 25-label 性能对比。

---

## 2. 代码结构与依赖

脚本位置：

- `ADAPCLA_ROOT/experiments/model1/run_pcla_learnable_bias.py`

依赖的原始工程组件（通过 `import` 引入，不改动原文件）：

- `fame/myfame/baseline/model7/config.py` 中的 `Model2Config`
- `fame/myfame/baseline/model7/model.py` 中的 `HALOModel`（作为基类）
- 任务数据：`fame/myfame/baseline/HALO/save` 下的
  - `codeToIndex.pkl`, `idToLabel.pkl`
  - `trainDataset.pkl`, `valDataset.pkl`
- 下游评估（仅在加 `--eval` 时使用）：
  - `fame/myfame/evaluate/evaluate_synthetic_training.py`

本目录下还会生成/使用以下子目录：

- `save_learnable_bias/`：保存带 learnable bias 的 PCLA 生成器及其生成的合成数据；
- `evaluate_learnable_bias/`：在 `--eval` 时保存下游 25-label 评估结果；
- `output/`：保存固定 bias PCLA 与 learnable bias PCLA 的对比 summary。

---

## 3. 模型改动：Learnable Output Bias

脚本中定义了新类：

```python
class HALOModelWithLearnableBias(_HALOModelBase):
    ...
```

其中 `_HALOModelBase` 即原始的 `HALOModel`。关键改动如下：

### 3.1 初始化：从统计先验构造可学习 bias

1. 首先在脚本中重写了与原 `train.py` 一致的 `compute_logit_adjust(train_data, config)`：
   - 基于 `trainDataset.pkl` 统计每个 code 在多少个 visit 中出现；
   - 计算 per-code 先验 `b_stat = log((1-π)/(π)) * tau`，并在 `[-clip, clip]` 区间截断；
   - 将它扩展到长度为 `total_vocab_size` 的向量 `adj_np`。

2. 然后在模型初始化时，将该向量作为可学习参数的初始值：

```python
adj_np, stats = compute_logit_adjust(train_data, config=cfg)
adj_init = torch.from_numpy(adj_np)
model = HALOModelWithLearnableBias(cfg, adj_init).to(device)
```

在 `__init__` 中：

```python
self.output_bias = nn.Parameter(init_bias.detach().clone().float())
```

### 3.2 前向传播：内部使用可学习 bias，而非外部 logit_adjust

在新模型的 `forward` 中，不再接收 `logit_adjust` 参数，而是直接使用内部参数：

```python
hidden_states = self.transformer(input_visits, position_ids, past)
logits = self.ehr_head(hidden_states, input_visits)

bias = self.output_bias.view(1, 1, -1).to(device=logits.device, dtype=logits.dtype)
out_logits = logits + bias
probs = torch.sigmoid(out_logits)
...
loss_logits = out_logits  # 已包含 bias
loss_elem = F.binary_cross_entropy_with_logits(
    loss_logits,
    shift_labels.to(dtype=loss_logits.dtype),
    pos_weight=pos_weight,
    reduction="none",
)
```

这样：

- `output_bias` 会像其他参数一样在训练中被更新；
- 不再依赖外部传入的 logit_adjust 向量；
- 几何上的「偏置纠偏」完全内化为可学习参数，有利于在不同数据设置下进行更灵活的调整。

---

## 4. 训练与生成流程（无 DDP、单 seed）

### 4.1 命令行参数

脚本入口 `parse_cli()` 中支持的主要参数：

- `--data_dir`：任务数据目录，默认  
  `DATA_MIMICIII`
- `--save_dir`：结果保存目录，默认  
  `./save_learnable_bias`
- `--seed`：随机种子，默认 `1`
- `--epochs`：训练 epoch 数，默认 `10`
- `--batch_size`：训练 batch size，默认使用 `Model2Config` 中的 `batch_size`（48）
- `--sample_batch_size`：采样时的 batch size，默认 `Model2Config.sample_batch_size`（256）
- `--num_workers`：DataLoader 线程数，默认 `4`
- `--lr`：学习率，默认从 `Model2Config.lr`（1e-4）
- `--pos_loss_weight`：BCE 中正类 reweight 系数，默认 `1.5`
- `--total_samples`：生成的合成病人数，默认 `33494`（与 MIMIC‑III 训练集等大）
- `--eval`：若加上此 flag，则在训练+生成后自动进行下游 25-label 评估并写对比表。

### 4.2 训练阶段

对应函数：`run_single_seed(args)` 的前半部分。

1. 加载 `codeToIndex.pkl`、`idToLabel.pkl`、`train/valDataset.pkl`；
2. 构造 `Model2Config`，设定 hyper-parameters 与 vocab 大小；
3. 用 `compute_logit_adjust` 计算 `adj_np`，作为 learnable bias 初值；
4. 构造 `HALOModelWithLearnableBias` + Adam 优化器；
5. 使用 `MIMICDataset` + `DataLoader` 按 epoch 训练：
   - 损失函数与原 PCLA 一致：`binary_cross_entropy_with_logits`；
   - 每个 batch 调用新模型的 `forward`，内部使用 `self.output_bias`；
   - 按验证集 loss 选 best checkpoint，保存到 `<save_dir>/model_lb.pt`。

### 4.3 生成阶段

训练结束后，脚本会：

1. 构造起始 token `stoken`（one-hot，`start_record_token` 位置为 1）；
2. 使用 `sample_sequence` 在单卡上生成 `total_samples` 条样本：
   - 每条样本的 max 长度为 `n_ctx`（默认 48）；
   - 每步调用 `model.sample(...)` 生成下一个 visit；
   - 若某条样本已经生成到 `end_record_token`，则提前停止该样本；
3. 用 `convert_ehr` 将 one-hot 序列还原为：

   ```python
   {"visits": [[code_ids...], [code_ids...], ...], "labels": label_vector}
   ```

4. 将所有样本写入：

   ```text
   <save_dir>/datasets/haloDataset.pkl
   ```

该 `haloDataset.pkl` 与原 PCLA、HALO 生成的数据结构兼容，可直接喂给 `evaluate_synthetic_training.py` 做下游评估。

---

## 5. `--eval`：与固定 bias PCLA 的下游性能对比

加上 `--eval` 参数后，脚本会在训练+生成之后自动做两件事：

### 5.1 调用原始下游评估脚本

```bash
python EVAL_PY \
  --base_data_dir <data_dir> \
  --mymodel2_path <save_dir>/datasets/haloDataset.pkl \
  --save_dir mywork/model1/evaluate_learnable_bias/seed<seed> \
  --sources MyModel2
```

- 只把本次 learnable-bias PCLA 当作 `MyModel2` 源；
- 使用 MIMIC‑III HALO/save 中的真实数据作为 Real 基线；
- 生成 `compare_real_halo_mymodel2.csv`，我们再从中抽取 `source=MyModel2` 的 25-label 均值 `(mean_acc, mean_f1)`。

### 5.2 与固定 bias PCLA 3-seed summary 对比

- 读取之前由 `run_pcla_best_3seeds.py` 生成的：

  ```text
  @mywork/model1/output/pcla_best_seed3_summary.csv
  ```

  并解析其中 `mean±std` 行，得到固定 bias PCLA 在 MIMIC‑III 上的：
  - `mean_acc` ± `std_acc`
  - `mean_f1` ± `std_f1`

- 与本次 learnable-bias PCLA 的单次结果对比，写入：

  ```text
  @mywork/model1/output/pcla_fixed_vs_learnable_bias_summary.csv
  @mywork/model1/output/pcla_fixed_vs_learnable_bias_summary.json
  ```

CSV 大致包括两行：

```csv
variant,mean_acc,std_acc,mean_f1,std_f1,note
PCLA_fixed_3seeds,0.901853,0.002840,0.902251,0.002746,"Original 3-seed fixed-bias PCLA summary (from pcla_best_seed3_summary.csv)"
PCLA_learnable_bias_seed1,0.90xxx,0.000000,0.90yyy,0.000000,"This run: single-seed PCLA with learnable output bias"
```

这样，你可以在论文中非常直观地对比：

- 原始 PCLA（固定 logit\_adjust，3 个 seed 平均）的下游性能；
- 新的 **PCLA + Learnable Output Bias**（单个 seed）在同一 MIMIC‑III 25-label 任务上的表现。

---

## 6. 典型运行示例

```bash
cd EXPERIMENTS_ROOT/model1

python run_pcla_learnable_bias.py \
  --data_dir DATA_MIMICIII \
  --save_dir ./save_learnable_bias/seed1 \
  --seed 1 \
  --epochs 10 \
  --batch_size 48 \
  --sample_batch_size 256 \
  --total_samples 33494 \
  --eval
```

运行结束后，你可以查看：

- `save_learnable_bias/seed1/model_lb.pt`：learnable-bias PCLA 生成器；
- `save_learnable_bias/seed1/datasets/haloDataset.pkl`：对应的合成 EHR；
- `evaluate_learnable_bias/seed1/compare_real_halo_mymodel2.csv`：下游评估明细；
- `output/pcla_fixed_vs_learnable_bias_summary.{csv,json}`：固定 bias vs learnable bias 的总体对比结果。



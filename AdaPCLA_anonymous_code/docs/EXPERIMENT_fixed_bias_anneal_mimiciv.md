# model5：MIMIC-IV data2 + 固定 Bias + 退火实验说明

## 1. 实验目的

在 **MIMIC-IV data2** 上复现与 model3 相同的 **固定 Bias + 退火** 方案，实现「同一方法、不同数据集」的对比：

- **数据**：MIMIC-IV 预处理数据（`fame/myfame/data2`），与 model8 基线一致。
- **初始化**：使用 **在 MIMIC-IV 上预训练好的 HALO checkpoint**（model8），而非 MIMIC-III 的 HALO，保证公平比较。
- **训练**：`out_logits = logits + α(epoch) * logit_adjust`，α 从 1.0 分段线性降到 0。
- **采样**：不加 bias（`logit_adjust=None`），即生成时完全不依赖先验向量。

---

## 2. 路径与配置

| 项目 | 路径/值 |
|------|---------|
| MIMIC-IV data2 | `DATA_MIMICIV` |
| 数据内容 | `trainDataset.pkl`, `valDataset.pkl`, `codeToIndex.pkl`, `idToLabel.pkl` |
| MIMIC-IV HALO checkpoint | `fame/myfame/baseline/model8/save_mimiciv_seed1_best/best_ckpt_lr0.0002_e10_seed1_20260115_021329/model8.pt` |
| 模型代码 | `fame/myfame/baseline/model8`（config.Model2Config, model.HALOModel） |
| 评估脚本 | `fame/myfame/evaluate/evaluate_synthetic_training.py`（`--base_data_dir` 指向 data2） |

---

## 3. 实验设计（与 model3 对齐）

- **固定 bias**：基于 MIMIC-IV 训练集 visit 统计计算 `logit_adjust`（tau=0.2, clip=15.0, eps=1e-8），只算一次。
- **退火**：α(epoch) 分段线性（前 30% α=1.0，30%–70% 线性降到 0，后 30% α=0）。
- **训练**：10 epoch，lr 等与 model8 一致；每个 epoch 用 `α(epoch) * logit_adjust` 参与 loss。
- **生成**：50k 条合成数据（与 model8 一致），采样时 `logit_adjust=None`。
- **评估**：下游 25-label 分类（`evaluate_synthetic_training.py`），`--base_data_dir` 为 data2。

---

## 4. 运行方式

```bash
cd EXPERIMENTS_ROOT/model5

# 仅训练 + 生成（默认 data_dir=data2, init_ckpt=model8 MIMIC-IV checkpoint）
python run_pcla_fixed_bias_anneal_mimiciv.py --save_dir ./save_anneal_mimiciv/seed1 --epochs 10

# 训练 + 生成 + 下游评估并写 summary
python run_pcla_fixed_bias_anneal_mimiciv.py \
  --data_dir DATA_MIMICIV \
  --save_dir ./save_anneal_mimiciv/seed1 \
  --epochs 10 \
  --eval
```

输出：

-  checkpoint：`save_anneal_mimiciv/seed1/model_anneal_mimiciv.pt`
- 合成数据：`save_anneal_mimiciv/seed1/datasets/haloDataset.pkl`
- 评估结果：`evaluate_anneal_mimiciv/seed1/compare_real_halo_mymodel2.csv`
- Summary：`output/pcla_fixed_bias_anneal_mimiciv_summary.csv` 与 `.json`

---

## 5. 与 model3 的对比

| 项目 | model3 | model5 |
|------|--------|--------|
| 数据集 | MIMIC-III（HALO/save） | MIMIC-IV（data2） |
| HALO 初始化 | model7（MIMIC-III 训好的 HALO） | model8（MIMIC-IV 训好的 HALO） |
| 方法 | 固定 bias + 退火，采样无 bias | 同左 |
| 合成数量 | 33,494 | 50,000 |
| 下游评估 | 25-label，HALO/save 的 test | 25-label，data2 的 test |

结果可并排写入论文：MIMIC-III 上「固定 bias + 退火」vs MIMIC-IV 上「固定 bias + 退火」，体现方法在另一数据集上的可迁移性。

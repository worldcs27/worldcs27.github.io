# model5：MIMIC-IV 固定 Bias + 退火 — 配置与结果记录（seed=1）

本文档记录本次 model5 实验的**运行配置**与**下游 25-label 评估结果**，并包含 **Real**（真实数据训练）作为对比。

---

## 1. 运行配置

| 配置项 | 值 |
|--------|-----|
| **数据目录** | `DATA_MIMICIV`（MIMIC-IV data2） |
| **保存目录** | `./save_anneal_mimiciv/seed1` |
| **随机种子** | 1 |
| **训练轮数** | 10 |
| **学习率** | 默认（Model2Config，与 model8 一致，如 1e-4） |
| **batch_size** | 默认（如 48） |
| **sample_batch_size** | 默认（如 256） |
| **pos_loss_weight** | 1.5 |
| **合成样本数** | 50,000 |
| **初始化 checkpoint** | `fame/myfame/baseline/model8/save_mimiciv_seed1_best/best_ckpt_lr0.0002_e10_seed1_20260115_021329/model8.pt`（MIMIC-IV 训好的 HALO） |
| **退火** | 分段线性：前 30% epoch α=1.0，30%–70% 线性降至 0，后 30% α=0 |
| **采样** | 无 bias（`logit_adjust=None`） |
| **logit_adjust** | tau=0.2, clip=15.0, eps=1e-8，基于 MIMIC-IV 训练集统计，只算一次 |

---

## 2. 下游评估设置

- **任务**：25-label CCS 二分类（与 model8 / HALO 评估一致）
- **评估脚本**：`fame/myfame/evaluate/evaluate_synthetic_training.py`
- **base_data_dir**：MIMIC-IV data2（同上）
- **评估结果目录**：`evaluate_anneal_mimiciv/seed1/`
- **详细 CSV**：`evaluate_anneal_mimiciv/seed1/compare_real_halo_mymodel2.csv`（每 label 的 Real / MyModel2 的 Accuracy、F1、AUROC、AUPRC）

---

## 3. 结果汇总（Real vs PCLA 固定 Bias + 退火）

下游分类器在 **Real**（真实 MIMIC-IV 数据训练）与 **MyModel2**（本次 model5 生成的 50k 合成数据训练）上的 **25-label 平均** 指标如下。

| 数据源 | mean Accuracy | mean F1 |
|--------|----------------|---------|
| **Real**（真实数据） | **0.9555** | **0.9564** |
| **MyModel2**（PCLA 固定 bias + 退火，seed=1） | **0.9158** | **0.9130** |

- **Real**：在 MIMIC-IV 真实数据上训练下游分类器，作为上界参考。
- **MyModel2**：在 model5 生成的合成数据上训练下游分类器；训练时使用固定 bias + 退火，**采样时无 bias**。

合成数据（MyModel2）相对 Real 的差距：mean Acc 约 −4.0 个百分点，mean F1 约 −4.3 个百分点，说明合成数据质量较高、与真实分布较接近。

---

## 4. 与 MIMIC-III 上 model3 的对比

| 项目 | model3（MIMIC-III） | model5（MIMIC-IV，本次） |
|------|---------------------|--------------------------|
| mean_acc（合成数据） | 0.9058 | **0.9158** |
| mean_f1（合成数据） | 0.9069 | **0.9130** |

MIMIC-IV 上本次 run 的 acc/F1 略高于 MIMIC-III 上的 model3，说明「固定 bias + 退火、采样无 bias」在 MIMIC-IV 上同样有效且略优。

---

## 5. 输出文件一览

| 文件 | 说明 |
|------|------|
| `save_anneal_mimiciv/seed1/model_anneal_mimiciv.pt` | 最佳验证 loss 对应的 checkpoint |
| `save_anneal_mimiciv/seed1/datasets/haloDataset.pkl` | 50k 条合成数据 |
| `evaluate_anneal_mimiciv/seed1/compare_real_halo_mymodel2.csv` | 每 label 的 Real / MyModel2 的 Accuracy、F1、AUROC、AUPRC |
| `output/pcla_fixed_bias_anneal_mimiciv_summary.csv` | 本次 run 的简要 summary |
| `output/pcla_fixed_bias_anneal_mimiciv_summary.json` | 同上，JSON 格式 |

---

## 6. 每 label 详细结果

25 个 label 的 Real / MyModel2 的 Accuracy、F1 Score、AUROC、AUPRC 见：

**`evaluate_anneal_mimiciv/seed1/compare_real_halo_mymodel2.csv`**

其中 Real 各 label Accuracy 多在 93%–98%，MyModel2 多在 86%–96%；少数类别（如 Fluid and electrolyte disorders、Other upper respiratory disease）MyModel2 略低，整体与上表均值一致。

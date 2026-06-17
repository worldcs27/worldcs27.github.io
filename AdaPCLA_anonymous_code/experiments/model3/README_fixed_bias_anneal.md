# model3：固定 bias + 退火 (Fixed Bias + Annealing)

## 目的

- **训练**：使用固定 logit_adjust（与原始 PCLA 相同，基于训练集统计算一次），但在训练过程中按 **α(epoch)** 缩放：前期 α=1.0，中后期线性退火到 α=0。
- **采样/生成**：**不加 bias**（α=0），即 `logit_adjust=None`，满足“生成时完全不依赖 bias”的要求。

## 退火 schedule

- 前 30% epoch：α = 1.0（与固定 PCLA 一致）
- 30%–70% epoch：α 从 1.0 线性降到 0.0
- 后 30% epoch：α = 0.0（无 bias 微调）

## 运行方式

```bash
cd EXPERIMENTS_ROOT/model3

# 仅训练 + 生成
python run_pcla_fixed_bias_anneal.py \
  --data_dir DATA_MIMICIII \
  --save_dir ./save_anneal/seed1

# 训练 + 生成 + 下游 25-label 评估并写 summary
python run_pcla_fixed_bias_anneal.py \
  --data_dir DATA_MIMICIII \
  --save_dir ./save_anneal/seed1 \
  --eval
```

## 输出

- `save_anneal/seed1/model_anneal.pt`：最佳 checkpoint（含 `logit_adjust` 与 `anneal_schedule` 信息）
- `save_anneal/seed1/datasets/haloDataset.pkl`：合成数据（**采样时未加 bias**）
- 若加 `--eval`：
  - `evaluate_anneal/seed1/compare_real_halo_mymodel2.csv`：下游评估明细
  - `output/pcla_fixed_bias_anneal_summary.csv`、`output/pcla_fixed_bias_anneal_summary.json`：与固定 bias 3-seed 的对比（mean_acc、mean_f1 等）

## 依赖

- 不修改 `dataset/` 与 `fame/` 下任何代码；复用 model7 的 `HALOModel` 与 `Model2Config`。
- 初始化 checkpoint 默认：`fame/myfame/baseline/model2/save/model2_halo_logit.pt`（与 model1 复现一致），可通过 `--init_ckpt_path` 覆盖。

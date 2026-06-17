# 2026-03-03 微观探测 & eNTK 实验记录（model6）

## 1. 代码改动概览

- **微观探测 logging（`model5/run_pcla_fixed_bias_anneal_mimiciv.py`）**
  - `_log_micro_probe`：
    - 改为只使用 **最后一个有效 time-step** 的概率：
      - 从后往前扫描 EHR，每个 time-step 以「是否有真实 code 或 `end_record_token` 激活」判定有效。
      - 找到最后一个有效步 `last_valid_idx` 后，仅取该步对应 `disease_id` 的 `prob / log_prob`。
      - 若找不到有效步，则 fallback 使用最后一行。
    - 新增参数 `logit_adjust`：
      - 允许在 micro-probe 前向时显式传入 bias（用于 Oracle 计算）。
    - `global_ckpt_idx < 0` 时写入固定文件 `micro_probe_oracle.csv`，否则写入 `micro_probe_ckpt_XXXX.csv`。
  - 在训练开始后、进入 epoch 循环之前：
    - 使用 **初始权重 θ₀ + 固定 bias `fixed_adj`** 对 micro-probe 集做一次前向：
      - 调用 `_log_micro_probe(..., global_ckpt_idx=-1, logit_adjust=fixed_adj)`。
      - 输出：`model6/micro_probe_logs/seed1/micro_probe_oracle.csv`（Oracle 水平线）。
  - 保留原有每 0.1 epoch 记录 1 次（`micro_probe_ckpt_per_epoch=10`）的逻辑，现改为记录最后一步概率。

- **eNTK 相似度写回配置（`model6/gen_micro_probe_config.py`）**
  - 新增选项 `--with_entk`：
    - 加载 data2 + 初始 HALO 模型（与训练相同的 MIMIC-IV model8 checkpoint）。
    - 对每一行 `(context_id, disease_id, type)`：
      - 计算 disease 的梯度向量 `g_d`（对 `f_y(x; θ₀)` 的参数梯度）。
      - 计算上下文 code 簇的平均梯度向量 `g_ctx`（对所有在该病例中出现过的 code 的梯度取平均）。
      - 计算余弦相似度 `initial_entk_sim = cos(g_d, g_ctx)`。
    - 将 `initial_entk_sim` 写入 CSV 新列。
  - 生成的新配置示例：
    - `micro_probe_configs/mimiciv_long_tail_triplets_seed1_with_entk.csv`
    - 共 90 行（30 个 context × 3 种 type），列为：`context_id, disease_id, type, initial_entk_sim`。

- **设备选择（避免占满 GPU 0）**
  - `run_pcla_fixed_bias_anneal_mimiciv.py` 新增 `--device` 参数：
    - 若未指定且有 ≥2 块 GPU：默认使用 `cuda:1`。
    - 否则默认 `cuda:0` 或 `cpu`。
    - 全文 `device = torch.device(args.device)`，不再自动抢 GPU 0。
  - `gen_micro_probe_config.py` 在 `--with_entk` 时优先使用 `cuda:1`（如存在）。
  - `analyze_entk_vs_dlogp.py`、`compute_entk_trajectory.py` 的 `--device` 默认也改为优先 `cuda:1`。

- **100 个模型 checkpoint（eNTK 轨迹用）**
  - 训练脚本在每个 micro-probe 时刻（每 0.1 epoch）保存模型参数：
    - 路径：`save_micro_probe_mimiciv/seed1/epoch_ckpts/micro_probe_ckpt_0000.pt` … `micro_probe_ckpt_0099.pt`。
  - 用于后续 100 点 eNTK 轨迹计算。

## 2. 本次重新训练（2026-03-03 晚）结果

- 命令（在 tmux 中执行）：

```bash
cd EXPERIMENTS_ROOT/model6
bash run_micro_probing_fixed_bias_anneal_mimiciv.sh
```

- 关键输出：
  - 主模型 checkpoint：
    - `save_micro_probe_mimiciv/seed1/model_anneal_mimiciv.pt`（时间：2026-03-03 20:56）。
  - 合成数据：
    - 已存在于 `save_micro_probe_mimiciv/seed1/datasets/haloDataset.pkl`（旧 run 保留）。
  - 微观探测轨迹（last-step 概率）：
    - `micro_probe_logs/seed1/micro_probe_ckpt_0000.csv` … `micro_probe_ckpt_0099.csv`
    - 每个文件 90 行，对应 30×3 个 `(context_id, disease_id, type)`。
  - Oracle 水平线：
    - `micro_probe_logs/seed1/micro_probe_oracle.csv`
    - 与 90 行探针一一对应，使用 θ₀ + `fixed_adj` 的 last-step log-prob。
  - 100 个 eNTK checkpoint：
    - `save_micro_probe_mimiciv/seed1/epoch_ckpts/micro_probe_ckpt_0000.pt` … `micro_probe_ckpt_0099.pt`
    - 与 100 个 micro-probe CSV 在 `global_ckpt_idx` 上对齐。

## 3. 分析与绘图脚本（今日完成）

- **单 context 概率轨迹 + Oracle 水平线**
  - `plot_micro_probe_single_context.py`：
    - 输入：`--context_id`（如 552）、`--logs_dir micro_probe_logs/seed1`。
    - 输出：`fig_micro_probe_ctx552.png`（三条轨迹，last-step log-prob）。
  - 新增临时 one-off 脚本绘图（inline Python）：
    - `fig_micro_probe_ctx552_with_oracle.png`：
      - 同一张图上画出：
        - related_rare / unrelated_rare / wrong 三条 last-step log-prob 轨迹；
        - 各自对应的 Oracle 水平虚线（来自 `micro_probe_oracle.csv`）。

- **eNTK 相似度 vs Δlog p**
  - `analyze_entk_vs_dlogp.py`：
    - 对 100 个 checkpoint 计算 Δlog p（末减首）。
    - 在 θ₀ 上计算每个 `(context_id, disease_id, type)` 的 `entk_sim_ctx_bundle`。
    - 输出：`entk_vs_dlogp.csv`。
  - `plot_entk_vs_dlogp.py`：
    - 输出：`fig_entk_vs_dlogp.png`（X: eNTK sim, Y: Δlog p，按 type 上色）。

- **100 点 eNTK 轨迹（单 context, pair）**
  - `compute_entk_trajectory.py`：
    - 自动从 `epoch_ckpts/micro_probe_ckpt_0000..0099.pt` 读入 100 个模型；
    - 对固定 `(context_id, disease1_id, disease2_id)` 计算：
      - \(K_t = g_{y_1,t}^T g_{y_2,t}\) 和 \(\cos(g_{y_1,t}, g_{y_2,t})\)；
    - 输出：
      - `entk_trajectory.csv`
      - `fig_entk_trajectory.png`（cos_sim vs t 的 100 点曲线）。

- **last-step vs mean-over-time 对比**
  - 临时绘图脚本（inline）：
    - 对 context 552 的 `related_rare`：
      - 用 100 个 checkpoint 的模型，重新对同一 EHR 做前向，分别计算：
        - 全 time-step 平均 log-prob；
        - 最后有效步的 log-prob。
      - 输出：`fig_last_vs_mean_ctx552_related.png`
      - 用于展示「最后一步」与「全序列平均」的定性差异。

- **Micro-probe 上的 bACC / GM 曲线**
  - 利用 90 个探针样本（30 related_rare vs 60 (unrelated_rare + wrong)）：
    - 正类：`type == related_rare`；
    - 负类：`type == unrelated_rare` 或 `wrong`；
    - 分数：micro-probe last-step `mean_prob`；
    - 固定阈值：0.5，将分数二值化，计算每个 checkpoint 的：
      - TPR, TNR, Balanced Accuracy = 0.5 (TPR + TNR)，Geometric Mean = sqrt(TPR·TNR)。
  - 输出：
    - `micro_probe_bacc_gm_over_time.csv`
    - `fig_micro_probe_bacc_gm.png`（bACC / GM 随 checkpoint 变化的曲线）。

## 4. 明日可继续的工作方向（建议）

1. **论文写作：KDD main 4.x 小节**
   - 用以下图支撑“从参数 ODE 到概率轨迹”的叙事：
     - `fig_micro_probe_ctx552_with_oracle.png`
     - `fig_entk_vs_dlogp.png`
     - `fig_entk_trajectory.png`
     - `fig_micro_probe_bacc_gm.png`
2. **补充说明文字（英文草稿）**
   - 解释：
     - Oracle 水平线如何构造（θ₀ + `fixed_adj`）；
     - 为什么只看最后一步概率；
     - eNTK sim（初始梯度桥梁）如何预测 Δlog p；
     - 微观探测 bACC / GM 曲线说明模型只“提拉”结构上相关的罕见并发症。
3. **如需扩展**：
   - 考虑 Softmax 对照组实验（可选，工作量大，视篇幅与时间而定）。


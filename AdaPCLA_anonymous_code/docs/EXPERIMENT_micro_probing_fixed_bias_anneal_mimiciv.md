# model5：MIMIC-IV data2 + 固定 Bias 退火的微观探测实验设计

> 说明：本文件只定义 **微观探测集（micro-probing dataset）+ 概率轨迹记录** 的实验设计，用于支撑 KDD 主文第 4 章中“从参数 ODE 到概率轨迹”的分析。不包含具体实现代码。

---

## 1. 目标与直觉

在已有的 **fixed Bias + 退火** 实验（见 `EXPERIMENT_fixed_bias_anneal_mimiciv.md`）基础上，我们增加一组专门服务于“学习动力学分析”的小规模实验：

- 构造一个规模很小、人工精挑的 **微观探测集（micro-probing dataset）**，包含若干典型 MIMIC-IV 病例上下文，以及每个上下文下三类候选疾病：
  - 真实罕见并发症（related\\_rare）
  - 无关罕见病（unrelated\\_rare）
  - 明显互斥/错误的疾病（wrong）
- 在 **偏置退火的训练过程中高频记录** 这些候选疾病的预测概率 / log-prob 轨迹：
  - 直观展示：偏置 + 退火不会“无脑拉高中所有尾部病”，而是**只抬高那些通过 K 矩阵桥梁连接的、有真实医学关联的罕见并发症**；
  - 通过引入 Oracle 基线（PCLA+LA）和 eNTK 相似度，把“参数层面的等效性定理”和“概率层面的动态轨迹”打通。

---

## 2. 微观探测集构造

### 2.1 上下文（contexts）的选取

- 数据域：MIMIC-IV data2，与 `EXPERIMENT_fixed_bias_anneal_mimiciv.md` 保持一致。
- 从训练集手工/规则筛选 **N_ctx ≈ 30–50** 个典型住院病例上下文 \(x^{(i)}\)：
  - 包含 1–3 个高频基础疾病（如：2 型糖尿病、高血压、心衰、CKD 等）；
  - 病例长度和信息量适中，避免极短或信息极少的样本；
  - 尽量覆盖若干不同的“基础病组合”场景。

记这些上下文为：
\[
\mathcal{X}_\\text{probe} = \{x^{(1)}, \dots, x^{(N_\\text{ctx})}\}.
\]

### 2.2 每个上下文的三类候选疾病

对每个上下文 \(x^{(i)}\)，在疾病空间（ICD/CCS code space）中，基于 K 矩阵与医学知识，指定三个候选疾病标签：

1. **相关罕见并发症（related\\_rare）** 记为 \(y^{(i)}_{\\text{rel}}\)
   - 与上下文中主体疾病在 K 矩阵中具有**较高相似度**（例如 top-k 相似度之一），
   - 在总体数据中为**长尾疾病**（出现频次较低），
   - 医学上确实属于当前基础病的合理并发症（如 T2DM → 特定视网膜病变/肾病）。

2. **无关罕见病（unrelated\\_rare）** 记为 \(y^{(i)}_{\\text{unrel}}\)
   - 同样是总体频率较低的尾部疾病；
   - 但与上下文主体疾病在 K 中的相似度**接近 0 或显著偏低**；
   - 临床逻辑上与该上下文关系较弱或无关。

3. **互斥/明显错误疾病（wrong）** 记为 \(y^{(i)}_{\\text{wrong}}\)
   - 与上下文关键信息在医学上**强互斥**，例如：
     - 妊娠并发症出现在老年男性心衰住院；
     - 儿科专属疾病出现在高龄 ICU 住院；
   - 或临床上几乎不可能与当前场景共存的疾病。

每个 context 最终关联一个三元组：
\[
\bigl(y^{(i)}_{\\text{rel}},\ y^{(i)}_{\\text{unrel}},\ y^{(i)}_{\\text{wrong}}\bigr).
\]

### 2.3 探测表结构（概念稿）

后续实现时，可将探测集存为一张小型表（如 csv）：

- 列字段示例：
  - `context_id`：指向具体病例 \(x^{(i)}\)
  - `disease_id`：候选疾病 code
  - `type`：\{`related_rare`, `unrelated_rare`, `wrong`\}
- 行数：\(3 \times N_\\text{ctx}\)。

---

## 3. eNTK 相似度验证（eNTK Similarity Verification）

为避免“仅凭医学常识硬猜桥梁”的质疑，我们在训练初期（\(t\\approx 0\) 或很早的 epoch）对微观探测集做一次 **eNTK 相似度验证**，定量证明：

> 在模型参数层面，相关罕见并发症与主体疾病之间，确实通过梯度/特征存在更强的“桥梁连接”。

### 3.1 梯度相似度的近似计算

在训练初期选取某一 checkpoint（例如 step 0 或第 1 epoch 末），对每个上下文 \(x^{(i)}\)：

- 选定一个“主体疾病”标签 \(y^{(i)}_{\\text{main}}\)（如上下文中的主要基础病之一），
- 对以下四个标签构造 loss 并计算梯度向量（在最后一层 Transformer 或生成头上）：
  - \(g^{(i)}_{\\text{main}} = \\\nabla_\\theta \\ell(x^{(i)}, y^{(i)}_{\\text{main}})\)
  - \(g^{(i)}_{\\text{rel}} = \\\nabla_\\theta \\ell(x^{(i)}, y^{(i)}_{\\text{rel}})\)
  - \(g^{(i)}_{\\text{unrel}} = \\\nabla_\\theta \\ell(x^{(i)}, y^{(i)}_{\\text{unrel}})\)
  - \(g^{(i)}_{\\text{wrong}} = \\\nabla_\\theta \\ell(x^{(i)}, y^{(i)}_{\\text{wrong}})\)

从而得到三类候选疾病相对于主体疾病的梯度余弦相似度：
\[
\\text{sim}^{(i)}_{\\bullet} =
\\frac{\\langle g^{(i)}_{\\text{main}}, g^{(i)}_{\\bullet} \\rangle}
     {\\|g^{(i)}_{\\text{main}}\\|\\,\\|g^{(i)}_{\\bullet}\\|},\\quad \\\bullet\\in\\{\\text{rel},\\text{unrel},\\text{wrong}\\}.
\]

这在形式上等价于 eNTK 矩阵 \(\\mathcal{K}^0(x^{(i)}_{\\text{main}}, y^{(i)}_{\\bullet})\) 的一个经验近似。

### 3.2 后续分析预期

在结果分析阶段，可构造如下散点图：

- X 轴：初始梯度相似度 \(\\text{sim}^{(i)}_{\\bullet}\)，
- Y 轴：退火结束时对应疾病 log-prob 的提升幅度 \(\\Delta \\log p = \\log p_{\\theta_T}(y\\mid x) - \\log p_{\\theta_0}(y\\mid x)\)。

预期现象：

- `related_rare` 样本集中在“初始相似度高、\\Delta\\log p 显著大”的区域；
- `unrelated_rare` 和 `wrong` 更集中在“相似度低、\\Delta\\log p 小甚至负”的区域。

这将从“梯度桥梁”的视角，把医学先验（K 矩阵）、模型内部特征以及概率轨迹三者焊接在一起，缓解“只是用医学直觉解释深度模型”的质疑。

---

## 4. 退火过程中的概率轨迹记录

### 4.1 退火 schedule 与 checkpoint 设计（高密度）

沿用 `EXPERIMENT_fixed_bias_anneal_mimiciv.md` 中的退火策略：

- 偏置权重 \(\\lambda(t)\) 从初值 \(\\lambda_0\) 退火到 0，训练总步数为 \(T\)。
- 因为微观探测集极小（\\(N_\\text{ctx}\\approx 30\\)–\\(50\\)，合计约百级样本），在其上做前向推理的额外开销可以忽略不计。

因此，我们不采用每 10% 训练进度只记录一次的粗粒度方案（共 11 个点），而是在退火阶段设置 **高密度 checkpoint**：

- 例如每隔固定 step（如每 100 个 optimization step，或每 0.5 个 epoch）记录一次，
- 对应 checkpoint 总数 \(K_\\text{chk}\\) 在 **100–500** 区间，根据实际训练步数调整。

这样可得到近似连续的轨迹 \(t \\\\mapsto \\log p_{\\theta_t}(y\\mid x)\)，使曲线视觉效果更接近一阶常微分方程解的平滑流形，而非稀疏折线拼接。

### 4.2 记录的概率/对数概率指标

在每个 checkpoint \(t_k\) 及每个探针 \((x^{(i)}, y^{(i)}_{\\bullet})\) 上，记录如下量：

- 生成头中对应疾病维度的 logit：
  - `logit_{t_k}(i, type) = f_{\\theta_{t_k}, d}(x^{(i)})`；
- 对应的概率与 log-prob（针对独立 Sigmoid 头）：
  - `prob_{t_k}(i, type) = \\sigma(f_{\\theta_{t_k}, d}(x^{(i)}))`；
  - `log_p_{t_k}(i, type) = \\log\\sigma(f_{\\theta_{t_k}, d}(x^{(i)}))`。

后续可对每一类 type 聚合为平均曲线：
\[
\\bar{\\ell}_{\\text{rel}}(t_k) = \\\nfrac{1}{N_\\text{ctx}} \\\n\sum_{i} \\log p_{\\theta_{t_k}}(y^{(i)}_{\\text{rel}}\\mid x^{(i)}),
\]
并类似定义 \\(\\bar{\\ell}_{\\text{unrel}}(t_k)\\)、\\(\\bar{\\ell}_{\\text{wrong}}(t_k)\\)。

### 4.3 Oracle Target（PCLA+LA）水平基准线

为在可视化上验证“退火后概率等效于基础模型 + 静态偏置”的理论结论，我们在单个 context 的轨迹图中引入 **Oracle Target 水平线**：

- 对于每个 related\\_rare 探针 \\((x^{(i)}, y^{(i)}_{\\text{rel}})\\)，额外计算一次：
  - 在同一基础 PCLA 模型下，于推理阶段直接施加 LA 偏置（PCLA+LA），得到目标 log-prob：
    \\
    \\\[ 
    \\log p_{\\text{oracle}}(y^{(i)}_{\\text{rel}} \\mid x^{(i)})
    = \\log p_{\\text{PCLA+LA}}(y^{(i)}_{\\text{rel}} \\mid x^{(i)}).
    \\
    \\
- 在绘制 AdaPCLA 训练过程中的 \\((t_k, \\log p_{\\theta_{t_k}}(y^{(i)}_{\\text{rel}}\\mid x^{(i)}))\\) 轨迹时，同时画一条以 \\log p_{\\text{oracle}} 为高度的**水平虚线**：
  - 这条虚线是“理想终态”的 Oracle 标尺，对应附录中“退火结束 \\Rightarrow 基础模型 + 静态偏置”的闭式解；
  - 预期现象：随着 \\(t/T \\to 1.0\\)，AdaPCLA 的 related\\_rare 轨迹应平滑逼近并贴合这条水平线，从视觉上印证偏置被充分吸收进参数的等效性定理。

### 4.4 预期可视化形式（摘要）

综合以上设计，KDD 主文中预计将展示如下几类图：

1. **单个 context 的三类候选轨迹 + Oracle 线**：
   - x 轴：训练进度（如 normalized step t/T）；
   - y 轴：log-prob 或 logit；
   - 曲线：related\\_rare / unrelated\\_rare / wrong 三条实线；
   - 附加：related\\_rare 的 Oracle 水平虚线（PCLA+LA）。

2. **所有 context 聚合后的平均轨迹**：
   - 三条平滑曲线对应三种 type 的 \\bar{\\ell}(t)；
   - 展示“只有结构上相关的罕见并发症被系统性抬高，其他两类保持低位”的模式。

3. **eNTK 相似度 vs. \\Delta\\log p 散点图**：
   - X 轴：初始梯度相似度 \\text{sim}^{(i)}_{\\bullet}；
   - Y 轴：退火结束时的 \\Delta\\log p；
   - 预期：related\\_rare 点云呈明显正相关趋势，而 unrelated\\_rare / wrong 更集中在相似度低、增益小的区域。

上述图形将与附录中的参数 ODE 推导形成互为支撑的“参数–概率”双视角，从而增强 KDD 主文第 4 章的说服力。

---

## 下一步操作（执行清单）

1. **生成微观探测配置 CSV**（若尚未生成）  
   ```bash
   cd EXPERIMENTS_ROOT/model6
   python gen_micro_probe_config.py --data_dir /path/to/data2 --out mimiciv_long_tail_triplets_seed1.csv
   ```  
   默认会写入 `micro_probe_configs/mimiciv_long_tail_triplets_seed1.csv`。可按需调整 `--n_ctx`、`--seed`，或事后手工编辑 CSV 以更符合“相关罕见 / 无关罕见 / 错误”的医学语义。

2. **重新跑训练 + 微观探测**  
   主流程（训练 + 合成数据）已跑完时，若需要**概率轨迹**（约 100 个 checkpoint 的 log-prob），需在配置就绪后重跑：  
   ```bash
   cd EXPERIMENTS_ROOT/model6
   bash run_micro_probing_fixed_bias_anneal_mimiciv.sh
   ```  
   完成后在 `micro_probe_logs/seed1/` 下会有 `micro_probe_ckpt_0000.csv` … `micro_probe_ckpt_0099.csv`，可用于画单 context 三轨迹、平均轨迹及 eNTK vs. Δlog p 图。

3. **（可选）仅用当前 checkpoint 做一次探测**  
   若不想重训，只要“当前模型在探测集上的概率快照”，可单独写一个小脚本：加载 `save_micro_probe_mimiciv/seed1/model_anneal_mimiciv.pt`，读入上述 CSV，对每个 (context_id, disease_id) 做一次前向，写出单份 CSV 供画图（无时间轨迹）。


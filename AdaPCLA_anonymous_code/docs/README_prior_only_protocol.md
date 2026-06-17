## model4：Prior-only 合成数据对比实验协议（方法草案）

本实验在 `model4` 中构造一个 **“无模型，仅先验 (prior-only)”** 的合成数据基线，用于与 Real / HALO / PCLA / Anneal 等结果对比，检验 **先验本身（logit_adjust / 统计频率）能支撑多高的下游表现**。

目标：**完全不使用 HALO/PCLA 等生成模型参数**，只利用训练集的统计先验生成一个符合 `haloDataset.pkl` 结构的合成数据集，并用现有 `evaluate_synthetic_training.py` 做 25-label 下游评估。

---

### 1. 先验的来源（沿用现有 `compute_logit_adjust`）

1. **数据与配置**
   - 使用与 PCLA 相同的 MIMIC-III 任务数据目录，例如：  
     `DATA_DIR = fame/myfame/baseline/HALO/save`。  
   - 加载：
     - `codeToIndex.pkl` → `code_to_index`（code 词表，大小 `V_code`）  
     - `idToLabel.pkl` → `id_to_label`（25 维 CCS 标签）  
     - `trainDataset.pkl` → `train_data`（列表，每个元素是字典 `{"visits": ..., "labels": ...}`）。
   - 构造 `Model2Config`，并设置：
     - `config.code_vocab_size = len(code_to_index)`  
     - `config.label_vocab_size = len(id_to_label)`  
     - `config.total_vocab_size = code_vocab + label_vocab + special_vocab`。

2. **先验 logit_adjust（code 级别）计算规则**  
   （完全沿用 `compute_logit_adjust(train_data, config)`）

   ```python
   code_vocab = int(config.code_vocab_size)
   total_visits = 0
   visit_counts = np.zeros((code_vocab,), dtype=np.int64)
   for p in train_data:
       visits = p.get("visits", [])
       total_visits += len(visits)
       for v in visits:
           if not v:
               continue
           for c in set(v):
               ci = int(c)
               if 0 <= ci < code_vocab:
                   visit_counts[ci] += 1

   eps = float(config.logit_adjust_eps)     # 1e-8
   pi = visit_counts.astype(np.float64) / float(total_visits)    # 每个 code 在多少个 visit 中出现
   b = np.log((1.0 - pi + eps) / (pi + eps)) * float(config.logit_adjust_tau)  # tau=0.2
   if config.logit_adjust_clip is not None:  # clip=15.0
       b = np.clip(b, -float(config.logit_adjust_clip), float(config.logit_adjust_clip))
   b = np.where(visit_counts > 0, b, 0.0)    # 只对出现过的 code 赋非零先验

   adj = np.zeros((config.total_vocab_size,), dtype=np.float32)
   adj[:code_vocab] = b.astype(np.float32)   # 扩展到 total_vocab_size，其它位置为 0
   ```

   - 先验向量 `b` 的含义：对每个 code 的 **log-odds 先验**，编码了 code 在 visit 中出现的长尾统计。
   - 对应的“先验出现概率”可以用：  
     \[
       p_i = \\sigma(b_i) = \\frac{1}{1+e^{-b_i}}
     \]

3. **label 先验**  
   - 独立于上面的 code 先验，再从 `train_data` 中统计 25 维 CCS 标签的边际频率：

   ```python
   # train_data: List[dict], 每个 p["labels"] 是长度=25 的 0/1 向量
   label_counts = np.zeros((config.label_vocab_size,), dtype=np.int64)
   for p in train_data:
       y = np.asarray(p["labels"], dtype=np.int64)
       label_counts += y

   num_patients = len(train_data)
   label_pi = label_counts.astype(np.float64) / float(num_patients)  # 每个 label 的边际概率
   ```

   - 对于 prior-only 方案，可以简单地把 25 维标签看作 **独立 Bernoulli**：  
     - 第 \(k\) 个 label 出现的概率 \(p_k = label\_pi[k]\)。

---

### 2. Prior-only 合成数据的生成协议（伪代码）

目标：生成一个与原 `haloDataset.pkl` 同结构的列表：

```python
synthetic = [
    {"visits": List[List[int]], "labels": np.ndarray(shape=(25,), dtype=int)},
    ...
]
```

完全不调用 HALO/PCLA 模型，只用上述 code/label 先验 + 简单的 visit 结构统计。

#### 2.1 辅助统计：visit 数分布与 visit 长度分布

从 `train_data` 抽取“病人层级结构”的经验分布，用于保持大致的结构感：

```python
num_visits_list = []      # 每个病人的 visit 数
visit_len_list = []       # 每个 visit 中 code 数（用于粗略控制 visit 稀疏度）

for p in train_data:
    visits = p.get("visits", [])
    num_visits_list.append(len(visits))
    for v in visits:
        visit_len_list.append(len(v))

# 可以记录经验分布或其分位数，用于随机采样
```

#### 2.2 生成单个病人（prior-only）

伪代码：

```python
import numpy as np
import random

def sample_patient_prior_only(
    *,
    code_probs: np.ndarray,     # shape = [V_code], 由 b 通过 sigmoid 得到
    label_probs: np.ndarray,    # shape = [25], 由 label_pi 得到
    num_visits_list: list[int],
    visit_len_list: list[int],
    max_visits: int | None = None,
) -> dict:
    \"\"\"根据先验（无模型）生成一个病人的 visits + labels。\"\"\"
    V_code = code_probs.shape[0]

    # 1) 抽样病人 visit 数量：从经验分布中随机取一个
    num_visits = random.choice(num_visits_list)
    if max_visits is not None:
        num_visits = min(num_visits, max_visits)

    visits_out: list[list[int]] = []

    for _ in range(num_visits):
        # 2) 抽样该 visit 中 code 数量（粗略控制稀疏度）
        visit_len = max(1, random.choice(visit_len_list))

        # 3) 根据 code 先验生成该 visit 的 code 集合
        #    简单做法 A：按 Bernoulli 独立采样，然后截断到指定长度
        flags = np.random.rand(V_code) < code_probs     # shape=[V_code], 布尔向量
        candidate_idxs = np.nonzero(flags)[0].tolist()
        random.shuffle(candidate_idxs)
        visit_codes = candidate_idxs[:visit_len]
        visit_codes = sorted(set(int(c) for c in visit_codes))

        if visit_codes:
            visits_out.append(visit_codes)

    # 4) 生成病人的 25 维 labels（独立 Bernoulli，来自边际先验）
    labels = (np.random.rand(label_probs.shape[0]) < label_probs).astype(np.int32)

    return {\"visits\": visits_out, \"labels\": labels}
```

要点：

- **不依赖任何 Transformer/生成器**：所有随机性只来自 code_probs、label_probs 与经验的结构分布。  
- `code_probs` 建议由 `b` 通过 `sigmoid` 得到：  
  `code_probs = 1.0 / (1.0 + np.exp(-b[:V_code]))`，并可根据需要做截断或重标定。  
- `num_visits_list`、`visit_len_list` 只负责给出大致“有几次就诊、每次几个 code”这类结构信息，不引入学习到的参数。

#### 2.3 生成完整数据集

```python
def generate_prior_only_dataset(
    *,
    train_data,
    config: Model2Config,
    total_samples: int,
) -> list[dict]:
    # 1) 计算 code 先验 logit_adjust
    adj_np, stats = compute_logit_adjust(train_data, config=config)
    V_code = int(config.code_vocab_size)
    b = adj_np[:V_code]
    code_probs = 1.0 / (1.0 + np.exp(-b))

    # 2) 计算 label 边际先验
    label_counts = np.zeros((config.label_vocab_size,), dtype=np.int64)
    for p in train_data:
        label_counts += np.asarray(p[\"labels\"], dtype=np.int64)
    label_probs = label_counts.astype(np.float64) / float(len(train_data))

    # 3) 抽样结构分布
    num_visits_list, visit_len_list = [], []
    for p in train_data:
        visits = p.get(\"visits\", [])
        num_visits_list.append(len(visits))
        for v in visits:
            visit_len_list.append(len(v))

    # 4) 生成合成数据
    synthetic: list[dict] = []
    for _ in range(total_samples):
        synthetic.append(
            sample_patient_prior_only(
                code_probs=code_probs,
                label_probs=label_probs,
                num_visits_list=num_visits_list,
                visit_len_list=visit_len_list,
                max_visits=config.n_ctx,   # 可选：上限控制
            )
        )
    return synthetic
```

生成完成后，只需将 `synthetic` 按与原 PCLA 一致的方式保存为 `haloDataset.pkl`，即可直接复用：

```bash
python evaluate_synthetic_training.py \
  --base_data_dir  <同 HALO/PCLA 的真实数据目录> \
  --mymodel2_path  path/to/model4/save_prior_only/datasets/haloDataset.pkl \
  --save_dir       path/to/model4/evaluate_prior_only/seed1 \
  --sources        MyModel2
```

---

### 3. 小结（写在 README 里便于落地）

- **不使用任何 HALO/PCLA 模型参数**，只用 `trainDataset.pkl` 的统计先验：  
  - code 层面：`compute_logit_adjust` 得到 log-odds 先验，再经 `sigmoid` 转为概率；  
  - label 层面：训练集边际频率转为 Bernoulli 概率。  
- **结构层面**（病人有多少次 visit、每次 visit 大致几个 code）完全来源于训练集的经验分布，不做建模。  
- 按上述协议生成的 `haloDataset.pkl` 可直接接入现有的 25-label 下游评估脚本，得到一个 **Prior-only (no model)** 的基线，与 Real / HALO / PCLA / Anneal 作全面对比。 

